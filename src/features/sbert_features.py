"""
SBERT (Sentence-BERT) Content-Based Features
Uses pre-trained Vietnamese SBERT for semantic embeddings

Advantages over TF-IDF:
1. Captures semantic meaning (not just keywords)
2. Better handling of synonyms and paraphrases
3. Dense embeddings (384-768 dim vs 10000 sparse)
4. Pre-trained on Vietnamese corpus
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import pickle
from src.features.text_processor import TextProcessor
from src.utils.logging_config import logger


class SBERTContentModel:
    """
    Content-Based Recommender using Sentence-BERT
    
    Process:
    1. Encode book descriptions with SBERT → dense embeddings
    2. Build user profiles: weighted average of interacted book embeddings
    3. Recommend: cosine similarity between user profile and all items
    """
    
    def __init__(self, model_name: str = 'keepitreal/vietnamese-sbert',
                 device: str = None):
        """
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.model_name = model_name
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading SBERT model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Item features
        self.embeddings = None  # (n_items, embedding_dim)
        self.book_ids = None
        self.id_to_idx = {}
        
        # User profiles
        self.user_profiles = {}  # {user_id: profile_embedding}
        self.user_interactions = {}  # {user_id: {book_id: strength}}
    
    def fit(self, books_df: pd.DataFrame, batch_size: int = 32):
        """
        Build SBERT embeddings for all books
        
        Args:
            books_df: DataFrame with book metadata
            batch_size: Encoding batch size
        """
        logger.info("Building SBERT embeddings...")
        
        # Build documents
        books_df = books_df.copy()
        books_df['document'] = books_df.apply(TextProcessor.build_document, axis=1)
        
        # Encode documents
        documents = books_df['document'].fillna('').tolist()
        
        logger.info(f"Encoding {len(documents)} documents...")
        self.embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        self.book_ids = books_df['book_id'].values
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        logger.info(f"SBERT embeddings: {self.embeddings.shape}")
    
    def build_user_profiles(self, interactions_df: pd.DataFrame):
        """
        Build user profile as weighted average of interacted book embeddings
        
        profile_u = Σ(strength_i × embedding_i) / Σ(strength_i)
        
        Args:
            interactions_df: DataFrame with [user_id, book_id, strength]
        """
        logger.info("Building user profiles from interactions...")
        
        # Store interactions
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            book_id = row['book_id']
            strength = row.get('strength', 1.0)
            
            if user_id not in self.user_interactions:
                self.user_interactions[user_id] = {}
            
            # Aggregate (sum) strengths for same book
            if book_id in self.user_interactions[user_id]:
                self.user_interactions[user_id][book_id] += strength
            else:
                self.user_interactions[user_id][book_id] = strength
        
        # Build profiles
        for user_id, books_dict in self.user_interactions.items():
            profile = self._compute_user_profile(books_dict)
            if profile is not None:
                self.user_profiles[user_id] = profile
        
        logger.info(f"Built profiles for {len(self.user_profiles)} users")
    
    def _compute_user_profile(self, books_dict: Dict[int, float]) -> Optional[np.ndarray]:
        """
        Compute user profile as weighted average of book embeddings
        
        Args:
            books_dict: {book_id: strength}
        
        Returns:
            profile_embedding or None
        """
        valid_embeddings = []
        valid_strengths = []
        
        for book_id, strength in books_dict.items():
            if book_id in self.id_to_idx:
                idx = self.id_to_idx[book_id]
                valid_embeddings.append(self.embeddings[idx])
                valid_strengths.append(strength)
        
        if not valid_embeddings:
            return None
        
        # Weighted average
        embeddings_array = np.array(valid_embeddings)  # (n_books, dim)
        strengths_array = np.array(valid_strengths).reshape(-1, 1)  # (n_books, 1)
        
        profile = (embeddings_array * strengths_array).sum(axis=0) / strengths_array.sum()
        
        # L2 normalize
        profile = profile / np.linalg.norm(profile)
        
        return profile
    
    def recommend_for_user(self, user_id: int, top_k: int = 10,
                          filter_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Recommend items using cosine similarity to user profile
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_items: Set of book IDs to exclude
        
        Returns:
            List of (book_id, score) tuples
        """
        if user_id not in self.user_profiles:
            logger.debug(f"No profile for user {user_id}")
            return []
        
        # Get user profile
        profile = self.user_profiles[user_id].reshape(1, -1)  # (1, dim)
        
        # Compute cosine similarity to all items
        # Since embeddings are L2-normalized, cosine = dot product
        scores = (self.embeddings @ profile.T).flatten()  # (n_items,)
        
        # Filter out interacted items
        if filter_items:
            for book_id in filter_items:
                if book_id in self.id_to_idx:
                    idx = self.id_to_idx[book_id]
                    scores[idx] = -np.inf
        else:
            # Default: filter already interacted
            interacted = self.user_interactions.get(user_id, {}).keys()
            for book_id in interacted:
                if book_id in self.id_to_idx:
                    idx = self.id_to_idx[book_id]
                    scores[idx] = -np.inf
        
        # Get top K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > -np.inf:
                book_id = self.book_ids[idx]
                score = float(scores[idx])
                results.append((book_id, score))
        
        return results
    
    def get_similar_items(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Item-to-item similarity using SBERT embeddings
        
        Args:
            book_id: Query book ID
            top_k: Number of similar items
        
        Returns:
            List of (book_id, similarity_score) tuples
        """
        if book_id not in self.id_to_idx:
            return []
        
        idx = self.id_to_idx[book_id]
        query_embedding = self.embeddings[idx].reshape(1, -1)
        
        # Cosine similarity (dot product for normalized vectors)
        scores = (self.embeddings @ query_embedding.T).flatten()
        scores[idx] = -1  # Exclude self
        
        # Get top K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.book_ids[i], float(scores[i])) 
                for i in top_indices if scores[i] > 0]
    
    def get_profile_keywords(self, user_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get semantic keywords for user profile (for explainability)
        
        Note: SBERT doesn't have explicit keywords like TF-IDF
        Returns top books user interacted with instead
        
        Args:
            user_id: User ID
            top_n: Number of items
        
        Returns:
            List of (book_title, strength) tuples
        """
        if user_id not in self.user_interactions:
            return []
        
        books_dict = self.user_interactions[user_id]
        
        # Sort by strength
        sorted_books = sorted(books_dict.items(), 
                            key=lambda x: x[1], reverse=True)[:top_n]
        
        # Format as (book_id_str, strength)
        return [(f"book_{book_id}", float(strength)) 
                for book_id, strength in sorted_books]
    
    def update_user_profile(self, user_id: int, book_id: int, strength: float = 1.0):
        """
        Update user profile with new interaction
        
        Args:
            user_id: User ID
            book_id: Book ID
            strength: Interaction strength
        """
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = {}
        
        if book_id in self.user_interactions[user_id]:
            self.user_interactions[user_id][book_id] += strength
        else:
            self.user_interactions[user_id][book_id] = strength
        
        # Recompute profile
        profile = self._compute_user_profile(self.user_interactions[user_id])
        if profile is not None:
            self.user_profiles[user_id] = profile
            logger.debug(f"Updated profile for user {user_id}")
    
    def save(self, path: Path):
        """Save model"""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'embeddings': self.embeddings,
                'book_ids': self.book_ids,
                'id_to_idx': self.id_to_idx,
                'user_profiles': self.user_profiles,
                'user_interactions': self.user_interactions
            }, f)
        
        logger.info(f"Saved SBERT content model to {path}")
    
    @classmethod
    def load(cls, path: Path, device: str = None):
        """Load saved model from trusted checkpoint"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(model_name=data['model_name'], device=device)
        model.embeddings = data['embeddings']
        model.book_ids = data['book_ids']
        model.id_to_idx = data['id_to_idx']
        model.user_profiles = data['user_profiles']
        model.user_interactions = data['user_interactions']
        
        logger.info(f"Loaded SBERT content model from {path}")
        return model
