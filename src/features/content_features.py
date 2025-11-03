# ==================== src/features/content_features.py ====================
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from src.features.text_processor import TextProcessor
from src.utils.logging_config import logger

from underthesea import word_tokenize

def vi_tokenizer(text: str):
    """Vietnamese tokenizer using underthesea"""
    return word_tokenize(text, format="text").split()

class ContentBasedModel:
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,    #  Loại từ chỉ xuất hiện 1 lần
            max_df=0.8,     #  Loại từ xuất hiện >80% (stop words)
            strip_accents=None,  # Keep Vietnamese diacritics
        )
        self.feature_matrix = None
        self.book_ids = None
        self.id_to_idx = {}
        
        # User profile components
        self.user_profiles = {}      # {user_id: weighted_avg_vector}
        self.user_interactions = {}  # {user_id: {book_id: strength}}
    
    def fit(self, books_df: pd.DataFrame):
        """Build TF-IDF features"""
        logger.info("Building content-based features...")
        
        # Build documents
        books_df = books_df.copy()
        books_df['document'] = books_df.apply(TextProcessor.build_document, axis=1)
        
        # Fit TF-IDF
        self.feature_matrix = self.vectorizer.fit_transform(books_df['document'])
        self.book_ids = books_df['book_id'].values
        df_tfidf = pd.DataFrame(
            self.feature_matrix.toarray(),
            index=self.book_ids,
            columns=self.vectorizer.get_feature_names_out()
        )
        print(f"TF-IDF DataFrame head:\n{df_tfidf.head()}")
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        logger.info(f"Content features: {self.feature_matrix.shape}")
    
    def get_similar(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get similar books via cosine similarity"""
        if book_id not in self.id_to_idx:
            return []
        
        idx = self.id_to_idx[book_id]
        query_vec = self.feature_matrix[idx]
        
        scores = cosine_similarity(query_vec, self.feature_matrix).flatten()
        
        # Exclude self
        scores[idx] = -1
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.book_ids[i], scores[i]) for i in top_indices if scores[i] > 0]
    
    def build_user_profiles(self, interactions_df: pd.DataFrame):
        """
        Build user profiles from interaction history
        
        User profile = Weighted average of TF-IDF vectors of books they interacted with
        
        Args:
            interactions_df: DataFrame with columns [user_id, book_id, strength]
        """
        logger.info("Building user profiles from interactions...")
        
        # Group interactions by user
        grouped = interactions_df.groupby('user_id')
        
        for user_id, group in grouped:
            # Get books and their strengths
            books_dict = dict(zip(group['book_id'], group['strength']))
            
            # Store interactions
            self.user_interactions[user_id] = books_dict
            
            # Compute weighted average profile
            user_profile = self._compute_user_profile(books_dict)
            
            if user_profile is not None:
                self.user_profiles[user_id] = user_profile
        
        logger.info(f"Built profiles for {len(self.user_profiles)} users")
    
    def _compute_user_profile(self, books_dict: Dict[int, float]) -> Optional[np.ndarray]:
        """
        Compute weighted average TF-IDF vector for a user
        
        Args:
            books_dict: {book_id: strength} mapping
        
        Returns:
            Weighted average vector or None if no valid books
        """
        vectors = []
        weights = []
        
        for book_id, strength in books_dict.items():
            if book_id in self.id_to_idx:
                idx = self.id_to_idx[book_id]
                vectors.append(self.feature_matrix[idx].toarray().flatten())
                weights.append(strength)
        
        if not vectors:
            return None
        
        # Weighted average
        vectors = np.array(vectors)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize to sum=1
        
        user_profile = np.average(vectors, axis=0, weights=weights)
        
        return user_profile
    
    def recommend_for_user(self, user_id: int, top_k: int = 10,
                          filter_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Get personalized recommendations for a user based on their profile
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_items: Set of book IDs to exclude (already interacted)
        
        Returns:
            List of (book_id, score) tuples
        """
        if user_id not in self.user_profiles:
            logger.warning(f"User {user_id} has no profile")
            return []
        
        # Get user profile vector
        user_vector = self.user_profiles[user_id].reshape(1, -1)
        
        # Compute similarity with all books
        scores = cosine_similarity(user_vector, self.feature_matrix).flatten()
        
        # Filter out already interacted items
        if filter_items:
            for book_id in filter_items:
                if book_id in self.id_to_idx:
                    idx = self.id_to_idx[book_id]
                    scores[idx] = -1
        
        # Get top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.book_ids[idx], scores[idx]))
        
        return results
    
    def get_profile_keywords(self, user_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top keywords from user's profile (for explainability)
        
        Args:
            user_id: User ID
            top_n: Number of keywords
        
        Returns:
            List of (keyword, weight) tuples
        """
        if user_id not in self.user_profiles:
            return []
        
        user_vector = self.user_profiles[user_id]
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features by weight
        top_indices = np.argsort(user_vector)[::-1][:top_n]
        
        keywords = []
        for idx in top_indices:
            if user_vector[idx] > 0:
                keywords.append((feature_names[idx], user_vector[idx]))
        
        return keywords
    
    def get_similar_items(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Alias for get_similar() for consistency with other models
        """
        return self.get_similar(book_id, top_k)
    
    def update_user_profile(self, user_id: int, book_id: int, strength: float = 1.0):
        """
        Incrementally update user profile with new interaction
        
        Args:
            user_id: User ID
            book_id: Book ID
            strength: Interaction strength
        """
        # Update interactions
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = {}
        
        self.user_interactions[user_id][book_id] = strength
        
        # Recompute profile
        user_profile = self._compute_user_profile(self.user_interactions[user_id])
        
        if user_profile is not None:
            self.user_profiles[user_id] = user_profile
            logger.debug(f"Updated profile for user {user_id}")

    
    def save(self, path: Path):
        """Save vectorizer and features"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'feature_matrix': self.feature_matrix,
                'book_ids': self.book_ids,
                'id_to_idx': self.id_to_idx,
                'user_profiles': self.user_profiles,
                'user_interactions': self.user_interactions
            }, f)
        logger.info(f"Saved content model to {path}")
    
    @classmethod
    def load(cls, path: Path):
        """Load saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.vectorizer = data['vectorizer']
        model.feature_matrix = data['feature_matrix']
        model.book_ids = data['book_ids']
        model.id_to_idx = data['id_to_idx']
        model.user_profiles = data.get('user_profiles', {})
        model.user_interactions = data.get('user_interactions', {})
        return model

# # ==================== src/features/content_features.py ====================
# from typing import Dict, List, Optional, Tuple
# import numpy as np
# import pandas as pd
# import pickle
# from pathlib import Path
# import torch
# from sentence_transformers import SentenceTransformer, util
# from src.features.text_processor import TextProcessor
# from src.utils.logging_config import logger
# from underthesea import word_tokenize

# def vi_tokenizer(text: str):
#     """Vietnamese tokenizer using underthesea"""
#     return word_tokenize(text, format="text").split()


# class ContentBasedModel:
#     def __init__(
#         self,
#         model_name: str = "keepitreal/vietnamese-sbert",
#         batch_size: int = 32,
#     ):
#         self.model_name = model_name
#         self.batch_size = batch_size
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model: Optional[SentenceTransformer] = None
#         self.embeddings: Optional[torch.Tensor] = None
#         self.feature_matrix: Optional[torch.Tensor] = None
#         self.book_ids: Optional[np.ndarray] = None
#         self.id_to_idx: Dict[int, int] = {}

#     def fit(self, books_df: pd.DataFrame):
#         """Build SBERT features"""
#         logger.info("Building content-based embeddings...")
#         self._ensure_model_loaded()

#         # Build documents
#         books_df = books_df.copy()
#         books_df["document"] = books_df.apply(TextProcessor.build_document, axis=1)

#         # Encode documents to dense embeddings
#         documents = books_df["document"].fillna("").astype(str).tolist()
#         embeddings_np = self.model.encode(
#             documents,
#             batch_size=self.batch_size,
#             show_progress_bar=False,
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#         )
#         self.embeddings = torch.from_numpy(embeddings_np).to(self.device)
#         self.feature_matrix = self.embeddings
#         self.book_ids = books_df["book_id"].astype(int).values
#         self.id_to_idx = {int(bid): idx for idx, bid in enumerate(self.book_ids)}

#         logger.info(f"Content embeddings: {self.embeddings.shape}")
#         logger.info(f"Content features: {self.embeddings.shape}")

#     def get_similar(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
#         """Get similar books via cosine similarity"""
#         if self.embeddings is None or book_id not in self.id_to_idx:
#             return []

#         idx = self.id_to_idx[book_id]
#         query_vec = self.embeddings[idx].unsqueeze(0)

#         scores = util.cos_sim(query_vec, self.embeddings).flatten()

#         # Exclude self
#         scores[idx] = -1.0

#         top_k = min(top_k, scores.shape[0])
#         if top_k <= 0:
#             return []

#         top_values, top_indices = torch.topk(scores, k=top_k)
#         similar_items = []
#         for score, index in zip(top_values.tolist(), top_indices.tolist()):
#             if score <= 0:
#                 continue
#             similar_items.append((int(self.book_ids[index]), float(score)))

#         return similar_items

#     def save(self, path: Path):
#         """Save embeddings and metadata"""
#         path.parent.mkdir(parents=True, exist_ok=True)
#         with open(path, "wb") as f:
#             pickle.dump(
#                 {
#                     "model_name": self.model_name,
#                     "embeddings": self.embeddings.detach().cpu().numpy()
#                     if self.embeddings is not None
#                     else None,
#                     "book_ids": self.book_ids,
#                     "id_to_idx": self.id_to_idx,
#                     "batch_size": self.batch_size,
#                 },
#                 f,
#             )
#         logger.info(f"Saved content model to {path}")

#     @classmethod
#     def load(cls, path: Path):
#         """Load saved model"""
#         with open(path, "rb") as f:
#             data = pickle.load(f)

#         model = cls(
#             model_name=data.get("model_name", "keepitreal/vietnamese-sbert"),
#             batch_size=data.get("batch_size", 32),
#         )
#         embeddings = data.get("embeddings")
#         if embeddings is not None:
#             model.embeddings = torch.from_numpy(embeddings).to(model.device)
#             model.feature_matrix = model.embeddings
#         model.book_ids = data.get("book_ids")
#         model.id_to_idx = data.get("id_to_idx", {})
#         return model

#     def _ensure_model_loaded(self):
#         """Lazily load the SBERT model to the appropriate device"""
#         if self.model is None:
#             logger.info(
#                 f"Loading SentenceTransformer model '{self.model_name}' on {self.device}"
#             )
#             self.model = SentenceTransformer(self.model_name, device=self.device)
