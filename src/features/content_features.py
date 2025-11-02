"""
Content-Based Filtering with User Profiles
"""
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
    """
    Content-Based Recommender with User Profiles
    
    Features:
    1. TF-IDF vectorization of book content
    2. User profile construction from interaction history
    3. Recommendations via user-item similarity
    """
    
    def __init__(self, max_features: int = 10000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=vi_tokenizer,
            preprocessor=None,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
        )
        self.feature_matrix = None  # Item features (N_items × N_features)
        self.book_ids = None
        self.id_to_idx = {}
        
        # User profiles
        self.user_profiles = {}  # {user_id: profile_vector}
        self.user_interactions = {}  # {user_id: {book_id: strength}}
    
    def fit(self, books_df: pd.DataFrame):
        """Build TF-IDF features for all books"""
        logger.info("Building content-based features...")
        
        # Build documents
        books_df = books_df.copy()
        books_df['document'] = books_df.apply(TextProcessor.build_document, axis=1)
        
        # Fit TF-IDF
        self.feature_matrix = self.vectorizer.fit_transform(books_df['document'])
        self.book_ids = books_df['book_id'].values
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        # Log sample
        result = vi_tokenizer(books_df['document'].iloc[0])
        logger.debug(f"Sample tokens: {result[:10]}")
        
        df_tfidf = pd.DataFrame(
            self.feature_matrix.toarray(),
            index=self.book_ids,
            columns=self.vectorizer.get_feature_names_out()
        )
        logger.debug(f"TF-IDF shape: {df_tfidf.shape}")
        
        logger.info(f"Content features: {self.feature_matrix.shape}")
    
    def build_user_profiles(self, interactions_df: pd.DataFrame):
        """
        Build user profiles from interaction history
        
        User profile = weighted average of interacted items' features
        Weight = interaction strength (rating, view duration, etc.)
        
        Args:
            interactions_df: DataFrame with [user_id, book_id, strength]
        """
        logger.info("Building user profiles from interactions...")
        
        # Store user interactions
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            book_id = row['book_id']
            strength = row.get('strength', 1.0)
            
            if user_id not in self.user_interactions:
                self.user_interactions[user_id] = {}
            
            # Aggregate strengths for same book
            if book_id in self.user_interactions[user_id]:
                self.user_interactions[user_id][book_id] += strength
            else:
                self.user_interactions[user_id][book_id] = strength
        
        # Build profiles
        for user_id, books_dict in self.user_interactions.items():
            profile = self._compute_user_profile(books_dict)
            if profile is not None:
                self.user_profiles[user_id] = profile
        
        logger.info(f"Built {len(self.user_profiles)} user profiles")
    
    def _compute_user_profile(self, books_dict: Dict[int, float]) -> Optional[np.ndarray]:
        """
        Compute user profile as weighted average of item features
        
        profile = Σ(strength_i × item_features_i) / Σ(strength_i)
        
        Args:
            books_dict: {book_id: strength}
        
        Returns:
            User profile vector (1 × N_features)
        """
        valid_books = []
        valid_strengths = []
        
        for book_id, strength in books_dict.items():
            if book_id in self.id_to_idx:
                valid_books.append(book_id)
                valid_strengths.append(strength)
        
        if not valid_books:
            return None
        
        # Get item feature vectors
        indices = [self.id_to_idx[bid] for bid in valid_books]
        item_features = self.feature_matrix[indices].toarray()  # (N_books × N_features)
        
        # Weighted average
        strengths = np.array(valid_strengths).reshape(-1, 1)  # (N_books × 1)
        weighted_features = item_features * strengths  # Element-wise multiplication
        
        profile = weighted_features.sum(axis=0) / strengths.sum()  # (N_features,)
        
        # Normalize to unit vector
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
        
        return profile
    
    def update_user_profile(self, user_id: int, book_id: int, strength: float = 1.0):
        """
        Update user profile with new interaction (for online learning)
        
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
    
    def recommend_for_user(self, user_id: int, top_k: int = 10, 
                          filter_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Recommend items for user based on their profile
        
        Computes similarity between user profile and all items:
        similarity(user, item) = cosine(user_profile, item_features)
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_items: Set of book IDs to exclude (e.g., already interacted)
        
        Returns:
            List of (book_id, score) tuples
        """
        if user_id not in self.user_profiles:
            logger.debug(f"No profile for user {user_id}, cannot recommend")
            return []
        
        user_profile = self.user_profiles[user_id].reshape(1, -1)  # (1 × N_features)
        
        # Compute similarity with all items
        scores = cosine_similarity(user_profile, self.feature_matrix).flatten()  # (N_items,)
        
        # Filter out already interacted items (unless explicitly allowed)
        if filter_items:
            for book_id in filter_items:
                if book_id in self.id_to_idx:
                    idx = self.id_to_idx[book_id]
                    scores[idx] = -1  # Exclude
        else:
            # Default: filter out already interacted
            interacted = self.user_interactions.get(user_id, {}).keys()
            for book_id in interacted:
                if book_id in self.id_to_idx:
                    idx = self.id_to_idx[book_id]
                    scores[idx] = -1
        
        # Get top K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only positive scores
                book_id = self.book_ids[idx]
                results.append((book_id, float(scores[idx])))
        
        return results
    
    def get_similar_items(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar items (item-to-item similarity)
        
        This is the OLD method (still useful for "similar books" feature)
        
        Args:
            book_id: Reference book ID
            top_k: Number of similar books
        
        Returns:
            List of (book_id, score) tuples
        """
        if book_id not in self.id_to_idx:
            return []
        
        idx = self.id_to_idx[book_id]
        query_vec = self.feature_matrix[idx]
        
        scores = cosine_similarity(query_vec, self.feature_matrix).flatten()
        
        # Exclude self
        scores[idx] = -1
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.book_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
    
    def get_profile_keywords(self, user_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top keywords from user profile (for explainability)
        
        Args:
            user_id: User ID
            top_n: Number of keywords
        
        Returns:
            List of (keyword, weight) tuples
        """
        if user_id not in self.user_profiles:
            return []
        
        profile = self.user_profiles[user_id]
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features
        top_indices = np.argsort(profile)[::-1][:top_n]
        
        keywords = []
        for idx in top_indices:
            if profile[idx] > 0:
                keywords.append((feature_names[idx], float(profile[idx])))
        
        return keywords
    
    def save(self, path: Path):
        """Save model including user profiles"""
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



#Sử dụng item-item
# # ==================== src/features/content_features.py ====================
# from typing import List, Tuple
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import pandas as pd
# import pickle
# from pathlib import Path
# from src.features.text_processor import TextProcessor
# from src.utils.logging_config import logger
# from underthesea import word_tokenize

# def vi_tokenizer(text: str):
#     # Trả về list token đã tách từ, có thể nối cụm bằng dấu gạch dưới
#     return word_tokenize(text, format="text").split()

# class ContentBasedModel:
#     def __init__(self, max_features: int = 10000):
#         self.vectorizer = TfidfVectorizer(
#             tokenizer=vi_tokenizer,
#             preprocessor=None,          # bỏ preprocessor mặc định
#             token_pattern=None,         # bắt buộc khi dùng tokenizer tùy biến
#             # max_features=max_features,
#             ngram_range=(1, 2),
#             min_df=2,    #  Loại từ chỉ xuất hiện 1 lần
#             max_df=0.8,     #  Loại từ xuất hiện >80% (stop words)
#             # strip_accents=None,  # Keep Vietnamese diacritics
#         )
#         self.feature_matrix = None
#         self.book_ids = None
#         self.id_to_idx = {}
    
#     def fit(self, books_df: pd.DataFrame):
#         """Build TF-IDF features"""
#         logger.info("Building content-based features...")
        
#         # Build documents
#         books_df = books_df.copy()
#         books_df['document'] = books_df.apply(TextProcessor.build_document, axis=1)
        
#         # Fit TF-IDF
#         self.feature_matrix = self.vectorizer.fit_transform(books_df['document'])
#         result = vi_tokenizer(books_df['document'].iloc[0])
#         print(f"Tokens of first document: {result}")
#         self.book_ids = books_df['book_id'].values
#         df_tfidf = pd.DataFrame(
#             self.feature_matrix.toarray(),
#             index=self.book_ids,
#             columns=self.vectorizer.get_feature_names_out()
#         )
#         print(f"TF-IDF DataFrame head:\n{df_tfidf.head()}")
#         self.id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
#         logger.info(f"Content features: {self.feature_matrix.shape}")
    
#     def get_similar(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
#         """Get similar books via cosine similarity"""
#         if book_id not in self.id_to_idx:
#             return []
        
#         idx = self.id_to_idx[book_id]
#         query_vec = self.feature_matrix[idx]
        
#         scores = cosine_similarity(query_vec, self.feature_matrix).flatten()
        
#         # Exclude self
#         scores[idx] = -1
        
#         top_indices = np.argsort(scores)[::-1][:top_k]
#         return [(self.book_ids[i], scores[i]) for i in top_indices if scores[i] > 0]
    
#     def save(self, path: Path):
#         """Save vectorizer and features"""
#         path.parent.mkdir(parents=True, exist_ok=True)
#         with open(path, 'wb') as f:
#             pickle.dump({
#                 'vectorizer': self.vectorizer,
#                 'feature_matrix': self.feature_matrix,
#                 'book_ids': self.book_ids,
#                 'id_to_idx': self.id_to_idx
#             }, f)
#         logger.info(f"Saved content model to {path}")
    
#     @classmethod
#     def load(cls, path: Path):
#         """Load saved model"""
#         with open(path, 'rb') as f:
#             data = pickle.load(f)
        
#         model = cls()
#         model.vectorizer = data['vectorizer']
#         model.feature_matrix = data['feature_matrix']
#         model.book_ids = data['book_ids']
#         model.id_to_idx = data['id_to_idx']
#         return model