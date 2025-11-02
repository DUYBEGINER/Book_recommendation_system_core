"""
Content-Based Filtering with Ridge Regression (MLCB approach)

User profile learning: Ŷ = XW + b
- X: Item features (TF-IDF)
- W: User weights (learned via Ridge)
- b: Bias term
"""
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from src.features.text_processor import TextProcessor
from src.utils.logging_config import logger
from underthesea import word_tokenize

def vi_tokenizer(text: str):
    """Vietnamese tokenizer"""
    return word_tokenize(text, format="text").split()

class RidgeContentModel:
    """
    Content-Based Recommender using Ridge Regression
    
    For each user n, learn a Ridge model:
        rating = X @ w_n + b_n
    
    Where:
    - X: Item features (TF-IDF) - shape (n_items, n_features)
    - w_n: User n's weights - shape (n_features,)
    - b_n: User n's bias - scalar
    
    Prediction:
        ŷ_ui = x_i^T @ w_u + b_u
    """
    
    def __init__(self, alpha: float = 1.0, max_features: int = 10000):
        """
        Args:
            alpha: Ridge regularization strength (higher = more regularization)
            max_features: Max TF-IDF features
        """
        self.alpha = alpha
        
        # TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=vi_tokenizer,
            preprocessor=None,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            max_features=max_features
        )
        
        # Item features
        self.feature_matrix = None  # (n_items, n_features)
        self.book_ids = None
        self.id_to_idx = {}
        
        # User models
        self.user_models = {}  # {user_id: Ridge model}
        self.user_interactions = {}  # {user_id: {book_id: strength}}
        
        # Scaler (optional, for better convergence)
        self.scaler = StandardScaler(with_mean=False)  # Sparse-safe
    
    def fit(self, books_df: pd.DataFrame):
        """Build TF-IDF features for all books"""
        logger.info("Building TF-IDF features...")
        
        # Build documents
        books_df = books_df.copy()
        books_df['document'] = books_df.apply(TextProcessor.build_document, axis=1)
        
        # Fit TF-IDF
        self.feature_matrix = self.vectorizer.fit_transform(books_df['document'])
        
        # Scale features (optional but recommended)
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
        
        self.book_ids = books_df['book_id'].values
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.book_ids)}
        
        logger.info(f"TF-IDF features: {self.feature_matrix.shape}")
    
    def train_user_models(self, interactions_df: pd.DataFrame, 
                         min_interactions: int = 5):
        """
        Train Ridge regression model for each user
        
        For user n with interactions on items I_n:
            X_n = feature_matrix[I_n]  # (|I_n|, n_features)
            y_n = ratings[I_n]          # (|I_n|,)
            
            Ridge fit: w_n, b_n = argmin ||y_n - (X_n @ w_n + b_n)||² + α||w_n||²
        
        Args:
            interactions_df: DataFrame with [user_id, book_id, strength]
            min_interactions: Minimum interactions required to train model
        """
        logger.info("Training Ridge models for users...")
        
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
        
        # Train Ridge model for each user
        trained = 0
        skipped = 0
        
        for user_id, books_dict in self.user_interactions.items():
            if len(books_dict) < min_interactions:
                skipped += 1
                continue
            
            # Prepare training data for this user
            X_user, y_user = self._prepare_user_training_data(books_dict)
            
            if X_user is None:
                skipped += 1
                continue
            
            # Train Ridge regression
            model = Ridge(alpha=self.alpha, fit_intercept=True, random_state=42)
            model.fit(X_user, y_user)
            
            self.user_models[user_id] = model
            trained += 1
        
        logger.info(f"Trained Ridge models for {trained} users (skipped {skipped} with < {min_interactions} interactions)")
    
    def _prepare_user_training_data(self, books_dict: Dict[int, float]):
        """
        Prepare training data for a single user
        
        Args:
            books_dict: {book_id: strength}
        
        Returns:
            X: Item features (n_interactions, n_features)
            y: Ratings (n_interactions,)
        """
        valid_books = []
        valid_strengths = []
        
        for book_id, strength in books_dict.items():
            if book_id in self.id_to_idx:
                valid_books.append(book_id)
                valid_strengths.append(strength)
        
        if not valid_books:
            return None, None
        
        # Get item features
        indices = [self.id_to_idx[bid] for bid in valid_books]
        X = self.feature_matrix[indices].toarray()  # (n_books, n_features)
        y = np.array(valid_strengths)  # (n_books,)
        
        return X, y
    
    def recommend_for_user(self, user_id: int, top_k: int = 10,
                          filter_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        Recommend items for user using learned Ridge model
        
        For each item i:
            score(u, i) = x_i^T @ w_u + b_u
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_items: Set of book IDs to exclude
        
        Returns:
            List of (book_id, predicted_rating) tuples
        """
        if user_id not in self.user_models:
            logger.debug(f"No Ridge model for user {user_id} - cold start user")
            # Return empty to let hybrid model handle fallback
            return []
        
        model = self.user_models[user_id]
        
        # Predict ratings for all items
        # X_all: (n_items, n_features)
        # predictions: (n_items,)
        X_all = self.feature_matrix.toarray()
        predictions = model.predict(X_all)
        
        # Log prediction stats
        logger.debug(f"Ridge predictions for user {user_id}: "
                    f"min={predictions.min():.2f}, "
                    f"max={predictions.max():.2f}, "
                    f"mean={predictions.mean():.2f}")
        
        # Filter out interacted items
        if filter_items:
            for book_id in filter_items:
                if book_id in self.id_to_idx:
                    idx = self.id_to_idx[book_id]
                    predictions[idx] = -np.inf
        else:
            # Default: filter already interacted
            interacted = self.user_interactions.get(user_id, {}).keys()
            for book_id in interacted:
                if book_id in self.id_to_idx:
                    idx = self.id_to_idx[book_id]
                    predictions[idx] = -np.inf
        
        # Get top K
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if predictions[idx] > -np.inf:
                book_id = self.book_ids[idx]
                score = float(predictions[idx])
                results.append((book_id, score))
        
        logger.debug(f"Ridge returned {len(results)} recommendations for user {user_id}")
        
        return results
    
    def get_user_weights(self, user_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top feature weights for user (for explainability)
        
        Args:
            user_id: User ID
            top_n: Number of top features
        
        Returns:
            List of (feature_name, weight) tuples
        """
        if user_id not in self.user_models:
            return []
        
        model = self.user_models[user_id]
        weights = model.coef_  # (n_features,)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top positive weights
        top_indices = np.argsort(weights)[::-1][:top_n]
        
        top_features = []
        for idx in top_indices:
            if weights[idx] > 0:
                top_features.append((feature_names[idx], float(weights[idx])))
        
        return top_features
    
    def get_similar_items(self, book_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Item-to-item similarity (fallback for when no user model)
        Uses cosine similarity on TF-IDF features
        """
        if book_id not in self.id_to_idx:
            return []
        
        idx = self.id_to_idx[book_id]
        query_vec = self.feature_matrix[idx]
        
        scores = cosine_similarity(query_vec, self.feature_matrix).flatten()
        scores[idx] = -1  # Exclude self
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(self.book_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
    
    def update_user_model(self, user_id: int, book_id: int, strength: float = 1.0):
        """
        Update user model with new interaction (online learning)
        
        Simple approach: Retrain Ridge model with new data point
        """
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = {}
        
        if book_id in self.user_interactions[user_id]:
            self.user_interactions[user_id][book_id] += strength
        else:
            self.user_interactions[user_id][book_id] = strength
        
        # Retrain model if enough data
        if len(self.user_interactions[user_id]) >= 5:
            X_user, y_user = self._prepare_user_training_data(
                self.user_interactions[user_id]
            )
            
            if X_user is not None:
                model = Ridge(alpha=self.alpha, fit_intercept=True, random_state=42)
                model.fit(X_user, y_user)
                self.user_models[user_id] = model
                logger.debug(f"Updated Ridge model for user {user_id}")
    
    def save(self, path: Path):
        """Save model"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'feature_matrix': self.feature_matrix,
                'scaler': self.scaler,
                'book_ids': self.book_ids,
                'id_to_idx': self.id_to_idx,
                'user_models': self.user_models,
                'user_interactions': self.user_interactions,
                'alpha': self.alpha
            }, f)
        logger.info(f"Saved Ridge content model to {path}")
    
    @classmethod
    def load(cls, path: Path):
        """Load saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(alpha=data.get('alpha', 1.0))
        model.vectorizer = data['vectorizer']
        model.feature_matrix = data['feature_matrix']
        model.scaler = data['scaler']
        model.book_ids = data['book_ids']
        model.id_to_idx = data['id_to_idx']
        model.user_models = data.get('user_models', {})
        model.user_interactions = data.get('user_interactions', {})
        
        return model