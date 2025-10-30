import implicit
import scipy.sparse as sp
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from src.utils.logging_config import logger

class CollaborativeModel:
    def __init__(self, factors: int = 64, iterations: int = 30, regularization: float = 0.01):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=42
        )
        self.user_item_matrix = None
        self.user_ids = None
        self.item_ids = None
        self.user_id_map = {}
        self.item_id_map = {}
        self.item_id_reverse = {}
    
    def fit(self, interactions_df: pd.DataFrame):
        """Build user-item matrix and train ALS"""
        logger.info("Training collaborative filtering model...")
        
        # Aggregate interactions (sum strengths per user-book)
        agg = interactions_df.groupby(['user_id', 'book_id'])['strength'].sum().reset_index()
        
        # Map IDs to matrix indices
        self.user_ids = sorted(agg['user_id'].unique())
        self.item_ids = sorted(agg['book_id'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.item_id_reverse = {idx: iid for iid, idx in self.item_id_map.items()}
        
        # Build sparse matrix
        rows = agg['user_id'].map(self.user_id_map).values
        cols = agg['book_id'].map(self.item_id_map).values
        data = agg['strength'].values
        
        self.user_item_matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )
        
        # Train (implicit expects item-user format)
        self.model.fit(self.user_item_matrix.T.tocsr())
        
        logger.info(f"CF matrix: {self.user_item_matrix.shape}, nnz={self.user_item_matrix.nnz}")
    
    def recommend(self, user_id: int, top_k: int = 10, filter_items: set = None) -> List[Tuple[int, float]]:
        """Get recommendations for a user"""
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Calculate safe N value
        # Get number of items user has already interacted with
        user_items = self.user_item_matrix[user_idx].nonzero()[1]
        num_interacted = len(user_items)
        num_available = len(self.item_ids) - num_interacted
        
        # Can't recommend more than available items
        safe_n = min(top_k, num_available)
        
        if safe_n <= 0:
            return []
        
        try:
            # Get recommendations
            ids, scores = self.model.recommend(
                user_idx,
                self.user_item_matrix[user_idx],
                N=safe_n,
                filter_already_liked_items=True
            )
        except IndexError as e:
            # If still IndexError, return empty (model needs retraining)
            logger.warning(f"IndexError in recommend for user {user_id}: {e}. Model may need retraining.")
            return []
        
        # Map back to book IDs
        results = []
        for idx, score in zip(ids, scores):
            book_id = self.item_id_reverse[idx]
            if filter_items and book_id in filter_items:
                continue
            results.append((book_id, float(score)))
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: Path):
        """Save model"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_item_matrix': self.user_item_matrix,
                'user_ids': self.user_ids,
                'item_ids': self.item_ids,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'item_id_reverse': self.item_id_reverse
            }, f)
        logger.info(f"Saved CF model to {path}")
    
    @classmethod
    def load(cls, path: Path):
        """Load saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.model = data['model']
        model.user_item_matrix = data['user_item_matrix']
        model.user_ids = data['user_ids']
        model.item_ids = data['item_ids']
        model.user_id_map = data['user_id_map']
        model.item_id_map = data['item_id_map']
        model.item_id_reverse = data['item_id_reverse']
        return model