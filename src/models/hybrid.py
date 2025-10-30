from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.collaborative import CollaborativeModel
from src.features.content_features import ContentBasedModel
from src.utils.logging_config import logger

class HybridRecommender:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha  # CF weight
        self.cf_model: Optional[CollaborativeModel] = None
        self.content_model: Optional[ContentBasedModel] = None
        self.books_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Train both models"""
        logger.info("Training hybrid recommender...")
        
        self.books_df = books_df
        self.interactions_df = interactions_df
        
        # Train content model
        self.content_model = ContentBasedModel()
        self.content_model.fit(books_df)
        
        # Train CF model
        self.cf_model = CollaborativeModel()
        self.cf_model.fit(interactions_df)
        
        logger.info("Hybrid model trained")
    
    def recommend(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Hybrid recommendations with fallback"""
        # Get user's interaction history for filtering
        user_history = set()
        if self.interactions_df is not None:
            user_books = self.interactions_df[self.interactions_df['user_id'] == user_id]['book_id'].unique()
            user_history = set(user_books)
        
        # Try CF first
        cf_results = []
        if self.cf_model and user_id in self.cf_model.user_id_map:
            # Request more items for better blending, but CF model will handle safe limit
            cf_results = self.cf_model.recommend(user_id, top_k=max(limit * 2, limit), filter_items=user_history)
        
        # If new user (no CF), fallback to popularity
        if not cf_results:
            return self._popularity_fallback(user_id, limit, user_history)
        
        # Get content scores for CF candidates
        hybrid_scores = []
        for book_id, cf_score in cf_results[:limit*2]:
            content_score = self._get_content_score_for_user(user_id, book_id)
            
            # Hybrid blend
            final_score = self.alpha * cf_score + (1 - self.alpha) * content_score
            
            hybrid_scores.append({
                'book_id': int(book_id),
                'score': float(final_score),
                'reasons': {
                    'cf': float(cf_score),
                    'content': float(content_score),
                    'pop': 0.0
                }
            })
        
        # Sort and return top K
        hybrid_scores.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_scores[:limit]
    
    def similar_books(self, book_id: int, limit: int = 10) -> List[Dict]:
        """Content-based similar books"""
        if not self.content_model:
            return []
        
        similar = self.content_model.get_similar(book_id, top_k=limit)
        return [{'book_id': int(bid), 'score': float(score)} for bid, score in similar]
    
    def _get_content_score_for_user(self, user_id: int, book_id: int) -> float:
        """Compute content score based on user's past preferences"""
        if not self.content_model or self.interactions_df is None:
            return 0.0
        
        # Get user's past books
        user_books = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) &
            (self.interactions_df['strength'] >= 3)  # Only high-rated
        ]['book_id'].unique()[:10]  # Recent 10
        
        if len(user_books) == 0:
            return 0.0
        
        # Average content similarity to user's books
        scores = []
        for ub in user_books:
            similar = self.content_model.get_similar(ub, top_k=50)
            for sid, score in similar:
                if sid == book_id:
                    scores.append(score)
                    break
        
        return np.mean(scores) if scores else 0.0
    
    def _popularity_fallback(self, user_id: int, limit: int, filter_items: set) -> List[Dict]:
        """Popularity-based fallback for cold start"""
        if self.interactions_df is None:
            return []
        
        # Compute popularity by interaction count
        pop = self.interactions_df.groupby('book_id').size().sort_values(ascending=False)
        
        results = []
        for book_id, count in pop.items():
            if book_id not in filter_items:
                results.append({
                    'book_id': int(book_id),
                    'score': float(count / pop.max()),  # Normalized
                    'reasons': {'cf': 0.0, 'content': 0.0, 'pop': 1.0}
                })
            if len(results) >= limit:
                break
        
        return results
    
    def save(self, artifacts_dir: Path):
        """Save all models"""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cf_model:
            self.cf_model.save(artifacts_dir / 'cf_model.pkl')
        if self.content_model:
            self.content_model.save(artifacts_dir / 'content_model.pkl')
        
        # Save book/interaction data for filtering
        if self.books_df is not None:
            self.books_df.to_pickle(artifacts_dir / 'books.pkl')
        if self.interactions_df is not None:
            self.interactions_df.to_pickle(artifacts_dir / 'interactions.pkl')
        
        logger.info(f"Saved hybrid model to {artifacts_dir}")
    
    @classmethod
    def load(cls, artifacts_dir: Path, alpha: float = 0.6):
        """Load saved models"""
        model = cls(alpha=alpha)
        
        cf_path = artifacts_dir / 'cf_model.pkl'
        content_path = artifacts_dir / 'content_model.pkl'
        
        if cf_path.exists():
            model.cf_model = CollaborativeModel.load(cf_path)
        if content_path.exists():
            model.content_model = ContentBasedModel.load(content_path)
        
        books_path = artifacts_dir / 'books.pkl'
        interactions_path = artifacts_dir / 'interactions.pkl'
        
        if books_path.exists():
            model.books_df = pd.read_pickle(books_path)
        if interactions_path.exists():
            model.interactions_df = pd.read_pickle(interactions_path)
        
        logger.info(f"Loaded hybrid model from {artifacts_dir}")
        return model