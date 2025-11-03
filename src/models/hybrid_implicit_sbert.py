"""
Hybrid Recommender: Implicit ALS + SBERT

Combines:
1. Implicit ALS - Fast collaborative filtering for implicit feedback
2. SBERT - Semantic content-based filtering with Vietnamese embeddings
3. Weighted fusion + popularity fallback
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from src.models.collaborative import CollaborativeModel
from src.features.sbert_features import SBERTContentModel
from src.utils.logging_config import logger


class HybridImplicitSBERTRecommender:
    """
    Hybrid Recommendation: Implicit ALS + SBERT
    
    Architecture:
    1. Implicit ALS: Fast collaborative filtering
    2. SBERT: Semantic content-based filtering
    3. Fusion: Weighted average
    4. Fallback: Popularity-based for cold start
    """
    
    def __init__(self, alpha: float = 0.6,
                 # ALS params
                 als_factors: int = 64, als_iterations: int = 30, 
                 als_regularization: float = 0.01,
                 # SBERT params
                 sbert_model: str = 'keepitreal/vietnamese-sbert',
                 device: str = None):
        """
        Args:
            alpha: Weight for ALS (0-1), SBERT weight = 1 - alpha
            als_factors: ALS latent factors
            als_iterations: ALS iterations
            als_regularization: ALS regularization parameter
            sbert_model: SBERT model name
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.alpha = alpha
        
        # ALS parameters
        self.als_factors = als_factors
        self.als_iterations = als_iterations
        self.als_regularization = als_regularization
        
        # SBERT parameters
        self.sbert_model_name = sbert_model
        self.device = device
        
        # Models
        self.als_model: Optional[CollaborativeModel] = None
        self.content_model: Optional[SBERTContentModel] = None
        
        # Popularity fallback
        self.popularity: Optional[Dict[int, int]] = None
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Train both ALS and SBERT models
        
        Args:
            books_df: DataFrame with book metadata
            interactions_df: DataFrame with [user_id, book_id, strength]
        """
        logger.info("="*70)
        logger.info("Training Hybrid Implicit ALS + SBERT Recommender")
        logger.info("="*70)
        
        # 1. Train Implicit ALS
        logger.info("\n1️⃣ Training Implicit ALS model...")
        self.als_model = CollaborativeModel(
            factors=self.als_factors,
            iterations=self.als_iterations,
            regularization=self.als_regularization
        )
        self.als_model.fit(interactions_df)
        logger.info("✅ ALS model trained")
        
        # 2. Train SBERT content model
        logger.info("\n2️⃣ Training SBERT content model...")
        self.content_model = SBERTContentModel(
            model_name=self.sbert_model_name,
            device=self.device
        )
        self.content_model.fit(books_df)
        self.content_model.build_user_profiles(interactions_df)
        logger.info("✅ SBERT model trained")
        
        # 3. Compute popularity
        logger.info("\n3️⃣ Computing popularity scores...")
        self._compute_popularity(interactions_df)
        logger.info("✅ Popularity scores computed")
        
        logger.info("\n" + "="*70)
        logger.info("✅ Hybrid Implicit ALS + SBERT training completed!")
        logger.info("="*70)
    
    def _compute_popularity(self, interactions_df: pd.DataFrame):
        """Compute item popularity from interactions"""
        pop_counts = interactions_df['book_id'].value_counts().to_dict()
        self.popularity = dict(sorted(pop_counts.items(), 
                                     key=lambda x: x[1], reverse=True))
    
    def recommend(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get hybrid recommendations
        
        Args:
            user_id: User ID
            limit: Number of recommendations
        
        Returns:
            List of recommendation dicts with scores and sources
        """
        # Get user's interaction history for filtering
        interacted_items = set()
        if self.content_model and user_id in self.content_model.user_interactions:
            interacted_items = set(self.content_model.user_interactions[user_id].keys())
        
        # 1. Get ALS recommendations
        als_results = []
        if self.als_model and user_id in self.als_model.user_id_map:
            als_results = self.als_model.recommend(
                user_id, 
                top_k=limit * 3,  # Get more for fusion
                filter_items=interacted_items
            )
        
        # 2. Get SBERT recommendations
        content_results = []
        if self.content_model and user_id in self.content_model.user_profiles:
            content_results = self.content_model.recommend_for_user(
                user_id,
                top_k=limit * 3,
                filter_items=interacted_items
            )
        
        # 3. Combine scores
        if als_results and content_results:
            combined = self._combine_scores(als_results, content_results)
            results = [
                {
                    'book_id': int(book_id),
                    'score': float(score),
                    'reasons': reasons
                }
                for book_id, score, reasons in combined[:limit]
            ]
        elif als_results:
            # ALS only
            results = [
                {
                    'book_id': int(book_id),
                    'score': float(score),
                    'reasons': {'als': float(score), 'sbert': 0.0, 'pop': 0.0}
                }
                for book_id, score in als_results[:limit]
            ]
        elif content_results:
            # SBERT only
            results = [
                {
                    'book_id': int(book_id),
                    'score': float(score),
                    'reasons': {'als': 0.0, 'sbert': float(score), 'pop': 0.0}
                }
                for book_id, score in content_results[:limit]
            ]
        else:
            # Fallback to popularity
            logger.debug(f"User {user_id} - using popularity fallback")
            results = self._get_popularity_recommendations(limit, interacted_items)
        
        return results
    
    def _combine_scores(self, als_results: List[Tuple[int, float]],
                       content_results: List[Tuple[int, float]]) -> List[Tuple[int, float, dict]]:
        """
        Combine ALS and SBERT scores with weighted average
        
        final_score = alpha * ALS_score + (1 - alpha) * SBERT_score
        
        Args:
            als_results: List of (book_id, als_score)
            content_results: List of (book_id, sbert_score)
        
        Returns:
            Combined list of (book_id, final_score, reasons_dict)
        """
        # Normalize scores to [0, 1]
        als_dict = self._normalize_scores(als_results)
        content_dict = self._normalize_scores(content_results)
        
        # Combine
        all_books = set(als_dict.keys()) | set(content_dict.keys())
        
        combined = []
        for book_id in all_books:
            als_score = als_dict.get(book_id, 0.0)
            sbert_score = content_dict.get(book_id, 0.0)
            
            # Weighted average
            final_score = self.alpha * als_score + (1 - self.alpha) * sbert_score
            
            # Build reasons breakdown
            reasons = {
                'als': float(als_score),
                'sbert': float(sbert_score),
                'pop': 0.0
            }
            
            combined.append((book_id, final_score, reasons))
        
        # Sort by score
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined
    
    def _normalize_scores(self, results: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalize scores to [0, 1] using min-max scaling"""
        if not results:
            return {}
        
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return {book_id: 1.0 for book_id, _ in results}
        
        normalized = {}
        for book_id, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized[book_id] = norm_score
        
        return normalized
    
    def _get_popularity_recommendations(self, limit: int, exclude: set) -> List[Dict]:
        """Fallback to popularity-based recommendations"""
        if self.popularity is None:
            return []
        
        results = []
        for book_id, count in self.popularity.items():
            if book_id not in exclude:
                results.append({
                    'book_id': int(book_id),
                    'score': float(count),
                    'reasons': {'als': 0.0, 'sbert': 0.0, 'pop': float(count)}
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def similar_books(self, book_id: int, limit: int = 10) -> List[Dict]:
        """Get similar books using SBERT embeddings"""
        if not self.content_model:
            return []
        
        similar = self.content_model.get_similar_items(book_id, top_k=limit)
        
        return [
            {
                'book_id': bid,
                'score': score,
                'source': 'sbert_similarity'
            }
            for bid, score in similar
        ]
    
    def get_user_profile_keywords(self, user_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get user's profile keywords (top interacted books for SBERT)"""
        if not self.content_model:
            return []
        return self.content_model.get_profile_keywords(user_id, top_n)
    
    def save(self, artifacts_dir: Path):
        """Save all models"""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save ALS model
        if self.als_model:
            self.als_model.save(artifacts_dir / 'als_model.pkl')
        
        # Save SBERT model
        if self.content_model:
            self.content_model.save(artifacts_dir / 'sbert_model.pkl')
        
        # Save metadata
        with open(artifacts_dir / 'hybrid_implicit_sbert_metadata.pkl', 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'als_factors': self.als_factors,
                'als_iterations': self.als_iterations,
                'als_regularization': self.als_regularization,
                'sbert_model_name': self.sbert_model_name,
                'popularity': self.popularity
            }, f)
        
        logger.info(f"Saved Hybrid Implicit ALS + SBERT model to {artifacts_dir}")
    
    @classmethod
    def load(cls, artifacts_dir: Path, alpha: float = None, device: str = None):
        """Load saved models"""
        # Load metadata
        metadata_path = artifacts_dir / 'hybrid_implicit_sbert_metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            alpha = alpha or metadata.get('alpha', 0.6)
            als_factors = metadata.get('als_factors', 64)
            als_iterations = metadata.get('als_iterations', 30)
            als_regularization = metadata.get('als_regularization', 0.01)
            sbert_model_name = metadata.get('sbert_model_name', 'keepitreal/vietnamese-sbert')
            popularity = metadata.get('popularity')
        else:
            alpha = alpha or 0.6
            als_factors = 64
            als_iterations = 30
            als_regularization = 0.01
            sbert_model_name = 'keepitreal/vietnamese-sbert'
            popularity = None
        
        model = cls(
            alpha=alpha,
            als_factors=als_factors,
            als_iterations=als_iterations,
            als_regularization=als_regularization,
            sbert_model=sbert_model_name,
            device=device
        )
        
        # Load ALS model
        als_path = artifacts_dir / 'als_model.pkl'
        if als_path.exists():
            model.als_model = CollaborativeModel.load(als_path)
        
        # Load SBERT model
        sbert_path = artifacts_dir / 'sbert_model.pkl'
        if sbert_path.exists():
            model.content_model = SBERTContentModel.load(sbert_path, device=device)
        
        model.popularity = popularity
        
        logger.info(f"Loaded Hybrid Implicit ALS + SBERT model (alpha={model.alpha})")
        return model
