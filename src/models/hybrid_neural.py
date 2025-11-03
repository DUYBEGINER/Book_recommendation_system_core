"""
Advanced Hybrid Recommender with Deep Learning

Combines:
1. Neural Collaborative Filtering (NCF) - deep user-item interactions
2. SBERT Content-Based - semantic embeddings
3. Weighted fusion + popularity fallback
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import pickle

from src.models.neural_cf import NeuralCFModel
from src.features.sbert_features import SBERTContentModel
from src.utils.logging_config import logger


class HybridNeuralRecommender:
    """
    Advanced Hybrid Recommendation with Deep Learning
    
    Architecture:
    1. NCF Model: Neural CF for collaborative signals
    2. SBERT Model: Semantic content-based filtering
    3. Fusion: Weighted average with learned embeddings
    4. Fallback: Popularity-based for cold start
    """
    
    def __init__(self, alpha: float = 0.6,
                 # NCF params
                 gmf_dim: int = 64, mlp_dims: List[int] = [128, 64, 32],
                 ncf_dropout: float = 0.2, ncf_lr: float = 0.001,
                 ncf_batch_size: int = 256, ncf_epochs: int = 20,
                 # SBERT params
                 sbert_model: str = 'keepitreal/vietnamese-sbert',
                 device: str = None):
        """
        Args:
            alpha: Weight for NCF vs SBERT (0.6 = 60% NCF, 40% SBERT)
            gmf_dim: GMF embedding dimension
            mlp_dims: MLP layer dimensions
            ncf_dropout: NCF dropout rate
            ncf_lr: NCF learning rate
            ncf_batch_size: NCF batch size
            ncf_epochs: NCF training epochs
            sbert_model: SBERT model name
            device: 'cuda' or 'cpu'
        """
        self.alpha = alpha
        
        # Models
        self.ncf_model = NeuralCFModel(
            gmf_dim=gmf_dim,
            mlp_dims=mlp_dims,
            dropout=ncf_dropout,
            learning_rate=ncf_lr,
            batch_size=ncf_batch_size,
            epochs=ncf_epochs,
            device=device
        )
        
        self.content_model = SBERTContentModel(
            model_name=sbert_model,
            device=device
        )
        
        self.popularity = None
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Train both NCF and SBERT models"""
        logger.info("Training Hybrid Neural Recommender...")
        
        # 1. SBERT Content Model
        logger.info("Building SBERT content features...")
        self.content_model.fit(books_df, batch_size=32)
        self.content_model.build_user_profiles(interactions_df)
        
        # 2. Neural CF Model
        logger.info("Training Neural Collaborative Filtering...")
        self.ncf_model.fit(interactions_df)
        
        # 3. Popularity
        self._compute_popularity(interactions_df)
        
        logger.info("Hybrid Neural model trained")
    
    def _compute_popularity(self, interactions_df: pd.DataFrame):
        """Compute item popularity for fallback"""
        popularity_counts = interactions_df.groupby('book_id').size()
        self.popularity = popularity_counts.sort_values(ascending=False)
        logger.info(f"Computed popularity for {len(self.popularity)} items")
    
    def recommend(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get hybrid neural recommendations
        
        Strategy:
        1. NCF predictions (collaborative)
        2. SBERT predictions (content-based)
        3. Weighted fusion
        4. Popularity fallback
        
        Args:
            user_id: User ID
            limit: Number of recommendations
        
        Returns:
            List of dicts with keys: book_id, score, reasons
        """
        ncf_results = []
        content_results = []
        
        # Get interacted items for filtering
        interacted_items = set()
        if user_id in self.content_model.user_interactions:
            interacted_items = set(self.content_model.user_interactions[user_id].keys())
        
        # 1. Neural CF
        if user_id in self.ncf_model.user_id_map:
            try:
                ncf_results = self.ncf_model.recommend(
                    user_id,
                    top_k=limit * 2,
                    filter_items=interacted_items
                )
                logger.debug(f"NCF returned {len(ncf_results)} results for user {user_id}")
            except Exception as e:
                logger.warning(f"NCF failed for user {user_id}: {e}")
        else:
            logger.debug(f"User {user_id} not in NCF training set")
        
        # 2. SBERT Content
        if user_id in self.content_model.user_profiles:
            try:
                content_results = self.content_model.recommend_for_user(
                    user_id,
                    top_k=limit * 2,
                    filter_items=interacted_items
                )
                logger.debug(f"SBERT returned {len(content_results)} results for user {user_id}")
            except Exception as e:
                logger.warning(f"SBERT failed for user {user_id}: {e}")
        else:
            logger.debug(f"No SBERT profile for user {user_id}")
        
        # 3. Combine NCF + SBERT
        if ncf_results or content_results:
            combined = self._combine_scores(ncf_results, content_results)
            results = [
                {
                    'book_id': book_id,
                    'score': score,
                    'reasons': reasons
                }
                for book_id, score, reasons in combined[:limit]
            ]
            
            logger.info(f"Hybrid Neural: {len(ncf_results)} NCF + {len(content_results)} SBERT -> {len(results)} final")
            return results
        
        # 4. Fallback to popularity
        logger.warning(f"No NCF or SBERT results for user {user_id}, falling back to popularity")
        pop_results = self._get_popularity_recommendations(limit, exclude=interacted_items)
        
        # Add reasons for popularity fallback
        for item in pop_results:
            item['reasons'] = {'ncf': 0.0, 'sbert': 0.0, 'pop': item['score']}
            del item['source']
        
        return pop_results
    
    def _combine_scores(self, ncf_results: List[Tuple[int, float]],
                       content_results: List[Tuple[int, float]]) -> List[Tuple[int, float, dict]]:
        """
        Combine NCF and SBERT scores with weighted average
        
        final_score = alpha * NCF_score + (1 - alpha) * SBERT_score
        
        Args:
            ncf_results: List of (book_id, ncf_score)
            content_results: List of (book_id, sbert_score)
        
        Returns:
            Combined list of (book_id, final_score, reasons_dict)
        """
        # Normalize scores to [0, 1]
        ncf_dict = self._normalize_scores(ncf_results)
        content_dict = self._normalize_scores(content_results)
        
        # Combine
        all_books = set(ncf_dict.keys()) | set(content_dict.keys())
        
        combined = []
        for book_id in all_books:
            ncf_score = ncf_dict.get(book_id, 0.0)
            sbert_score = content_dict.get(book_id, 0.0)
            
            # Weighted average
            final_score = self.alpha * ncf_score + (1 - self.alpha) * sbert_score
            
            # Build reasons breakdown
            reasons = {
                'ncf': float(ncf_score),
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
                    'source': 'popularity'
                })
                if len(results) >= limit:
                    break
        
        return results
    
    def similar_books(self, book_id: int, limit: int = 10) -> List[Dict]:
        """Get similar books using SBERT embeddings"""
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
        return self.content_model.get_profile_keywords(user_id, top_n)
    
    def save(self, artifacts_dir: Path):
        """Save all models"""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save NCF model
        self.ncf_model.save(artifacts_dir / 'ncf_model.pt')
        
        # Save SBERT model
        self.content_model.save(artifacts_dir / 'sbert_model.pkl')
        
        # Save metadata
        with open(artifacts_dir / 'hybrid_neural_metadata.pkl', 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'popularity': self.popularity
            }, f)
        
        logger.info(f"Saved Hybrid Neural model to {artifacts_dir}")
    
    @classmethod
    def load(cls, artifacts_dir: Path, alpha: float = None, device: str = None):
        """Load saved models"""
        # Load metadata
        metadata_path = artifacts_dir / 'hybrid_neural_metadata.pkl'
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            alpha = alpha or metadata.get('alpha', 0.6)
            popularity = metadata.get('popularity')
        else:
            alpha = alpha or 0.6
            popularity = None
        
        model = cls(alpha=alpha, device=device)
        
        # Load NCF model
        ncf_path = artifacts_dir / 'ncf_model.pt'
        if ncf_path.exists():
            model.ncf_model = NeuralCFModel.load(ncf_path, device=device)
        
        # Load SBERT model
        sbert_path = artifacts_dir / 'sbert_model.pkl'
        if sbert_path.exists():
            model.content_model = SBERTContentModel.load(sbert_path, device=device)
        
        model.popularity = popularity
        
        logger.info(f"Loaded Hybrid Neural model (alpha={model.alpha})")
        return model
