"""
Hybrid Recommender combining CF and Content-Based with User Profiles
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

from src.models.collaborative import CollaborativeModel
from src.features.content_features import ContentBasedModel
from src.utils.logging_config import logger
from src.features.content_features_Ridge import RidgeContentModel

class HybridRecommender:
    """
    Hybrid recommendation combining:
    1. Collaborative Filtering (CF) - user-item interactions
    2. Content-Based Filtering (CBF) - user profiles + item features
    3. Popularity fallback
    """
    
    def __init__(self, alpha: float = 0.6, use_ridge: bool = True, ridge_alpha: float = 1.0):
        """
        Args:
            alpha: Weight for CF vs Content
            use_ridge: Use Ridge regression (True) or weighted average (False)
            ridge_alpha: Ridge regularization parameter
        """
        self.alpha = alpha
        self.use_ridge = use_ridge
        self.ridge_alpha = ridge_alpha
        
        self.cf_model = None
        self.content_model = None
        self.popularity = None
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Train both CF and Content models"""
        logger.info(f"Training hybrid recommender (Ridge={self.use_ridge})...")
        
        # 1. Content-Based Model
        if self.use_ridge:
            logger.info("Using Ridge Regression for content-based...")
            self.content_model = RidgeContentModel(alpha=self.ridge_alpha)
            self.content_model.fit(books_df)
            self.content_model.train_user_models(interactions_df, min_interactions=5)
        else:
            logger.info("Using weighted average for content-based...")
            from src.features.content_features import ContentBasedModel
            self.content_model = ContentBasedModel()
            self.content_model.fit(books_df)
            self.content_model.build_user_profiles(interactions_df)
        
        # 2. CF and popularity (same as before)
        logger.info("Training collaborative filtering model...")
        self.cf_model = CollaborativeModel(factors=64, iterations=30)
        self.cf_model.fit(interactions_df)
        
        self._compute_popularity(interactions_df)
        logger.info("Hybrid model trained")
    
    def _compute_popularity(self, interactions_df: pd.DataFrame):
        """Compute item popularity for fallback"""
        popularity_counts = interactions_df.groupby('book_id').size()
        self.popularity = popularity_counts.sort_values(ascending=False)
        logger.info(f"Computed popularity for {len(self.popularity)} items")
    
    def recommend(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get hybrid recommendations
        
        Strategy:
        1. Try CF recommendations (if user has interactions)
        2. Try Content-Based recommendations (if user has profile)
        3. Combine CF + CBF scores with weighted average
        4. Fallback to popularity if both fail
        
        Args:
            user_id: User ID
            limit: Number of recommendations
        
        Returns:
            List of dicts with keys: book_id, score, source
        """
        cf_results = []
        content_results = []
        
        # Get interacted items for filtering
        interacted_items = set()
        if self.content_model and user_id in self.content_model.user_interactions:
            interacted_items = set(self.content_model.user_interactions[user_id].keys())
        
        # 1. Collaborative Filtering
        if self.cf_model and user_id in self.cf_model.user_id_map:
            try:
                cf_results = self.cf_model.recommend(
                    user_id, 
                    top_k=limit * 2,
                    filter_items=interacted_items
                )
                logger.debug(f"CF returned {len(cf_results)} results for user {user_id}")
            except Exception as e:
                logger.warning(f"CF failed for user {user_id}: {e}")
        else:
            logger.debug(f"No CF results for user {user_id}, user not in training set")
        
        # 2. Content-Based Filtering (using User Profile or Ridge Model)
        if self.use_ridge:
            # Ridge model: check user_models
            if self.content_model and user_id in self.content_model.user_models:
                try:
                    content_results = self.content_model.recommend_for_user(
                        user_id,
                        top_k=limit * 2,
                        filter_items=interacted_items
                    )
                    logger.debug(f"Ridge returned {len(content_results)} results for user {user_id}")
                except Exception as e:
                    logger.warning(f"Ridge failed for user {user_id}: {e}")
            else:
                logger.debug(f"No Ridge model for user {user_id}")
        else:
            # Weighted average: check user_profiles
            if self.content_model and user_id in self.content_model.user_profiles:
                try:
                    content_results = self.content_model.recommend_for_user(
                        user_id,
                        top_k=limit * 2,
                        filter_items=interacted_items
                    )
                    logger.debug(f"Content returned {len(content_results)} results for user {user_id}")
                except Exception as e:
                    logger.warning(f"Content-based failed for user {user_id}: {e}")
            else:
                logger.debug(f"No content profile for user {user_id}")
        
        # 3. Combine CF + Content with weighted average
        if cf_results or content_results:
            combined = self._combine_scores(cf_results, content_results)
            results = [
                {
                    'book_id': book_id,
                    'score': score,
                    'reasons': reasons
                }
                for book_id, score, reasons in combined[:limit]
            ]
            
            logger.info(f"Hybrid: {len(cf_results)} CF + {len(content_results)} Content -> {len(results)} final")
            return results
        
        # 4. Fallback to popularity
        logger.warning(f"No CF or Content results for user {user_id}, falling back to popularity")
        pop_results = self._get_popularity_recommendations(limit, exclude=interacted_items)
        
        # Add reasons for popularity fallback
        for item in pop_results:
            item['reasons'] = {'cf': 0.0, 'content': 0.0, 'pop': item['score']}
            del item['source']  # Remove old 'source' field
        
        return pop_results
    
    def _combine_scores(self, cf_results: List[Tuple[int, float]], 
                       content_results: List[Tuple[int, float]]) -> List[Tuple[int, float, dict]]:
        """
        Combine CF and Content scores with weighted average
        
        final_score = alpha * CF_score + (1 - alpha) * Content_score
        
        Args:
            cf_results: List of (book_id, cf_score)
            content_results: List of (book_id, content_score)
        
        Returns:
            Combined list of (book_id, final_score, reasons_dict) sorted by final score
        """
        # Normalize scores to [0, 1] range
        cf_dict = self._normalize_scores(cf_results)
        content_dict = self._normalize_scores(content_results)
        
        # Combine
        all_books = set(cf_dict.keys()) | set(content_dict.keys())
        
        combined = []
        for book_id in all_books:
            cf_score = cf_dict.get(book_id, 0.0)
            content_score = content_dict.get(book_id, 0.0)
            
            # Weighted average
            final_score = self.alpha * cf_score + (1 - self.alpha) * content_score
            
            # Build reasons breakdown
            reasons = {
                'cf': float(cf_score),
                'content': float(content_score),
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
            # All same scores
            return {book_id: 1.0 for book_id, _ in results}
        
        # Min-max normalization
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
        """Get similar books (item-to-item similarity)"""
        if not self.content_model:
            return []
        
        similar = self.content_model.get_similar_items(book_id, top_k=limit)
        
        return [
            {
                'book_id': bid,
                'score': score,
                'source': 'content_similarity'
            }
            for bid, score in similar
        ]
    
    def get_user_profile_keywords(self, user_id: int, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get user's profile keywords (for explainability)
        
        Args:
            user_id: User ID
            top_n: Number of keywords
        
        Returns:
            List of (keyword, weight) tuples
        """
        if not self.content_model:
            return []
        
        return self.content_model.get_profile_keywords(user_id, top_n)
    
    def save(self, artifacts_dir: Path):
        """Save all models"""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cf_model:
            self.cf_model.save(artifacts_dir / 'cf_model.pkl')
        
        if self.content_model:
            self.content_model.save(artifacts_dir / 'content_model.pkl')
        
        # Save hybrid metadata including use_ridge flag
        import pickle
        with open(artifacts_dir / 'hybrid_metadata.pkl', 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'popularity': self.popularity,
                'use_ridge': self.use_ridge,  # ✅ Save model type
                'ridge_alpha': self.ridge_alpha
            }, f)
        
        logger.info(f"Saved hybrid model to {artifacts_dir} (Ridge={self.use_ridge})")
    
    @classmethod
    def load(cls, artifacts_dir: Path, alpha: float = None, use_ridge: bool = None):
        """Load saved models"""
        # Load metadata
        metadata_path = artifacts_dir / 'hybrid_metadata.pkl'
        if metadata_path.exists():
            import pickle
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            alpha = alpha or metadata.get('alpha', 0.6)
            use_ridge = use_ridge if use_ridge is not None else metadata.get('use_ridge', False)
            ridge_alpha = metadata.get('ridge_alpha', 1.0)
            popularity = metadata.get('popularity')
        else:
            alpha = alpha or 0.6
            use_ridge = use_ridge if use_ridge is not None else False
            ridge_alpha = 1.0
            popularity = None
        
        model = cls(alpha=alpha, use_ridge=use_ridge, ridge_alpha=ridge_alpha)
        
        # Load CF model
        cf_path = artifacts_dir / 'cf_model.pkl'
        if cf_path.exists():
            model.cf_model = CollaborativeModel.load(cf_path)
        
        # Load Content model (auto-detect type)
        content_path = artifacts_dir / 'content_model.pkl'
        if content_path.exists():
            # Try to detect model type from file
            import pickle
            with open(content_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if it's Ridge model (has user_models) or Weighted (has user_profiles)
            if 'user_models' in data:
                # Ridge model
                from src.features.content_features_Ridge import RidgeContentModel
                model.content_model = RidgeContentModel.load(content_path)
                model.use_ridge = True
                logger.info("Loaded Ridge content model")
            elif 'user_profiles' in data:
                # Weighted average model
                model.content_model = ContentBasedModel.load(content_path)
                model.use_ridge = False
                logger.info("Loaded weighted average content model")
            else:
                logger.warning("Unknown content model format")
        
        model.popularity = popularity
        
        logger.info(f"Loaded hybrid model (alpha={model.alpha}, use_ridge={model.use_ridge})")
        
        return model




# from typing import List, Dict, Tuple, Optional
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from src.models.collaborative import CollaborativeModel
# from src.features.content_features import ContentBasedModel
# from src.utils.logging_config import logger

# class HybridRecommender:
#     def __init__(self, alpha: float = 0.6):
#         self.alpha = alpha  # CF weight
#         self.cf_model: Optional[CollaborativeModel] = None
#         self.content_model: Optional[ContentBasedModel] = None
#         self.books_df: Optional[pd.DataFrame] = None
#         self.interactions_df: Optional[pd.DataFrame] = None
    
#     def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
#         """Train both models"""
#         logger.info("Training hybrid recommender...")
        
#         self.books_df = books_df
#         self.interactions_df = interactions_df
        
#         # Train content model
#         self.content_model = ContentBasedModel()
#         self.content_model.fit(books_df)
#         print("Content feature matrix shape:", self.content_model.feature_matrix.shape)
#         # Train CF model
#         self.cf_model = CollaborativeModel()
#         self.cf_model.fit(interactions_df)
        
#         logger.info("Hybrid model trained")
    
#     def recommend(self, user_id: int, limit: int = 10) -> List[Dict]:
#         """Hybrid recommendations with fallback"""
#         # Get user's interaction history for filtering
#         user_history = set()
#         if self.interactions_df is not None:
#             logger.warning("Content model or interactions data available.")
#             user_books = self.interactions_df[self.interactions_df['user_id'] == user_id]['book_id'].unique()
#             user_history = set(user_books)
        
#         # Try CF first
#         cf_results = []
#         if self.cf_model and user_id in self.cf_model.user_id_map:
#             # Request more items for better blending, but CF model will handle safe limit
#             cf_results = self.cf_model.recommend(user_id, top_k=max(limit * 2, limit), filter_items=user_history)
        

#         # If new user (no CF), fallback to popularity
#         if not cf_results:
#             logger.warning(f"No CF results for user {user_id}, falling back to popularity")
#             return self._popularity_fallback(user_id, limit, user_history)

#         logger.info(f"result {cf_results}")
#         # Get content scores for CF candidates
#         hybrid_scores = []
#         for book_id, cf_score in cf_results[:limit*2]:
#             content_score = self._get_content_score_for_user(user_id, book_id)
#             # Hybrid blend
#             final_score = self.alpha * cf_score + (1 - self.alpha) * content_score
#             logger.debug(f"User {user_id}, Book {book_id}: CF={cf_score:.4f}, Content={content_score:.4f}, Hybrid={final_score:.4f}")
#             hybrid_scores.append({
#                 'book_id': int(book_id),
#                 'score': float(final_score),
#                 'reasons': {
#                     'cf': float(cf_score),
#                     'content': float(content_score),
#                     'pop': 0.0
#                 }
#             })
        
#         # Sort and return top K
#         hybrid_scores.sort(key=lambda x: x['score'], reverse=True)
#         return hybrid_scores[:limit]
    
#     def similar_books(self, book_id: int, limit: int = 10) -> List[Dict]:
#         """Content-based similar books"""
#         if not self.content_model:
#             return []
        
#         similar = self.content_model.get_similar(book_id, top_k=limit)
#         return [{'book_id': int(bid), 'score': float(score)} for bid, score in similar]
    
#     def _get_content_score_for_user(self, user_id: int, book_id: int) -> float:
#         """Compute content score based on user's past preferences"""
#         if not self.content_model or self.interactions_df is None:
#             logger.warning("Content model or interactions data not available.")
#             return 0.0
        
#         # Get user's past books
#         user_books = self.interactions_df[
#             (self.interactions_df['user_id'] == user_id) &
#             (self.interactions_df['strength'] >= 3)  # Only high-rated
#         ]['book_id'].unique()[:10]  # Recent 10
        
#         if len(user_books) == 0:
#             return 0.0
        
#         # Average content similarity to user's books
#         scores = []
#         for ub in user_books:
#             similar = self.content_model.get_similar(ub, top_k=50)
#             for sid, score in similar:
#                 if sid == book_id:
#                     scores.append(score)
#                     break
        
#         return np.mean(scores) if scores else 0.0
    
#     def _popularity_fallback(self, user_id: int, limit: int, filter_items: set) -> List[Dict]:
#         """Popularity-based fallback for cold start"""
#         if self.interactions_df is None:
#             return []
        
#         # Compute popularity by interaction count
#         pop = self.interactions_df.groupby('book_id').size().sort_values(ascending=False)
        
#         results = []
#         for book_id, count in pop.items():
#             if book_id not in filter_items:
#                 results.append({
#                     'book_id': int(book_id),
#                     'score': float(count / pop.max()),  # Normalized
#                     'reasons': {'cf': 0.0, 'content': 0.0, 'pop': 1.0}
#                 })
#             if len(results) >= limit:
#                 break
        
#         return results
    
#     def save(self, artifacts_dir: Path):
#         """Save all models"""
#         artifacts_dir.mkdir(parents=True, exist_ok=True)
        
#         if self.cf_model:
#             self.cf_model.save(artifacts_dir / 'cf_model.pkl')
#         if self.content_model:
#             self.content_model.save(artifacts_dir / 'content_model.pkl')
        
#         # Save book/interaction data for filtering
#         if self.books_df is not None:
#             self.books_df.to_pickle(artifacts_dir / 'books.pkl')
#         if self.interactions_df is not None:
#             self.interactions_df.to_pickle(artifacts_dir / 'interactions.pkl')
        
#         logger.info(f"Saved hybrid model to {artifacts_dir}")
    
#     @classmethod
#     def load(cls, artifacts_dir: Path, alpha: float = 0.6):
#         """Load saved models"""
#         model = cls(alpha=alpha)
        
#         cf_path = artifacts_dir / 'cf_model.pkl'
#         content_path = artifacts_dir / 'content_model.pkl'
        
#         if cf_path.exists():
#             model.cf_model = CollaborativeModel.load(cf_path)
#         if content_path.exists():
#             model.content_model = ContentBasedModel.load(content_path)
        
#         books_path = artifacts_dir / 'books.pkl'
#         interactions_path = artifacts_dir / 'interactions.pkl'
        
#         if books_path.exists():
#             model.books_df = pd.read_pickle(books_path)
#         if interactions_path.exists():
#             model.interactions_df = pd.read_pickle(interactions_path)
        
#         logger.info(f"Loaded hybrid model from {artifacts_dir}")
#         return model