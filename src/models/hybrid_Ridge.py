"""
Hybrid Recommender combining CF and Content-Based with User Profiles
With Online Learning support for incremental updates
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque

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
    
    def __init__(self, alpha: float = 0.6, ridge_alpha: float = 1.0,
                 online_learning: bool = True, buffer_size: int = 100):
        """
        Args:
            alpha: Weight for CF vs Content
            ridge_alpha: Ridge regularization parameter
            online_learning: Enable online learning (incremental updates)
            buffer_size: Number of interactions to buffer before batch update
        """
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.online_learning = online_learning
        self.buffer_size = buffer_size
        
        self.cf_model = None
        self.content_model = None  # RidgeContentModel
        self.popularity = None
        
        # Online learning components
        if self.online_learning:
            self.interaction_buffer = deque(maxlen=buffer_size)
            self.buffer_stats = {
                'total_added': 0,
                'total_updates': 0,
                'last_update': None
            }
            logger.info(f"Online learning enabled with buffer size {buffer_size}")
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Train both CF and Content models"""
        logger.info("Training hybrid recommender with Ridge regression...")
        
        # 1. Content-Based Model (Ridge only)
        logger.info("Building Ridge content-based model...")
        self.content_model = RidgeContentModel(alpha=self.ridge_alpha)
        self.content_model.fit(books_df)
        self.content_model.train_user_models(interactions_df, min_interactions=5)
        
        # 2. Collaborative Filtering
        logger.info("Training collaborative filtering model...")
        self.cf_model = CollaborativeModel(factors=64, iterations=30)
        self.cf_model.fit(interactions_df)
        
        # 3. Popularity scores
        self._compute_popularity(interactions_df)
        logger.info("Hybrid model trained successfully")
    
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
        
        # 2. Content-Based Filtering (using Ridge Model)
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
        
        # Ridge model uses get_user_weights()
        return self.content_model.get_user_weights(user_id, top_n)
    
    def save(self, artifacts_dir: Path):
        """Save all models"""
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        if self.cf_model:
            self.cf_model.save(artifacts_dir / 'cf_model.pkl')
        
        if self.content_model:
            self.content_model.save(artifacts_dir / 'content_model.pkl')
        
        # Save hybrid metadata
        import pickle
        with open(artifacts_dir / 'hybrid_metadata.pkl', 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'popularity': self.popularity,
                'ridge_alpha': self.ridge_alpha
            }, f)
        
        logger.info(f"Saved hybrid model to {artifacts_dir}")
    
    @classmethod
    def load(cls, artifacts_dir: Path, alpha: float = None):
        """Load saved models"""
        # Load metadata
        metadata_path = artifacts_dir / 'hybrid_metadata.pkl'
        if metadata_path.exists():
            import pickle
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            alpha = alpha or metadata.get('alpha', 0.6)
            ridge_alpha = metadata.get('ridge_alpha', 1.0)
            popularity = metadata.get('popularity')
        else:
            alpha = alpha or 0.6
            ridge_alpha = 1.0
            popularity = None
        
        model = cls(alpha=alpha, ridge_alpha=ridge_alpha)
        
        # Load CF model
        cf_path = artifacts_dir / 'cf_model.pkl'
        if cf_path.exists():
            model.cf_model = CollaborativeModel.load(cf_path)
        
        # Load Ridge Content model
        content_path = artifacts_dir / 'content_model.pkl'
        if content_path.exists():
            from src.features.content_features_Ridge import RidgeContentModel
            model.content_model = RidgeContentModel.load(content_path)
            logger.info("Loaded Ridge content model")
        
        model.popularity = popularity
        
        logger.info(f"Loaded hybrid model (alpha={model.alpha})")
        
        return model
    
    # ==================== Online Learning Methods ====================
    
    def add_interaction(self, user_id: int, book_id: int, strength: float, 
                       interaction_type: str = 'implicit'):
        """
        Add a new interaction for online learning
        
        Args:
            user_id: User ID
            book_id: Book ID
            strength: Interaction strength (1-5 for ratings, 1.0-3.0 for implicit)
            interaction_type: Type of interaction (rating, view, favorite, bookmark)
        
        Returns:
            bool: True if buffer triggered update, False otherwise
        """
        if not self.online_learning:
            logger.warning("Online learning is disabled. Call enable_online_learning() first.")
            return False
        
        # Add to buffer
        interaction = {
            'user_id': user_id,
            'book_id': book_id,
            'strength': strength,
            'type': interaction_type,
            'ts': datetime.now()
        }
        self.interaction_buffer.append(interaction)
        self.buffer_stats['total_added'] += 1
        
        logger.debug(f"Added interaction: user={user_id}, book={book_id}, "
                    f"strength={strength}, type={interaction_type}")
        
        # Check if buffer is full → trigger update
        if len(self.interaction_buffer) >= self.buffer_size:
            logger.info(f"Buffer full ({len(self.interaction_buffer)} interactions), "
                       f"triggering incremental update...")
            self.incremental_update()
            return True
        
        return False
    
    def incremental_update(self, force: bool = False):
        """
        Perform incremental model update with buffered interactions
        
        Args:
            force: Force update even if buffer is not full
        """
        if not self.online_learning:
            logger.warning("Online learning is disabled")
            return
        
        if len(self.interaction_buffer) == 0:
            logger.info("No interactions in buffer to update")
            return
        
        if not force and len(self.interaction_buffer) < self.buffer_size:
            logger.info(f"Buffer has only {len(self.interaction_buffer)} interactions, "
                       f"skipping update (threshold: {self.buffer_size})")
            return
        
        logger.info(f"Starting incremental update with {len(self.interaction_buffer)} interactions...")
        
        # Convert buffer to DataFrame
        buffer_df = pd.DataFrame(list(self.interaction_buffer))
        
        # 1. Update Content-Based Model (Ridge or Weighted)
        self._update_content_model(buffer_df)
        
        # 2. Update Popularity
        self._update_popularity(buffer_df)
        
        # 3. Update CF model (more complex, requires ALS re-fitting)
        # For now, we'll collect these for next full retrain
        # In production, you might use incremental matrix factorization
        logger.info("CF model update skipped (requires full retrain for ALS)")
        logger.info("💡 For real incremental CF, consider switching to SGD-based models")
        
        # Clear buffer
        self.interaction_buffer.clear()
        self.buffer_stats['total_updates'] += 1
        self.buffer_stats['last_update'] = datetime.now()
        
        logger.info(f"Incremental update completed! Total updates: {self.buffer_stats['total_updates']}")
    
    def _update_content_model(self, new_interactions_df: pd.DataFrame):
        """Update content-based model with new interactions"""
        if not self.content_model:
            logger.warning("Content model not initialized")
            return
        
        logger.info("Updating Ridge content-based model...")
        
        # Ridge model: Retrain user models for affected users
        affected_users = new_interactions_df['user_id'].unique()
        logger.info(f"Retraining Ridge models for {len(affected_users)} affected users...")
        
        for user_id in affected_users:
            # Get user's OLD interactions
            old_interactions = self.content_model.user_interactions.get(user_id, {})
            
            # Merge with NEW interactions
            user_new = new_interactions_df[new_interactions_df['user_id'] == user_id]
            for _, row in user_new.iterrows():
                old_interactions[row['book_id']] = row['strength']
            
            # Update user_interactions
            self.content_model.user_interactions[user_id] = old_interactions
            
            # Retrain this user's Ridge model
            if len(old_interactions) >= 5:  # min_interactions threshold
                X_train, y_train = self.content_model._prepare_user_training_data(old_interactions)
                
                if X_train is not None and len(X_train) > 0:
                    from sklearn.linear_model import Ridge
                    ridge_model = Ridge(alpha=self.content_model.alpha)
                    ridge_model.fit(X_train, y_train)
                    self.content_model.user_models[user_id] = ridge_model
        
        logger.info(f"✅ Updated Ridge models for {len(affected_users)} users")
    
    def _update_popularity(self, new_interactions_df: pd.DataFrame):
        """Update popularity statistics with new interactions"""
        if self.popularity is None:
            self.popularity = pd.Series(dtype=int)
        
        logger.info("Updating popularity scores...")
        
        # Count new interactions per book
        new_counts = new_interactions_df['book_id'].value_counts()
        
        # Merge with existing popularity
        for book_id, count in new_counts.items():
            if book_id in self.popularity.index:
                self.popularity[book_id] += count
            else:
                self.popularity[book_id] = count
        
        # Re-sort
        self.popularity = self.popularity.sort_values(ascending=False)
        
        logger.info(f"Popularity updated for {len(new_counts)} books")
    
    def get_buffer_status(self) -> Dict:
        """Get online learning buffer status"""
        if not self.online_learning:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'buffer_size': len(self.interaction_buffer),
            'buffer_capacity': self.buffer_size,
            'fill_percentage': len(self.interaction_buffer) / self.buffer_size * 100,
            'total_added': self.buffer_stats['total_added'],
            'total_updates': self.buffer_stats['total_updates'],
            'last_update': self.buffer_stats['last_update'].isoformat() if self.buffer_stats['last_update'] else None
        }
    
    def enable_online_learning(self, buffer_size: int = 100):
        """Enable online learning after model is loaded"""
        self.online_learning = True
        self.buffer_size = buffer_size
        self.interaction_buffer = deque(maxlen=buffer_size)
        self.buffer_stats = {
            'total_added': 0,
            'total_updates': 0,
            'last_update': None
        }
        logger.info(f"Online learning enabled with buffer size {buffer_size}")
    
    def disable_online_learning(self):
        """Disable online learning"""
        self.online_learning = False
        logger.info("Online learning disabled")




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