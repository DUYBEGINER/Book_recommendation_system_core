from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from src.models.collaborative import CollaborativeModel
from src.features.content_features import ContentBasedModel
from src.models.diversity import DiversityRecommender
from src.utils.logging_config import logger

class HybridRecommender:
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha  # CF weight
        self.cf_model: Optional[CollaborativeModel] = None
        self.content_model: Optional[ContentBasedModel] = None
        self.diversity_model: Optional[DiversityRecommender] = None
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
        print("Content feature matrix shape:", self.content_model.feature_matrix.shape)
        # Train CF model
        self.cf_model = CollaborativeModel()
        self.cf_model.fit(interactions_df)

        self._build_diversity_model()
        
        logger.info("Hybrid model trained")
    
    def recommend(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Hybrid recommendations with fallback"""
        # Get user's interaction history for filtering
        user_history = set()
        if self.interactions_df is not None:
            logger.warning("Content model or interactions data available.")
            user_books = self.interactions_df[self.interactions_df['user_id'] == user_id]['book_id'].unique()
            user_history = set(user_books)
        
        # Try CF first
        cf_results = []
        if self.cf_model and user_id in self.cf_model.user_id_map:
            # Request more items for better blending, but CF model will handle safe limit
            cf_results = self.cf_model.recommend(user_id, top_k=max(limit * 2, limit), filter_items=user_history)
        

        # If new user (no CF), fallback to popularity
        if not cf_results:
            logger.warning(f"No CF results for user {user_id}, falling back to popularity")
            return self._popularity_fallback(user_id, limit, user_history)

        logger.info(f"result {cf_results}")
        # Get content scores for CF candidates
        hybrid_scores = []
        for book_id, cf_score in cf_results[:limit*2]:
            content_score = self._get_content_score_for_user(user_id, book_id)
            # Hybrid blend
            final_score = self.alpha * cf_score + (1 - self.alpha) * content_score
            logger.debug(f"User {user_id}, Book {book_id}: CF={cf_score:.4f}, Content={content_score:.4f}, Hybrid={final_score:.4f}")
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

    def diversity_recommendations(
        self,
        book_id: int,
        limit: int = 5,
    ) -> Dict[str, List[Dict]]:
        """Return diverse recommendations for the provided book."""
        if not self.diversity_model:
            raise RuntimeError("Diversity model not initialized")

        results = self.diversity_model.recommend(
            book_id=book_id,
            limit=limit,
        )

        return {
            'book_id': int(book_id),
            'items': [
                {
                    'book_id': int(item.book_id),
                    'rating': float(item.rating),
                    'score': float(item.score),
                    'metadata': {k: float(v) for k, v in item.metadata.items()} if item.metadata else {},
                }
                for item in results
            ],
        }
    
    def similar_books(self, book_id: int, limit: int = 10) -> List[Dict]:
        """Content-based similar books"""
        if not self.content_model:
            return []
        
        similar = self.content_model.get_similar(book_id, top_k=limit)
        return [{'book_id': int(bid), 'score': float(score)} for bid, score in similar]
    
    def _get_content_score_for_user(self, user_id: int, book_id: int) -> float:
        """Compute content score based on user's past preferences"""
        if not self.content_model or self.interactions_df is None:
            logger.warning("Content model or interactions data not available.")
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

        model._build_diversity_model()
        
        logger.info(f"Loaded hybrid model from {artifacts_dir}")
        return model

    def _build_diversity_model(self):
        """Initialize diversity recommender if data is available."""
        if self.books_df is None:
            logger.warning("Cannot initialize diversity model without books data")
            self.diversity_model = None
            return

        try:
            embeddings = None
            if (
                self.content_model
                and self.content_model.feature_matrix is not None
                and self.content_model.book_ids is not None
            ):
                try:
                    feature_matrix = self.content_model.feature_matrix.detach().cpu().numpy()
                    content_ids = [int(bid) for bid in self.content_model.book_ids]
                    id_to_idx = {bid: idx for idx, bid in enumerate(content_ids)}

                    ordered_vectors = []
                    missing_ids = []
                    for bid in self.books_df["book_id"].astype(int):
                        idx = id_to_idx.get(int(bid))
                        if idx is None:
                            missing_ids.append(int(bid))
                            continue
                        ordered_vectors.append(feature_matrix[idx])

                    if missing_ids:
                        logger.warning(
                            "Diversity embeddings missing for %d books. Falling back to TF-IDF for those entries.",
                            len(missing_ids),
                        )
                    if ordered_vectors and len(ordered_vectors) == len(self.books_df):
                        embeddings = np.vstack(ordered_vectors)
                except Exception as embedding_exc:
                    logger.warning(f"Failed to prepare SBERT embeddings for diversity: {embedding_exc}")

            self.diversity_model = DiversityRecommender(
                books_df=self.books_df,
                interactions_df=self.interactions_df,
                embeddings=embeddings,
            )
        except Exception as exc:
            logger.error(f"Failed to initialize diversity model: {exc}")
            self.diversity_model = None
