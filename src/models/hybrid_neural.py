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
import numpy as np

from src.models.neural_cf import NeuralCFModel
from src.features.sbert_features import SBERTContentModel
from src.models.diversity import DiversityRecommender
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
        self.diversity_model: Optional[DiversityRecommender] = None
        self.books_df: Optional[pd.DataFrame] = None
        self._diversity_rating_map: Dict[int, float] = {}
        
        # Online learning (SBERT profiles only)
        self.online_learning: bool = False
        self.interaction_buffer: List[Dict] = []
        self.buffer_size: int = 100
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Train both NCF and SBERT models"""
        logger.info("Training Hybrid Neural Recommender...")
        
        self.books_df = self._prepare_books_for_diversity(books_df)
        self._diversity_rating_map = self._compute_diversity_rating_map(interactions_df)

        # 1. SBERT Content Model
        logger.info("Building SBERT content features...")
        self.content_model.fit(books_df, batch_size=32)
        self.content_model.build_user_profiles(interactions_df)
        
        # 2. Neural CF Model
        logger.info("Training Neural Collaborative Filtering...")
        self.ncf_model.fit(interactions_df)
        
        # 3. Popularity
        self._compute_popularity(interactions_df)
        self._build_diversity_model()
        
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
                'popularity': self.popularity,
                'diversity_rating_map': self._diversity_rating_map,
                'online_learning': self.online_learning,
                'buffer_size': self.buffer_size,
            }, f)
        
        if self.books_df is not None:
            self.books_df.to_pickle(artifacts_dir / 'books.pkl')
        
        logger.info(f"Saved Hybrid Neural model to {artifacts_dir}")
    
    @classmethod
    def load(cls, artifacts_dir: Path, alpha: float = None, device: str = None):
        """Load saved models"""
        metadata = {}
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
        model._diversity_rating_map = metadata.get('diversity_rating_map', {})
        model.online_learning = metadata.get('online_learning', False)
        model.buffer_size = metadata.get('buffer_size', 100)
        model.interaction_buffer = []

        books_path = artifacts_dir / 'books.pkl'
        if books_path.exists():
            model.books_df = pd.read_pickle(books_path)
        else:
            logger.warning("Books metadata not found in artifacts; diversity recommendations disabled until retrain.")

        model._build_diversity_model()
        
        logger.info(f"Loaded Hybrid Neural model (alpha={model.alpha})")
        return model

    def _build_diversity_model(self):
        """Initialize diversity recommender leveraging SBERT embeddings."""
        if self.books_df is None or self.books_df.empty:
            logger.warning("Cannot initialize diversity model without books metadata")
            self.diversity_model = None
            return

        embeddings = None
        if (
            self.content_model
            and getattr(self.content_model, "embeddings", None) is not None
            and getattr(self.content_model, "book_ids", None) is not None
        ):
            try:
                id_to_idx = {int(bid): idx for idx, bid in enumerate(self.content_model.book_ids)}
                vectors = []
                missing = []
                for bid in self.books_df["book_id"]:
                    idx = id_to_idx.get(int(bid))
                    if idx is None:
                        missing.append(int(bid))
                        continue
                    vectors.append(self.content_model.embeddings[idx])

                if missing:
                    logger.warning(
                        "Diversity embeddings missing for %d books; falling back to TF-IDF for those entries.",
                        len(missing),
                    )

                if vectors and len(vectors) == len(self.books_df):
                    embeddings = np.vstack(vectors).astype(np.float32)
                else:
                    embeddings = None
            except Exception as exc:
                logger.warning(f"Failed to prepare SBERT embeddings for diversity: {exc}")

        tag_candidates = [
            column
            for column in (
                "genres_text",
                "tags",
                "tag_name",
                "categories",
                "category",
                "authors",
                "title",
                "description",
                "summary",
            )
            if column in self.books_df.columns
        ]

        try:
            self.diversity_model = DiversityRecommender(
                books_df=self.books_df,
                interactions_df=None,
                tag_columns=tag_candidates or None,
                embeddings=embeddings,
            )
            if self._diversity_rating_map:
                self.diversity_model.rating_map = {
                    int(book_id): float(rating)
                    for book_id, rating in self._diversity_rating_map.items()
                }
                ratings = list(self.diversity_model.rating_map.values())
                self.diversity_model.global_rating = (
                    float(np.nanmean(ratings)) if ratings else 0.0
                )
        except Exception as exc:
            logger.error(f"Failed to initialize diversity model: {exc}")
            self.diversity_model = None

    def _prepare_books_for_diversity(self, books_df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant columns for diversity recommendations."""
        if books_df is None or books_df.empty:
            return pd.DataFrame()

        baseline_columns = ["book_id"]
        candidate_columns = [
            "genres_text",
            "tags",
            "tag_name",
            "categories",
            "category",
            "authors",
            "title",
            "description",
            "summary",
        ]

        columns = baseline_columns + [col for col in candidate_columns if col in books_df.columns]
        return books_df[columns].copy()

    def _compute_diversity_rating_map(self, interactions_df: pd.DataFrame) -> Dict[int, float]:
        """Compute average rating per book for diversity ranking."""
        if interactions_df is None or interactions_df.empty:
            return {}

        rating_column = None
        for candidate in ("rating_value", "rating", "strength"):
            if candidate in interactions_df.columns:
                rating_column = candidate
                break

        if rating_column is None:
            return {}

        ratings_df = interactions_df.dropna(subset=["book_id", rating_column])
        grouped = ratings_df.groupby("book_id")[rating_column].mean()
        return {int(book_id): float(value) for book_id, value in grouped.items()}

    # ==================== Online Learning (SBERT only) ====================

    def enable_online_learning(self, buffer_size: int = 100):
        """Enable incremental updates for SBERT user profiles."""
        self.online_learning = True
        self.buffer_size = buffer_size
        self.interaction_buffer = []
        logger.info(
            "Online learning enabled for Hybrid Neural recommender "
            f"(buffer_size={buffer_size}). Only SBERT profiles will update incrementally."
        )
        logger.info("NCF model still requires full retrain to refresh collaborative signals.")

    def disable_online_learning(self):
        """Disable online learning and clear buffered interactions."""
        self.online_learning = False
        self.interaction_buffer = []
        logger.info("Online learning disabled for Hybrid Neural recommender.")

    def add_interaction(
        self,
        user_id: int,
        book_id: int,
        strength: float = 1.0,
        interaction_type: str = "history",
    ) -> bool:
        """
        Add a new interaction to the buffer.

        Returns True if buffer capacity reached and an update was triggered.
        """
        if not self.online_learning:
            logger.warning("Attempted to add interaction while online learning disabled.")
            return False

        self.interaction_buffer.append(
            {
                "user_id": user_id,
                "book_id": book_id,
                "strength": float(strength),
                "type": interaction_type,
            }
        )

        logger.debug(
            "Buffered interaction (user=%s, book=%s, strength=%s, type=%s) "
            "[%d/%d]",
            user_id,
            book_id,
            strength,
            interaction_type,
            len(self.interaction_buffer),
            self.buffer_size,
        )

        if len(self.interaction_buffer) >= self.buffer_size:
            logger.info(
                "Online learning buffer full (%d/%d). Triggering SBERT incremental update.",
                len(self.interaction_buffer),
                self.buffer_size,
            )
            self.incremental_update(force=True)
            return True

        return False

    def incremental_update(self, force: bool = False):
        """
        Apply buffered interactions to SBERT user profiles.

        NCF model is not updated here (requires full retraining).
        """
        if not self.online_learning:
            logger.warning("Online learning disabled; skipping incremental update.")
            return

        if not self.interaction_buffer:
            logger.info("Online learning buffer empty; nothing to update.")
            return

        if not force and len(self.interaction_buffer) < self.buffer_size:
            logger.info(
                "Buffer not full (%d/%d). Skipping incremental update "
                "(pass force=True to override).",
                len(self.interaction_buffer),
                self.buffer_size,
            )
            return

        logger.info(
            "Applying %d buffered interactions to SBERT user profiles...",
            len(self.interaction_buffer),
        )

        if self.content_model:
            for entry in self.interaction_buffer:
                self.content_model.update_user_profile(
                    user_id=entry["user_id"],
                    book_id=entry["book_id"],
                    strength=entry["strength"],
                    interaction_type=entry["type"],
                )
            logger.info("✅ SBERT user profiles updated incrementally.")

        if self.popularity is not None:
            for entry in self.interaction_buffer:
                book_id = entry["book_id"]
                self.popularity[book_id] = self.popularity.get(book_id, 0) + 1

        processed = len(self.interaction_buffer)
        self.interaction_buffer = []
        logger.info(
            "Online learning update complete. Cleared %d buffered interactions. "
            "Reminder: NCF model still uses the previous training snapshot.",
            processed,
        )

    def get_buffer_status(self) -> Dict[str, object]:
        """Return current buffer statistics for monitoring."""
        return {
            "enabled": self.online_learning,
            "buffer_size": len(self.interaction_buffer),
            "buffer_capacity": self.buffer_size,
            "buffer_full": len(self.interaction_buffer) >= self.buffer_size
            if self.online_learning
            else False,
            "note": "Only SBERT profiles are updated incrementally. NCF model requires full retrain.",
        }
