"""
Hybrid Recommender combining CF and Content-Based with User Profiles
"""
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

from src.models.collaborative import CollaborativeModel
from src.features.content_features import ContentBasedModel
from src.models.diversity import DiversityRecommender
from src.utils.logging_config import logger

class HybridRecommender:
    """
    Hybrid recommendation combining:
    1. Collaborative Filtering (CF) - user-item interactions
    2. Content-Based Filtering (CBF) - user profiles + item features
    3. Popularity fallback
    """
    
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha  # CF weight
        self.cf_model: Optional[CollaborativeModel] = None
        self.content_model: Optional[ContentBasedModel] = None
        self.diversity_model: Optional[DiversityRecommender] = None
        self.books_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None
    
    def train(self, books_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Train both CF and Content models"""
        logger.info("Training hybrid recommender...")
        
        # 1. Content-Based Model
        logger.info("Building content-based features...")
        self.content_model = ContentBasedModel()
        self.content_model.fit(books_df)
        
        # 2. Build user profiles from interactions
        self.content_model.build_user_profiles(interactions_df)
        
        # 3. Collaborative Filtering
        logger.info("Training collaborative filtering model...")
        self.cf_model = CollaborativeModel(factors=64, iterations=30)
        self.cf_model.fit(interactions_df)

        self._build_diversity_model()
        
        # 4. Popularity baseline
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
        
        # 2. Content-Based Filtering (using User Profile)
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
                    'source': 'hybrid'
                }
                for book_id, score in combined[:limit]
            ]
            
            logger.info(f"Hybrid: {len(cf_results)} CF + {len(content_results)} Content -> {len(results)} final")
            return results
        
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
        
        # Save hybrid metadata
        import pickle
        with open(artifacts_dir / 'hybrid_metadata.pkl', 'wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'popularity': self.popularity
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
            popularity = metadata.get('popularity')
        else:
            alpha = alpha or 0.6
            popularity = None
        
        model = cls(alpha=alpha)
        
        # Load CF model
        cf_path = artifacts_dir / 'cf_model.pkl'
        if cf_path.exists():
            model.cf_model = CollaborativeModel.load(cf_path)
        
        # Load Content model
        content_path = artifacts_dir / 'content_model.pkl'
        if content_path.exists():
            model.content_model = ContentBasedModel.load(content_path)
        
        model.popularity = popularity
        
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
