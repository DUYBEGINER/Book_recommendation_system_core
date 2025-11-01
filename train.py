# ==================== train.py ====================
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from src.data.db_loader import DatabaseLoader
from src.models.hybrid import HybridRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger

def compute_metrics(recommender: HybridRecommender, interactions_df: pd.DataFrame, k: int = 10):
    """Compute HR@K and NDCG@K on holdout set"""
    
    # Split by timestamp (last 20% as test)
    interactions_sorted = interactions_df.sort_values('ts')
    split_idx = int(len(interactions_sorted) * 0.8)
    train_df = interactions_sorted.iloc[:split_idx]
    test_df = interactions_sorted.iloc[split_idx:]
    
    # Group test by user
    test_grouped = test_df.groupby('user_id')['book_id'].apply(set).to_dict()
    
    hits = 0
    ndcg_scores = []
    users_evaluated = 0
    
    for user_id, true_items in test_grouped.items():
        if len(true_items) == 0:
            continue
        
        # Get recommendations
        recs = recommender.recommend(user_id, limit=k)
        rec_items = [r['book_id'] for r in recs]
        
        # Hit rate
        if any(item in true_items for item in rec_items):
            hits += 1
        
        # NDCG
        relevance = [1 if item in true_items else 0 for item in rec_items]
        if sum(relevance) > 0:
            ideal = sorted(relevance, reverse=True)
            ndcg = ndcg_score([ideal], [relevance])
            ndcg_scores.append(ndcg)
        
        users_evaluated += 1
    
    hr = hits / users_evaluated if users_evaluated > 0 else 0
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0
    
    return {
        'HR@K': hr,
        'NDCG@K': avg_ndcg,
        'users_evaluated': users_evaluated
    }

def compute_coverage(recommender: HybridRecommender, interactions_df: pd.DataFrame, num_samples: int = 100):
    """Compute catalog coverage"""
    all_items = set(interactions_df['book_id'].unique())
    recommended_items = set()
    
    user_ids = interactions_df['user_id'].unique()[:num_samples]
    
    for user_id in user_ids:
        recs = recommender.recommend(user_id, limit=10)
        recommended_items.update([r['book_id'] for r in recs])
    
    coverage = len(recommended_items) / len(all_items) if all_items else 0
    return coverage
def main():
    parser = argparse.ArgumentParser(description='Train Book Recommendation System')
    parser.add_argument('--db-uri', type=str, help='Database URI')
    parser.add_argument('--schema', type=str, default='book_recommendation_system')
    parser.add_argument('--alpha', type=float, default=0.6, help='CF weight (0-1)')
    parser.add_argument('--artifacts-dir', type=str, default='./artifacts')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    
    args = parser.parse_args()
    settings = get_settings()
    
    # Use CLI args or fall back to settings
    db_uri = args.db_uri or settings.db_uri
    schema = args.schema or settings.db_schema
    alpha = args.alpha
    artifacts_dir = Path(args.artifacts_dir)
    
    logger.info(f"Training with alpha={alpha}, artifacts={artifacts_dir}")
    
    # Load data
    loader = DatabaseLoader(db_uri, schema)
    books_df, interactions_df = loader.load_all()
    
    logger.info(f"Loaded {len(books_df)} books, {len(interactions_df)} interactions")
    logger.info(f"Unique users: {interactions_df['user_id'].nunique()}")
    logger.info(f"Unique books in interactions: {interactions_df['book_id'].nunique()}")
    
    if len(books_df) == 0 or len(interactions_df) == 0:
        logger.error("Insufficient data for training")
        sys.exit(1)
    
    # Check if retraining needed
    old_model_path = artifacts_dir / 'cf_model.pkl'
    if old_model_path.exists():
        logger.warning("Existing model found. It will be overwritten.")
        logger.info("Deleting old artifacts to ensure clean training...")
        import shutil
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Train hybrid model
    recommender = HybridRecommender(alpha=alpha)
    recommender.train(books_df, interactions_df)
    
    # Evaluate
    if args.evaluate:
        logger.info("Evaluating model...")
        
        # Validate model matches data
        if recommender.cf_model:
            model_users = set(recommender.cf_model.user_ids)
            data_users = set(interactions_df['user_id'].unique())
            missing_users = data_users - model_users
            if missing_users:
                logger.warning(f"Model missing {len(missing_users)} users from current data")
        
        try:
            metrics_5 = compute_metrics(recommender, interactions_df, k=5)
            metrics_10 = compute_metrics(recommender, interactions_df, k=10)
            coverage = compute_coverage(recommender, interactions_df)
            
            logger.info(f"Metrics @5: HR={metrics_5['HR@K']:.4f}, NDCG={metrics_5['NDCG@K']:.4f}")
            logger.info(f"Metrics @10: HR={metrics_10['HR@K']:.4f}, NDCG={metrics_10['NDCG@K']:.4f}")
            logger.info(f"Coverage: {coverage:.4f}")
            logger.info(f"Users evaluated: {metrics_10['users_evaluated']}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.warning("Skipping evaluation, but model is saved")
    
    # Save models
    recommender.save(artifacts_dir)
    logger.info("Training complete!")
# def main():
#     parser = argparse.ArgumentParser(description='Train Book Recommendation System')
#     parser.add_argument('--db-uri', type=str, help='Database URI')
#     parser.add_argument('--schema', type=str, default='book_recommendation_system')
#     parser.add_argument('--alpha', type=float, default=0.6, help='CF weight (0-1)')
#     parser.add_argument('--artifacts-dir', type=str, default='./artifacts')
#     parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    
#     args = parser.parse_args()
#     settings = get_settings()
    
#     # Use CLI args or fall back to settings
#     db_uri = args.db_uri or settings.db_uri
#     schema = args.schema or settings.db_schema
#     alpha = args.alpha
#     artifacts_dir = Path(args.artifacts_dir)
    
#     logger.info(f"Training with alpha={alpha}, artifacts={artifacts_dir}")
    
#     # Load data
#     loader = DatabaseLoader(db_uri, schema)
#     books_df, interactions_df = loader.load_all()
    
#     logger.info(f"Books: {len(books_df)}, Interactions: {len(interactions_df)}")
    
#     if len(books_df) == 0 or len(interactions_df) == 0:
#         logger.error("Insufficient data for training")
#         sys.exit(1)
    
#     # Train hybrid model
#     recommender = HybridRecommender(alpha=alpha)
#     recommender.train(books_df, interactions_df)
    
#     # Evaluate
#     if args.evaluate:
#         logger.info("Evaluating model...")
        
#         metrics_5 = compute_metrics(recommender, interactions_df, k=5)
#         metrics_10 = compute_metrics(recommender, interactions_df, k=10)
#         coverage = compute_coverage(recommender, interactions_df)
        
#         logger.info(f"Metrics @5: HR={metrics_5['HR@K']:.4f}, NDCG={metrics_5['NDCG@K']:.4f}")
#         logger.info(f"Metrics @10: HR={metrics_10['HR@K']:.4f}, NDCG={metrics_10['NDCG@K']:.4f}")
#         logger.info(f"Coverage: {coverage:.4f}")
#         logger.info(f"Users evaluated: {metrics_10['users_evaluated']}")
    
#     # Save models
#     recommender.save(artifacts_dir)
#     logger.info("Training complete!")

if __name__ == "__main__":
    main()