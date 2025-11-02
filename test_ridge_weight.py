"""
Compare Ridge Regression vs Weighted Average for Content-Based
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.db_loader import DatabaseLoader
from src.models.hybrid_Ridge import HybridRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger
import pandas as pd

def evaluate_approach(recommender, test_df, approach_name):
    """Evaluate one approach"""
    from sklearn.metrics import ndcg_score
    import numpy as np
    
    test_grouped = test_df.groupby('user_id')['book_id'].apply(set).to_dict()
    
    hits = []
    ndcgs = []
    skipped_users = []
    
    for user_id, true_items in test_grouped.items():
        try:
            recs = recommender.recommend(user_id, limit=10)
            
            # Skip if no recommendations (cold-start user)
            if not recs:
                skipped_users.append(user_id)
                logger.debug(f"Skipping user {user_id} - no recommendations")
                continue
            
            rec_items = {rec['book_id'] for rec in recs}
            
            # Hit rate
            hit = 1 if len(rec_items & true_items) > 0 else 0
            hits.append(hit)
            
            # NDCG
            relevance = [1 if rec['book_id'] in true_items else 0 for rec in recs]
            if sum(relevance) > 0:
                ideal = sorted(relevance, reverse=True)
                ndcg = ndcg_score([ideal], [relevance])
                ndcgs.append(ndcg)
        
        except Exception as e:
            logger.warning(f"Error evaluating user {user_id}: {e}")
            skipped_users.append(user_id)
            continue
    
    hr = np.mean(hits) if hits else 0
    ndcg_avg = np.mean(ndcgs) if ndcgs else 0
    
    print(f"\n{approach_name} Results:")
    print(f"  HR@10:   {hr:.4f}")
    print(f"  NDCG@10: {ndcg_avg:.4f}")
    print(f"  Users evaluated: {len(hits)}")
    print(f"  Users skipped: {len(skipped_users)} (cold-start)")
    if skipped_users:
        print(f"  Skipped user IDs: {skipped_users[:5]}{'...' if len(skipped_users) > 5 else ''}")
    
    return {"HR": hr, "NDCG": ndcg_avg, "users_evaluated": len(hits)}

def main():
    settings = get_settings()
    
    # Load data
    loader = DatabaseLoader(settings.db_uri, settings.db_schema)
    books_df, interactions_df = loader.load_all()
    
    # Split 80/20
    interactions_sorted = interactions_df.sort_values('ts')
    split_idx = int(len(interactions_sorted) * 0.8)
    train_df = interactions_sorted.iloc[:split_idx]
    test_df = interactions_sorted.iloc[split_idx:]
    
    print(f"\n{'='*60}")
    print("COMPARING CONTENT-BASED APPROACHES")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} interactions")
    print(f"Test:  {len(test_df)} interactions")
    
    # 1. Weighted Average approach
    print(f"\n{'='*60}")
    print("1. Training with Weighted Average...")
    print(f"{'='*60}")
    
    rec_weighted = HybridRecommender(alpha=0.6, use_ridge=False)
    rec_weighted.train(books_df, train_df)
    results_weighted = evaluate_approach(rec_weighted, test_df, "Weighted Average")
    
    # 2. Ridge Regression approach
    print(f"\n{'='*60}")
    print("2. Training with Ridge Regression...")
    print(f"{'='*60}")
    
    rec_ridge = HybridRecommender(alpha=0.6, use_ridge=True, ridge_alpha=1.0)
    rec_ridge.train(books_df, train_df)
    results_ridge = evaluate_approach(rec_ridge, test_df, "Ridge Regression")
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Weighted':<12} {'Ridge':<12} {'Winner':<15}")
    print("-" * 60)
    
    for metric in ['HR', 'NDCG']:
        w_val = results_weighted[metric]
        r_val = results_ridge[metric]
        winner = "Ridge" if r_val > w_val else "Weighted" if w_val > r_val else "Tie"
        improvement = ((r_val - w_val) / w_val * 100) if w_val > 0 else 0
        
        print(f"{metric + '@10':<15} {w_val:<12.4f} {r_val:<12.4f} {winner:<15} ({improvement:+.1f}%)")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()