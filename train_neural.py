"""
Training script for Hybrid Neural Recommender (NCF + SBERT)

Usage:
    python train_neural.py --evaluate --alpha 0.6
    python train_neural.py --ncf-epochs 50 --gmf-dim 128
"""
import argparse
import sys
from pathlib import Path

from src.data.db_loader import DatabaseLoader
from src.models.hybrid_neural import HybridNeuralRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger
from src.utils.evaluation import RecommenderEvaluator, DataSplitter


def main():
    # ==================== Parse Arguments ====================
    parser = argparse.ArgumentParser(description='Train Hybrid Neural Recommender (NCF + SBERT)')
    
    # Data args
    parser.add_argument('--db-uri', type=str, help='Database URI')
    parser.add_argument('--schema', type=str, default='book_recommendation_system')
    
    # Model args
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='NCF weight (0-1), SBERT weight = 1-alpha')
    parser.add_argument('--gmf-dim', type=int, default=64,
                       help='GMF embedding dimension')
    parser.add_argument('--ncf-epochs', type=int, default=20,
                       help='NCF training epochs')
    parser.add_argument('--ncf-batch-size', type=int, default=256,
                       help='NCF batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu (auto-detect if None)')
    
    # Training args
    parser.add_argument('--artifacts-dir', type=str, default='./artifacts_neural',
                       help='Directory to save models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation on test set')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Test set ratio for evaluation (default: 0.2)')
    parser.add_argument('--split-mode', type=str, default='temporal',
                       choices=['temporal', 'random'],
                       help='Split mode: temporal or random')
    
    args = parser.parse_args()
    settings = get_settings()
    
    # Use CLI args or fall back to settings
    db_uri = args.db_uri or settings.db_uri
    schema = args.schema or settings.db_schema
    artifacts_dir = Path(args.artifacts_dir)
    
    # ==================== Print Configuration ====================
    logger.info("="*60)
    logger.info("TRAINING HYBRID NEURAL RECOMMENDER (NCF + SBERT)")
    logger.info("="*60)
    logger.info(f"Model Configuration:")
    logger.info(f"  Alpha (NCF weight): {args.alpha}")
    logger.info(f"  GMF dimension: {args.gmf_dim}")
    logger.info(f"  NCF epochs: {args.ncf_epochs}")
    logger.info(f"  NCF batch size: {args.ncf_batch_size}")
    logger.info(f"  Device: {args.device or 'auto-detect'}")
    logger.info(f"  Artifacts dir: {artifacts_dir}")
    logger.info(f"  Evaluation: {args.evaluate}")
    if args.evaluate:
        logger.info(f"  Test ratio: {args.test_ratio}")
        logger.info(f"  Split mode: {args.split_mode}")
    logger.info("="*60)
    
    # ==================== Load Data ====================
    logger.info("Loading data from database...")
    loader = DatabaseLoader(db_uri, schema)
    books_df, interactions_df = loader.load_all()
    
    logger.info(f"Loaded {len(books_df)} books, {len(interactions_df)} interactions")
    logger.info(f"Unique users: {interactions_df['user_id'].nunique()}")
    logger.info(f"Unique books in interactions: {interactions_df['book_id'].nunique()}")
    
    if len(books_df) == 0 or len(interactions_df) == 0:
        logger.error("Insufficient data for training")
        sys.exit(1)
    
    # ==================== Train/Test Split ====================
    if args.evaluate:
        if args.split_mode == 'temporal':
            train_df, test_df = DataSplitter.temporal_split(
                interactions_df,
                test_ratio=args.test_ratio,
                timestamp_col='ts'
            )
        else:
            train_df, test_df = DataSplitter.random_split(
                interactions_df,
                test_ratio=args.test_ratio
            )
    else:
        # Production mode: train on all data
        train_df = interactions_df
        test_df = None
    
    # ==================== Train Model ====================
    logger.info("Initializing Hybrid Neural Recommender...")
    recommender = HybridNeuralRecommender(
        alpha=args.alpha,
        gmf_dim=args.gmf_dim,
        ncf_epochs=args.ncf_epochs,
        ncf_batch_size=args.ncf_batch_size,
        device=args.device
    )
    
    logger.info("Training model...")
    recommender.train(books_df, train_df)
    logger.info("Training completed!")
    
    # ==================== Evaluate ====================
    if args.evaluate and test_df is not None:
        logger.info("\n" + "="*60)
        logger.info("EVALUATION ON HELD-OUT TEST SET")
        logger.info("="*60)
        
        evaluator = RecommenderEvaluator()
        
        # Compute metrics @5 and @10
        metrics_5 = evaluator.compute_metrics(recommender, test_df, train_df, k=5)
        metrics_10 = evaluator.compute_metrics(recommender, test_df, train_df, k=10)
        
        # Compute coverage
        coverage = evaluator.compute_coverage(recommender, train_df, num_samples=100)
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Metrics @5:")
        logger.info(f"  HR@5:   {metrics_5['HR@K']:.4f}")
        logger.info(f"  NDCG@5: {metrics_5['NDCG@K']:.4f}")
        logger.info(f"\nMetrics @10:")
        logger.info(f"  HR@10:   {metrics_10['HR@K']:.4f}")
        logger.info(f"  NDCG@10: {metrics_10['NDCG@K']:.4f}")
        logger.info(f"\nDiversity:")
        logger.info(f"  Coverage: {coverage:.4f}")
        logger.info(f"\nUsers:")
        logger.info(f"  Evaluated: {metrics_10['users_evaluated']}")
        logger.info(f"  Skipped: {metrics_10['users_skipped']}")
        logger.info("="*60)
    
    # ==================== Save Model ====================
    logger.info(f"\nSaving model to {artifacts_dir}...")
    recommender.save(artifacts_dir)
    logger.info("Model saved successfully!")
    
    # ==================== Test Sample Recommendations ====================
    logger.info("\n" + "="*60)
    logger.info("SAMPLE RECOMMENDATIONS")
    logger.info("="*60)
    
    evaluator = RecommenderEvaluator()
    test_user = train_df['user_id'].iloc[0]
    evaluator.print_sample_recommendations(recommender, books_df, test_user, limit=5)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
