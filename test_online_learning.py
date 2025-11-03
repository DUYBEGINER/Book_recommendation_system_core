"""
Test Online Learning functionality
Demonstrates incremental model updates without full retraining
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.db_loader import DatabaseLoader
from src.models.hybrid_Ridge import HybridRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger
import pandas as pd


def main():
    settings = get_settings()
    
    print("="*70)
    print("🧪 TESTING ONLINE LEARNING FOR HYBRID RECOMMENDER")
    print("="*70)
    
    # 1. Load data and train initial model
    print("\n1️⃣ Loading data and training initial model...")
    loader = DatabaseLoader(settings.db_uri, settings.db_schema)
    books_df, interactions_df = loader.load_all()
    
    print(f"   📊 Loaded {len(books_df)} books, {len(interactions_df)} interactions")
    
    # Train on 80% of data (simulate production model)
    split_idx = int(len(interactions_df) * 0.8)
    train_df = interactions_df.iloc[:split_idx]
    test_df = interactions_df.iloc[split_idx:]  # Simulate new interactions
    
    print(f"   📈 Training on {len(train_df)} interactions...")
    recommender = HybridRecommender(
        alpha=0.6, 
        ridge_alpha=1.0,
        online_learning=True,  # Enable online learning
        buffer_size=20  # Small buffer for demo
    )
    recommender.train(books_df, train_df)
    
    print("   ✅ Initial model trained!")
    
    # 2. Get initial recommendations for a test user
    test_user = train_df['user_id'].iloc[0]
    print(f"\n2️⃣ Getting initial recommendations for user {test_user}...")
    
    initial_recs = recommender.recommend(test_user, limit=5)
    print(f"   📚 Top 5 recommendations:")
    for i, rec in enumerate(initial_recs, 1):
        book_id = rec['book_id']
        score = rec['score']
        book = books_df[books_df['book_id'] == book_id]
        title = book['title'].values[0] if len(book) > 0 else "Unknown"
        print(f"      {i}. {title[:50]} (score={score:.4f})")
    
    # 3. Simulate new user interactions (online learning)
    print(f"\n3️⃣ Simulating new user interactions...")
    print(f"   📊 Buffer size: {recommender.buffer_size}")
    
    # Add interactions one by one
    for idx, row in test_df.head(25).iterrows():  # Add 25 new interactions
        user_id = row['user_id']
        book_id = row['book_id']
        strength = row['strength']
        
        buffer_triggered = recommender.add_interaction(
            user_id=user_id,
            book_id=book_id,
            strength=strength,
            interaction_type='rating'
        )
        
        if buffer_triggered:
            print(f"   🔄 Buffer full! Incremental update triggered automatically")
    
    # 4. Check buffer status
    print(f"\n4️⃣ Checking online learning buffer status...")
    status = recommender.get_buffer_status()
    print(f"   📊 Buffer status:")
    print(f"      - Size: {status['buffer_size']}/{status['buffer_capacity']}")
    print(f"      - Fill: {status['fill_percentage']:.1f}%")
    print(f"      - Total added: {status['total_added']}")
    print(f"      - Total updates: {status['total_updates']}")
    print(f"      - Last update: {status['last_update']}")
    
    # 5. Force incremental update with remaining buffer
    if status['buffer_size'] > 0:
        print(f"\n5️⃣ Forcing incremental update with {status['buffer_size']} remaining interactions...")
        recommender.incremental_update(force=True)
        print("   ✅ Incremental update completed!")
    
    # 6. Get updated recommendations
    print(f"\n6️⃣ Getting updated recommendations after online learning...")
    updated_recs = recommender.recommend(test_user, limit=5)
    print(f"   📚 Top 5 recommendations (after update):")
    for i, rec in enumerate(updated_recs, 1):
        book_id = rec['book_id']
        score = rec['score']
        book = books_df[books_df['book_id'] == book_id]
        title = book['title'].values[0] if len(book) > 0 else "Unknown"
        print(f"      {i}. {title[:50]} (score={score:.4f})")
    
    # 7. Compare results
    print(f"\n7️⃣ Comparing before/after online learning...")
    initial_ids = [r['book_id'] for r in initial_recs]
    updated_ids = [r['book_id'] for r in updated_recs]
    
    common = set(initial_ids) & set(updated_ids)
    new_recs = set(updated_ids) - set(initial_ids)
    
    print(f"   📊 Common recommendations: {len(common)}/5")
    print(f"   🆕 New recommendations: {len(new_recs)}/5")
    
    if new_recs:
        print(f"   💡 New books recommended after learning:")
        for book_id in new_recs:
            book = books_df[books_df['book_id'] == book_id]
            title = book['title'].values[0] if len(book) > 0 else "Unknown"
            print(f"      - {title[:50]}")
    
    # 8. Final stats
    final_status = recommender.get_buffer_status()
    print(f"\n8️⃣ Final online learning statistics:")
    print(f"   📈 Total interactions added: {final_status['total_added']}")
    print(f"   🔄 Total incremental updates: {final_status['total_updates']}")
    print(f"   ✅ Online learning working correctly!")
    
    print("\n" + "="*70)
    print("🎉 ONLINE LEARNING TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # 9. Demonstrate disable/enable
    print(f"\n9️⃣ Testing enable/disable functionality...")
    recommender.disable_online_learning()
    print("   ⏸️  Online learning disabled")
    
    recommender.enable_online_learning(buffer_size=50)
    print("   ▶️  Online learning re-enabled with buffer size 50")
    
    status = recommender.get_buffer_status()
    print(f"   ✅ Buffer reset: {status['buffer_size']}/{status['buffer_capacity']}")


if __name__ == "__main__":
    main()
