"""
Script để test model recommendation đã train
"""
from pathlib import Path
from src.models.hybrid import HybridRecommender
from src.data.db_loader import DatabaseLoader
from src.utils.config import get_settings
import pandas as pd

def test_recommendations():
    """Test personalized recommendations"""
    settings = get_settings()
    
    print("=" * 60)
    print("🔍 LOADING MODEL...")
    print("=" * 60)
    
    # Load model
    artifacts_dir = Path(settings.artifacts_dir)
    recommender = HybridRecommender.load(artifacts_dir, alpha=settings.alpha)
    
    print(f"✅ Model loaded successfully from {artifacts_dir}")
    print()
    
    # Load data to get user/book info
    loader = DatabaseLoader(settings.db_uri, settings.db_schema)
    books_df, interactions_df = loader.load_all()
    
    print("=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)
    print(f"Total Books: {len(books_df)}")
    print(f"Total Interactions: {len(interactions_df)}")
    print(f"Unique Users: {interactions_df['user_id'].nunique()}")
    print(f"Unique Books in interactions: {interactions_df['book_id'].nunique()}")
    print()
    
    # Get some test users
    test_users = interactions_df['user_id'].unique()[:3]
    
    print("=" * 60)
    print("🎯 TEST PERSONALIZED RECOMMENDATIONS")
    print("=" * 60)
    
    for user_id in test_users:
        print(f"\n👤 User ID: {user_id}")
        print("-" * 60)
        
        # Get user's history
        user_history = interactions_df[interactions_df['user_id'] == user_id]
        print(f"📚 User's interaction history ({len(user_history)} items):")
        for _, row in user_history.head(5).iterrows():
            book_info = books_df[books_df['book_id'] == row['book_id']]
            if not book_info.empty:
                title = book_info.iloc[0]['title']
                print(f"  - {title} (strength: {row['strength']:.2f}, type: {row['type']})")
        
        # Get recommendations
        recs = recommender.recommend(user_id, limit=5)
        
        print(f"\n💡 Top 5 Recommendations:")
        if recs:
            for i, rec in enumerate(recs, 1):
                book_info = books_df[books_df['book_id'] == rec['book_id']]
                if not book_info.empty:
                    book = book_info.iloc[0]
                    print(f"  {i}. {book['title']}")
                    print(f"     Score: {rec['score']:.4f}")
                    print(f"     Reasons: CF={rec['reasons']['cf']:.4f}, "
                          f"Content={rec['reasons']['content']:.4f}, "
                          f"Pop={rec['reasons']['pop']:.4f}")
                    print(f"     Author: {book['authors_text']}")
                    print(f"     Genre: {book['genres_text']}")
                else:
                    print(f"  {i}. Book ID {rec['book_id']} (Score: {rec['score']:.4f})")
        else:
            print("  ⚠️ No recommendations available")
        
        print()
    
    print("=" * 60)
    print("🔗 TEST SIMILAR BOOKS (Content-Based)")
    print("=" * 60)
    
    # Test similar books for a random book
    test_book_id = interactions_df['book_id'].iloc[0]
    test_book = books_df[books_df['book_id'] == test_book_id].iloc[0]
    
    print(f"\n📖 Source Book: {test_book['title']}")
    print(f"   Author: {test_book['authors_text']}")
    print(f"   Genre: {test_book['genres_text']}")
    print()
    
    similar = recommender.similar_books(test_book_id, limit=5)
    
    print(f"🔍 Top 5 Similar Books:")
    if similar:
        for i, sim in enumerate(similar, 1):
            book_info = books_df[books_df['book_id'] == sim['book_id']]
            if not book_info.empty:
                book = book_info.iloc[0]
                print(f"  {i}. {book['title']}")
                print(f"     Similarity Score: {sim['score']:.4f}")
                print(f"     Author: {book['authors_text']}")
                print(f"     Genre: {book['genres_text']}")
            else:
                print(f"  {i}. Book ID {sim['book_id']} (Score: {sim['score']:.4f})")
    else:
        print("  ⚠️ No similar books found")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_recommendations()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
