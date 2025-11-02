"""
Test User Profile functionality
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.db_loader import DatabaseLoader
from src.models.hybrid import HybridRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger

def main():
    settings = get_settings()
    
    # Load data
    loader = DatabaseLoader(settings.db_uri, settings.db_schema)
    books_df, interactions_df = loader.load_all()
    
    # Train model
    recommender = HybridRecommender(alpha=0.6)
    recommender.train(books_df, interactions_df)
    
    # Test user profile
    test_user = interactions_df['user_id'].iloc[0]
    
    print(f"\n{'='*60}")
    print(f"USER PROFILE TEST - User {test_user}")
    print(f"{'='*60}\n")
    
    # 1. Show user's interactions
    if test_user in recommender.content_model.user_interactions:
        interactions = recommender.content_model.user_interactions[test_user]
        print(f"📚 User {test_user} has interacted with {len(interactions)} books:")
        for book_id, strength in sorted(interactions.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            book_title = books_df[books_df['book_id'] == book_id]['title'].values
            title = book_title[0] if len(book_title) > 0 else "Unknown"
            print(f"  - Book {book_id}: {title[:50]} (strength: {strength:.2f})")
    
    # 2. Show user's profile keywords
    keywords = recommender.get_user_profile_keywords(test_user, top_n=15)
    print(f"\n🔑 User {test_user}'s Top Keywords (Interests):")
    for keyword, weight in keywords:
        print(f"  - {keyword}: {weight:.4f}")
    
    # 3. Get recommendations
    print(f"\n💡 Recommendations for User {test_user}:")
    recs = recommender.recommend(test_user, limit=10)
    for i, rec in enumerate(recs, 1):
        book_id = rec['book_id']
        score = rec['score']
        source = rec['source']
        
        book_title = books_df[books_df['book_id'] == book_id]['title'].values
        title = book_title[0] if len(book_title) > 0 else "Unknown"
        
        print(f"  {i}. {title[:50]} (score: {score:.4f}, source: {source})")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()