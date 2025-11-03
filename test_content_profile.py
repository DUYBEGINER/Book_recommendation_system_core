"""
Test ContentBasedModel with User Profiles
Demonstrates weighted average TF-IDF user profiles
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.db_loader import DatabaseLoader
from src.features.content_features import ContentBasedModel
from src.utils.config import get_settings
from src.utils.logging_config import logger
import pandas as pd


def main():
    settings = get_settings()
    
    print("="*70)
    print("🧪 TESTING CONTENT-BASED MODEL WITH USER PROFILES")
    print("="*70)
    
    # 1. Load data
    print("\n1️⃣ Loading data from database...")
    loader = DatabaseLoader(settings.db_uri, settings.db_schema)
    books_df, interactions_df = loader.load_all()
    
    print(f"   📊 Loaded {len(books_df)} books, {len(interactions_df)} interactions")
    print(f"   👥 Unique users: {interactions_df['user_id'].nunique()}")
    
    # 2. Build content model
    print("\n2️⃣ Building TF-IDF content model...")
    content_model = ContentBasedModel(max_features=10000)
    content_model.fit(books_df)
    
    print(f"   ✅ Feature matrix shape: {content_model.feature_matrix.shape}")
    print(f"   📚 Books indexed: {len(content_model.book_ids)}")
    
    # 3. Build user profiles
    print("\n3️⃣ Building user profiles from interactions...")
    content_model.build_user_profiles(interactions_df)
    
    print(f"   ✅ Built profiles for {len(content_model.user_profiles)} users")
    print(f"   📊 User interactions tracked: {len(content_model.user_interactions)}")
    
    # 4. Test user profile for a specific user
    test_user = interactions_df['user_id'].iloc[0]
    print(f"\n4️⃣ Analyzing user profile for user {test_user}...")
    
    # Show user's interactions
    if test_user in content_model.user_interactions:
        interactions = content_model.user_interactions[test_user]
        print(f"\n   📚 User {test_user} has interacted with {len(interactions)} books:")
        
        sorted_interactions = sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:5]
        for book_id, strength in sorted_interactions:
            book = books_df[books_df['book_id'] == book_id]
            if len(book) > 0:
                title = book['title'].values[0]
                genre = book['genres_text'].values[0]
                print(f"      - {title[:50]} (strength={strength:.1f}, genre={genre})")
    
    # Show user's profile keywords
    print(f"\n   🔑 User {test_user}'s Top Keywords (Interests):")
    keywords = content_model.get_profile_keywords(test_user, top_n=20)
    for i, (keyword, weight) in enumerate(keywords, 1):
        print(f"      {i:2d}. {keyword:30s} (weight={weight:.4f})")
    
    # 5. Get personalized recommendations
    print(f"\n5️⃣ Getting personalized recommendations for user {test_user}...")
    
    # Filter out already interacted books
    interacted = set(content_model.user_interactions.get(test_user, {}).keys())
    
    recs = content_model.recommend_for_user(test_user, top_k=10, filter_items=interacted)
    
    print(f"\n   💡 Top 10 Content-Based Recommendations:")
    for i, (book_id, score) in enumerate(recs, 1):
        book = books_df[books_df['book_id'] == book_id]
        if len(book) > 0:
            title = book['title'].values[0]
            genre = book['genres_text'].values[0]
            author = book['authors_text'].values[0]
            print(f"      {i:2d}. {title[:45]:45s} (score={score:.4f})")
            print(f"          Genre: {genre}, Author: {author}")
    
    # 6. Compare with item-to-item similarity
    print(f"\n6️⃣ Comparing with item-to-item similarity...")
    
    # Pick a book the user liked
    if interacted:
        liked_book_id = list(interacted)[0]
        liked_book = books_df[books_df['book_id'] == liked_book_id]
        
        if len(liked_book) > 0:
            print(f"\n   📖 Reference book: {liked_book['title'].values[0]}")
            
            similar = content_model.get_similar(liked_book_id, top_k=5)
            
            print(f"\n   🔗 Similar books (item-to-item):")
            for i, (book_id, score) in enumerate(similar, 1):
                book = books_df[books_df['book_id'] == book_id]
                if len(book) > 0:
                    title = book['title'].values[0]
                    print(f"      {i}. {title[:50]} (similarity={score:.4f})")
    
    # 7. Test incremental profile update
    print(f"\n7️⃣ Testing incremental profile update...")
    
    # Simulate new interaction
    new_book_id = books_df[~books_df['book_id'].isin(interacted)]['book_id'].iloc[0]
    new_book = books_df[books_df['book_id'] == new_book_id]
    
    print(f"\n   ➕ Adding new interaction:")
    print(f"      Book: {new_book['title'].values[0]}")
    print(f"      Strength: 5.0 (loved it!)")
    
    # Update profile
    content_model.update_user_profile(test_user, new_book_id, strength=5.0)
    
    # Get new keywords
    updated_keywords = content_model.get_profile_keywords(test_user, top_n=10)
    print(f"\n   🔑 Updated Top Keywords:")
    for i, (keyword, weight) in enumerate(updated_keywords[:10], 1):
        print(f"      {i:2d}. {keyword:30s} (weight={weight:.4f})")
    
    # Get new recommendations
    updated_recs = content_model.recommend_for_user(
        test_user, 
        top_k=5, 
        filter_items=interacted | {new_book_id}
    )
    
    print(f"\n   💡 Updated Recommendations:")
    for i, (book_id, score) in enumerate(updated_recs, 1):
        book = books_df[books_df['book_id'] == book_id]
        if len(book) > 0:
            title = book['title'].values[0]
            print(f"      {i}. {title[:50]} (score={score:.4f})")
    
    # 8. Statistics
    print(f"\n8️⃣ Model Statistics:")
    print(f"   📊 TF-IDF Features: {content_model.feature_matrix.shape[1]}")
    print(f"   📚 Books indexed: {len(content_model.book_ids)}")
    print(f"   👥 User profiles: {len(content_model.user_profiles)}")
    print(f"   📈 Total interactions: {sum(len(v) for v in content_model.user_interactions.values())}")
    
    # Profile vector stats
    if test_user in content_model.user_profiles:
        profile_vec = content_model.user_profiles[test_user]
        non_zero = (profile_vec > 0).sum()
        print(f"   🎯 User {test_user} profile:")
        print(f"      - Vector dimension: {len(profile_vec)}")
        print(f"      - Non-zero features: {non_zero}")
        print(f"      - Sparsity: {(1 - non_zero/len(profile_vec))*100:.1f}%")
    
    print("\n" + "="*70)
    print("🎉 CONTENT-BASED USER PROFILE TEST COMPLETED!")
    print("="*70)
    
    # 9. Save/Load test
    print(f"\n9️⃣ Testing save/load functionality...")
    
    test_path = Path("./test_artifacts/content_model_test.pkl")
    test_path.parent.mkdir(exist_ok=True)
    
    content_model.save(test_path)
    print(f"   💾 Saved model to {test_path}")
    
    loaded_model = ContentBasedModel.load(test_path)
    print(f"   ✅ Loaded model successfully")
    print(f"   📊 Loaded profiles: {len(loaded_model.user_profiles)}")
    print(f"   📊 Loaded interactions: {len(loaded_model.user_interactions)}")
    
    # Verify loaded model works
    test_recs = loaded_model.recommend_for_user(test_user, top_k=3, filter_items=interacted)
    print(f"   ✅ Recommendations work: {len(test_recs)} items")


if __name__ == "__main__":
    main()
