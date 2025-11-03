"""
Test Hybrid Implicit ALS + SBERT Recommender API endpoints

Make sure to run server first:
    python server_implicit_sbert.py

Then run this test:
    python test_implicit_sbert_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8003/api/v1"


def test_implicit_sbert_api():
    print("="*70)
    print("🧪 TESTING HYBRID IMPLICIT ALS + SBERT RECOMMENDER API")
    print("="*70)
    
    # 1. Check health
    print("\n1️⃣ Checking API health...")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {resp.json()['status']}")
    print(f"   Models loaded: {resp.json()['models_loaded']}")
    
    # 2. Get model info
    print("\n2️⃣ Getting model info...")
    resp = requests.get(f"{BASE_URL}/model/info")
    info = resp.json()
    print(f"   Model type: {info['model_type']}")
    print(f"   Alpha (ALS:SBERT): {info['alpha']}:{1-info['alpha']}")
    
    if info['als_model']:
        print(f"\n   ALS Model:")
        print(f"     Users: {info['als_model']['num_users']}")
        print(f"     Items: {info['als_model']['num_items']}")
        print(f"     Factors: {info['als_model']['factors']}")
        print(f"     Iterations: {info['als_model']['iterations']}")
    
    if info['sbert_model']:
        print(f"\n   SBERT Model:")
        print(f"     Books: {info['sbert_model']['num_books']}")
        print(f"     User profiles: {info['sbert_model']['num_user_profiles']}")
        print(f"     Embedding dim: {info['sbert_model']['embedding_dim']}")
        print(f"     Model name: {info['sbert_model']['model_name']}")
    
    # 3. Get recommendations for a test user
    test_user = 1
    print(f"\n3️⃣ Getting recommendations for user {test_user}...")
    resp = requests.get(f"{BASE_URL}/recommendations?user_id={test_user}&limit=5")
    
    if resp.status_code == 200:
        recs = resp.json()
        print(f"   User ID: {recs['user_id']}")
        print(f"   Recommendations:")
        for item in recs['items']:
            print(f"     Book {item['book_id']}: score={item['score']:.4f}")
            print(f"       ALS={item['reasons']['als']:.4f}, SBERT={item['reasons']['sbert']:.4f}, Pop={item['reasons']['pop']:.4f}")
    else:
        print(f"   ❌ Error: {resp.json()['detail']}")
    
    # 4. Get similar books
    test_book = 1
    print(f"\n4️⃣ Getting similar books to book {test_book}...")
    resp = requests.get(f"{BASE_URL}/similar?book_id={test_book}&limit=5")
    
    if resp.status_code == 200:
        similar = resp.json()
        print(f"   Similar to book {similar['book_id']}:")
        for item in similar['items']:
            print(f"     Book {item['book_id']}: score={item['score']:.4f}")
    else:
        print(f"   ❌ Error: {resp.json()['detail']}")
    
    # 5. Get user profile
    print(f"\n5️⃣ Getting user profile for user {test_user}...")
    resp = requests.get(f"{BASE_URL}/user/profile/{test_user}")
    
    if resp.status_code == 200:
        profile = resp.json()
        print(f"   User ID: {profile['user_id']}")
        print(f"   Num interactions: {profile['num_interactions']}")
        print(f"   Profile dimension: {profile['profile_dimension']}")
        print(f"   Top books:")
        for book_id, strength in profile['top_books'][:5]:
            print(f"     Book {book_id}: strength={strength:.2f}")
    else:
        print(f"   ⚠️  User profile not found (may be cold start user)")
    
    # 6. Record feedback
    print(f"\n6️⃣ Recording feedback...")
    resp = requests.post(f"{BASE_URL}/feedback", json={
        "user_id": test_user,
        "book_id": 10,
        "event": "favorite"
    })
    
    if resp.status_code == 200:
        feedback = resp.json()
        print(f"   Status: {feedback['status']}")
        print(f"   Strength: {feedback['strength']}")
        print(f"   Message: {feedback['message']}")
    else:
        print(f"   ❌ Error: {resp.json()['detail']}")
    
    # 7. Get user interactions
    print(f"\n7️⃣ Getting user interactions for user {test_user}...")
    resp = requests.get(f"{BASE_URL}/user/interactions/{test_user}")
    
    if resp.status_code == 200:
        interactions = resp.json()
        print(f"   User ID: {interactions['user_id']}")
        print(f"   Num interactions: {interactions['num_interactions']}")
        print(f"   Top interactions:")
        for book in interactions['books'][:5]:
            print(f"     Book {book['book_id']}: strength={book['strength']:.2f}")
    else:
        print(f"   ⚠️  No interactions found")
    
    print("\n" + "="*70)
    print("✅ API TEST COMPLETED")
    print("="*70)


if __name__ == "__main__":
    try:
        test_implicit_sbert_api()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server")
        print("Make sure the server is running:")
        print("  python server_implicit_sbert.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")
