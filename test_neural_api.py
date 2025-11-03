"""
Test Hybrid Neural Recommender API endpoints

Make sure to run server first:
    python server_neural.py

Then run this test:
    python test_neural_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8002/api/v1"


def test_neural_api():
    print("="*70)
    print("🧪 TESTING HYBRID NEURAL RECOMMENDER API")
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
    print(f"   Alpha: {info['alpha']}")
    
    if info['ncf_model']:
        print(f"   NCF users: {info['ncf_model']['num_users']}")
        print(f"   NCF items: {info['ncf_model']['num_items']}")
        print(f"   GMF dim: {info['ncf_model']['gmf_dim']}")
        print(f"   MLP dims: {info['ncf_model']['mlp_dims']}")
        print(f"   Device: {info['ncf_model']['device']}")
    
    if info['sbert_model']:
        print(f"   SBERT books: {info['sbert_model']['num_books']}")
        print(f"   SBERT user profiles: {info['sbert_model']['num_user_profiles']}")
        print(f"   Embedding dim: {info['sbert_model']['embedding_dim']}")
        print(f"   Model name: {info['sbert_model']['model_name']}")
    
    # 3. Get recommendations for a test user
    test_user = 1
    print(f"\n3️⃣ Getting recommendations for user {test_user}...")
    resp = requests.get(f"{BASE_URL}/recommendations?user_id={test_user}&limit=5")
    
    if resp.status_code == 200:
        recs = resp.json()['items']
        print(f"   📚 Top 5 recommendations:")
        for i, rec in enumerate(recs, 1):
            print(f"      {i}. Book {rec['book_id']} - Score: {rec['score']:.4f}")
            if rec['reasons']:
                print(f"         Reasons: NCF={rec['reasons'].get('ncf', 0):.4f}, "
                      f"SBERT={rec['reasons'].get('sbert', 0):.4f}")
    else:
        print(f"   ❌ Error: {resp.status_code} - {resp.text}")
    
    # 4. Get similar books
    test_book = 1
    print(f"\n4️⃣ Getting similar books to book {test_book}...")
    resp = requests.get(f"{BASE_URL}/similar?book_id={test_book}&limit=5")
    
    if resp.status_code == 200:
        similar = resp.json()['items']
        print(f"   📖 Top 5 similar books:")
        for i, item in enumerate(similar, 1):
            print(f"      {i}. Book {item['book_id']} - Similarity: {item['score']:.4f}")
    else:
        print(f"   ❌ Error: {resp.status_code} - {resp.text}")
    
    # 5. Get user profile
    print(f"\n5️⃣ Getting user profile for user {test_user}...")
    resp = requests.get(f"{BASE_URL}/user/profile/{test_user}")
    
    if resp.status_code == 200:
        profile = resp.json()
        print(f"   👤 User {test_user} profile:")
        print(f"      Interactions: {profile['num_interactions']}")
        print(f"      Profile dimension: {profile['profile_dimension']}")
        print(f"      Top books:")
        for book_id, strength in profile['top_books'][:5]:
            print(f"         - Book {book_id}: strength={strength:.2f}")
    else:
        print(f"   ❌ Error: {resp.status_code} - {resp.text}")
    
    # 6. Send feedback
    print(f"\n6️⃣ Sending user feedback...")
    feedback_data = {
        "user_id": test_user,
        "book_id": 99,
        "event": "favorite"
    }
    resp = requests.post(f"{BASE_URL}/feedback", json=feedback_data)
    
    if resp.status_code == 200:
        result = resp.json()
        print(f"   ✅ {result['status']}: {result['message']}")
    else:
        print(f"   ❌ Error: {resp.status_code}")
    
    # 7. Get user interactions
    print(f"\n7️⃣ Getting user interactions for user {test_user}...")
    resp = requests.get(f"{BASE_URL}/user/interactions/{test_user}")
    
    if resp.status_code == 200:
        interactions = resp.json()
        print(f"   📊 User {test_user} interactions:")
        print(f"      Total: {interactions['num_interactions']}")
        print(f"      Top 5 books:")
        for book in interactions['books'][:5]:
            print(f"         - Book {book['book_id']}: strength={book['strength']:.2f}")
    else:
        print(f"   ❌ Error: {resp.status_code}")
    
    print("\n" + "="*70)
    print("✅ TESTING COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    try:
        test_neural_api()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server!")
        print("Please start the server first:")
        print("  python server_neural.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
