"""
Test script để demo retrain model qua API
"""
import requests
import time

BASE_URL = "http://localhost:8001/api/v1"

def test_retrain_flow():
    """Test complete retrain flow"""
    
    print("=" * 60)
    print("🧪 TESTING MODEL RETRAIN FLOW")
    print("=" * 60)
    
    # 1. Check initial health
    print("\n1️⃣ Checking initial health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 2. Get model info
    print("\n2️⃣ Getting current model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    if response.status_code == 200:
        info = response.json()
        print(f"   Alpha: {info['alpha']}")
        print(f"   CF Users: {info['cf_model']['num_users']}")
        print(f"   CF Items: {info['cf_model']['num_items']}")
        print(f"   Content Features: {info['content_model']['feature_dim']}")
    else:
        print(f"   Error: {response.status_code}")
    
    # 3. Test recommendation before retrain
    print("\n3️⃣ Testing recommendation (before retrain)...")
    response = requests.get(f"{BASE_URL}/recommendations?user_id=1&limit=5")
    if response.status_code == 200:
        recs = response.json()
        print(f"   Got {len(recs['items'])} recommendations for user {recs['user_id']}")
        for item in recs['items'][:3]:
            print(f"   - Book {item['book_id']}: score={item['score']:.4f}")
    else:
        print(f"   Error: {response.status_code}")
    
    # 4. Trigger retrain
    print("\n4️⃣ Triggering model retrain...")
    response = requests.post(f"{BASE_URL}/retrain")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # 5. Check health during retrain
    print("\n5️⃣ Checking health during retrain...")
    for i in range(5):
        time.sleep(2)
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"   [{i+1}/5] Status: {data['status']}, Models loaded: {data['models_loaded']}")
        
        if data['status'] != 'retraining':
            print("   ✅ Retraining completed!")
            break
    
    # 6. Get updated model info
    print("\n6️⃣ Getting updated model info...")
    response = requests.get(f"{BASE_URL}/model/info")
    if response.status_code == 200:
        info = response.json()
        print(f"   CF Users: {info['cf_model']['num_users']}")
        print(f"   CF Items: {info['cf_model']['num_items']}")
        print(f"   Is Retraining: {info['is_retraining']}")
    
    # 7. Test recommendation after retrain
    print("\n7️⃣ Testing recommendation (after retrain)...")
    response = requests.get(f"{BASE_URL}/recommendations?user_id=1&limit=5")
    if response.status_code == 200:
        recs = response.json()
        print(f"   Got {len(recs['items'])} recommendations")
        for item in recs['items'][:3]:
            print(f"   - Book {item['book_id']}: score={item['score']:.4f}")
    
    # 8. Try to retrain again (should fail - already in progress)
    print("\n8️⃣ Testing concurrent retrain (should fail if still running)...")
    response = requests.post(f"{BASE_URL}/retrain")
    print(f"   Status: {response.status_code}")
    if response.status_code == 409:
        print("   ✅ Correctly rejected concurrent retrain")
    else:
        print(f"   Response: {response.json()}")
    
    print("\n" + "=" * 60)
    print("✅ TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_retrain_flow()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server")
        print("   Make sure server is running: python server.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
