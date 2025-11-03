"""
Test Online Learning API endpoints
"""
import requests
import time

BASE_URL = "http://localhost:8001/api/v1"


def test_online_learning_api():
    print("="*70)
    print("🧪 TESTING ONLINE LEARNING API")
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
    print(f"   Alpha: {info['alpha']}")
    print(f"   Online learning: {info['online_learning']}")
    
    # 3. Check online learning status
    print("\n3️⃣ Checking online learning status...")
    resp = requests.get(f"{BASE_URL}/online-learning/status")
    status = resp.json()
    print(f"   Enabled: {status['enabled']}")
    if status['enabled']:
        print(f"   Buffer: {status['buffer_size']}/{status['buffer_capacity']}")
        print(f"   Fill: {status['fill_percentage']:.1f}%")
    
    # 4. Get initial recommendations
    test_user = 11
    print(f"\n4️⃣ Getting initial recommendations for user {test_user}...")
    resp = requests.get(f"{BASE_URL}/recommendations?user_id={test_user}&limit=5")
    initial_recs = resp.json()['items']
    print(f"   📚 Top 5 recommendations:")
    for i, rec in enumerate(initial_recs, 1):
        print(f"      {i}. Book {rec['book_id']} (score={rec['score']:.4f})")
    
    # 5. Send feedback (simulate user interactions)
    print(f"\n5️⃣ Sending user feedback (simulating interactions)...")
    
    # Simulate various interactions
    feedbacks = [
        {'user_id': test_user, 'book_id': 120, 'event': 'view'},
        {'user_id': test_user, 'book_id': 121, 'event': 'bookmark'},
        {'user_id': test_user, 'book_id': 122, 'event': 'favorite'},
        {'user_id': test_user, 'book_id': 123, 'event': 'rate', 'rating_value': 5},
        {'user_id': 12, 'book_id': 124, 'event': 'view'},
        {'user_id': 12, 'book_id': 125, 'event': 'favorite'},
    ]
    
    for fb in feedbacks:
        resp = requests.post(f"{BASE_URL}/feedback", json=fb)
        result = resp.json()
        print(f"   ✅ Feedback recorded: {fb['event']} - Buffer triggered: {result.get('buffer_triggered_update', False)}")
        time.sleep(0.1)  # Small delay
    
    # 6. Check buffer status after feedback
    print(f"\n6️⃣ Checking buffer status after feedback...")
    resp = requests.get(f"{BASE_URL}/online-learning/status")
    status = resp.json()
    print(f"   Buffer: {status['buffer_size']}/{status['buffer_capacity']}")
    print(f"   Total added: {status['total_added']}")
    print(f"   Total updates: {status['total_updates']}")
    
    # 7. Force incremental update
    if status['buffer_size'] > 0:
        print(f"\n7️⃣ Forcing incremental update with {status['buffer_size']} interactions...")
        resp = requests.post(f"{BASE_URL}/online-learning/update?force=true")
        result = resp.json()
        print(f"   ✅ Update completed!")
        print(f"   Before: {result['before']['buffer_size']} interactions")
        print(f"   After: {result['after']['buffer_size']} interactions")
    
    # 8. Get updated recommendations
    print(f"\n8️⃣ Getting updated recommendations after online learning...")
    resp = requests.get(f"{BASE_URL}/recommendations?user_id={test_user}&limit=5")
    updated_recs = resp.json()['items']
    print(f"   📚 Top 5 recommendations (after update):")
    for i, rec in enumerate(updated_recs, 1):
        print(f"      {i}. Book {rec['book_id']} (score={rec['score']:.4f})")
    
    # 9. Compare results
    print(f"\n9️⃣ Comparing before/after...")
    initial_ids = [r['book_id'] for r in initial_recs]
    updated_ids = [r['book_id'] for r in updated_recs]
    
    common = set(initial_ids) & set(updated_ids)
    new_recs = set(updated_ids) - set(initial_ids)
    
    print(f"   📊 Common: {len(common)}/5")
    print(f"   🆕 New: {len(new_recs)}/5")
    if new_recs:
        print(f"   💡 New books: {new_recs}")
    
    # 10. Test disable/enable
    print(f"\n🔟 Testing disable/enable...")
    
    resp = requests.post(f"{BASE_URL}/online-learning/disable")
    print(f"   ⏸️  Disabled: {resp.json()['status']}")
    
    resp = requests.get(f"{BASE_URL}/online-learning/status")
    print(f"   Status: {resp.json()}")
    
    resp = requests.post(f"{BASE_URL}/online-learning/enable?buffer_size=50")
    print(f"   ▶️  Enabled: {resp.json()['status']}")
    
    resp = requests.get(f"{BASE_URL}/online-learning/status")
    status = resp.json()
    print(f"   Buffer capacity: {status['buffer_capacity']}")
    
    print("\n" + "="*70)
    print("🎉 ONLINE LEARNING API TEST COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    try:
        test_online_learning_api()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API server")
        print("💡 Please start the server first: python server.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
