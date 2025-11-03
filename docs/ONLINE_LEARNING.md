# Online Learning for Hybrid Recommender

## 📚 Overview

Online Learning cho phép model học tăng dần từ user interactions mới **mà không cần retrain toàn bộ**. Điều này quan trọng cho production vì:

1. ⚡ **Nhanh chóng**: Cập nhật model trong vài giây thay vì vài phút
2. 🔄 **Real-time**: Phản ánh user behavior ngay lập tức
3. 💰 **Tiết kiệm**: Không cần compute resources để retrain
4. 🎯 **Chính xác**: Model luôn up-to-date với user preferences mới nhất

## 🏗️ Architecture

```
User Interaction → Buffer (100 interactions) → Incremental Update
                                               ↓
                                    Update Content Model (Ridge/Weighted)
                                    Update Popularity Scores
                                    (CF skipped - requires full retrain)
```

### Components:

1. **Interaction Buffer**: Queue FIFO, tự động trigger update khi đầy
2. **Incremental Content Update**: 
   - Ridge: Retrain user models với data mới
   - Weighted: Recompute weighted average profiles
3. **Popularity Update**: Cộng dồn interaction counts
4. **CF Update**: Skipped (ALS requires full retrain, consider SGD-based CF)

## 🚀 Usage

### 1. Enable Online Learning (Python)

```python
from src.models.hybrid_Ridge import HybridRecommender

# Train initial model
recommender = HybridRecommender(
    alpha=0.6,
    use_ridge=True,
    online_learning=True,  # Enable
    buffer_size=100        # Buffer capacity
)
recommender.train(books_df, interactions_df)

# Add new interactions
recommender.add_interaction(
    user_id=123,
    book_id=456,
    strength=5.0,
    interaction_type='rating'
)

# Force update
recommender.incremental_update(force=True)

# Check status
status = recommender.get_buffer_status()
print(status)
```

### 2. API Endpoints

#### Record Feedback (Auto-trigger)
```bash
curl -X POST http://localhost:8001/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123,
    "book_id": 456,
    "event": "rate",
    "rating_value": 5
  }'
```

**Events & Strengths:**
- `view`: 1.0 (implicit signal)
- `bookmark`: 2.0 (saved for later)
- `favorite`: 3.0 (strong positive)
- `rate`: 1-5 (explicit rating)

#### Check Buffer Status
```bash
curl http://localhost:8001/api/v1/online-learning/status
```

Response:
```json
{
  "enabled": true,
  "buffer_size": 45,
  "buffer_capacity": 100,
  "fill_percentage": 45.0,
  "total_added": 245,
  "total_updates": 2,
  "last_update": "2025-11-03T12:34:56"
}
```

#### Force Incremental Update
```bash
curl -X POST http://localhost:8001/api/v1/online-learning/update?force=true
```

#### Enable/Disable
```bash
# Enable with custom buffer size
curl -X POST http://localhost:8001/api/v1/online-learning/enable?buffer_size=200

# Disable
curl -X POST http://localhost:8001/api/v1/online-learning/disable
```

## 📊 Monitoring

### Model Info (includes online learning stats)
```bash
curl http://localhost:8001/api/v1/model/info
```

Response:
```json
{
  "status": "loaded",
  "alpha": 0.6,
  "use_ridge": true,
  "online_learning": {
    "enabled": true,
    "buffer_size": 45,
    "buffer_capacity": 100,
    "total_updates": 2
  },
  "content_model": {
    "num_books": 250,
    "feature_dim": 10000,
    "model_type": "Ridge"
  }
}
```

## 🧪 Testing

### Local Test (Python)
```bash
python test_online_learning.py
```

Output:
```
🧪 TESTING ONLINE LEARNING FOR HYBRID RECOMMENDER
1️⃣ Loading data and training initial model...
   ✅ Initial model trained!
2️⃣ Getting initial recommendations...
3️⃣ Simulating new user interactions...
   🔄 Buffer full! Incremental update triggered
4️⃣ Checking buffer status...
   📊 Total updates: 1
5️⃣ Getting updated recommendations...
   💡 New books recommended after learning
🎉 ONLINE LEARNING TEST COMPLETED SUCCESSFULLY!
```

### API Test
```bash
# Start server
python server.py

# In another terminal
python test_online_api.py
```

## ⚙️ Configuration

### Buffer Size Tuning

| Buffer Size | Update Frequency | Latency | Use Case |
|-------------|------------------|---------|----------|
| 10-50 | Very high | Low | Real-time apps, A/B testing |
| 100-200 | High | Medium | **Recommended for production** |
| 500-1000 | Low | High | Batch processing, low traffic |

**Default: 100** (good balance)

### When to Use Full Retrain vs Online Learning

| Scenario | Approach |
|----------|----------|
| New user interactions | Online Learning ✅ |
| User profile updates | Online Learning ✅ |
| Popularity changes | Online Learning ✅ |
| New books added | Full Retrain 🔄 |
| Model architecture change | Full Retrain 🔄 |
| Weekly/Monthly refresh | Full Retrain 🔄 |

## 🔬 How It Works

### Ridge Content Model Update

```python
# User có interaction mới
old_interactions = {book1: 4.0, book2: 3.0}
new_interactions = {book3: 5.0, book4: 4.0}

# Merge
all_interactions = {book1: 4.0, book2: 3.0, book3: 5.0, book4: 4.0}

# Retrain Ridge model cho user này
X_train = feature_matrix[book_ids]  # TF-IDF features
y_train = [4.0, 3.0, 5.0, 4.0]      # Ratings

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# User's new weights learned! ✅
```

### Popularity Update

```python
# Old popularity
popularity = {book1: 100, book2: 50}

# New interactions
new_counts = {book1: 5, book3: 2}

# Update
popularity[book1] += 5  # 105
popularity[book3] = 2   # New entry

# Re-sort ✅
```

## 📈 Performance

**Benchmark** (250 books, 586 interactions):

| Operation | Time | Speedup vs Full Retrain |
|-----------|------|------------------------|
| Full Retrain | 15-20s | 1x baseline |
| Incremental Update (100 interactions) | **2-3s** | **5-7x faster** |
| Add Single Interaction | <10ms | 1500x faster |

**Memory**: O(buffer_size) additional overhead

## 🚨 Limitations

1. **CF Model Not Updated**: ALS requires full retrain
   - Solution: Consider SGD-based CF (e.g., SVD++, Neural CF)
   
2. **New Books**: Cannot be recommended until full retrain
   - Solution: Use content-based for cold-start items
   
3. **Buffer Loss**: If server crashes before update, buffer is lost
   - Solution: Persist buffer to disk or database

## 🔮 Future Enhancements

1. ✅ **Implemented**: Ridge/Weighted content update, Popularity update
2. 🚧 **TODO**: Incremental CF with SGD
3. 🚧 **TODO**: Persist buffer to database
4. 🚧 **TODO**: A/B testing framework
5. 🚧 **TODO**: User-level learning rate tuning

## 📚 References

- [Online Learning for Recommender Systems](https://arxiv.org/abs/1711.03705)
- [Incremental Matrix Factorization](https://ieeexplore.ieee.org/document/6748996)
- [Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

## 💡 Production Checklist

- [x] Enable online learning at startup
- [x] Monitor buffer fill rate
- [x] Set up alerts for update failures
- [ ] Persist buffer to Redis/Database
- [ ] Schedule periodic full retrains (weekly)
- [ ] A/B test online vs offline models
- [ ] Track online learning metrics (update latency, model drift)

---

**Author**: DUYBEGINER  
**Date**: 2025-11-03  
**Version**: 1.0.0
