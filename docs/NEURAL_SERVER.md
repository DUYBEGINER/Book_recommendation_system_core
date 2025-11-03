# Hybrid Neural Recommender Server

Server API cho Hybrid Neural Recommender sử dụng **NCF (Neural Collaborative Filtering) + SBERT (Semantic BERT)**.

## 🏗️ Kiến trúc

```
User Request → FastAPI Server (port 8002)
                     ↓
          HybridNeuralRecommender
                     ↓
         ┌───────────┴───────────┐
         ↓                       ↓
    NCF Model              SBERT Model
  (User-Item Deep)      (Semantic Content)
         ↓                       ↓
    Predictions            Embeddings
         └───────────┬───────────┘
                     ↓
            Weighted Fusion (alpha)
                     ↓
              Recommendations
```

## 🚀 Khởi động Server

### 1. Train Neural Model (nếu chưa có)

```bash
python train_neural.py --evaluate --alpha 0.6 --ncf-epochs 20
```

Model sẽ được lưu vào `./artifacts_neural/`

### 2. Khởi động Server

```bash
python server_neural.py
```

Server sẽ chạy tại: **http://localhost:8002**

### 3. Test API

```bash
python test_neural_api.py
```

## 📡 API Endpoints

### Health Check
```bash
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### Get Recommendations
```bash
GET /api/v1/recommendations?user_id=1&limit=10
```

**Response:**
```json
{
  "user_id": 1,
  "limit": 10,
  "items": [
    {
      "book_id": 42,
      "score": 0.8532,
      "reasons": {
        "ncf": 0.7821,
        "sbert": 0.9243,
        "pop": 0.0
      }
    }
  ]
}
```

### Get Similar Books (SBERT)
```bash
GET /api/v1/similar?book_id=1&limit=10
```

**Response:**
```json
{
  "book_id": 1,
  "items": [
    {
      "book_id": 15,
      "score": 0.9234
    }
  ]
}
```

### Record Feedback
```bash
POST /api/v1/feedback
Content-Type: application/json

{
  "user_id": 1,
  "book_id": 42,
  "event": "favorite"
}
```

**Events:** `view`, `bookmark`, `favorite`, `rate`

**Response:**
```json
{
  "status": "recorded",
  "message": "Feedback logged. Neural model requires batch retraining.",
  "user_id": 1,
  "book_id": 42,
  "event": "favorite",
  "strength": 3.0
}
```

### Get Model Info
```bash
GET /api/v1/model/info
```

**Response:**
```json
{
  "status": "loaded",
  "model_type": "HybridNeuralRecommender",
  "alpha": 0.6,
  "ncf_model": {
    "num_users": 100,
    "num_items": 500,
    "gmf_dim": 64,
    "mlp_dims": [128, 64, 32],
    "device": "cpu"
  },
  "sbert_model": {
    "num_books": 500,
    "num_user_profiles": 100,
    "embedding_dim": 768,
    "model_name": "keepitreal/vietnamese-sbert"
  }
}
```

### Get User Profile
```bash
GET /api/v1/user/profile/1?top_n=20
```

**Response:**
```json
{
  "user_id": 1,
  "num_interactions": 25,
  "profile_dimension": 768,
  "top_books": [
    [42, 5.0],
    [15, 4.5]
  ],
  "message": "SBERT uses dense embeddings (no sparse keywords like TF-IDF)"
}
```

### Get User Interactions
```bash
GET /api/v1/user/interactions/1
```

### Retrain Model
```bash
POST /api/v1/retrain
```

⚠️ **Warning:** Retraining là tốn tài nguyên (NCF training + SBERT encoding)

## ⚙️ Configuration

### Model Hyperparameters

File `train_neural.py`:
- `--alpha`: NCF weight (0-1), SBERT weight = 1-alpha
- `--gmf-dim`: GMF embedding dimension (default: 64)
- `--ncf-epochs`: NCF training epochs (default: 20)
- `--ncf-batch-size`: NCF batch size (default: 256)
- `--device`: cuda or cpu

### Server Settings

File `server_neural.py`:
- **Port:** 8002 (không conflict với server.py port 8001)
- **Artifacts:** `./artifacts_neural/`
- **CORS:** Enabled for all origins

## 🔄 So sánh với Classic Server

| Feature | Classic Server (8001) | Neural Server (8002) |
|---------|----------------------|---------------------|
| **CF Model** | ALS (Matrix Factorization) | NCF (Deep Learning) |
| **Content Model** | Ridge Regression | SBERT (Transformers) |
| **Online Learning** | ✅ Incremental Updates | ❌ Batch Retrain Only |
| **Training Speed** | Fast (seconds) | Slow (minutes) |
| **Prediction Speed** | Very Fast | Fast (GPU) / Moderate (CPU) |
| **Semantic Understanding** | TF-IDF Keywords | Dense Embeddings |
| **Cold Start** | Popularity | Semantic Similarity |

## 🎯 Khi nào dùng Neural Server?

### ✅ Nên dùng khi:
1. **Semantic understanding quan trọng** - Cần hiểu nghĩa sâu của nội dung
2. **Có GPU** - Training và inference nhanh hơn
3. **Data đủ lớn** - Neural models cần nhiều data
4. **Offline batch processing** - Không cần real-time updates

### ❌ Không nên dùng khi:
1. **Cần online learning** - NCF không hỗ trợ incremental updates
2. **Resource hạn chế** - Training tốn GPU/RAM
3. **Data nhỏ** - Neural models dễ overfit
4. **Cần inference siêu nhanh** - Classic ALS nhanh hơn

## 🐛 Troubleshooting

### "Model not loaded yet"
```bash
# Train model trước
python train_neural.py --evaluate
```

### "CUDA out of memory"
```bash
# Giảm batch size hoặc dùng CPU
python train_neural.py --device cpu --ncf-batch-size 128
```

### Port 8002 đã được dùng
```bash
# Tìm process đang dùng port
netstat -ano | findstr :8002

# Hoặc đổi port trong server_neural.py
uvicorn.run(app, host="0.0.0.0", port=8003)
```

## 📊 Performance Benchmarks

Dựa trên dataset 10 users, 50 books:

| Metric | Classic (ALS+Ridge) | Neural (NCF+SBERT) |
|--------|---------------------|-------------------|
| **Training Time** | ~15s | ~2-3 minutes |
| **Inference Time** | ~50ms | ~150ms (CPU) / ~30ms (GPU) |
| **HR@10** | 0.45-0.50 | 0.50-0.55 |
| **NDCG@10** | 0.18-0.22 | 0.19-0.25 |
| **Coverage** | 60-70% | 70-80% |

## 🔮 Future Improvements

- [ ] **Online Learning for NCF**: Implement incremental matrix updates
- [ ] **Model Ensemble**: Combine Classic + Neural predictions
- [ ] **A/B Testing**: Framework để test Neural vs Classic
- [ ] **Caching**: Redis cache cho SBERT embeddings
- [ ] **Batch Inference**: Optimize cho multiple user requests
- [ ] **Model Serving**: TorchServe hoặc TensorFlow Serving

## 📝 License

MIT License - Tự do sử dụng và chỉnh sửa
