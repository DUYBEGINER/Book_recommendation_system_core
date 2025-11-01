# API Retrain Guide

## Endpoints mới

### 1. GET `/api/v1/model/info`
Lấy thông tin về model hiện tại.

**Response:**
```json
{
  "status": "loaded",
  "alpha": 0.6,
  "cf_model": {
    "num_users": 14,
    "num_items": 175,
    "matrix_nnz": 291
  },
  "content_model": {
    "num_books": 250,
    "feature_dim": 10000
  },
  "is_retraining": false
}
```

### 2. POST `/api/v1/retrain`
Trigger retraining model với data mới từ database.

**Response (Success):**
```json
{
  "status": "accepted",
  "message": "Model retraining started in background. Check /health for status."
}
```

**Response (Conflict - đang retrain):**
```json
{
  "detail": "Retraining already in progress"
}
```
Status code: `409 Conflict`

### 3. GET `/api/v1/health` (Updated)
Health check với status retraining.

**Response:**
```json
{
  "status": "retraining",  // hoặc "ok"
  "models_loaded": true
}
```

## Cách sử dụng

### 1. Start server
```bash
python server.py
```

Server sẽ chạy ở `http://localhost:8001`

### 2. Check model hiện tại
```bash
curl http://localhost:8001/api/v1/model/info
```

### 3. Trigger retrain
```bash
curl -X POST http://localhost:8001/api/v1/retrain
```

### 4. Monitor progress
```bash
# Check status
curl http://localhost:8001/api/v1/health

# Hoặc xem logs trong terminal của server
```

### 5. Test recommendations
```bash
# Trước khi retrain
curl "http://localhost:8001/api/v1/recommendations?user_id=1&limit=5"

# Sau khi retrain
curl "http://localhost:8001/api/v1/recommendations?user_id=1&limit=5"
```

## Automated Test

Chạy script test tự động:

```bash
python test_retrain.py
```

Script này sẽ:
1. ✅ Check health ban đầu
2. ✅ Lấy thông tin model
3. ✅ Test recommendations trước retrain
4. ✅ Trigger retrain
5. ✅ Monitor progress
6. ✅ Test recommendations sau retrain
7. ✅ Verify concurrent retrain prevention

## Flow hoạt động

```
1. API nhận POST /retrain
   ↓
2. Set is_retraining = True
   ↓
3. Background task:
   - Load data từ database
   - Train model mới
   - Save artifacts
   - Hot-swap model cũ → mới
   ↓
4. Set is_retraining = False
   ↓
5. API tiếp tục dùng model mới
```

## Notes

### Hot-swap model
- Model cũ vẫn phục vụ requests trong lúc train
- Khi train xong, swap sang model mới ngay lập tức
- Không cần restart server
- Zero downtime!

### Concurrent protection
- Chỉ cho phép 1 retrain task chạy cùng lúc
- Nếu đang retrain, POST /retrain sẽ trả về `409 Conflict`

### Production considerations

1. **Authentication**: Thêm API key hoặc JWT vào `/retrain` endpoint
2. **Scheduling**: Dùng cron job hoặc Celery để retrain định kỳ
3. **Monitoring**: Track metrics trước/sau retrain
4. **Rollback**: Lưu model cũ để rollback nếu model mới kém hơn

## Example with curl

```bash
# 1. Check initial state
curl http://localhost:8001/api/v1/model/info

# 2. Trigger retrain
curl -X POST http://localhost:8001/api/v1/retrain

# 3. Monitor (repeat until status != "retraining")
while true; do
  curl http://localhost:8001/api/v1/health
  sleep 2
done

# 4. Verify new model
curl http://localhost:8001/api/v1/model/info
```

## Troubleshooting

### Lỗi: "Model not loaded"
- Kiểm tra artifacts directory có tồn tại không
- Chạy `python train.py` để train model lần đầu

### Lỗi: "Retraining already in progress"
- Đợi retrain hiện tại hoàn thành
- Check `/health` để xem status

### Model không update sau retrain
- Check logs trong server terminal
- Verify database có data mới không
- Check artifacts directory có được update không
