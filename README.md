# Book Recommendation System

Production-ready hybrid recommendation microservice for book e-commerce platform.

## Features

- **Dual Recommendation Engines**:
  - **Classic Server** (Port 8001): ALS + Ridge Regression with Online Learning
  - **Neural Server** (Port 8002): NCF + SBERT with Deep Semantic Understanding
- **Hybrid Recommendations**: Combines collaborative filtering and content-based filtering
- **Multi-signal**: Ratings (explicit), reading history, favorites, bookmarks (implicit)
- **Cold-start handling**: Popularity and content-based fallbacks
- **Online Learning**: Incremental updates without full retraining (Classic only)
- **Fast inference**: <200ms for top-10 recommendations with 100k+ books
- **RESTful API**: FastAPI with OpenAPI documentation
- **Production-ready**: Docker, PostgreSQL, logging, metrics

## Quick Start

### Option 1: Launch Both Servers

```bash
# Train both models
python train.py --evaluate
python train_neural.py --evaluate

# Start both servers
python start_servers.py

# Classic API: http://localhost:8001
# Neural API:  http://localhost:8002
```

### Option 2: Classic Server Only (Fast, Online Learning)

```bash
# Train classic model
python train.py --evaluate

# Start server
python server.py

# Test API
python test_online_api.py
```

### Option 3: Neural Server Only (Semantic, Deep Learning)

### Option 3: Neural Server Only (Semantic, Deep Learning)

```bash
# Train neural model
python train_neural.py --evaluate

# Start server
python server_neural.py

# Test API
python test_neural_api.py
```

### Docker Compose (Recommended for Production)

```bash
# Clone and setup
git clone <repo>
cd book-recsys

# Copy environment variables
cp .env.example .env

# Start services (trains model automatically)
docker-compose up --build

# API available at http://localhost:8001
```

### Local Development

```bash
# Install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export DB_URI="postgresql://user:pass@localhost:5432/bookdb"
export DB_SCHEMA="book_recommendation_system"

# Train model
python train.py --evaluate

# Start API
uvicorn serve:app --reload --port 8001
```

## API Documentation

### Classic Server (Port 8001) - ALS + Ridge

See full documentation: [Online Learning Guide](docs/ONLINE_LEARNING.md)

**Key Features:**
- ✅ Online Learning (incremental updates)
- ✅ Fast training (~15 seconds)
- ✅ Fast inference (<50ms)
- ✅ Low resource usage

### Neural Server (Port 8002) - NCF + SBERT

See full documentation: [Neural Server Guide](docs/NEURAL_SERVER.md)

**Key Features:**
- ✅ Semantic understanding (SBERT embeddings)
- ✅ Deep learning (Neural Collaborative Filtering)
- ✅ Better accuracy on large datasets
- ⚠️ Requires batch retraining (no online learning)

### Endpoints (Both Servers)

#### GET /api/v1/health
Health check

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true
}
```

#### GET /api/v1/recommendations
Get personalized recommendations

**Parameters:**
- `user_id` (int): User ID
- `limit` (int, default=10): Number of recommendations

**Response:**
```json
{
  "user_id": 123,
  "limit": 10,
  "items": [
    {
      "book_id": 42,
      "score": 0.912,
      "reasons": {"cf": 0.62, "content": 0.29, "pop": 0.01}
    }
  ]
}
```

#### GET /api/v1/similar
Get similar books (content-based)

**Parameters:**
- `book_id` (int): Book ID
- `limit` (int, default=10): Number of results

**Response:**
```json
{
  "book_id": 42,
  "items": [
    {"book_id": 99, "score": 0.81}
  ]
}
```

#### POST /api/v1/feedback
Record user interaction

**Body:**
```json
{
  "user_id": 123,
  "book_id": 42,
  "event": "view",  // "view", "favorite", "bookmark", "rate"
  "rating_value": 5  // optional, only for "rate" event
}
```

#### POST /api/v1/retrain
Trigger model retraining (admin)

**Response:**
```json
{
  "status": "retraining scheduled"
}
```

## Training

### Classic Model (ALS + Ridge)

```bash
python train.py \
  --alpha 0.6 \
  --ridge-alpha 1.0 \
  --evaluate \
  --test-ratio 0.2
```

### Neural Model (NCF + SBERT)

```bash
python train_neural.py \
  --alpha 0.6 \
  --gmf-dim 64 \
  --ncf-epochs 20 \
  --evaluate \
  --test-ratio 0.2
```

**Metrics reported:**
- HR@K (Hit Rate)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Coverage (catalog coverage)

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Client Application                      │
└────────────┬────────────────────────────┬────────────────┘
             │                            │
             ↓                            ↓
   ┌─────────────────┐        ┌─────────────────────┐
   │ Classic Server  │        │  Neural Server      │
   │  Port 8001      │        │   Port 8002         │
   │                 │        │                     │
   │ • ALS + Ridge   │        │ • NCF + SBERT       │
   │ • Online Learn  │        │ • Deep Learning     │
   │ • Fast          │        │ • Semantic          │
   └────────┬────────┘        └──────────┬──────────┘
            │                            │
            └────────────┬───────────────┘
                         ↓
              ┌──────────────────────┐
              │   PostgreSQL DB      │
              │                      │
              │ • Books              │
              │ • Interactions       │
              │ • User Profiles      │
              └──────────────────────┘
```

### Components

1. **Data Layer** (`src/data/`)
   - SQL loaders for books and interactions
   - Supports ratings, reading history, favorites, bookmarks

2. **Feature Engineering** (`src/features/`)
   - Text processing (Vietnamese diacritics handling)
   - TF-IDF vectorization (unigrams + bigrams)

3. **Models** (`src/models/`)
   - **Classic**: 
     - Collaborative Filtering: Implicit ALS
     - Content-based: Ridge Regression
     - Hybrid: `HybridRecommender` (with online learning)
   - **Neural**:
     - Collaborative Filtering: Neural CF (NeuMF)
     - Content-based: SBERT Embeddings
     - Hybrid: `HybridNeuralRecommender`

4. **API** (`src/api/`)
   - FastAPI routes
   - Pydantic schemas
   - OpenAPI documentation

### Hybrid Scoring

```python
final_score = alpha * cf_score + (1 - alpha) * content_score
```

Default `alpha=0.6` (tune on validation set)

## Integration Examples

### Java (Spring Boot)

```java
@Service
public class RecommendationService {
    private final RestTemplate restTemplate;
    private final String recsysUrl = "http://localhost:8001/api/v1";
    
    public List<Recommendation> getRecommendations(Long userId, int limit) {
        String url = String.format("%s/recommendations?user_id=%d&limit=%d", 
                                   recsysUrl, userId, limit);
        RecommendationsResponse response = restTemplate.getForObject(
            url, RecommendationsResponse.class);
        return response.getItems();
    }
}
```

### Node.js (Express)

```javascript
const axios = require('axios');

const RECSYS_URL = 'http://localhost:8001/api/v1';

async function getRecommendations(userId, limit = 10) {
  const response = await axios.get(`${RECSYS_URL}/recommendations`, {
    params: { user_id: userId, limit }
  });
  return response.data.items;
}

async function recordFeedback(userId, bookId, event, ratingValue = null) {
  await axios.post(`${RECSYS_URL}/feedback`, {
    user_id: userId,
    book_id: bookId,
    event,
    rating_value: ratingValue
  });
}
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## Performance

### Classic Server (ALS + Ridge)
- **Latency**: <50ms for top-10 (warm start, 100k books)
- **Throughput**: ~500 req/s (single instance)
- **Memory**: ~1GB (with 100k books, 1M interactions)
- **Training**: ~15 seconds (full retrain)
- **Online Update**: ~2-3 seconds (incremental)

### Neural Server (NCF + SBERT)
- **Latency**: ~150ms CPU / ~30ms GPU for top-10
- **Throughput**: ~100 req/s CPU / ~300 req/s GPU
- **Memory**: ~3GB (with SBERT embeddings)
- **Training**: ~2-3 minutes (with GPU)
- **Accuracy**: 5-10% better NDCG on large datasets

### When to Use Which?

| Criteria | Classic Server | Neural Server |
|----------|---------------|---------------|
| **Real-time updates** | ✅ Yes | ❌ No |
| **Low latency** | ✅ <50ms | ⚠️ ~150ms |
| **Semantic understanding** | ⚠️ Limited | ✅ Excellent |
| **Small dataset (<10k)** | ✅ Better | ⚠️ Overfit risk |
| **Large dataset (>100k)** | ✅ Good | ✅ Better |
| **Resource constrained** | ✅ Low | ❌ High |

## Configuration

All settings in `.env` file:

```bash
DB_URI=postgresql://...
DB_SCHEMA=book_recommendation_system
ALPHA=0.6  # CF weight
CF_FACTORS=64
CF_ITERATIONS=30
ARTIFACTS_DIR=./artifacts
```

## License

MIT