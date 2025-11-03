"""
FastAPI Server for Hybrid Implicit ALS + SBERT Recommender

Usage:
    python server_implicit_sbert.py

API will be available at: http://localhost:8003
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes_implicit_sbert import router, recommender as global_recommender
from src.models.hybrid_implicit_sbert import HybridImplicitSBERTRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger
from pathlib import Path
import uvicorn
import time

settings = get_settings()

app = FastAPI(
    title="Book Recommendation API - Implicit ALS + SBERT",
    version="1.0.0",
    description="Hybrid Recommender using Implicit ALS + SBERT"
)

# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    
    # Log request
    logger.info(f"[REQUEST] {request.method} {request.url.path}")
    logger.info(f"  Client: {request.client.host if request.client else 'Unknown'}")
    logger.info(f"  Query params: {dict(request.query_params)}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    logger.info(f"[RESPONSE] Status: {response.status_code} | Duration: {duration:.3f}s")
    logger.info("-" * 60)
    
    return response

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models at startup"""
    global global_recommender
    from src.api import routes_implicit_sbert
    
    artifacts_dir = Path("./artifacts_implicit_sbert")
    
    if artifacts_dir.exists():
        logger.info(f"Loading Implicit ALS + SBERT models from {artifacts_dir}...")
        try:
            routes_implicit_sbert.recommender = HybridImplicitSBERTRecommender.load(artifacts_dir)
            logger.info("✅ Models loaded successfully!")
            
            # Log model info
            if routes_implicit_sbert.recommender.als_model:
                logger.info(f"  ALS users: {len(routes_implicit_sbert.recommender.als_model.user_id_map)}")
                logger.info(f"  ALS items: {len(routes_implicit_sbert.recommender.als_model.item_id_map)}")
            if routes_implicit_sbert.recommender.content_model:
                logger.info(f"  SBERT books: {len(routes_implicit_sbert.recommender.content_model.book_ids)}")
                logger.info(f"  SBERT profiles: {len(routes_implicit_sbert.recommender.content_model.user_profiles)}")
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            logger.info("Server will start without pre-trained models")
    else:
        logger.warning(f"Artifacts directory not found: {artifacts_dir}")
        logger.info("Server will start without pre-trained models")
        logger.info("Train model with: python train_implicit_sbert.py")

app.include_router(router, prefix="/api/v1", tags=["implicit-sbert-recommendations"])

if __name__ == "__main__":
    logger.info("🚀 Starting Hybrid Implicit ALS + SBERT Recommender Server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
