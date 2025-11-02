from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router, recommender as global_recommender
from src.models.hybrid_Ridge import HybridRecommender
from src.utils.config import get_settings
from src.utils.logging_config import logger
from pathlib import Path
import time

settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version
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
    from src.api import routes
    
    artifacts_dir = Path(settings.artifacts_dir)
    
    if artifacts_dir.exists():
        try:
            routes.recommender = HybridRecommender.load(artifacts_dir, alpha=settings.alpha)
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    else:
        logger.warning(f"Artifacts directory not found: {artifacts_dir}")

app.include_router(router, prefix="/api/v1", tags=["recommendations"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)