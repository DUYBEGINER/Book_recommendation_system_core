from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.schemas import *
from src.models.hybrid import HybridRecommender
from src.utils.logging_config import logger
from pathlib import Path

router = APIRouter()

# Global model instance (loaded at startup)
recommender: Optional[HybridRecommender] = None

def get_recommender():
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return recommender

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        models_loaded=recommender is not None
    )

@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(user_id: int, limit: int = 10):
    """Get personalized recommendations"""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    rec = get_recommender()
    items = rec.recommend(user_id, limit)
    
    return RecommendationsResponse(
        user_id=user_id,
        limit=limit,
        items=[RecommendationItem(**item) for item in items]
    )

@router.get("/similar", response_model=SimilarResponse)
async def get_similar(book_id: int, limit: int = 10):
    """Get similar books"""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
    rec = get_recommender()
    items = rec.similar_books(book_id, limit)
    
    return SimilarResponse(
        book_id=book_id,
        items=[SimilarItem(**item) for item in items]
    )

@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """Record user feedback (for online learning - simplified)"""
    # In production, this would update interaction logs and trigger retraining
    logger.info(f"Feedback: user={request.user_id}, book={request.book_id}, event={request.event}")
    return {"status": "recorded"}

@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining (admin endpoint)"""
    # In production, add authentication
    background_tasks.add_task(retrain_models)
    return {"status": "retraining scheduled"}

def retrain_models():
    """Background task to retrain models"""
    # This would be called by train.py in production
    logger.info("Retraining models...")