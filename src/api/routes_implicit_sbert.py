"""
API Routes for Hybrid Implicit ALS + SBERT Recommender
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.schemas import *
from src.models.hybrid_implicit_sbert import HybridImplicitSBERTRecommender
from src.data.db_loader import DatabaseLoader
from src.utils.config import get_settings
from src.utils.logging_config import logger
from pathlib import Path
from typing import Optional

router = APIRouter()

# Global model instance (loaded at startup)
recommender: Optional[HybridImplicitSBERTRecommender] = None
is_retraining: bool = False  # Track retraining status

def get_recommender():
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return recommender

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if models are loaded"""
    try:
        rec = get_recommender()
        return HealthResponse(
            status="healthy",
            models_loaded=True
        )
    except:
        return HealthResponse(
            status="unhealthy",
            models_loaded=False
        )

@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(user_id: int, limit: int = 10):
    """
    Get hybrid recommendations (ALS + SBERT)
    
    Combines:
    - Implicit ALS collaborative filtering
    - SBERT semantic content-based filtering
    - Popularity fallback for cold start
    """
    rec = get_recommender()
    
    try:
        results = rec.recommend(user_id, limit=limit)
        
        items = [
            RecommendationItem(
                book_id=r['book_id'],
                score=r['score'],
                reasons=r['reasons']
            )
            for r in results
        ]
        
        return RecommendationsResponse(
            user_id=user_id,
            limit=limit,
            items=items
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar", response_model=SimilarResponse)
async def get_similar(book_id: int, limit: int = 10):
    """
    Get semantically similar books using SBERT embeddings
    
    Uses cosine similarity between SBERT embeddings
    """
    rec = get_recommender()
    
    try:
        if not rec.content_model:
            raise HTTPException(status_code=503, detail="SBERT model not loaded")
        
        similar = rec.content_model.get_similar_items(book_id, top_k=limit)
        
        items = [
            SimilarItem(book_id=int(bid), score=float(score))
            for bid, score in similar
        ]
        
        return SimilarResponse(book_id=book_id, items=items)
    except Exception as e:
        logger.error(f"Error getting similar books: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """
    Record user feedback (for future batch retraining)
    
    Event types and their strengths (consistent with training data):
    - view: 1.0 (implicit view signal)
    - favorite: 5.0 (strong positive signal)
    - rate: rating_value (1-5 explicit rating)
    
    Note: This model doesn't support online learning.
    Feedback is logged for future batch retraining.
    """
    rec = get_recommender()
    
    # Map event to strength (MUST match db_loader.py!)
    strength_map = {
        'view': 1.0,
        'favorite': 5.0,  # Matches training data
        'rate': request.rating_value if request.rating_value else 3.0
    }
    
    strength = strength_map.get(request.event, 1.0)
    
    # Log feedback (for future retraining)
    logger.info(f"📝 Feedback logged: user={request.user_id}, book={request.book_id}, "
               f"event={request.event}, strength={strength}")
    
    return {
        "status": "recorded",
        "user_id": request.user_id,
        "book_id": request.book_id,
        "event": request.event,
        "strength": strength,
        "message": "Feedback logged for future batch retraining"
    }

@router.get("/model/info")
async def get_model_info():
    """Get information about current loaded model"""
    rec = get_recommender()
    
    info = {
        "status": "loaded",
        "alpha": rec.alpha,
        "online_learning": {"enabled": False},  # This model doesn't support online learning
        "cf_model": {
            "num_users": len(rec.als_model.user_id_map) if rec.als_model and rec.als_model.user_id_map else 0,
            "num_items": len(rec.als_model.item_id_map) if rec.als_model and rec.als_model.item_id_map else 0,
            "matrix_nnz": rec.als_model.user_item_matrix.nnz if rec.als_model and rec.als_model.user_item_matrix is not None else 0,
            "model_type": "Implicit ALS",
            "factors": rec.als_factors,
            "iterations": rec.als_iterations,
            "regularization": rec.als_regularization
        } if rec.als_model else None,
        "content_model": {
            "num_books": len(rec.content_model.book_ids) if rec.content_model and rec.content_model.book_ids is not None else 0,
            "num_user_profiles": len(rec.content_model.user_profiles) if rec.content_model and rec.content_model.user_profiles else 0,
            "feature_dim": rec.content_model.embeddings.shape[1] if rec.content_model and rec.content_model.embeddings is not None else 0,
            "model_type": "SBERT",
            "model_name": rec.content_model.model_name if rec.content_model else None
        } if rec.content_model else None,
        "is_retraining": is_retraining
    }
    
    return info

@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger background retraining
    
    Reloads data from database and retrains both ALS and SBERT models
    """
    global is_retraining
    
    if is_retraining:
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    background_tasks.add_task(retrain_models)
    
    return {
        "status": "retraining_started",
        "message": "Models are being retrained in background. Check /model/info for status."
    }

@router.get("/user/profile/{user_id}")
async def get_user_profile(user_id: int, top_n: int = 20):
    """
    Get user's semantic profile from SBERT
    
    Returns top keywords/concepts from user's reading history
    """
    rec = get_recommender()
    
    if not rec.content_model:
        raise HTTPException(status_code=503, detail="SBERT model not loaded")
    
    if user_id not in rec.content_model.user_profiles:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in SBERT profiles")
    
    try:
        # Get user's embedding profile
        profile_embedding = rec.content_model.user_profiles[user_id]
        
        # Get user's interaction history
        user_books = rec.content_model.user_interactions.get(user_id, {})
        
        return {
            "user_id": user_id,
            "num_interactions": len(user_books),
            "profile_dimension": len(profile_embedding),
            "top_books": sorted(user_books.items(), key=lambda x: x[1], reverse=True)[:10],
            "message": "SBERT uses dense embeddings (no sparse keywords like TF-IDF)"
        }
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/interactions/{user_id}")
async def get_user_interactions(user_id: int):
    """Get user's interaction history"""
    rec = get_recommender()
    
    # Try to get from SBERT model first
    if rec.content_model and user_id in rec.content_model.user_interactions:
        interactions = rec.content_model.user_interactions.get(user_id, {})
    elif rec.als_model and user_id in rec.als_model.user_id_map:
        # Fallback to ALS model (no strength info, just presence)
        interactions = {}
        logger.info(f"User {user_id} found in ALS but interaction strengths not available")
    else:
        raise HTTPException(status_code=404, detail=f"No interactions found for user {user_id}")
    
    if not interactions:
        raise HTTPException(status_code=404, detail=f"No interactions found for user {user_id}")
    
    return {
        "user_id": user_id,
        "num_interactions": len(interactions),
        "books": [
            {"book_id": book_id, "strength": strength}
            for book_id, strength in sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        ]
    }

async def retrain_models():
    """Background task to retrain models"""
    global is_retraining, recommender
    
    try:
        is_retraining = True
        logger.info("🔄 Starting background retraining...")
        
        # Load fresh data
        settings = get_settings()
        loader = DatabaseLoader(settings.db_uri, settings.db_schema)
        books_df, interactions_df = loader.load_all()
        
        # Retrain
        recommender.train(books_df, interactions_df)
        
        # Save updated models
        artifacts_dir = Path("./artifacts_implicit_sbert")
        recommender.save(artifacts_dir)
        
        logger.info("✅ Background retraining completed!")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
    finally:
        is_retraining = False
