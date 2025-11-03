from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.schemas import *
from src.models.hybrid_Ridge import HybridRecommender
from src.data.db_loader import DatabaseLoader
from src.utils.config import get_settings
from src.utils.logging_config import logger
from pathlib import Path
from typing import Optional

router = APIRouter()

# Global model instance (loaded at startup)
recommender: Optional[HybridRecommender] = None
is_retraining: bool = False  # Track retraining status

def get_recommender():
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return recommender

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="retraining" if is_retraining else "ok",
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

@router.get("/diversity", response_model=DiversityResponse)
async def get_diversity(book_id: int, limit: int = 5):
    """Get diverse book recommendations for a reference book"""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 100"
        )

    rec = get_recommender()

    if rec.diversity_model is None:
        raise HTTPException(status_code=503, detail="Diversity model not loaded")

    try:
        result = rec.diversity_recommendations(
            book_id=book_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to compute diversity recommendations: {exc}")
        raise HTTPException(status_code=500, detail="Failed to compute diversity recommendations")

    return DiversityResponse(**result)

@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """
    Record user feedback and trigger online learning
    
    Event types and their strengths:
    - view: 1.0 (implicit positive signal)
    - bookmark: 2.0 (saved for later)
    - favorite: 3.0 (strong positive signal)
    - rate: rating_value (1-5 explicit rating)
    """
    rec = get_recommender()
    
    # Map event to strength
    strength_map = {
        'view': 1.0,
        'bookmark': 2.0,
        'favorite': 3.0,
        'rate': request.rating_value if request.rating_value else 3.0
    }
    
    strength = strength_map.get(request.event, 1.0)
    
    # Validate rating_value for 'rate' event
    if request.event == 'rate' and not request.rating_value:
        raise HTTPException(
            status_code=400, 
            detail="rating_value is required for 'rate' event"
        )
    
    logger.info(f"Feedback: user={request.user_id}, book={request.book_id}, "
               f"event={request.event}, strength={strength}")
    
    # Add to online learning buffer
    if rec.online_learning:
        buffer_triggered = rec.add_interaction(
            user_id=request.user_id,
            book_id=request.book_id,
            strength=strength,
            interaction_type=request.event
        )
        
        buffer_status = rec.get_buffer_status()
        
        return {
            "status": "recorded",
            "online_learning": True,
            "buffer_triggered_update": buffer_triggered,
            "buffer_status": buffer_status
        }
    else:
        # Fallback: just log (online learning disabled)
        return {
            "status": "recorded",
            "online_learning": False,
            "message": "Feedback logged, but online learning is disabled"
        }

@router.get("/model/info")
async def get_model_info():
    """Get information about current loaded model"""
    rec = get_recommender()
    
    info = {
        "status": "loaded",
        "alpha": rec.alpha,
        "online_learning": rec.get_buffer_status() if rec.online_learning else {"enabled": False},
        "cf_model": {
            "num_users": len(rec.cf_model.user_ids) if rec.cf_model else 0,
            "num_items": len(rec.cf_model.item_ids) if rec.cf_model else 0,
            "matrix_nnz": rec.cf_model.user_item_matrix.nnz if rec.cf_model else 0
        } if rec.cf_model else None,
        "content_model": {
            "num_books": len(rec.content_model.book_ids) if rec.content_model else 0,
            "feature_dim": rec.content_model.feature_matrix.shape[1] if rec.content_model else 0,
            "model_type": "Ridge"
        } if rec.content_model else None,
        "is_retraining": is_retraining
    }
    
    return info

@router.post("/online-learning/update")
async def trigger_incremental_update(force: bool = False):
    """
    Trigger incremental model update with buffered interactions
    
    Args:
        force: Force update even if buffer is not full
    """
    rec = get_recommender()
    
    if not rec.online_learning:
        raise HTTPException(
            status_code=400,
            detail="Online learning is disabled. Enable it first with POST /online-learning/enable"
        )
    
    buffer_status_before = rec.get_buffer_status()
    
    rec.incremental_update(force=force)
    
    buffer_status_after = rec.get_buffer_status()
    
    return {
        "status": "updated",
        "before": buffer_status_before,
        "after": buffer_status_after
    }

@router.post("/online-learning/enable")
async def enable_online_learning(buffer_size: int = 100):
    """Enable online learning"""
    if buffer_size < 10 or buffer_size > 1000:
        raise HTTPException(
            status_code=400,
            detail="buffer_size must be between 10 and 1000"
        )
    
    rec = get_recommender()
    rec.enable_online_learning(buffer_size=buffer_size)
    
    return {
        "status": "enabled",
        "buffer_size": buffer_size
    }

@router.post("/online-learning/disable")
async def disable_online_learning():
    """Disable online learning"""
    rec = get_recommender()
    rec.disable_online_learning()
    
    return {"status": "disabled"}

@router.get("/online-learning/status")
async def get_online_learning_status():
    """Get online learning buffer status"""
    rec = get_recommender()
    return rec.get_buffer_status()

@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Trigger model retraining (admin endpoint)"""
    global is_retraining
    
    # Check if already retraining
    if is_retraining:
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    # In production, add authentication here
    # e.g., check API key or JWT token
    
    background_tasks.add_task(retrain_models)
    return {
        "status": "accepted",
        "message": "Model retraining started in background. Check /health for status."
    }

@router.get("/user/profile/{user_id}")
async def get_user_profile(user_id: int):
    """
    Get user's content profile (interests/keywords)
    
    Useful for:
    - Explainability: "We recommend this because you like [keywords]"
    - User dashboard: Show user's detected interests
    """
    rec = get_recommender()
    
    if not rec.content_model:
        raise HTTPException(status_code=503, detail="Content model not loaded")
    
    keywords = rec.get_user_profile_keywords(user_id, top_n=20)
    
    if not keywords:
        raise HTTPException(status_code=404, detail=f"No profile found for user {user_id}")
    
    return {
        "user_id": user_id,
        "keywords": [
            {"keyword": kw, "weight": float(weight)}
            for kw, weight in keywords
        ]
    }

@router.get("/user/interactions/{user_id}")
async def get_user_interactions(user_id: int):
    """Get user's interaction history (for debugging/explainability)"""
    rec = get_recommender()
    
    if not rec.content_model:
        raise HTTPException(status_code=503, detail="Content model not loaded")
    
    interactions = rec.content_model.user_interactions.get(user_id, {})
    
    if not interactions:
        raise HTTPException(status_code=404, detail=f"No interactions for user {user_id}")
    
    return {
        "user_id": user_id,
        "interactions": [
            {"book_id": book_id, "strength": strength}
            for book_id, strength in sorted(interactions.items(), 
                                           key=lambda x: x[1], reverse=True)
        ]
    }

async def retrain_models():
    """Background task to retrain models"""
    global recommender, is_retraining
    
    is_retraining = True
    
    try:
        logger.info("🔄 Starting model retraining...")
        settings = get_settings()
        
        # 1. Load fresh data from database
        logger.info("📊 Loading data from database...")
        loader = DatabaseLoader(settings.db_uri, settings.db_schema)
        books_df, interactions_df = loader.load_all()
        
        logger.info(f"Loaded {len(books_df)} books, {len(interactions_df)} interactions")
        logger.info(f"Unique users: {interactions_df['user_id'].nunique()}")
        
        if len(books_df) == 0 or len(interactions_df) == 0:
            logger.error("❌ Insufficient data for retraining")
            return
        
        # 2. Train new model
        logger.info("🤖 Training new hybrid model...")
        new_recommender = HybridRecommender(alpha=settings.alpha)
        new_recommender.train(books_df, interactions_df)
        
        # 3. Save new model
        artifacts_dir = Path(settings.artifacts_dir)
        logger.info(f"💾 Saving new model to {artifacts_dir}...")
        new_recommender.save(artifacts_dir)
        
        # 4. Replace old model in memory (hot-swap)
        logger.info("♻️ Replacing old model with new model...")
        recommender = new_recommender
        
        logger.info("✅ Model retraining completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        is_retraining = False
