"""
API Routes for Hybrid Neural Recommender (NCF + SBERT)
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from src.api.schemas import *
from src.models.hybrid_neural import HybridNeuralRecommender
from src.data.db_loader import DatabaseLoader
from src.utils.config import get_settings
from src.utils.logging_config import logger
from pathlib import Path
from typing import Optional

router = APIRouter()

# Global model instance (loaded at startup)
recommender: Optional[HybridNeuralRecommender] = None
is_retraining: bool = False  # Track retraining status

def get_recommender():
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return recommender

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if neural models are loaded and ready"""
    try:
        rec = get_recommender()
        return HealthResponse(status="healthy", models_loaded=True)
    except HTTPException:
        return HealthResponse(status="no_model", models_loaded=False)

@router.get("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(user_id: int, limit: int = 10):
    """
    Get hybrid neural recommendations for a user
    
    Combines NCF (neural collaborative filtering) + SBERT (semantic content)
    """
    rec = get_recommender()
    
    try:
        results = rec.recommend(user_id, limit=limit)
        
        if not results:
            logger.warning(f"No recommendations for user {user_id}")
            return RecommendationsResponse(
                user_id=user_id,
                limit=limit,
                items=[]
            )
        
        items = [
            RecommendationItem(
                book_id=r['book_id'],
                score=r['score'],
                reasons=r.get('reasons', {})
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
    Record user feedback (for future online learning)
    
    Event types and their strengths (consistent with training data):
    - view: 1.0 (implicit view signal)
    - favorite: 5.0 (strong positive signal)
    - rate: rating_value (1-5 explicit rating)
    
    Note: HybridNeuralRecommender doesn't support online learning yet.
    This endpoint logs the feedback for future batch retraining.
    """
    rec = get_recommender()
    
    # Map event to strength (MUST match db_loader.py!)
    strength_map = {
        'view': 1.0,
        'favorite': 5.0,  # Matches training data
        'rate': request.rating_value if request.rating_value else 3.0
    }
    
    strength = strength_map.get(request.event, 1.0)
    
    logger.info(f"📝 Feedback recorded: User {request.user_id} -> Book {request.book_id} "
                f"(event={request.event}, strength={strength})")
    logger.info("⚠️  Neural model requires batch retraining. Use /retrain endpoint.")
    
    return {
        "status": "recorded",
        "message": "Feedback logged. Neural model requires batch retraining.",
        "user_id": request.user_id,
        "book_id": request.book_id,
        "event": request.event,
        "strength": strength
    }

@router.get("/model/info")
async def get_model_info():
    """Get information about current loaded neural model"""
    rec = get_recommender()
    
    info = {
        "status": "loaded",
        "model_type": "HybridNeuralRecommender",
        "alpha": rec.alpha,
        "ncf_model": {
            "num_users": len(rec.ncf_model.user_id_map) if rec.ncf_model and rec.ncf_model.user_id_map else 0,
            "num_items": len(rec.ncf_model.item_id_map) if rec.ncf_model and rec.ncf_model.item_id_map else 0,
            "gmf_dim": rec.ncf_model.gmf_dim if rec.ncf_model else 0,
            "mlp_dims": rec.ncf_model.mlp_dims if rec.ncf_model else [],
            "device": str(rec.ncf_model.device) if rec.ncf_model else "unknown"
        } if rec.ncf_model else None,
        "sbert_model": {
            "num_books": len(rec.content_model.book_ids) if rec.content_model and rec.content_model.book_ids is not None else 0,
            "num_user_profiles": len(rec.content_model.user_profiles) if rec.content_model and rec.content_model.user_profiles else 0,
            "embedding_dim": rec.content_model.embeddings.shape[1] if rec.content_model and rec.content_model.embeddings is not None else 0,
            "model_name": rec.content_model.model_name if rec.content_model else None
        } if rec.content_model else None,
        "is_retraining": is_retraining
    }
    
    return info

@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Trigger neural model retraining in background
    
    Warning: This is computationally expensive (NCF training + SBERT encoding)
    """
    global is_retraining
    
    if is_retraining:
        raise HTTPException(status_code=409, detail="Model is already retraining")
    
    background_tasks.add_task(retrain_neural_models)
    
    return {
        "status": "started",
        "message": "Neural model retraining started in background"
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
    """Get user's interaction history from SBERT model"""
    rec = get_recommender()
    
    if not rec.content_model:
        raise HTTPException(status_code=503, detail="SBERT model not loaded")
    
    interactions = rec.content_model.user_interactions.get(user_id, {})
    
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

async def retrain_neural_models():
    """Background task to retrain neural models"""
    global is_retraining, recommender
    
    try:
        is_retraining = True
        logger.info("🔄 Starting neural model retraining...")
        
        settings = get_settings()
        loader = DatabaseLoader(settings.db_uri, settings.db_schema)
        books_df, interactions_df = loader.load_all()
        
        logger.info(f"Loaded {len(books_df)} books, {len(interactions_df)} interactions")
        
        # Train new model with same hyperparameters
        new_recommender = HybridNeuralRecommender(
            alpha=recommender.alpha if recommender else 0.6,
            gmf_dim=recommender.gmf_dim if recommender else 64,
            mlp_dims=recommender.mlp_dims if recommender else [128, 64, 32],
            ncf_epochs=20,
            ncf_batch_size=256,
            device=str(recommender.device) if recommender else None
        )
        
        new_recommender.train(books_df, interactions_df)
        
        # Save new model
        artifacts_dir = Path("./artifacts_neural")
        new_recommender.save(artifacts_dir)
        
        # Swap models
        recommender = new_recommender
        
        logger.info("✅ Neural model retraining completed!")
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
    finally:
        is_retraining = False
