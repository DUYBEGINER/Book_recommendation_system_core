"""
API Routes for Hybrid Neural Recommender (NCF + SBERT)
"""
import math
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
    if is_retraining:
        return HealthResponse(status="retraining", models_loaded=True)

    if recommender is None:
        return HealthResponse(status="no_model", models_loaded=False)

    return HealthResponse(status="ok", models_loaded=True)

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
    Record user feedback and optionally trigger SBERT online learning
    
    Event types and their strengths:
    - rating: rating_value (1-5 explicit rating)
    - history: 1.0 (implicit signal)
    - favorite: 5.0 to add, 0.0 to remove (rating_value = 0)
    
    Note: Online learning updates SBERT user profiles only.
    NCF model still requires batch retraining.
    """
    rec = get_recommender()
    
    # Map event to interaction strength
    if request.event == 'rating':
        if not request.rating_value or request.rating_value < 1:
            raise HTTPException(
                status_code=400,
                detail="rating_value (1-5) is required for 'rating' event"
            )
        strength = float(request.rating_value)
    elif request.event == 'favorite':
        strength = 0.0 if request.rating_value == 0 else 5.0
    elif request.event == 'history':
        strength = 1.0
    else:
        strength = 1.0
    
    logger.info(
        "Feedback: user=%s, book=%s, event=%s, strength=%.2f",
        request.user_id,
        request.book_id,
        request.event,
        strength,
    )
    
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
            "buffer_status": buffer_status,
            "note": "SBERT profiles updated incrementally. NCF model requires retrain for collaborative signals."
        }
    
    return {
        "status": "recorded",
        "online_learning": False,
        "message": "Feedback logged. Enable online learning or trigger retrain to apply updates."
    }

@router.get("/model/info")
async def get_model_info():
    """Get information about current loaded neural model"""
    rec = get_recommender()
    
    info = {
        "status": "loaded",
        "model_type": "HybridNeuralRecommender",
        "alpha": rec.alpha,
        "online_learning": rec.get_buffer_status() if rec.online_learning else {"enabled": False},
        "ncf_model": None,
        "content_model": {
            "num_books": len(rec.content_model.book_ids) if rec.content_model and rec.content_model.book_ids is not None else 0,
            "num_user_profiles": len(rec.content_model.user_profiles) if rec.content_model and rec.content_model.user_profiles else 0,
            "embedding_dim": rec.content_model.embeddings.shape[1] if rec.content_model and rec.content_model.embeddings is not None else 0,
            "model_name": rec.content_model.model_name if rec.content_model else None
        } if rec.content_model else None,
        "is_retraining": is_retraining
    }

    if rec.ncf_model:
        def _safe_number(value):
            if value is None:
                return None
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return value

        def _safe_metrics(metrics):
            if not metrics:
                return None
            return {key: _safe_number(value) for key, value in metrics.items()}

        training_history = getattr(rec.ncf_model, "training_history", [])
        info["ncf_model"] = {
            "num_users": len(rec.ncf_model.user_id_map) if rec.ncf_model.user_id_map else 0,
            "num_items": len(rec.ncf_model.item_id_map) if rec.ncf_model.item_id_map else 0,
            "gmf_dim": rec.ncf_model.gmf_dim,
            "mlp_dims": rec.ncf_model.mlp_dims,
            "training_params": {
                "epochs": rec.ncf_model.epochs,
                "dropout": rec.ncf_model.dropout,
                "learning_rate": rec.ncf_model.learning_rate,
                "weight_decay": rec.ncf_model.weight_decay,
                "batch_size": rec.ncf_model.batch_size,
            },
            "training_history": [_safe_number(value) for value in training_history] if training_history else [],
            "last_training_loss": _safe_number(getattr(rec.ncf_model, "last_training_loss", None)),
            "evaluation_metrics": _safe_metrics(getattr(rec.ncf_model, "last_evaluation_metrics", None))
        }
    
    return info

@router.post("/online-learning/update")
async def trigger_incremental_update(force: bool = False):
    """
    Trigger incremental SBERT updates using buffered interactions.
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
    processed = max(
        0,
        int(buffer_status_before.get("buffer_size") or 0)
        - int(buffer_status_after.get("buffer_size") or 0)
    )
    status = "updated" if processed > 0 else "skipped"
    
    response = {
        "status": status,
        "before": buffer_status_before,
        "after": buffer_status_after,
        "interactions_processed": processed,
        "note": "SBERT profiles refreshed. NCF model still uses previous training snapshot."
    }
    
    if status == "skipped":
        response["message"] = "Buffer chưa đạt ngưỡng, chưa có cập nhật nào được thực hiện."
    
    return response

@router.post("/online-learning/enable")
async def enable_online_learning(buffer_size: int = 100):
    """Enable SBERT online learning for hybrid neural recommender."""
    if buffer_size < 10 or buffer_size > 1000:
        raise HTTPException(
            status_code=400,
            detail="buffer_size must be between 10 and 1000"
        )
    
    rec = get_recommender()
    rec.enable_online_learning(buffer_size=buffer_size)
    
    return {
        "status": "enabled",
        "buffer_size": buffer_size,
        "note": "Only SBERT profiles update incrementally. NCF model requires full retrain."
    }

@router.post("/online-learning/disable")
async def disable_online_learning():
    """Disable SBERT online learning."""
    rec = get_recommender()
    rec.disable_online_learning()
    
    return {"status": "disabled"}

@router.get("/online-learning/status")
async def get_online_learning_status():
    """Return current online learning buffer status."""
    rec = get_recommender()
    return rec.get_buffer_status()

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
