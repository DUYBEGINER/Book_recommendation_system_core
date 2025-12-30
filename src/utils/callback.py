"""
Callback utilities for notifying Java backend about model updates.
Used for cache invalidation when models are retrained or updated.
"""
import httpx
from typing import List, Optional
from src.utils.config import get_settings
from src.utils.logging_config import logger


async def notify_retrain_complete(model_key: Optional[str] = None) -> bool:
    """
    Notify Java backend that model retraining is complete.
    This triggers cache invalidation for all recommendations.
    
    Args:
        model_key: The key of the model that was retrained (e.g., 'implicit', 'neural')
    
    Returns:
        True if callback was successful, False otherwise
    """
    settings = get_settings()
    
    if not settings.callback_enabled:
        logger.info("Callback disabled, skipping retrain notification")
        return True
    
    url = f"{settings.java_backend_url}/internal/recsys/callback/retrain-complete"
    payload = {"model_key": model_key} if model_key else {}
    
    try:
        async with httpx.AsyncClient(timeout=settings.callback_timeout) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully notified backend about retrain completion: {response.json()}")
                return True
            else:
                logger.warning(f"Backend callback returned status {response.status_code}: {response.text}")
                return False
                
    except httpx.TimeoutException:
        logger.warning(f"Timeout while notifying backend about retrain completion")
        return False
    except Exception as e:
        logger.warning(f"Failed to notify backend about retrain completion: {e}")
        return False


async def notify_incremental_update(user_ids: List[int]) -> bool:
    """
    Notify Java backend that incremental update was performed.
    This triggers cache invalidation for affected users.
    
    Args:
        user_ids: List of user IDs whose profiles were updated
    
    Returns:
        True if callback was successful, False otherwise
    """
    settings = get_settings()
    
    if not settings.callback_enabled:
        logger.info("Callback disabled, skipping incremental update notification")
        return True
    
    if not user_ids:
        logger.info("No user IDs to notify, skipping callback")
        return True
    
    url = f"{settings.java_backend_url}/internal/recsys/callback/incremental-update"
    payload = {"user_ids": user_ids}
    
    try:
        async with httpx.AsyncClient(timeout=settings.callback_timeout) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully notified backend about incremental update for {len(user_ids)} users")
                return True
            else:
                logger.warning(f"Backend callback returned status {response.status_code}: {response.text}")
                return False
                
    except httpx.TimeoutException:
        logger.warning(f"Timeout while notifying backend about incremental update")
        return False
    except Exception as e:
        logger.warning(f"Failed to notify backend about incremental update: {e}")
        return False


def notify_retrain_complete_sync(model_key: Optional[str] = None) -> bool:
    """
    Synchronous version of notify_retrain_complete.
    Used for background tasks that run in a sync context.
    """
    settings = get_settings()
    
    if not settings.callback_enabled:
        logger.info("Callback disabled, skipping retrain notification")
        return True
    
    url = f"{settings.java_backend_url}/internal/recsys/callback/retrain-complete"
    payload = {"model_key": model_key} if model_key else {}
    
    try:
        with httpx.Client(timeout=settings.callback_timeout) as client:
            response = client.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully notified backend about retrain completion: {response.json()}")
                return True
            else:
                logger.warning(f"Backend callback returned status {response.status_code}: {response.text}")
                return False
                
    except httpx.TimeoutException:
        logger.warning(f"Timeout while notifying backend about retrain completion")
        return False
    except Exception as e:
        logger.warning(f"Failed to notify backend about retrain completion: {e}")
        return False


def notify_incremental_update_sync(user_ids: List[int]) -> bool:
    """
    Synchronous version of notify_incremental_update.
    """
    settings = get_settings()
    
    if not settings.callback_enabled:
        logger.info("Callback disabled, skipping incremental update notification")
        return True
    
    if not user_ids:
        logger.info("No user IDs to notify, skipping callback")
        return True
    
    url = f"{settings.java_backend_url}/internal/recsys/callback/incremental-update"
    payload = {"user_ids": user_ids}
    
    try:
        with httpx.Client(timeout=settings.callback_timeout) as client:
            response = client.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully notified backend about incremental update for {len(user_ids)} users")
                return True
            else:
                logger.warning(f"Backend callback returned status {response.status_code}: {response.text}")
                return False
                
    except httpx.TimeoutException:
        logger.warning(f"Timeout while notifying backend about incremental update")
        return False
    except Exception as e:
        logger.warning(f"Failed to notify backend about incremental update: {e}")
        return False
