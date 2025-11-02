# ==================== src/utils/config.py ====================
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    db_uri: str = "postgresql://postgres:123@localhost:5432/recommendation_book_system_db"
    db_schema: str = "book_recommendation_system"
    
    # Model parameters
    cf_factors: int = 64
    cf_iterations: int = 30
    cf_regularization: float = 0.01
    alpha: float = 0.6  # hybrid blend weight
    
    # API
    api_title: str = "Book Recommendation API"
    api_version: str = "1.0.0"
    
    # Paths
    artifacts_dir: str = "./artifacts"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()