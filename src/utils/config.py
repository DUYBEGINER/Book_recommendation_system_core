# ==================== src/utils/config.py ====================
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "book_recommendation_db"
    db_user: str = "postgres"
    db_password: str = "123"
    db_uri: str = "postgresql://postgres:123@localhost:5432/book_recommendation_db"
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
    
    # Java Backend callback URL for cache invalidation
    java_backend_url: str = "http://localhost:8080/api/v1"
    callback_enabled: bool = True
    callback_timeout: int = 5  # seconds
    
    # Python environment
    pythonunbuffered: int = 1

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()