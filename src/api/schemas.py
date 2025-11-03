# ==================== src/api/schemas.py ====================
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class RecommendationItem(BaseModel):
    book_id: int
    score: float
    reasons: Dict[str, float]

class RecommendationsResponse(BaseModel):
    user_id: int
    limit: int
    items: List[RecommendationItem]

class SimilarItem(BaseModel):
    book_id: int
    score: float

class SimilarResponse(BaseModel):
    book_id: int
    items: List[SimilarItem]

class DiversityItem(BaseModel):
    book_id: int
    rating: float
    score: float
    metadata: Dict[str, float]

class DiversityResponse(BaseModel):
    book_id: int
    items: List[DiversityItem]

class FeedbackRequest(BaseModel):
    user_id: int
    book_id: int
    event: str = Field(..., pattern="^(view|favorite|rate)$")
    rating_value: Optional[int] = Field(None, ge=1, le=5)

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
