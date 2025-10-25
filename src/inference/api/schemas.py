from typing import List, Optional

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class EmbedRequest(BaseModel):
    """Request schema for embedding generation"""
    sentence: str = Field(  # Singular!
        ...,
        min_length=1,
        description="Single sentence to embed"
    )

    @field_validator("sentence")
    @classmethod
    def validate_sentence(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Sentence cannot be empty")
        return v.strip()


class EmbedResponse(BaseModel):
    """Response schema for embedding generation"""
    embedding: List[float]  # Single embedding
    model: str


class EmbedResponse(BaseModel):
    """Response schema for embedding generation"""
    embedding: List[float]
    model: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    device: str
    queue_size: int
    inflight_batches: int


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
