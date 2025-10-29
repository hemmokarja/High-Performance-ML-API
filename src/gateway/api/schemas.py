from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

import structlog

logger = structlog.get_logger(__name__)


class EmbedRequest(BaseModel):
    """Request schema for embedding generation"""
    input_text: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Input text to embed"
    )

    @field_validator("input_text")
    @classmethod
    def validate_input_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Input text cannot be empty")
        return v.strip()


class EmbedResponse(BaseModel):
    """Response schema for embedding generation"""
    embedding: List[float]
    model: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    gateway_version: str
    inference_service: dict


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(
        ...,
        description="Error type or message"
    )
    detail: Optional[str] = Field(
        None,
        description="Detailed error message"
    )
    code: Optional[str] = Field(
        None,
        description="Error code for programmatic handling"
    )


class RateLimitError(ErrorResponse):
    """Rate limit error with retry information"""
    retry_after: int = Field(
        ...,
        description="Seconds until rate limit resets"
    )
    limit: int = Field(
        ...,
        description="Rate limit threshold"
    )
    limit_type: str = Field(
        ...,
        description="Type of rate limit (minute/hour)"
    )