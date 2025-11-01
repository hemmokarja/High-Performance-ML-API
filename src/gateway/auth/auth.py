import os

import structlog
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from gateway.auth.api_key_db import ApiKeyDB
from gateway.auth.rate_limiter import RateLimiter, RateLimitExceeded

logger = structlog.get_logger(__name__)

security = HTTPBearer()


class AuthService:
    """
    Authentication and rate limiting middleware for API Gateway.

    Validates API keys and enforces rate limits on a per-user basis.
    """
    def __init__(self, api_key_db: ApiKeyDB, rate_limiter: RateLimiter):
        self.api_key_db = api_key_db
        self.rate_limiter = rate_limiter
        self.bypass_rate_limits = (
            os.getenv("BYPASS_RATE_LIMITS", "false").lower() == "true"
        )
        logger.info(
            "AuthService initialized", bypass_rate_limits=self.bypass_rate_limits
        )

    async def verify_api_key(
        self, credentials: HTTPAuthorizationCredentials = Security(security)
    ) -> dict:
        """
        Verify API key and enforce rate limits.

        This function should be used as a FastAPI dependency:

        @app.post("/v1/embed")
        async def embed(
            request: EmbedRequest,
            user: dict = Depends(auth_middleware.verify_api_key)
        ):
            ...

        Args:
            credentials: HTTP Bearer token from request

        Returns:
            User information dictionary

        Raises:
            HTTPException: 401 for invalid key, 429 for rate limit exceeded
        """
        api_key = credentials.credentials
        user_info = self.api_key_db.get_key_info(api_key)
        if not user_info:
            logger.warning(
                "Invalid API key attempt",
                key_prefix=api_key[:16] if len(api_key) >= 16 else "***"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if self.bypass_rate_limits:
            user_info["rate_limit_info"] = {
                "requests_this_minute": 0,
                "requests_this_hour": 0,
                "minute_limit": float("inf"),
                "hour_limit": float("inf")
            }
            return user_info

        try:
            rate_limit_info = self.rate_limiter.check_rate_limit(
                user_id=user_info["user_id"],
                minute_limit=user_info["rate_limit_per_minute"],
                hour_limit=user_info["rate_limit_per_hour"]
            )

            user_info["rate_limit_info"] = {
                "requests_this_minute": rate_limit_info.requests_this_minute,
                "requests_this_hour": rate_limit_info.requests_this_hour,
                "minute_limit": rate_limit_info.minute_limit,
                "hour_limit": rate_limit_info.hour_limit
            }

            logger.info(
                "Request authenticated",
                user_id=user_info["user_id"],
                requests_minute=rate_limit_info.requests_this_minute,
                requests_hour=rate_limit_info.requests_this_hour
            )

            return user_info

        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(e),
                headers={
                    "Retry-After": str(e.retry_after),
                    "X-RateLimit-Limit": str(e.limit),
                    "X-RateLimit-Reset": str(e.retry_after)
                }
            )
