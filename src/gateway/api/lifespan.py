from contextlib import asynccontextmanager
import structlog
import httpx
from fastapi import FastAPI

from gateway.auth.api_key_db import ApiKeyDB
from gateway.auth.rate_limiter import SlidingWindowRateLimiter

logger = structlog.get_logger(__name__)


def create_lifespan(
    inference_service_url: str,
    api_key_db: ApiKeyDB,
    rate_limiter: SlidingWindowRateLimiter
):
    """
    Create a lifespan context manager for the gateway app.
    
    Usage:
        lifespan_fn = create_lifespan(
            inference_service_url="http://localhost:8001",
            api_key_db=api_key_db,
            rate_limiter=rate_limiter
        )
        app = FastAPI(lifespan=lifespan_fn)
    
    Args:
        inference_service_url: URL of the inference service
        api_key_db: API key database instance
        rate_limiter: Rate limiter instance
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        """Manage application lifecycle - startup and shutdown"""
        try:
            app.state.inference_url = inference_service_url
            app.state.api_key_db = api_key_db
            app.state.rate_limiter = rate_limiter

            limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
            app.state.http_client = httpx.AsyncClient(timeout=30.0, limits=limits)

            logger.info(
                "API Gateway started successfully",
                inference_url=inference_service_url
            )

        except Exception as e:
            logger.error("Failed to start API Gateway", error=str(e))
            raise

        yield
        
        logger.info("Shutting down API Gateway")

        # close persistent HTTP client
        if hasattr(app.state, "http_client"):
            await app.state.http_client.aclose()
            logger.info("HTTP client closed")

        logger.info("API Gateway shutdown complete")
    
    return _lifespan