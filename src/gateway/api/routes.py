import httpx
import structlog
from fastapi import FastAPI, HTTPException, Request, Depends, status

from gateway.api.schemas import EmbedRequest, EmbedResponse, HealthResponse
from shared import correlation_ids

logger = structlog.get_logger(__name__)


def _ensure_service_ready(request: Request):
    """
    Shared readiness check function.
    Raises HTTP 503 if gateway components are not ready.
    """
    app_state = request.app.state

    if not hasattr(app_state, "http_client"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gateway not ready: HTTP client not initialized"
        )

    if not hasattr(app_state, "inference_url"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gateway not ready: inference URL not configured"
        )

    return app_state


async def _health_check(request: Request):
    """
    Health check endpoint.
    Returns gateway status and inference service health.
    """
    app_state = _ensure_service_ready(request)

    inference_health = {"status": "unknown"}
    try:
        # include correlation ID in health check to inference service
        headers = {}
        correlation_id = correlation_ids.get_correlation_id()
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id

        response = await app_state.http_client.get(
            f"{app_state.inference_url}/health", headers=headers, timeout=2.0
        )
        if response.status_code == 200:
            inference_health = response.json()
    except Exception as e:
        logger.warning("Inference service health check failed", error=str(e))
        inference_health = {"status": "unhealthy", "error": str(e)}

    return HealthResponse(
        status="healthy",
        gateway_version="1.0.0",
        inference_service=inference_health
    )


async def _readiness_check(request: Request):
    """
    Readiness check for Kubernetes/load balancers.
    Returns 200 if service can accept requests, 503 otherwise.
    """
    _ensure_service_ready(request)
    return {"status": "ready"}


def register_routes(app: FastAPI, verify_api_key_dependency):
    """
    Factory function to register all routes on a FastAPI app instance.

    Routes are defined inside this function so they can close over the
    verify_api_key_dependency.

    Args:
        app: FastAPI application instance
        verify_api_key_dependency: Dependency function for API key verification
    """
    async def _embed_text(
        embed_request: EmbedRequest,
        request: Request,
        user: dict = Depends(verify_api_key_dependency)
    ):
        """
        Generate embeddings for input text.
        
        This endpoint accepts input text and returns its embeddings.
        Requires valid API key authentication and enforces rate limits.
        """
        app_state = request.app.state
        
        try:
            headers = {"Content-Type": "application/json"}
            correlation_id = correlation_ids.get_correlation_id()
            if correlation_id:
                headers["X-Correlation-ID"] = correlation_id

            # Forward request to inference service with correlation ID
            response = await app_state.http_client.post(
                f"{app_state.inference_url}/embed",
                json={"input_text": embed_request.input_text},
                headers=headers,
                timeout=30.0
            )

            if response.status_code != 200:
                logger.error(
                    "Inference service error",
                    status_code=response.status_code,
                    response=response.text
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Inference service error"
                )
            
            inference_result = response.json()

            logger.info(
                "Embedding generated",
                user_id=user["user_id"],
                text_length=len(embed_request.input_text),
            )
            
            return EmbedResponse(
                embedding=inference_result["embedding"],
                model=inference_result["model"],
            )

        except httpx.TimeoutException:
            logger.error("Inference service timeout")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Inference service timeout"
            )
        except httpx.RequestError as e:
            logger.error("Inference service connection error", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Cannot connect to inference service"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e), exc_info=e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding generation failed"
            )

    async def _get_usage(
        request: Request,
        user: dict = Depends(verify_api_key_dependency)
    ):
        """Get current rate limit usage for the authenticated user."""
        app_state = request.app.state

        usage = app_state.rate_limiter.get_usage(user["user_id"])
        return {
            "user_id": user["user_id"],
            "usage": usage,
            "limits": {
                "per_minute": user["rate_limit_per_minute"],
                "per_hour": user["rate_limit_per_hour"]
            }
        }

    # public endpoints (no auth required)
    app.add_api_route(
        "/health",
        _health_check,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["system"]
    )

    app.add_api_route(
        "/ready",
        _readiness_check,
        methods=["GET"],
        tags=["system"]
    )

    # protected endpoints (auth required via Depends in function signature)
    app.add_api_route(
        "/v1/embed",
        _embed_text,
        methods=["POST"],
        response_model=EmbedResponse,
        tags=["embeddings"]
    )
    
    app.add_api_route(
        "/v1/usage",
        _get_usage,
        methods=["GET"],
        tags=["account"]
    )

    logger.info("Routes registered successfully")
    return app
