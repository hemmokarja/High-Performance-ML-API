import structlog
import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, status
from gateway.api.schemas import EmbedRequest, EmbedResponse, HealthResponse

logger = structlog.get_logger(__name__)


def _ensure_service_ready(request: Request):
    """
    Shared readiness check function.
    Raises HTTP 503 if gateway components are not ready.
    """
    app_state = request.app.state

    if not hasattr(app_state, "inference_client"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gateway not ready: inference client not initialized"
        )

    if not hasattr(app_state, "verify_api_key"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gateway not ready: authentication not initialized"
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
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{app_state.inference_url}/health")
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


async def _embed_text(
    embed_request: EmbedRequest,
    request: Request,
    user: dict = Depends(lambda: None)  # Placeholder, will be replaced
):
    """
    Generate embeddings for input text.
    
    This endpoint accepts input text and returns its embeddings.
    Requires valid API key authentication and enforces rate limits.
    """
    app_state = request.app.state

    try:
        # forward request to inference service
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{app_state.inference_url}/embed",
                json={"input_text": embed_request.input_text},
                headers={"Content-Type": "application/json"}
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
            user_id=user.get("user_id") if user else "anonymous",
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
    user: dict = Depends(lambda: None)  # Placeholder, will be replaced
):
    """
    Get current rate limit usage for the authenticated user.
    """
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


def register_routes(app: FastAPI, verify_api_key_dependency):
    """
    Factory function to register all routes on a FastAPI app instance.
    
    Args:
        app: FastAPI application instance
        verify_api_key_dependency: Dependency function for API key verification
    """
    # Public endpoints (no auth required)
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
    
    # Protected endpoints (auth required)
    app.add_api_route(
        "/v1/embed",
        _embed_text,
        methods=["POST"],
        response_model=EmbedResponse,
        dependencies=[Depends(verify_api_key_dependency)],
        tags=["embeddings"]
    )
    
    app.add_api_route(
        "/v1/usage",
        _get_usage,
        methods=["GET"],
        dependencies=[Depends(verify_api_key_dependency)],
        tags=["account"]
    )
    
    logger.info("Routes registered successfully")
    return app