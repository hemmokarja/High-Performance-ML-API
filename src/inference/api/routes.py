import asyncio

import structlog
from fastapi import FastAPI, HTTPException, Request, status

from inference.api.schemas import EmbedResponse, EmbedRequest, HealthResponse

logger = structlog.get_logger(__name__)


def _ensure_service_ready(request: Request):
    """
    Shared readiness/health check function.
    Raises HTTP 503 if model or batcher are not ready.
    """
    app_state = request.app.state

    if (
        not hasattr(app_state, "model")
        or not hasattr(app_state, "batcher")
        or not app_state.batcher.is_started()
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

    return app_state.model, app_state.batcher


async def _health_check(request: Request):
    """
    Health check endpoint.
    
    Returns model status and queue metrics.
    """
    model, batcher = _ensure_service_ready(request)
    return HealthResponse(
        status="healthy",
        model=model.model_name,
        device=model.device_str,
        queue_size=batcher.request_queue.qsize(),
        inflight_batches=batcher.inflight_batches
    )


async def _readiness_check(request: Request):
    """
    Readiness check for Kubernetes/load balancers.

    Returns 200 if service can accept requests, 503 otherwise.
    """
    _, _ = _ensure_service_ready(request)
    return {"status": "ready"}


async def _embed_sentence(embed_request: EmbedRequest, request: Request):
    """
    Generate embeddings for sentence.
    
    This endpoint accepts a sentence and returns its embeddings.
    Requests are automatically batched for efficient inference.
    """
    model, batcher = _ensure_service_ready(request)
    try:
        emb = await batcher.predict(embed_request.sentence)
        return EmbedResponse(embedding=emb.tolist(), model=model.model_name)

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timeout"
        )
    except Exception as e:
        logger.error("Embedding generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding generation failed"
        )


async def _metrics(request: Request):
    """
    Expose metrics for monitoring (Prometheus format).

    NOTE: Later, we'll use e.g., prometheus_client library for proper metrics.
    """
    _, batcher = _ensure_service_ready(request)
    return {
        "queue_size": batcher.request_queue.qsize(),
        "inflight_batches": batcher.inflight_batches,
        "num_workers": batcher.num_workers,
        "max_batch_size": batcher.max_batch_size,
    }


def register_routes(app: FastAPI):
    """
    Factory function to register all routes on a FastAPI app instance.
    """
    app.add_api_route(
        "/health",
        _health_check,
        methods=["GET"],
        response_model=HealthResponse,
    )

    app.add_api_route(
        "/ready",
        _readiness_check,
        methods=["GET"],
    )

    app.add_api_route(
        "/embed",
        _embed_sentence,
        methods=["POST"],
        response_model=EmbedResponse,
    )

    app.add_api_route(
        "/metrics",
        _metrics,
        methods=["GET"],
    )

    return app