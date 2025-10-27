import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from gateway.api.schemas import ErrorResponse, RateLimitError
from gateway.auth.rate_limiter import RateLimitExceeded

logger = structlog.get_logger(__name__)


async def _validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    error_details = "; ".join(
        [
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
            for err in exc.errors()
        ]
    )

    logger.warning(
        "Validation error",
        path=request.url.path,
        errors=exc.errors()
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            detail=error_details,
            code="VALIDATION_ERROR"
        ).model_dump()
    )


async def _value_error_handler(request: Request, exc: ValueError):
    """Handle value errors from business logic"""
    logger.warning(
        "Value error",
        path=request.url.path,
        error=str(exc)
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Invalid value",
            detail=str(exc),
            code="VALUE_ERROR"
        ).model_dump()
    )


async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors"""
    logger.warning(
        "Rate limit exceeded",
        path=request.url.path,
        limit_type=exc.limit_type,
        limit=exc.limit
    )

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content=RateLimitError(
            error="Rate limit exceeded",
            detail=str(exc),
            code="RATE_LIMIT_EXCEEDED",
            retry_after=exc.retry_after,
            limit=exc.limit,
            limit_type=exc.limit_type
        ).model_dump(),
        headers={
            "Retry-After": str(exc.retry_after),
            "X-RateLimit-Limit": str(exc.limit),
            "X-RateLimit-Reset": str(exc.retry_after)
        }
    )


async def _general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(
        "Unexpected error",
        path=request.url.path,
        error=str(exc),
        exc_info=exc
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            code="INTERNAL_ERROR"
        ).model_dump()
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI app"""
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(ValueError, _value_error_handler)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_exception_handler(Exception, _general_exception_handler)
    
    logger.info("Exception handlers registered")
    return app
