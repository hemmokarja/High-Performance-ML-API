import structlog
from fastapi import FastAPI
from fastapi import status
from fastapi.responses import JSONResponse

from inference.api.schemas import ErrorResponse
from shared import correlation_ids

logger = structlog.get_logger(__name__)


async def _value_error_handler(request, exc):
    """Handle validation errors"""

    logger.warning("Validation error", error=str(exc))

    response_data = ErrorResponse(
        error="Validation error",
        detail=str(exc)
    ).model_dump()

    correlation_id = correlation_ids.get_correlation_id()
    if correlation_id:
        response_data["correlation_id"] = correlation_id
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data
    )


async def _general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error("Unexpected error", error=str(exc), exc_info=exc)

    response_data = ErrorResponse(
        error="Internal server error",
        detail="An unexpected error occurred"
    ).model_dump()

    correlation_id = correlation_ids.get_correlation_id()
    if correlation_id:
        response_data["correlation_id"] = correlation_id
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ValueError, _value_error_handler)
    app.add_exception_handler(Exception, _general_exception_handler)
    return app
