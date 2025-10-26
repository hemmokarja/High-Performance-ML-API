import structlog
from fastapi import FastAPI
from fastapi import status
from fastapi.responses import JSONResponse

from inference.api.schemas import ErrorResponse

logger = structlog.get_logger(__name__)


async def _value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            detail=str(exc)
        ).model_dump()
    )


async def _general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error("Unexpected error", error=str(exc), exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred"
        ).model_dump()
    )

def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ValueError, _value_error_handler)
    app.add_exception_handler(Exception, _general_exception_handler)
    return app
