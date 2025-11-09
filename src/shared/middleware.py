import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from shared import correlation_ids

logger = structlog.get_logger(__name__)

CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"  # alternative name some clients use


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle correlation IDs for request tracing.

    Extracts or generates correlation IDs for each request and ensures they're available
    throughout the request lifecycle.

    Behavior:
    1. Extracts correlation ID from incoming request headers (X-Correlation-ID
       or X-Request-ID)
    2. If not present, generates a new one with "gw" or "inf" prefix
    3. Sets correlation ID in ContextVar for the request lifecycle
    4. Adds correlation ID to response headers
    5. Logs request start/end with correlation ID
    """

    def __init__(self, app: ASGIApp, prefix: str = "gw"):
        super().__init__(app)
        self.prefix = prefix
        logger.info("CorrelationIdMiddleware initialized")

    async def dispatch(self, request: Request, call_next):
        """
        Process each request and inject correlation ID.

        Args:
            request: Incoming request
            call_next: Next middleware or route handler

        Returns:
            Response with correlation ID header
        """
        correlation_id = (
            request.headers.get(CORRELATION_ID_HEADER)
            or request.headers.get(REQUEST_ID_HEADER)
        )

        if not correlation_id:
            correlation_id = correlation_ids.generate_correlation_id(prefix=self.prefix)
            logger.debug(
                "Generated new correlation ID",
                path=request.url.path
            )
        else:
            logger.debug(
                "Using client-provided correlation ID",
                path=request.url.path
            )

        # set in ContextVar for this request's async context
        correlation_ids.set_correlation_id(correlation_id)

        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
        )
        
        # process request
        try:
            response: Response = await call_next(request)

            # add correlation ID to response headers so that client gets them back
            response.headers[CORRELATION_ID_HEADER] = correlation_id

            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Request failed with exception",
                method=request.method,
                path=request.url.path,
                error=str(e),
                exc_info=e
            )
            raise
