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

        # set in ContextVar for this request's async context
        correlation_ids.set_correlation_id(correlation_id)
        
        # process request
        try:
            response: Response = await call_next(request)

            # add correlation ID to response headers so that client gets them back
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            
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


from starlette.types import ASGIApp, Scope, Receive, Send
from fastapi import Request, Response

class CorrelationIdASGIMiddleware:
    """
    Lightweight ASGI middleware for correlation IDs.

    - Extracts correlation ID from incoming request headers
    - Generates a new one if missing
    - Stores it in a ContextVar for the async request lifecycle
    - Adds it to response headers
    """

    def __init__(self, app: ASGIApp, prefix: str = "gw"):
        self.app = app
        self.prefix = prefix
        logger.info("CorrelationIdASGIMiddleware initialized")

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            # Pass through non-HTTP requests (e.g., websocket)
            await self.app(scope, receive, send)
            return

        # Convert headers to dict for easy access
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        correlation_id = headers.get(CORRELATION_ID_HEADER.lower()) or headers.get(REQUEST_ID_HEADER.lower())

        if not correlation_id:
            correlation_id = correlation_ids.generate_correlation_id(prefix=self.prefix)
            logger.debug("Generated new correlation ID", correlation_id=correlation_id)
        else:
            logger.debug("Using client-provided correlation ID", correlation_id=correlation_id)

        # Set in ContextVar for this request
        correlation_ids.set_correlation_id(correlation_id)

        async def send_wrapper(message):
            # Inject correlation ID into response headers
            if message["type"] == "http.response.start":
                # ensure headers exist
                message.setdefault("headers", [])
                # append correlation ID header
                message["headers"].append(
                    (CORRELATION_ID_HEADER.encode(), correlation_id.encode())
                )
            await send(message)

        # Call the next app in the chain
        await self.app(scope, receive, send_wrapper)
