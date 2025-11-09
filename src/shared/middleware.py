import structlog
from starlette.types import ASGIApp, Scope, Receive, Send

from shared import correlation_ids

logger = structlog.get_logger(__name__)

CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"  # alternative name some clients use


class CorrelationIdASGIMiddleware:
    """
    Lightweight middleware to handle correlation IDs for request tracing.

    Extracts or generates correlation IDs for each request and ensures they're available
    throughout the request lifecycle.

    This implementation works directly with the ASGI protocol interface, making it
    significantly faster than BaseHTTPMiddleware-based approaches (typically 10ms+
    faster). It avoids the overhead of Request/Response object creation and response
    buffering by intercepting and modifying headers at the protocol level as they
    stream through.

    Behavior:
    1. Extracts correlation ID from incoming request headers (X-Correlation-ID
       or X-Request-ID)
    2. If not present, generates a new one with "gw" or "inf" prefix
    3. Sets correlation ID in ContextVar for the request lifecycle
    4. Adds correlation ID to response headers
    """

    def __init__(self, app: ASGIApp, prefix: str = "gw"):
        self.app = app
        self.prefix = prefix
        logger.info("CorrelationIdASGIMiddleware initialized")

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            # pass through non-HTTP requests (e.g., websocket)
            await self.app(scope, receive, send)
            return

        try:
            # ASGI headers are stored as a list of tuples
            headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
        except UnicodeDecodeError:
            headers = {}

        correlation_id = (
            headers.get(CORRELATION_ID_HEADER) or headers.get(REQUEST_ID_HEADER)
        )
        if not correlation_id:
            correlation_id = correlation_ids.generate_correlation_id(prefix=self.prefix)

        correlation_ids.set_correlation_id(correlation_id)

        async def _send_wrapper(message):
            if message["type"] == "http.response.start":
                # remove existing correlation header to prevent duplicates
                headers_list = [
                    (k, v) for k, v in message.get("headers", [])
                    if k.lower() != CORRELATION_ID_HEADER.lower().encode()
                ]
                headers_list.append(
                    (CORRELATION_ID_HEADER.encode(), correlation_id.encode())
                )
                message["headers"] = headers_list
            await send(message)

        try:
            await self.app(scope, receive, _send_wrapper)
        except Exception as e:
            logger.error(
                "Request failed with exception",
                error=str(e),
                exc_info=e
            )
            raise
