from typing import Any, Dict

import structlog

from shared import correlation_ids


def _add_correlation_id_processor(
    logger: Any, method_name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Structlog processor to inject correlation ID into every log entry.

    This processor runs for every log call and automatically adds the
    correlation_id field if one exists in the current context.

    Args:
        logger: Logger instance
        method_name: Log method name (info, warning, error, etc.)
        event_dict: Dictionary of log event data

    Returns:
        Modified event_dict with correlation_id added
    """
    correlation_id = correlation_ids.get_correlation_id()
    if correlation_id:
        # Add to beginning of dict for visibility
        event_dict = {"correlation_id": correlation_id, **event_dict}
    return event_dict


def configure_structlog() -> None:
    """
    Configure structlog with correlation ID support.

    Call this once at application startup, before creating the FastAPI app.
    """
    structlog.configure(
        processors=[
            # add correlation ID first, so it appears early in logs
            _add_correlation_id_processor,
            # standard processors
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
