import uuid
from contextvars import ContextVar
from typing import Optional

# contextVar to store correlation ID for the current async context, this automatically
# propagates across async calls within the same request
correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> Optional[str]:
    """Get the correlation ID for the current request context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current request context."""
    correlation_id_var.set(correlation_id)


def generate_correlation_id(prefix: str = "gw") -> str:
    """
    Generate a new correlation ID with a prefix.

    Prefix distinguishes where the correlation id is generated. In principle, it should 
    always be generated in gateway.
    
    Args:
        prefix: Prefix for the correlation ID (default: "gw" for gateway)
                Use "inf" for inference service
    """
    return f"{prefix}-{uuid.uuid4()}"


def clear_correlation_id() -> None:
    """
    Clear the correlation ID from the current context.
    Useful for cleanup in testing.
    """
    correlation_id_var.set(None)
