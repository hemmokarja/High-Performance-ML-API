import hashlib
import secrets
import structlog
from datetime import datetime, timezone
from typing import Optional

logger = structlog.get_logger(__name__)


class ApiKeyDB:
    """
    In-memory API key database with user metadata.
    
    In production, this should be backed by PostgreSQL/Redis.
    Keys are stored as SHA-256 hashes for security.
    """
    def __init__(self):
        self._keys: dict[str, dict] = {}

    def add_key(
        self,
        key: str,
        user_id: str,
        name: str,
        rate_limit_per_minute: int = 60,
        rate_limit_per_hour: int = 1000,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Add a new API key to the database.

        Args:
            key: The raw API key (will be hashed)
            user_id: Unique user identifier
            name: Human-readable name for the key
            rate_limit_per_minute: Requests allowed per minute
            rate_limit_per_hour: Requests allowed per hour
            metadata: Optional additional metadata

        Returns:
            The SHA-256 hash of the key
        """
        key_hash = self._hash_key(key)

        self._keys[key_hash] = {
            "user_id": user_id,
            "name": name,
            "rate_limit_per_minute": rate_limit_per_minute,
            "rate_limit_per_hour": rate_limit_per_hour,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "is_active": True
        }

        logger.info(
            "API key added",
            user_id=user_id,
            name=name,
            key_hash=key_hash[:16]  # log only prefix for security
        )

        return key_hash

    def get_key_info(self, key: str) -> Optional[dict]:
        """
        Retrieve information for an API key.

        Args:
            key: The raw API key

        Returns:
            Key information dict if found and active, None otherwise
        """
        key_hash = self._hash_key(key)
        key_info = self._keys.get(key_hash)

        if key_info and key_info.get("is_active", False):
            return key_info

        return None

    def revoke_key(self, key: str) -> bool:
        """
        Revoke an API key (soft delete).

        Args:
            key: The raw API key

        Returns:
            True if key was found and revoked, False otherwise
        """
        key_hash = self._hash_key(key)

        if key_hash in self._keys:
            self._keys[key_hash]["is_active"] = False
            logger.info("API key revoked", key_hash=key_hash[:16])
            return True

        return False

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key using SHA-256"""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def generate_key(prefix: str = "sk_live") -> str:
        """
        Generate a new API key with secure random bytes.

        Args:
            prefix: Key prefix (e.g., 'sk_live' for production, 'sk_test' for testing)

        Returns:
            A new API key string
        """
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"
