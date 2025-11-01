import datetime
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import structlog
import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    limit_type: str  # "minute" or "hour"
    limit: int
    retry_after: int  # seconds until limit resets

    def __str__(self):
        return (
            f"Rate limit exceeded: {self.limit} requests per {self.limit_type}. "
            f"Retry after {self.retry_after} seconds."
        )


@dataclass
class RateLimitInfo:
    """Rate limit information for a user"""
    requests_this_minute: int
    requests_this_hour: int
    minute_window_start: float
    hour_window_start: float
    minute_limit: int
    hour_limit: int


class RateLimiter(ABC):

    @abstractmethod
    def check_rate_limit(
        self, user_id: str, minute_limit: int, hour_limit: int
    ) -> RateLimitInfo:
        pass

    @abstractmethod
    def get_usage(self, user_id: str) -> dict:
        pass

    @abstractmethod
    def reset_user(self, user_id: str) -> None:
        pass

    @abstractmethod
    def is_redis_available(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class RedisSlidingWindowRateLimiter(RateLimiter):
    """
    Redis-based sliding window rate limiter with per-minute and per-hour limits.

    Uses Redis sorted sets for distributed rate limiting with sliding window algorithm.

    In production environments with Redis, this provides:
    - Distributed rate limiting across multiple gateway instances
    - Atomic operations via Lua scripts
    - Automatic key expiration
    """

    # Lua script for atomic rate limit check and increment
    # This ensures thread-safety and reduces round-trips to Redis
    CHECK_AND_INCREMENT_SCRIPT = """
    local minute_key = KEYS[1]
    local hour_key = KEYS[2]
    local now = tonumber(ARGV[1])
    local minute_limit = tonumber(ARGV[2])
    local hour_limit = tonumber(ARGV[3])
    local minute_window = 60
    local hour_window = 3600

    -- Remove expired entries
    redis.call('ZREMRANGEBYSCORE', minute_key, '-inf', now - minute_window)
    redis.call('ZREMRANGEBYSCORE', hour_key, '-inf', now - hour_window)

    -- Count current requests
    local minute_count = redis.call('ZCOUNT', minute_key, now - minute_window, '+inf')
    local hour_count = redis.call('ZCOUNT', hour_key, now - hour_window, '+inf')

    -- Check limits
    if minute_count >= minute_limit then
        local oldest = redis.call('ZRANGE', minute_key, 0, 0, 'WITHSCORES')
        local retry_after = 1
        if #oldest > 0 then
            retry_after = math.max(1, math.ceil(tonumber(oldest[2]) + minute_window - now))
        end
        return {-1, minute_count, hour_count, retry_after, 'minute'}
    end

    if hour_count >= hour_limit then
        local oldest = redis.call('ZRANGE', hour_key, 0, 0, 'WITHSCORES')
        local retry_after = 1
        if #oldest > 0 then
            retry_after = math.max(1, math.ceil(tonumber(oldest[2]) + hour_window - now))
        end
        return {-2, minute_count, hour_count, retry_after, 'hour'}
    end

    -- Add request with microsecond precision for uniqueness
    local score = now + math.random() / 1000000
    redis.call('ZADD', minute_key, score, score)
    redis.call('ZADD', hour_key, score, score)

    -- Set expiration (2x window for safety margin)
    redis.call('EXPIRE', minute_key, minute_window * 2)
    redis.call('EXPIRE', hour_key, hour_window * 2)
    
    return {0, minute_count + 1, hour_count + 1, 0, ''}
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "ratelimit",
        socket_connect_timeout: float = 1.0,
        socket_timeout: float = 1.0,
        retry_on_timeout: bool = False,
    ):
        """
        Initialize rate limiter with Redis connection.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for Redis keys
            socket_connect_timeout: Timeout for initial connection
            socket_timeout: Timeout for operations
            retry_on_timeout: Whether to retry on timeout
        """
        self.key_prefix = key_prefix
        self._redis_client: Optional[redis.Redis] = None
        self._script_sha: Optional[str] = None

        self._redis_client = redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            retry_on_timeout=retry_on_timeout,
            health_check_interval=30,
        )

        # test connection and load script
        self._redis_client.ping()
        self._script_sha = self._redis_client.script_load(
            self.CHECK_AND_INCREMENT_SCRIPT
        )

        logger.info(
            "Redis rate limiter initialized successfully",
            redis_url=redis_url,
            mode="distributed"
        )

    def check_rate_limit(
        self, user_id: str, minute_limit: int, hour_limit: int
    ) -> RateLimitInfo:
        """
        Check if a request should be allowed based on rate limits.

        Args:
            user_id: Unique identifier for the user
            minute_limit: Maximum requests allowed per minute
            hour_limit: Maximum requests allowed per hour

        Returns:
            RateLimitInfo with current usage stats
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        now = time.time()
        minute_key = f"{self.key_prefix}:minute:{user_id}"
        hour_key = f"{self.key_prefix}:hour:{user_id}"

        result = self._redis_client.evalsha(
            self._script_sha,
            2,  # number of keys
            minute_key,
            hour_key,
            now,
            minute_limit,
            hour_limit,
        )

        status, minute_count, hour_count, retry_after, limit_type = result

        if status == -1:  # minute limit exceeded
            logger.warning(
                "Rate limit exceeded (minute)",
                user_id=user_id,
                count=minute_count,
                limit=minute_limit
            )
            raise RateLimitExceeded(
                limit_type="minute",
                limit=minute_limit,
                retry_after=int(retry_after)
            )
        elif status == -2:  # hour limit exceeded
            logger.warning(
                "Rate limit exceeded (hour)",
                user_id=user_id,
                count=hour_count,
                limit=hour_limit
            )
            raise RateLimitExceeded(
                limit_type="hour",
                limit=hour_limit,
                retry_after=int(retry_after)
            )

        return RateLimitInfo(
            requests_this_minute=minute_count,
            requests_this_hour=hour_count,
            minute_window_start=now - 60,
            hour_window_start=now - 3600,
            minute_limit=minute_limit,
            hour_limit=hour_limit
        )

    def get_usage(self, user_id: str) -> dict:
        """Get current usage stats for a user without incrementing counters."""
        now = time.time()

        minute_key = f"{self.key_prefix}:minute:{user_id}"
        hour_key = f"{self.key_prefix}:hour:{user_id}"

        # clean old entries and count
        pipe = self._redis_client.pipeline()
        pipe.zremrangebyscore(minute_key, "-inf", now - 60)
        pipe.zcount(minute_key, now - 60, "+inf")
        pipe.zremrangebyscore(hour_key, "-inf", now - 3600)
        pipe.zcount(hour_key, now - 3600, "+inf")
        results = pipe.execute()

        minute_count = results[1]
        hour_count = results[3]

        return {
            "requests_last_minute": minute_count,
            "requests_last_hour": hour_count,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "backend": "redis"
        }

    def reset_user(self, user_id: str) -> None:
        """Reset rate limit counters for a user"""
        minute_key = f"{self.key_prefix}:minute:{user_id}"
        hour_key = f"{self.key_prefix}:hour:{user_id}"
        self._redis_client.delete(minute_key, hour_key)

        logger.info("Rate limit reset", user_id=user_id)

    def is_redis_available(self) -> bool:
        try:
            self._redis_client.ping()
            return True
        except (RedisConnectionError, RedisError):
            return False

    def close(self) -> None:
        if self._redis_client:
            try:
                self._redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning("Error closing Redis connection", error=str(e))


class NoOpRateLimiter(RateLimiter):
    """
    No-operation rate limiter that always allows requests.

    Used when rate limiting is disabled (e.g., when Redis is unavailable or
    BYPASS_RATE_LIMITS is set to true).
    """
    def __init__(self):
        logger.info("NoOp rate limiter initialized - all requests will be allowed")

    def check_rate_limit(
        self, user_id: str, minute_limit: int, hour_limit: int
    ) -> RateLimitInfo:
        return RateLimitInfo(
            requests_this_minute=0,
            requests_this_hour=0,
            minute_window_start=0.0,
            hour_window_start=0.0,
            minute_limit=minute_limit,
            hour_limit=hour_limit
        )

    def get_usage(self, user_id: str) -> dict:
        return {
            "requests_last_minute": 0,
            "requests_last_hour": 0,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "backend": "disabled"
        }

    def reset_user(self, user_id: str) -> None:
        pass

    def is_redis_available(self) -> bool:
        return False

    def close(self) -> None:
        pass


def create_rate_limiter(redis_url: str, bypass_rate_limits: bool):
    """
    Create appropriate rate limiter based on configuration and Redis availability.

    Priority:
    1. If BYPASS_RATE_LIMITS is True, use NoOpRateLimiter
    2. Try to connect to Redis at redis_url
    3. If Redis fails, use NoOpRateLimiter and log warning

    Args:
        redis_url: Redis connection URL
        bypass_rate_limits: Whether to bypass rate limiting entirely

    Returns:
        Rate limiter instance (Redis-based or NoOp)
    """
    if bypass_rate_limits:
        logger.warning("Rate limiting disabled via configuration")
        return NoOpRateLimiter()

    rate_limiter = RedisSlidingWindowRateLimiter(redis_url=redis_url)
    
    if not rate_limiter.is_redis_available():
        logger.warning(
            "Redis not available, rate limiting disabled",
            redis_url=redis_url,
            hint="Install Redis locally or start Redis container to enable rate limiting"
        )
        return NoOpRateLimiter()

    logger.info(
        "Redis rate limiter active",
        redis_url=redis_url,
        mode="distributed"
    )
    return rate_limiter