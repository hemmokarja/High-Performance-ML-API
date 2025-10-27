import datetime
import time
from collections import defaultdict
from dataclasses import dataclass

import structlog

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


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter with per-minute and per-hour limits.

    Uses a sliding window algorithm for more accurate rate limiting compared to fixed
    windows. 

    TODO: in production, use Redis with sorted sets for distributed rate limiting.
    """
    def __init__(self):
        # structure: {user_id: {timestamp: request_count}}
        self._minute_windows: dict[str, dict[int, int]] = defaultdict(dict)
        self._hour_windows: dict[str, dict[int, int]] = defaultdict(dict)

        logger.info("SlidingWindowRateLimiter initialized")

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

        self._cleanup_old_windows(user_id, now)
        
        # count requests in current windows
        minute_count = self._count_requests(
            self._minute_windows[user_id], now, window_seconds=60
        )
        hour_count = self._count_requests(
            self._hour_windows[user_id], now, window_seconds=3600
        )

        if minute_count >= minute_limit:
            retry_after = self._calculate_retry_after(
                self._minute_windows[user_id], now, window_seconds=60
            )
            logger.warning(
                "Rate limit exceeded (minute)",
                user_id=user_id,
                count=minute_count,
                limit=minute_limit
            )
            raise RateLimitExceeded(
                limit_type="minute", limit=minute_limit, retry_after=retry_after
            )

        if hour_count >= hour_limit:
            retry_after = self._calculate_retry_after(
                self._hour_windows[user_id], now, window_seconds=3600
            )
            logger.warning(
                "Rate limit exceeded (hour)",
                user_id=user_id,
                count=hour_count,
                limit=hour_limit
            )
            raise RateLimitExceeded(
                limit_type="hour", limit=hour_limit, retry_after=retry_after
            )

        current_second = int(now)
        self._minute_windows[user_id][current_second] = (
            self._minute_windows[user_id].get(current_second, 0) + 1
        )
        self._hour_windows[user_id][current_second] = (
            self._hour_windows[user_id].get(current_second, 0) + 1
        )

        return RateLimitInfo(
            requests_this_minute=minute_count + 1,
            requests_this_hour=hour_count + 1,
            minute_window_start=now - 60,
            hour_window_start=now - 3600,
            minute_limit=minute_limit,
            hour_limit=hour_limit
        )
    
    def get_usage(self, user_id: str) -> dict:
        """
        Get current usage stats for a user without incrementing counters.

        Args:
            user_id: Unique identifier for the user

        Returns:
            Dictionary with current usage statistics
        """
        now = time.time()
        minute_count = self._count_requests(
            self._minute_windows.get(user_id, {}), now, window_seconds=60
        )
        hour_count = self._count_requests(
            self._hour_windows.get(user_id, {}), now, window_seconds=3600
        )
        return {
            "requests_last_minute": minute_count,
            "requests_last_hour": hour_count,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

    def reset_user(self, user_id: str) -> None:
        """Reset rate limit counters for a user"""
        self._minute_windows.pop(user_id, None)
        self._hour_windows.pop(user_id, None)
        logger.info("Rate limit reset", user_id=user_id)

    def _count_requests(
        self, window: dict[int, int], now: float, window_seconds: int
    ) -> int:
        """Count requests within a time window"""
        cutoff = now - window_seconds
        return sum(count for timestamp, count in window.items() if timestamp > cutoff)

    def _calculate_retry_after(
        self, window: dict[int, int], now: float, window_seconds: int
    ) -> int:
        """Calculate seconds until oldest request falls out of window"""
        if not window:
            return 1

        oldest_timestamp = min(window.keys())
        retry_after = int(oldest_timestamp + window_seconds - now) + 1
        return max(1, retry_after)
    
    def _cleanup_old_windows(self, user_id: str, now: float) -> None:
        """Remove expired entries from sliding windows"""

        # clean minute window (keep last 2 minutes for safety margin)
        minute_cutoff = now - 120
        if user_id in self._minute_windows:
            self._minute_windows[user_id] = {
                ts: count
                for ts, count in self._minute_windows[user_id].items()
                if ts > minute_cutoff
            }

        # clean hour window (keep last 2 hours for safety margin)
        hour_cutoff = now - 7200
        if user_id in self._hour_windows:
            self._hour_windows[user_id] = {
                ts: count
                for ts, count in self._hour_windows[user_id].items()
                if ts > hour_cutoff
            }
