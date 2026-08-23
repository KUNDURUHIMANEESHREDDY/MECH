"""Rate Limiting Configuration for MECH Platform."""

import asyncio
from collections import defaultdict
from typing import Optional

import redis.asyncio as aioredis

from backend.core.config.settings import get_settings


class RateLimitStore:
    """Abstract rate limit store interface."""

    async def check_limit(self, key: str, window: int, max_requests: int) -> bool:
        """Check if request is within rate limit. Returns True if allowed."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close connections."""
        pass


class InMemoryRateLimitStore(RateLimitStore):
    """In-memory rate limit store (fallback for local dev)."""

    def __init__(self, max_keys: int = 500) -> None:
        self._store: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._max_keys = max_keys

    async def check_limit(self, key: str, window: int, max_requests: int) -> bool:
        import time

        now = time.time()
        cutoff = now - window

        async with self._lock:
            window_list = self._store[key]
            # Remove expired entries
            while window_list and window_list[0] < cutoff:
                window_list.pop(0)

            # Check limit
            if len(window_list) >= max_requests:
                return False

            # Add current request
            window_list.append(now)

            # Periodic cleanup
            if len(self._store) > self._max_keys:
                stale_keys = [
                    k for k, v in self._store.items() if not v or v[-1] < cutoff
                ]
                for k in stale_keys:
                    del self._store[k]

        return True

    async def close(self) -> None:
        self._store.clear()


class RedisRateLimitStore(RateLimitStore):
    """Redis-backed rate limit store."""

    def __init__(
        self,
        redis_url: str,
        encoding: str = "utf-8",
        decode_responses: bool = True,
        connect_timeout: int = 2,
        socket_timeout: int = 2,
        fail_closed: bool = True,
    ) -> None:
        self._redis_url = redis_url
        self._client: Optional[aioredis.Redis] = None
        self._encoding = encoding
        self._decode_responses = decode_responses
        self._connect_timeout = connect_timeout
        self._socket_timeout = socket_timeout
        self._fail_closed = fail_closed

    async def _ensure_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(
                self._redis_url,
                encoding=self._encoding,
                decode_responses=self._decode_responses,
                socket_connect_timeout=self._connect_timeout,
                socket_timeout=self._socket_timeout,
            )
        return self._client

    async def check_limit(self, key: str, window: int, max_requests: int) -> bool:
        import time

        client = await self._ensure_client()
        now = time.time()
        cutoff = now - window

        try:
            pipe = client.pipeline()
            pipe.zremrangebyscore(key, "-inf", cutoff)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, window * 2)
            results = await pipe.execute()
            count = results[2]
            return count <= max_requests
        except Exception as e:
            import logging
            logging.getLogger("MECH").warning("Redis rate limit error: %s; fail_closed=%s", e, self._fail_closed)
            return not self._fail_closed

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


def create_rate_limit_store() -> RateLimitStore:
    """Create rate limit store based on configuration."""
    settings = get_settings()

    if settings.rate_limit_redis_url:
        return RedisRateLimitStore(
            redis_url=settings.rate_limit_redis_url,
            encoding=settings.redis_encoding,
            decode_responses=settings.redis_decode_responses,
            connect_timeout=settings.rate_limit_redis_connect_timeout,
            socket_timeout=settings.rate_limit_redis_socket_timeout,
            fail_closed=settings.rate_limit_fail_closed,
        )
    return InMemoryRateLimitStore(max_keys=settings.rate_limit_in_memory_max_keys)


def get_rate_limit_config() -> dict:
    """Get rate limiting configuration values."""
    settings = get_settings()
    return {
        "window_seconds": settings.rate_limit_window_seconds,
        "max_requests": settings.rate_limit_max_requests,
    }