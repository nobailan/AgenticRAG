"""
redis_client.py -- Lightweight Redis connection wrapper for semantic caching.

Provides a thin wrapper around redis-py with automatic connection management
and graceful degradation when Redis is unavailable.

Usage:
    from src.cache.redis_client import RedisClient
    client = RedisClient(redis_url="redis://localhost:6379/0")
    client.set("key", "value", ttl=3600)
    value = client.get("key")
"""

import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class RedisClient:
    """Minimal Redis client with graceful fallback.

    If Redis is unavailable (connection refused, module not installed),
    all operations silently return None and log a warning once.

    Attributes:
        redis_url: Redis connection URL (e.g., 'redis://localhost:6379/0').
        connected: Whether the Redis connection is alive.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize Redis client.

        Args:
            redis_url: Redis connection URL string.
        """
        self.redis_url = redis_url
        self._client: Any = None
        self._connected: Optional[bool] = None  # None = not yet checked
        self._warned = False

    @property
    def connected(self) -> bool:
        """Check if Redis is available and connected."""
        if self._connected is None:
            self._connect()
        return self._connected or False

    def _connect(self) -> None:
        """Attempt to connect to Redis. Silently degrades on failure."""
        try:
            import redis
            self._client = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            self._client.ping()
            self._connected = True
            logger.info("Redis connected: %s", self.redis_url)
        except ImportError:
            if not self._warned:
                logger.warning("redis-py not installed. Semantic cache disabled.")
                self._warned = True
            self._connected = False
        except Exception as e:
            if not self._warned:
                logger.warning("Redis unavailable (%s). Semantic cache disabled.", e)
                self._warned = True
            self._connected = False
            self._client = None

    def get(self, key: str) -> Optional[str]:
        """Get a value from Redis.

        Args:
            key: Redis key.

        Returns:
            Stored string value, or None if not found or Redis unavailable.
        """
        if not self.connected or self._client is None:
            return None
        try:
            return self._client.get(key)
        except Exception as e:
            logger.warning("Redis GET failed: %s", e)
            return None

    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set a key-value pair in Redis with TTL.

        Args:
            key: Redis key.
            value: String value to store.
            ttl: Time-to-live in seconds (default: 1 hour).

        Returns:
            True on success, False on failure.
        """
        if not self.connected or self._client is None:
            return False
        try:
            self._client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.warning("Redis SET failed: %s", e)
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis.

        Args:
            key: Redis key.

        Returns:
            True if key exists, False otherwise or if Redis unavailable.
        """
        if not self.connected or self._client is None:
            return False
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.warning("Redis EXISTS failed: %s", e)
            return False

    def get_json(self, key: str) -> Optional[dict]:
        """Get a JSON value from Redis (auto-deserialized).

        Args:
            key: Redis key.

        Returns:
            Deserialized dict, or None.
        """
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    def set_json(self, key: str, value: dict, ttl: int = 3600) -> bool:
        """Set a JSON-serializable dict in Redis.

        Args:
            key: Redis key.
            value: Dict to store.
            ttl: TTL in seconds.

        Returns:
            True on success.
        """
        try:
            return self.set(key, json.dumps(value, ensure_ascii=False), ttl)
        except (TypeError, ValueError):
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._connected = False
