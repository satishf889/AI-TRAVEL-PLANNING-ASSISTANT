from __future__ import annotations

"""Response and FAQ caching layer.

Uses Redis when available with automatic in-memory fallback to cache
frequently asked questions and responses, reducing LLM calls and latency.
"""

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache manager supporting Redis with an in-memory dictionary fallback."""

    def __init__(
        self,
        redis_enabled: bool = True,
        redis_host: str = "redis",
        redis_port: int = 6379,
        ttl_seconds: int = 3600,
        redis_client: Any = None,
    ) -> None:
        """Initialise ResponseCache."""
        self.redis_enabled = redis_enabled
        self.ttl_seconds = ttl_seconds
        self._memory_cache: dict[str, Any] = {}
        self.redis_client = redis_client

        if self.redis_enabled and self.redis_client is None:
            import redis

            hosts_to_try = [redis_host]
            if redis_host != "redis":
                hosts_to_try.append("redis")
            if redis_host != "localhost":
                hosts_to_try.append("localhost")

            for host in hosts_to_try:
                try:
                    client = redis.Redis(
                        host=host,
                        port=redis_port,
                        decode_responses=True,
                        socket_connect_timeout=1,
                    )
                    client.ping()
                    self.redis_client = client
                    logger.info(f"Connected to Redis response cache at {host}:{redis_port}.")
                    break
                except Exception:
                    continue

            if self.redis_client is None:
                logger.warning("Could not connect to Redis, using in-memory cache.")


    def _generate_key(self, query: str) -> str:
        """Generate a deterministic cache key from a query."""
        normalized = query.strip().lower()
        query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"faq_cache:{query_hash}"

    def get(self, query: str) -> dict[str, Any] | None:
        """Get cached response for a query, if present."""
        key = self._generate_key(query)

        if self.redis_client is not None:
            try:
                val = self.redis_client.get(key)
                if val:
                    return json.loads(val)  # type: ignore[no-any-return]
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        return self._memory_cache.get(key)

    def set(self, query: str, data: dict[str, Any]) -> None:
        """Store response data in cache."""
        key = self._generate_key(query)

        if self.redis_client is not None:
            try:
                serialized = json.dumps(data)
                self.redis_client.setex(key, self.ttl_seconds, serialized)
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        self._memory_cache[key] = data

    def clear(self) -> None:
        """Clear all cached responses."""
        if self.redis_client is not None:
            try:
                keys = self.redis_client.keys("faq_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")

        self._memory_cache.clear()
