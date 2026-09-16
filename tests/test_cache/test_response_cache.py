"""Tests for ResponseCache."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from features.cache.response_cache import ResponseCache


@pytest.mark.unit
class TestResponseCache:
    """Tests for Redis-backed response cache with in-memory fallback."""

    def test_in_memory_cache_get_set(self) -> None:
        """Test getting and setting in in-memory fallback mode."""
        cache = ResponseCache(redis_enabled=False)
        query = "What is Gardens by the Bay?"
        data = {"answer": "Gardens by the Bay is a park.", "sources": []}

        assert cache.get(query) is None
        cache.set(query, data)
        cached = cache.get(query)
        assert cached is not None
        assert cached["answer"] == data["answer"]

    def test_cache_clear(self) -> None:
        """Test clearing the cache."""
        cache = ResponseCache(redis_enabled=False)
        cache.set("test", {"answer": "val"})
        assert cache.get("test") is not None
        cache.clear()
        assert cache.get("test") is None

    def test_redis_cache_get_set(self) -> None:
        """Test Redis interactions when redis client is available."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = '{"answer": "from redis", "sources": []}'

        cache = ResponseCache(redis_enabled=True, redis_client=mock_redis)
        result = cache.get("Singapore attractions")

        assert result is not None
        assert result["answer"] == "from redis"
        mock_redis.get.assert_called_once()

    def test_redis_set_calls_setex(self) -> None:
        """Test that setting in Redis sets TTL."""
        mock_redis = MagicMock()
        cache = ResponseCache(redis_enabled=True, redis_client=mock_redis, ttl_seconds=3600)
        cache.set("Singapore weather", {"answer": "Sunny"})

        mock_redis.setex.assert_called_once()
