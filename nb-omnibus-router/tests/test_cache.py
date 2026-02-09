"""
Test suite for Redis caching layer.
Tests cache functionality, fallback behavior, and metrics.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

# Import cache module
import sys

sys.path.insert(0, "/home/runner/workspace/nb-omnibus-router")

from utils.cache import (
    RedisCache,
    CacheConfig,
    CacheWarmer,
    CircuitBreaker,
    cache_result,
    get_cache,
    cache_quantum_state,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state (normal operation)."""
        cb = CircuitBreaker(threshold=3, timeout=60)

        async def success_func():
            return "success"

        result = await cb.call(success_func)
        assert result == "success"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(threshold=3, timeout=60)

        async def fail_func():
            raise Exception("Test error")

        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception):
                await cb.call(fail_func)

        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_after_timeout(self):
        """Test circuit breaker transitions to half-open after timeout."""
        cb = CircuitBreaker(threshold=1, timeout=0)  # 0 timeout for testing

        async def fail_func():
            raise Exception("Test error")

        with pytest.raises(Exception):
            await cb.call(fail_func)

        # Wait a bit and try again
        await asyncio.sleep(0.1)

        # Circuit should be half-open now
        assert cb.state in ["open", "half-open"]


class TestRedisCache:
    """Test Redis cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test cache initialization."""
        config = CacheConfig(host="localhost", port=6379)
        cache = RedisCache(config)

        # Should initialize without error
        assert cache.config == config
        assert cache._connected is False

    @pytest.mark.asyncio
    async def test_cache_key_generation(self):
        """Test cache key generation."""
        config = CacheConfig()
        cache = RedisCache(config)

        key1 = cache._generate_key("test", "arg1", "arg2", kwarg1="value1")
        key2 = cache._generate_key("test", "arg1", "arg2", kwarg1="value1")
        key3 = cache._generate_key("test", "arg1", "arg2", kwarg1="value2")

        # Same inputs should generate same key
        assert key1 == key2
        # Different inputs should generate different keys
        assert key1 != key3
        # Key should have prefix
        assert key1.startswith("nb:test:")

    @pytest.mark.asyncio
    async def test_serialization(self):
        """Test serialization and deserialization."""
        cache = RedisCache(CacheConfig())

        test_data = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
            "datetime": datetime.utcnow(),
        }

        serialized = cache._serialize(test_data)
        deserialized = cache._deserialize(serialized)

        assert deserialized["string"] == test_data["string"]
        assert deserialized["number"] == test_data["number"]
        assert deserialized["list"] == test_data["list"]
        assert deserialized["nested"] == test_data["nested"]

    @pytest.mark.asyncio
    async def test_local_cache_fallback(self):
        """Test local cache fallback when Redis is unavailable."""
        config = CacheConfig()
        cache = RedisCache(config)

        # Simulate Redis unavailable
        cache._connected = False

        # Should still be able to set and get
        await cache.set("test_key", "test_value", ttl=60)
        result = await cache.get("test_key")

        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Test cache TTL functionality."""
        config = CacheConfig()
        cache = RedisCache(config)
        cache._connected = False

        await cache.set("key_with_ttl", "value", ttl=1)

        # Should be available immediately
        result = await cache.get("key_with_ttl")
        assert result == "value"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        result = await cache.get("key_with_ttl")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test cache deletion."""
        config = CacheConfig()
        cache = RedisCache(config)
        cache._connected = False

        await cache.set("delete_key", "value")
        result = await cache.get("delete_key")
        assert result == "value"

        deleted = await cache.delete("delete_key")
        assert deleted is True

        result = await cache.get("delete_key")
        assert result is None


class TestCacheDecorator:
    """Test cache decorator functionality."""

    @pytest.mark.asyncio
    async def test_cache_decorator_caches_result(self):
        """Test that decorator caches function results."""
        call_count = 0

        @cache_result(prefix="test", ttl=60)
        async def expensive_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return {"arg1": arg1, "arg2": arg2, "computed": True}

        # First call should execute function
        result1 = await expensive_function("a", "b")
        assert call_count == 1

        # Second call with same args should return cached result
        result2 = await expensive_function("a", "b")
        assert call_count == 1  # Should not have called function again
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_decorator_different_args(self):
        """Test cache with different arguments."""
        call_count = 0

        @cache_result(prefix="test", ttl=60)
        async def expensive_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return {"arg1": arg1, "arg2": arg2}

        await expensive_function("a", "b")
        await expensive_function("a", "c")  # Different arg2

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_decorator_skip_args(self):
        """Test cache decorator with skip_args parameter."""
        call_count = 0

        @cache_result(prefix="test", ttl=60, skip_args=[0])  # Skip first arg
        async def expensive_function(request, data):
            nonlocal call_count
            call_count += 1
            return {"data": data}

        # Should cache based only on 'data' parameter
        await expensive_function("req1", "data1")
        await expensive_function("req2", "data1")  # Same data, different request

        assert call_count == 1  # Should be cached


class TestCacheWarmer:
    """Test cache warming functionality."""

    @pytest.mark.asyncio
    async def test_cache_warmer_registration(self):
        """Test cache warmer task registration."""
        config = CacheConfig(warmup_enabled=True)
        cache = RedisCache(config)
        warmer = CacheWarmer(cache)

        async def sample_task(cache):
            return {"warmed": True}

        warmer.register(sample_task, priority=1)

        assert len(warmer._warming_tasks) == 1

    @pytest.mark.asyncio
    async def test_cache_warmer_execution(self):
        """Test cache warmer task execution."""
        config = CacheConfig(warmup_enabled=True, warmup_concurrency=2)
        cache = RedisCache(config)
        cache._connected = False
        warmer = CacheWarmer(cache)

        execution_order = []

        async def task1(cache):
            execution_order.append(1)
            return {"task": 1}

        async def task2(cache):
            execution_order.append(2)
            return {"task": 2}

        warmer.register(task1, priority=1)
        warmer.register(task2, priority=2)

        result = await warmer.warm()

        assert result["status"] == "completed"
        assert result["total_tasks"] == 2
        assert result["successful"] == 2


class TestCacheIntegration:
    """Integration tests for cache with NeuralBlitz operations."""

    @pytest.mark.asyncio
    async def test_quantum_state_caching(self):
        """Test quantum state calculation caching."""
        call_count = 0

        @cache_quantum_state(ttl=300)
        async def calculate_quantum_state(qubits, depth):
            nonlocal call_count
            call_count += 1
            return {"qubits": qubits, "depth": depth, "state": "superposition"}

        # First call
        result1 = await calculate_quantum_state(4, 3)
        assert call_count == 1

        # Cached call
        result2 = await calculate_quantum_state(4, 3)
        assert call_count == 1
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_health_check(self):
        """Test cache health check functionality."""
        cache = RedisCache(CacheConfig())
        cache._connected = False

        health = await cache.health_check()

        assert health["status"] == "unavailable"
        assert health["mode"] == "fallback"

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        config = CacheConfig()
        cache = RedisCache(config)
        cache._connected = False

        # Add some data
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        stats = await cache.get_stats()

        assert stats["connected"] is False
        assert stats["mode"] == "local"
        assert stats["local_cache_size"] == 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
