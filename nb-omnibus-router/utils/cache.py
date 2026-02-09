"""
Redis Caching Utility Module
NeuralBlitz Omnibus Router - High Performance Caching Layer

Provides:
- Redis connection management with connection pooling
- Decorator-based function result caching
- Cache invalidation strategies
- Fallback to no-cache mode when Redis is unavailable
- Cache warming and pre-loading capabilities
"""

import asyncio
import hashlib
import json
import logging
import pickle
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass, asdict

import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis.asyncio as redis
    from redis.asyncio.connection import ConnectionPool

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis package not installed. Caching will be disabled.")

# Try to import Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not installed. Cache metrics will be disabled.")


# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    CACHE_HITS = Counter(
        "neuralblitz_cache_hits_total",
        "Total number of cache hits",
        ["cache_type", "operation"],
    )
    CACHE_MISSES = Counter(
        "neuralblitz_cache_misses_total",
        "Total number of cache misses",
        ["cache_type", "operation"],
    )
    CACHE_ERRORS = Counter(
        "neuralblitz_cache_errors_total",
        "Total number of cache errors",
        ["cache_type", "error_type"],
    )
    CACHE_DURATION = Histogram(
        "neuralblitz_cache_operation_duration_seconds",
        "Cache operation duration in seconds",
        ["cache_type", "operation"],
    )
    CACHE_SIZE = Gauge(
        "neuralblitz_cache_size_bytes", "Current cache size in bytes", ["cache_type"]
    )
    CACHE_KEYS = Gauge(
        "neuralblitz_cache_keys_total", "Total number of keys in cache", ["cache_type"]
    )
    CACHE_WARMUP_TIME = Histogram(
        "neuralblitz_cache_warmup_duration_seconds",
        "Cache warmup duration in seconds",
        ["cache_type"],
    )


@dataclass
class CacheConfig:
    """Configuration for Redis cache."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30

    # Cache policies
    default_ttl: int = 3600  # 1 hour
    quantum_state_ttl: int = 300  # 5 minutes
    multi_reality_ttl: int = 600  # 10 minutes
    consciousness_ttl: int = 180  # 3 minutes
    api_response_ttl: int = 900  # 15 minutes

    # Warming configuration
    warmup_enabled: bool = True
    warmup_batch_size: int = 100
    warmup_concurrency: int = 5

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


class CircuitBreaker:
    """Circuit breaker pattern for Redis failures."""

    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "open":
                if (
                    self.last_failure_time
                    and (datetime.utcnow() - self.last_failure_time).seconds
                    > self.timeout
                ):
                    self.state = "half-open"
                    self.failure_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                self.failure_count = 0
                if self.state == "half-open":
                    self.state = "closed"
            return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = datetime.utcnow()
                if self.failure_count >= self.threshold:
                    self.state = "open"
                    logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures"
                    )
            raise e


class RedisCache:
    """
    High-performance Redis cache manager with fallback support.

    Features:
    - Connection pooling
    - Automatic reconnection
    - Circuit breaker pattern
    - Graceful degradation to no-cache mode
    - Comprehensive metrics
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
        self._circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
        )
        self._local_cache: Dict[str, Any] = {}
        self._local_cache_ttl: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
        self._connected = False
        self._metrics_enabled = PROMETHEUS_AVAILABLE

    async def connect(self) -> bool:
        """Initialize Redis connection pool."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using local cache fallback")
            return False

        try:
            self._pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
            )
            self._client = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Close Redis connection pool."""
        if self._pool:
            await self._pool.disconnect()
            self._connected = False
            logger.info("Disconnected from Redis")

    async def health_check(self) -> Dict[str, Any]:
        """Check Redis health status."""
        if not self._connected or not self._client:
            return {
                "status": "unavailable",
                "mode": "fallback",
                "local_cache_keys": len(self._local_cache),
            }

        try:
            info = await self._client.info()
            return {
                "status": "healthy",
                "mode": "redis",
                "version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            return {"status": "unhealthy", "mode": "fallback", "error": str(e)}

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = json.dumps(
            {"args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
        hash_value = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"nb:{prefix}:{hash_value}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self._metrics_enabled:
            with CACHE_DURATION.labels(cache_type="redis", operation="get").time():
                return await self._get_internal(key)
        return await self._get_internal(key)

    async def _get_internal(self, key: str) -> Optional[Any]:
        """Internal get implementation."""
        # Try Redis first
        if self._connected and self._client:
            try:
                data = await self._circuit_breaker.call(self._client.get, key)
                if data:
                    if self._metrics_enabled:
                        CACHE_HITS.labels(cache_type="redis", operation="get").inc()
                    return self._deserialize(data)
            except Exception as e:
                if self._metrics_enabled:
                    CACHE_ERRORS.labels(cache_type="redis", error_type="get").inc()
                logger.warning(f"Redis get failed, using local cache: {e}")

        # Fallback to local cache
        async with self._lock:
            if key in self._local_cache:
                expiry = self._local_cache_ttl.get(key)
                if expiry and datetime.utcnow() < expiry:
                    if self._metrics_enabled:
                        CACHE_HITS.labels(cache_type="local", operation="get").inc()
                    return self._local_cache[key]
                else:
                    # Expired, remove from cache
                    del self._local_cache[key]
                    del self._local_cache_ttl[key]

        if self._metrics_enabled:
            CACHE_MISSES.labels(
                cache_type="redis" if self._connected else "local", operation="get"
            ).inc()
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,  # Only set if key doesn't exist
    ) -> bool:
        """Set value in cache."""
        if self._metrics_enabled:
            with CACHE_DURATION.labels(cache_type="redis", operation="set").time():
                return await self._set_internal(key, value, ttl, nx)
        return await self._set_internal(key, value, ttl, nx)

    async def _set_internal(
        self, key: str, value: Any, ttl: Optional[int] = None, nx: bool = False
    ) -> bool:
        """Internal set implementation."""
        ttl = ttl or self.config.default_ttl
        serialized = self._serialize(value)

        # Try Redis first
        if self._connected and self._client:
            try:
                if nx:
                    result = await self._circuit_breaker.call(
                        self._client.setnx, key, serialized
                    )
                    if result:
                        await self._client.expire(key, ttl)
                    return bool(result)
                else:
                    await self._circuit_breaker.call(
                        self._client.setex, key, ttl, serialized
                    )
                    return True
            except Exception as e:
                if self._metrics_enabled:
                    CACHE_ERRORS.labels(cache_type="redis", error_type="set").inc()
                logger.warning(f"Redis set failed, using local cache: {e}")

        # Fallback to local cache
        async with self._lock:
            self._local_cache[key] = value
            self._local_cache_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)

        return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        deleted = False

        if self._connected and self._client:
            try:
                await self._circuit_breaker.call(self._client.delete, key)
                deleted = True
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

        # Also remove from local cache
        async with self._lock:
            if key in self._local_cache:
                del self._local_cache[key]
                if key in self._local_cache_ttl:
                    del self._local_cache_ttl[key]
                deleted = True

        return deleted

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        count = 0

        if self._connected and self._client:
            try:
                keys = []
                async for key in self._client.scan_iter(match=pattern):
                    keys.append(key)

                if keys:
                    await self._circuit_breaker.call(self._client.delete, *keys)
                    count = len(keys)
            except Exception as e:
                logger.warning(f"Redis delete_pattern failed: {e}")

        # Also clean local cache
        async with self._lock:
            local_keys = list(self._local_cache.keys())
            for key in local_keys:
                if pattern.replace("*", "") in key:
                    del self._local_cache[key]
                    if key in self._local_cache_ttl:
                        del self._local_cache_ttl[key]
                    count += 1

        return count

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self._connected and self._client:
            try:
                return await self._circuit_breaker.call(self._client.exists, key) > 0
            except Exception:
                pass

        async with self._lock:
            return key in self._local_cache

    async def ttl(self, key: str) -> int:
        """Get TTL of a key."""
        if self._connected and self._client:
            try:
                return await self._circuit_breaker.call(self._client.ttl, key)
            except Exception:
                pass

        async with self._lock:
            if key in self._local_cache_ttl:
                expiry = self._local_cache_ttl[key]
                remaining = (expiry - datetime.utcnow()).total_seconds()
                return max(0, int(remaining))
        return -2  # Key doesn't exist

    async def clear(self) -> bool:
        """Clear all cache."""
        if self._connected and self._client:
            try:
                await self._circuit_breaker.call(self._client.flushdb)
            except Exception as e:
                logger.warning(f"Redis flush failed: {e}")

        async with self._lock:
            self._local_cache.clear()
            self._local_cache_ttl.clear()

        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "connected": self._connected,
            "mode": "redis" if self._connected else "local",
            "local_cache_size": len(self._local_cache),
        }

        if self._connected and self._client:
            try:
                info = await self._client.info()
                stats.update(
                    {
                        "redis_version": info.get("redis_version"),
                        "used_memory": info.get("used_memory"),
                        "used_memory_human": info.get("used_memory_human"),
                        "connected_clients": info.get("connected_clients"),
                        "total_keys": info.get("db0", {}).get("keys", 0)
                        if "db0" in info
                        else 0,
                    }
                )
            except Exception as e:
                stats["error"] = str(e)

        return stats


# Global cache instance
_cache_instance: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get or create global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
        await _cache_instance.connect()
    return _cache_instance


def cache_result(
    prefix: str,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
    skip_args: Optional[List[int]] = None,
    condition: Optional[Callable] = None,
):
    """
    Decorator for caching function results.

    Args:
        prefix: Cache key prefix
        ttl: Time-to-live in seconds
        key_func: Custom function to generate cache key
        skip_args: Argument indices to skip when generating key
        condition: Function to determine if result should be cached

    Example:
        @cache_result(prefix="quantum_state", ttl=300)
        async def calculate_quantum_state(data):
            return expensive_calculation(data)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = await get_cache()

            # Check condition
            if condition and not condition(*args, **kwargs):
                return await func(*args, **kwargs)

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Filter out arguments to skip
                filtered_args = args
                if skip_args:
                    filtered_args = tuple(
                        arg for i, arg in enumerate(args) if i not in skip_args
                    )
                cache_key = cache._generate_key(prefix, *filtered_args, **kwargs)

            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            if result is not None:
                await cache.set(cache_key, result, ttl=ttl)

            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        # Attach cache invalidation helper
        async_wrapper.invalidate = lambda *args, **kwargs: _invalidate_cache(
            prefix, key_func, skip_args, *args, **kwargs
        )
        async_wrapper.invalidate_all = lambda: _invalidate_cache_pattern(
            f"nb:{prefix}:*"
        )

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


async def _invalidate_cache(
    prefix: str,
    key_func: Optional[Callable],
    skip_args: Optional[List[int]],
    *args,
    **kwargs,
):
    """Invalidate specific cache entry."""
    cache = await get_cache()

    if key_func:
        cache_key = key_func(*args, **kwargs)
    else:
        filtered_args = args
        if skip_args:
            filtered_args = tuple(
                arg for i, arg in enumerate(args) if i not in skip_args
            )
        cache_key = cache._generate_key(prefix, *filtered_args, **kwargs)

    await cache.delete(cache_key)


async def _invalidate_cache_pattern(pattern: str):
    """Invalidate all cache entries matching pattern."""
    cache = await get_cache()
    await cache.delete_pattern(pattern)


class CacheInvalidator:
    """Helper class for cache invalidation strategies."""

    def __init__(self):
        self._invalidation_handlers: Dict[str, List[Callable]] = {}

    def register(self, event: str, handler: Callable):
        """Register an invalidation handler for an event."""
        if event not in self._invalidation_handlers:
            self._invalidation_handlers[event] = []
        self._invalidation_handlers[event].append(handler)

    async def invalidate(self, event: str, **context):
        """Trigger invalidation for an event."""
        handlers = self._invalidation_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(**context)
                else:
                    handler(**context)
            except Exception as e:
                logger.error(f"Cache invalidation handler failed: {e}")


# Global invalidator instance
_invalidator = CacheInvalidator()


def on_event(event: str):
    """Decorator to register cache invalidation handler."""

    def decorator(func: Callable) -> Callable:
        _invalidator.register(event, func)
        return func

    return decorator


class CacheWarmer:
    """
    Cache warming strategies for startup performance.

    Pre-loads frequently accessed data into cache on startup.
    """

    def __init__(self, cache: RedisCache):
        self.cache = cache
        self._warming_tasks: List[Callable] = []
        self._config = cache.config

    def register(self, task: Callable, priority: int = 0):
        """Register a warming task with priority (lower = higher priority)."""
        self._warming_tasks.append((priority, task))
        self._warming_tasks.sort(key=lambda x: x[0])

    async def warm(self) -> Dict[str, Any]:
        """Execute all warming tasks."""
        if not self._config.warmup_enabled:
            logger.info("Cache warming disabled")
            return {"status": "disabled"}

        logger.info(f"Starting cache warming with {len(self._warming_tasks)} tasks")
        start_time = datetime.utcnow()
        results = []

        # Execute tasks with limited concurrency
        semaphore = asyncio.Semaphore(self._config.warmup_concurrency)

        async def run_task(priority: int, task: Callable):
            async with semaphore:
                task_start = datetime.utcnow()
                try:
                    result = (
                        await task(self.cache)
                        if asyncio.iscoroutinefunction(task)
                        else task(self.cache)
                    )
                    duration = (datetime.utcnow() - task_start).total_seconds()
                    return {
                        "task": task.__name__,
                        "priority": priority,
                        "status": "success",
                        "duration": duration,
                        "result": result,
                    }
                except Exception as e:
                    duration = (datetime.utcnow() - task_start).total_seconds()
                    return {
                        "task": task.__name__,
                        "priority": priority,
                        "status": "error",
                        "duration": duration,
                        "error": str(e),
                    }

        # Run all tasks concurrently with semaphore limit
        tasks = [run_task(p, t) for p, t in self._warming_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = (datetime.utcnow() - start_time).total_seconds()

        if PROMETHEUS_AVAILABLE:
            CACHE_WARMUP_TIME.observe(total_duration)

        success_count = sum(
            1 for r in results if isinstance(r, dict) and r.get("status") == "success"
        )

        return {
            "status": "completed",
            "total_tasks": len(self._warming_tasks),
            "successful": success_count,
            "failed": len(self._warming_tasks) - success_count,
            "duration_seconds": total_duration,
            "tasks": results,
        }


# Specialized cache decorators for NeuralBlitz operations


def cache_quantum_state(ttl: int = 300):
    """Cache decorator for quantum neuron state calculations."""
    return cache_result(prefix="quantum_state", ttl=ttl)


def cache_multi_reality_evolution(ttl: int = 600):
    """Cache decorator for multi-reality network evolution results."""
    return cache_result(prefix="multi_reality", ttl=ttl)


def cache_consciousness_metrics(ttl: int = 180):
    """Cache decorator for consciousness level metrics."""
    return cache_result(prefix="consciousness", ttl=ttl)


def cache_api_response(ttl: int = 900):
    """Cache decorator for API responses."""
    return cache_result(
        prefix="api_response",
        ttl=ttl,
        skip_args=[0],  # Skip request object in key generation
    )


# Cache warming tasks for NeuralBlitz


async def warm_quantum_capabilities(cache: RedisCache):
    """Pre-load quantum capabilities into cache."""
    from engines.quantum import QuantumEngine

    engine = QuantumEngine()
    capabilities = await engine.get_capabilities()
    await cache.set("nb:warm:quantum_capabilities", capabilities, ttl=3600)
    return {"quantum_capabilities": True}


async def warm_neuralblitz_capabilities(cache: RedisCache):
    """Pre-load NeuralBlitz capabilities into cache."""
    from engines.neuralblitz import NeuralBlitzCore

    nb = NeuralBlitzCore()
    capabilities = await nb.get_capabilities()
    await cache.set("nb:warm:neuralblitz_capabilities", capabilities, ttl=3600)
    return {"neuralblitz_capabilities": True}


async def warm_consciousness_level(cache: RedisCache):
    """Pre-load consciousness level into cache."""
    level_data = {
        "level": 7,
        "max_level": 8,
        "percentage": 87.5,
        "status": "active",
        "dimensions": 11,
        "integration": "high",
    }
    await cache.set("nb:warm:consciousness_level", level_data, ttl=180)
    return {"consciousness_level": True}


# Initialize cache warming
def register_default_warming_tasks(warmer: CacheWarmer):
    """Register default NeuralBlitz cache warming tasks."""
    warmer.register(warm_quantum_capabilities, priority=1)
    warmer.register(warm_neuralblitz_capabilities, priority=2)
    warmer.register(warm_consciousness_level, priority=3)


# Context manager for cache operations
@asynccontextmanager
async def cached_operation(key: str, ttl: int = 3600):
    """Context manager for caching operations."""
    cache = await get_cache()
    value = await cache.get(key)

    if value is not None:
        yield value, True  # cached = True
    else:
        result = yield None, False  # cached = False
        if result is not None:
            await cache.set(key, result, ttl=ttl)
