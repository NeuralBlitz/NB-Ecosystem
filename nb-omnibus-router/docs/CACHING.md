# NeuralBlitz Redis Caching Layer

## Overview

The NeuralBlitz API now includes a high-performance Redis caching layer that provides:

- **Sub-millisecond response times** for cached data
- **Graceful fallback** to local cache when Redis is unavailable
- **Circuit breaker pattern** to prevent cascade failures
- **Automatic cache warming** on startup
- **Comprehensive metrics** for monitoring cache performance
- **Flexible TTL policies** for different data types

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   NeuralBlitz API                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
   ┌─────▼─────┐                ┌─────▼─────┐
   │  Cache    │                │  Circuit  │
   │  Manager  │                │  Breaker  │
   └─────┬─────┘                └───────────┘
         │
   ┌─────▼────────────────┐
   │    Redis Cache       │
   │  (High Performance)  │
   └─────┬────────────────┘
         │
         │ Fallback
         ▼
   ┌─────────────────────┐
   │   Local In-Memory   │
   │   Cache (Fallback)  │
   └─────────────────────┘
```

## Features

### 1. Connection Management

- **Connection pooling** for efficient Redis connections
- **Automatic reconnection** with exponential backoff
- **Health checking** every 30 seconds
- **Graceful degradation** to local cache on failure

### 2. Cache Decorators

```python
from utils.cache import cache_quantum_state, cache_consciousness_metrics

@cache_quantum_state(ttl=300)  # 5 minutes
async def calculate_quantum_state(data):
    return expensive_calculation(data)

@cache_consciousness_metrics(ttl=180)  # 3 minutes
async def get_consciousness_level():
    return fetch_consciousness_metrics()
```

### 3. Cache Invalidation

```python
from utils.cache import on_event, get_cache

# Register invalidation handler
@on_event("quantum_state_updated")
async def invalidate_quantum_cache(**context):
    cache = await get_cache()
    await cache.delete_pattern("nb:quantum_state:*")
```

### 4. Cache Warming

Automatic pre-loading of frequently accessed data on startup:

- Quantum capabilities
- NeuralBlitz capabilities  
- Consciousness level metrics
- API response templates

### 5. Circuit Breaker

Prevents cascade failures when Redis is unavailable:

- Opens after 5 consecutive failures
- Half-open state after 60 seconds
- Automatic recovery detection

## Configuration

### settings.yaml

```yaml
REDIS:
  host: "redis"                    # Redis hostname
  port: 6379                       # Redis port
  db: 0                            # Database number
  password: null                   # Auth password (optional)
  max_connections: 50              # Connection pool size
  socket_timeout: 5.0              # Socket timeout (seconds)
  socket_connect_timeout: 5.0      # Connect timeout (seconds)
  retry_on_timeout: true           # Retry failed operations
  health_check_interval: 30        # Health check interval (seconds)
  
  # TTL Configuration (seconds)
  default_ttl: 3600                # Default: 1 hour
  quantum_state_ttl: 300           # Quantum states: 5 minutes
  multi_reality_ttl: 600           # Multi-reality: 10 minutes
  consciousness_ttl: 180           # Consciousness: 3 minutes
  api_response_ttl: 900            # API responses: 15 minutes
  
  # Cache Warming
  warmup_enabled: true             # Enable warmup on startup
  warmup_batch_size: 100           # Batch size for warming
  warmup_concurrency: 5            # Concurrent warming tasks
  
  # Circuit Breaker
  circuit_breaker_enabled: true    # Enable circuit breaker
  circuit_breaker_threshold: 5     # Failures before opening
  circuit_breaker_timeout: 60      # Seconds before half-open
```

### Environment Variables

```bash
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
```

## API Endpoints

### Cache Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cache/stats` | GET | Get cache statistics |
| `/cache/clear` | POST | Clear all cache entries |
| `/cache/health` | GET | Get cache health status |
| `/metrics/cache` | GET | Get cache performance metrics |
| `/metrics/cache/hit-rate` | GET | Get cache hit rate statistics |
| `/metrics/cache/keys` | GET | List cache keys (admin only) |

### Example Response: `/cache/stats`

```json
{
  "connected": true,
  "mode": "redis",
  "redis_version": "7.2.0",
  "used_memory_human": "1.23M",
  "connected_clients": 5,
  "total_keys": 1247,
  "local_cache_size": 0
}
```

## Usage Examples

### Basic Caching

```python
from utils.cache import cache_result, get_cache

# Cache function result
@cache_result(prefix="my_data", ttl=600)
async def fetch_expensive_data(id: int):
    return await database.query(id)

# Manual cache operations
cache = await get_cache()
await cache.set("key", "value", ttl=300)
value = await cache.get("key")
await cache.delete("key")
```

### Conditional Caching

```python
from utils.cache import cache_result

def should_cache(result):
    return result is not None and result.get("valid")

@cache_result(prefix="filtered", ttl=300, condition=should_cache)
async def get_filtered_data(filter_params):
    return await fetch_data(filter_params)
```

### Cache Invalidation

```python
from utils.cache import cache_result

@cache_result(prefix="user_data", ttl=3600)
async def get_user_data(user_id: int):
    return await fetch_user(user_id)

# Invalidate specific entry
await get_user_data.invalidate(user_id=123)

# Invalidate all user data
await get_user_data.invalidate_all()
```

## Prometheus Metrics

The following metrics are automatically exported:

| Metric | Type | Description |
|--------|------|-------------|
| `neuralblitz_cache_hits_total` | Counter | Total cache hits |
| `neuralblitz_cache_misses_total` | Counter | Total cache misses |
| `neuralblitz_cache_errors_total` | Counter | Cache errors |
| `neuralblitz_cache_operation_duration_seconds` | Histogram | Operation latency |
| `neuralblitz_cache_size_bytes` | Gauge | Cache size in bytes |
| `neuralblitz_cache_keys_total` | Gauge | Total keys in cache |
| `neuralblitz_cache_warmup_duration_seconds` | Histogram | Warmup duration |

## Alert Rules

The following Prometheus alerts are configured:

```yaml
- alert: CacheHitRateLow
  expr: rate(neuralblitz_cache_hits_total[5m]) / 
        (rate(neuralblitz_cache_hits_total[5m]) + 
         rate(neuralblitz_cache_misses_total[5m])) < 0.5
  for: 5m
  
- alert: CacheConnectionFailed
  expr: neuralblitz_cache_errors_total > 10
  for: 1m
```

## Testing

Run the cache test suite:

```bash
cd /home/runner/workspace/nb-omnibus-router
pytest tests/test_cache.py -v
```

## Docker Compose

The Redis service is automatically included in docker-compose.yml:

```yaml
redis:
  image: redis:7-alpine
  container_name: neuralblitz-redis
  ports:
    - "6379:6379"
  volumes:
    - redis-data:/data
    - ./config/redis.conf:/usr/local/etc/redis/redis.conf:ro
```

## Performance Tuning

### Redis Configuration

The included `redis.conf` is optimized for:
- 512MB memory limit with LRU eviction
- Active rehashing for large datasets
- Lazy freeing for better performance
- TCP keepalive for connection stability

### TTL Recommendations

| Data Type | Recommended TTL | Rationale |
|-----------|----------------|-----------|
| Quantum states | 5 minutes | Rapidly changing |
| Multi-reality evolution | 10 minutes | Moderate change rate |
| Consciousness metrics | 3 minutes | Frequently accessed |
| API capabilities | 1 hour | Rarely changes |
| Partner config | 1 hour | Semi-static |

## Troubleshooting

### Redis Connection Failed

Check the logs for:
```bash
docker logs neuralblitz-redis
docker logs neuralblitz-router
```

Verify Redis is running:
```bash
docker-compose ps redis
redis-cli ping
```

### High Cache Miss Rate

1. Check TTL configuration - may be too short
2. Verify cache warming is enabled
3. Monitor `neuralblitz_cache_hit_rate` metric
4. Review cache key generation for uniqueness

### Memory Issues

Monitor Redis memory usage:
```bash
redis-cli info memory
```

The configuration includes:
- `maxmemory 512mb`
- `maxmemory-policy allkeys-lru`

## Migration Guide

### Adding Caching to Existing Endpoints

1. Import cache decorators:
```python
from utils.cache import cache_api_response
```

2. Add decorator to endpoint:
```python
@router.get("/endpoint")
@cache_api_response(ttl=900)
async def my_endpoint():
    return await expensive_operation()
```

3. Configure TTL in `settings.yaml`:
```yaml
REDIS:
  api_response_ttl: 900
```

## Security Considerations

- Redis runs without password in internal Docker network
- Access restricted to neuralblitz-network
- Cache keys prefixed with `nb:` to avoid collisions
- Sensitive data should use shorter TTL
- Consider encrypting cached sensitive data

## Roadmap

- [ ] Distributed cache invalidation
- [ ] Cache analytics dashboard
- [ ] Machine learning-based TTL optimization
- [ ] Multi-tier caching (Redis + local + CDN)
- [ ] Cache coherency across multiple API instances
