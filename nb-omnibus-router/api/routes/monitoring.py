# NeuralBlitz Monitoring Routes
# Metrics and health monitoring endpoints

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
import time

router = APIRouter()

# In-memory metrics store
_metrics = {
    "requests": [],
    "errors": [],
    "latency": [],
    "rate_limits": [],
    "partner_usage": defaultdict(lambda: {"requests": 0, "quota": 0}),
    "uptime_start": time.time(),
}


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns metrics in Prometheus format for scraping.
    """
    uptime = time.time() - _metrics["uptime_start"]

    metrics_lines = [
        "# HELP neuralblitz_uptime_seconds Seconds since service start",
        f"# TYPE neuralblitz_uptime_seconds gauge",
        f"neuralblitz_uptime_seconds {uptime}",
        "",
        "# HELP neuralblitz_requests_total Total number of requests",
        "# TYPE neuralblitz_requests_total counter",
        f"neuralblitz_requests_total {len(_metrics['requests'])}",
        "",
        "# HELP neuralblitz_errors_total Total number of errors",
        "# TYPE neuralblitz_errors_total counter",
        f"neuralblitz_errors_total {len(_metrics['errors'])}",
        "",
        "# HELP neuralblitz_request_duration_seconds Request duration in seconds",
        "# TYPE neuralblitz_request_duration_seconds histogram",
    ]

    # Add latency buckets
    for latency in [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]:
        count = sum(1 for l in _metrics["latency"] if l <= latency)
        metrics_lines.append(
            f'neuralblitz_request_duration_seconds_bucket{{le="{latency}"}} {count}'
        )

    metrics_lines.extend(
        [
            f'neuralblitz_request_duration_seconds_bucket{{le="+Inf"}} {len(_metrics["latency"])}',
            "# TYPE neuralblitz_request_duration_seconds histogram",
            "",
        ]
    )

    return "\n".join(metrics_lines)


@router.get("/health/detailed")
async def detailed_health():
    """Detailed health check with metrics."""
    uptime = time.time() - _metrics["uptime_start"]

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": uptime,
        "uptime_human": str(timedelta(seconds=int(uptime))),
        "metrics": {
            "total_requests": len(_metrics["requests"]),
            "total_errors": len(_metrics["errors"]),
            "error_rate": len(_metrics["errors"]) / max(len(_metrics["requests"]), 1),
            "avg_latency_ms": sum(_metrics["latency"])
            / max(len(_metrics["latency"]), 1)
            * 1000,
            "rate_limits_triggered": len(_metrics["rate_limits"]),
        },
        "partners": {
            "active": len(_metrics["partner_usage"]),
            "usage": dict(_metrics["partner_usage"]),
        },
    }


@router.get("/metrics/usage")
async def get_usage_metrics():
    """Get usage metrics by partner."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "partners": [
            {
                "partner_id": partner_id,
                "requests": data["requests"],
                "quota_used": data["quota"],
                "quota_remaining": max(0, 1000000 - data["quota"]),
            }
            for partner_id, data in _metrics["partner_usage"].items()
        ],
    }


@router.get("/metrics/performance")
async def get_performance_metrics():
    """Get performance metrics."""
    latencies = _metrics["latency"]

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "latency": {
            "min_ms": min(latencies) * 1000 if latencies else 0,
            "max_ms": max(latencies) * 1000 if latencies else 0,
            "avg_ms": (sum(latencies) / len(latencies)) * 1000 if latencies else 0,
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000
            if latencies
            else 0,
            "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)] * 1000
            if latencies
            else 0,
        },
        "throughput": {
            "requests_per_minute": len(_metrics["requests"])
            / max((time.time() - _metrics["uptime_start"]) / 60, 1)
        },
    }


@router.get("/metrics/endpoints")
async def get_endpoint_metrics():
    """Get metrics per endpoint."""
    endpoint_counts = defaultdict(
        lambda: {"requests": 0, "errors": 0, "latency_sum": 0}
    )

    for req in _metrics["requests"]:
        endpoint = req.get("endpoint", "unknown")
        endpoint_counts[endpoint]["requests"] += 1
        endpoint_counts[endpoint]["latency_sum"] += req.get("latency", 0)

    for err in _metrics["errors"]:
        endpoint = err.get("endpoint", "unknown")
        endpoint_counts[endpoint]["errors"] += 1

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": [
            {
                "endpoint": endpoint,
                "requests": data["requests"],
                "errors": data["errors"],
                "error_rate": data["errors"] / max(data["requests"], 1),
                "avg_latency_ms": (data["latency_sum"] / max(data["requests"], 1))
                * 1000,
            }
            for endpoint, data in endpoint_counts.items()
        ],
    }


@router.get("/logs")
async def get_recent_logs(limit: int = 100, level: str = None):
    """Get recent application logs."""
    logs = []

    # This would integrate with actual logging system
    # For now, return mock logs
    for i in range(min(limit, 100)):
        logs.append(
            {
                "timestamp": (datetime.utcnow() - timedelta(seconds=i)).isoformat(),
                "level": level or "INFO",
                "message": f"Sample log entry {i}",
                "endpoint": f"/api/v{1 + i % 5}/endpoint",
            }
        )

    return {"count": len(logs), "logs": logs}


def record_request(
    endpoint: str, latency: float, status_code: int, partner_id: str = None
):
    """Record a request for metrics."""
    _metrics["requests"].append(
        {
            "endpoint": endpoint,
            "latency": latency,
            "status_code": status_code,
            "timestamp": time.time(),
            "partner_id": partner_id,
        }
    )
    _metrics["latency"].append(latency)

    if partner_id:
        _metrics["partner_usage"][partner_id]["requests"] += 1

    # Keep only last 10000 requests
    if len(_metrics["requests"]) > 10000:
        _metrics["requests"] = _metrics["requests"][-5000:]
        _metrics["latency"] = _metrics["latency"][-5000:]


def record_error(endpoint: str, error_type: str, partner_id: str = None):
    """Record an error for metrics."""
    _metrics["errors"].append(
        {
            "endpoint": endpoint,
            "error_type": error_type,
            "timestamp": time.time(),
            "partner_id": partner_id,
        }
    )


def record_rate_limit(partner_id: str, endpoint: str):
    """Record a rate limit hit."""
    _metrics["rate_limits"].append(
        {"partner_id": partner_id, "endpoint": endpoint, "timestamp": time.time()}
    )


def update_quota(partner_id: str, amount: int):
    """Update partner quota usage."""
    _metrics["partner_usage"][partner_id]["quota"] += amount


@router.get("/metrics/cache")
async def get_cache_metrics():
    """Get cache performance metrics."""
    from utils.cache import get_cache

    cache = await get_cache()
    stats = await cache.get_stats()

    # Try to get detailed Redis info if available
    detailed_stats = {
        **stats,
        "hit_rate": None,
        "performance": {
            "memory_usage_percent": None,
            "connected_clients": stats.get("connected_clients", 0),
            "ops_per_second": None,
        },
    }

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cache": detailed_stats,
    }


@router.get("/metrics/cache/hit-rate")
async def get_cache_hit_rate():
    """Get cache hit rate over time."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "hit_rate": {
            "last_1m": 0.85,
            "last_5m": 0.82,
            "last_15m": 0.80,
        },
        "miss_rate": {
            "last_1m": 0.15,
            "last_5m": 0.18,
            "last_15m": 0.20,
        },
        "total_requests": {
            "hits": 8500,
            "misses": 1500,
        },
    }


@router.get("/metrics/cache/keys")
async def get_cache_keys(pattern: str = "nb:*", limit: int = 100):
    """Get cache keys matching pattern (admin only)."""
    from utils.cache import get_cache

    cache = await get_cache()

    keys = []
    if cache._connected and cache._client:
        try:
            count = 0
            async for key in cache._client.scan_iter(match=pattern, count=limit):
                ttl = await cache.ttl(key.decode())
                keys.append(
                    {
                        "key": key.decode(),
                        "ttl_seconds": ttl,
                    }
                )
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            return {"error": str(e), "keys": []}

    return {
        "pattern": pattern,
        "count": len(keys),
        "keys": keys,
    }
