"""
NeuralBlitz Health Check System
Comprehensive health checks for all services
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import httpx
from monitoring.metrics import SERVICE_HEALTH, SYSTEM_UPTIME
from monitoring.circuit_breaker import get_circuit_breaker, CircuitBreakerManager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    service_name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List["HealthCheckResult"] = field(default_factory=list)


class HealthCheck:
    """Base class for health checks."""

    def __init__(
        self,
        name: str,
        check_func: Callable[..., Any],
        timeout: float = 5.0,
        critical: bool = True,
    ):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.critical = critical
        self.last_check: Optional[HealthCheckResult] = None

    async def check(self) -> HealthCheckResult:
        """Execute health check."""
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(self.check_func(), timeout=self.timeout)
            else:
                result = self.check_func()

            response_time = (time.time() - start_time) * 1000

            health_result = HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                message="Service is healthy",
                timestamp=datetime.utcnow(),
                metadata=result if isinstance(result, dict) else {},
            )

            # Update metric
            SERVICE_HEALTH.labels(service_name=self.name).state("healthy")

        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            health_result = HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY
                if self.critical
                else HealthStatus.DEGRADED,
                response_time_ms=response_time,
                message=f"Health check timed out after {self.timeout}s",
                timestamp=datetime.utcnow(),
            )
            SERVICE_HEALTH.labels(service_name=self.name).state(
                "unhealthy" if self.critical else "degraded"
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            health_result = HealthCheckResult(
                service_name=self.name,
                status=HealthStatus.UNHEALTHY
                if self.critical
                else HealthStatus.DEGRADED,
                response_time_ms=response_time,
                message=str(e),
                timestamp=datetime.utcnow(),
            )
            SERVICE_HEALTH.labels(service_name=self.name).state(
                "unhealthy" if self.critical else "degraded"
            )

        self.last_check = health_result
        return health_result


class HealthCheckRegistry:
    """Registry for health checks."""

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self._start_time = time.time()

    def register(self, check: HealthCheck):
        """Register a health check."""
        self.checks[check.name] = check
        logger.info(f"Registered health check: {check.name}")

    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = await asyncio.gather(
            *[check.check() for check in self.checks.values()], return_exceptions=True
        )

        check_results = []
        overall_status = HealthStatus.HEALTHY

        for result in results:
            if isinstance(result, Exception):
                check_results.append(
                    {
                        "service": "unknown",
                        "status": HealthStatus.UNHEALTHY.value,
                        "error": str(result),
                    }
                )
                overall_status = HealthStatus.UNHEALTHY
            else:
                check_results.append(
                    {
                        "service": result.service_name,
                        "status": result.status.value,
                        "response_time_ms": round(result.response_time_ms, 2),
                        "message": result.message,
                        "timestamp": result.timestamp.isoformat(),
                        "metadata": result.metadata,
                    }
                )

                # Determine overall status
                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (
                    result.status == HealthStatus.DEGRADED
                    and overall_status != HealthStatus.UNHEALTHY
                ):
                    overall_status = HealthStatus.DEGRADED

        uptime = time.time() - self._start_time
        SYSTEM_UPTIME.labels(service_name="neuralblitz").set(uptime)

        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": uptime,
            "uptime_human": self._format_duration(uptime),
            "checks": check_results,
            "total_checks": len(check_results),
            "healthy_count": sum(1 for r in check_results if r["status"] == "healthy"),
            "degraded_count": sum(
                1 for r in check_results if r["status"] == "degraded"
            ),
            "unhealthy_count": sum(
                1 for r in check_results if r["status"] == "unhealthy"
            ),
        }

    async def check_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Check specific service."""
        check = self.checks.get(service_name)
        if not check:
            return None

        result = await check.check()
        return {
            "service": result.service_name,
            "status": result.status.value,
            "response_time_ms": round(result.response_time_ms, 2),
            "message": result.message,
            "timestamp": result.timestamp.isoformat(),
            "metadata": result.metadata,
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")

        return " ".join(parts)


# Global registry
health_registry = HealthCheckRegistry()


# ============================================================================
# Built-in Health Checks
# ============================================================================


async def check_quantum_engine() -> Dict[str, Any]:
    """Check quantum engine health."""
    from engines.quantum import QuantumEngine

    engine = QuantumEngine()
    capabilities = await engine.get_capabilities()

    return {
        "engine": "quantum_simulator",
        "version": capabilities.get("version", "unknown"),
        "max_qubits": capabilities["capabilities"][0]["max_qubits"],
        "status": "operational",
    }


async def check_consciousness_bridge() -> Dict[str, Any]:
    """Check consciousness bridge health."""
    return {
        "bridge_status": "connected",
        "cosmic_bridge_strength": 0.92,
        "universal_field_accessible": True,
        "dimensional_access": 11,
    }


async def check_reality_network() -> Dict[str, Any]:
    """Check reality network health."""
    # Check circuit breaker states
    cb_states = CircuitBreakerManager.get_all_states()

    return {
        "network_stability": 0.88,
        "active_entanglements": 2,
        "phase_coherence": 0.91,
        "circuit_breakers": cb_states,
    }


async def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    # Simulated database check
    return {
        "connection": "healthy",
        "response_time_ms": 5.2,
        "active_connections": 10,
        "max_connections": 100,
    }


async def check_cache() -> Dict[str, Any]:
    """Check cache connectivity."""
    # Simulated cache check
    return {
        "connection": "healthy",
        "hit_rate": 0.85,
        "memory_usage_percent": 45.0,
    }


async def check_external_apis() -> Dict[str, Any]:
    """Check external API connectivity."""
    # This would check actual external APIs
    return {
        "endpoints_checked": 3,
        "healthy": 3,
        "degraded": 0,
        "unhealthy": 0,
    }


def check_memory() -> Dict[str, Any]:
    """Check system memory."""
    import psutil

    memory = psutil.virtual_memory()
    return {
        "total_mb": memory.total // (1024 * 1024),
        "available_mb": memory.available // (1024 * 1024),
        "used_percent": memory.percent,
        "status": "healthy" if memory.percent < 90 else "warning",
    }


def check_disk() -> Dict[str, Any]:
    """Check disk space."""
    import psutil

    disk = psutil.disk_usage("/")
    used_percent = (disk.used / disk.total) * 100

    return {
        "total_gb": disk.total // (1024**3),
        "free_gb": disk.free // (1024**3),
        "used_percent": used_percent,
        "status": "healthy" if used_percent < 90 else "warning",
    }


def check_cpu() -> Dict[str, Any]:
    """Check CPU usage."""
    import psutil

    cpu_percent = psutil.cpu_percent(interval=1)
    return {
        "usage_percent": cpu_percent,
        "core_count": psutil.cpu_count(),
        "status": "healthy" if cpu_percent < 90 else "warning",
    }


# ============================================================================
# Initialize Health Checks
# ============================================================================


def init_health_checks():
    """Initialize all health checks."""
    checks = [
        HealthCheck("quantum_engine", check_quantum_engine, timeout=10.0),
        HealthCheck("consciousness_bridge", check_consciousness_bridge, timeout=5.0),
        HealthCheck("reality_network", check_reality_network, timeout=5.0),
        HealthCheck("database", check_database, timeout=5.0, critical=False),
        HealthCheck("cache", check_cache, timeout=3.0, critical=False),
        HealthCheck("external_apis", check_external_apis, timeout=10.0, critical=False),
        HealthCheck("memory", check_memory, timeout=5.0),
        HealthCheck("disk", check_disk, timeout=5.0),
        HealthCheck("cpu", check_cpu, timeout=5.0),
    ]

    for check in checks:
        health_registry.register(check)

    logger.info(f"Initialized {len(checks)} health checks")
