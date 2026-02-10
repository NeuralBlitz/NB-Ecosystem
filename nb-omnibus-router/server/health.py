"""
Health check and readiness probe management
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result"""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Complete health report"""

    status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime
    version: str
    uptime_seconds: float


class HealthManager:
    """
    Manages health checks and readiness probes
    """

    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.checks: Dict[str, Callable] = {}
        self.check_results: Dict[str, HealthCheck] = {}
        self._start_time = datetime.utcnow()
        self._cache = None

    def register_check(self, name: str, check_func: Callable, critical: bool = True):
        """
        Register a health check function

        Args:
            name: Check name
            check_func: Async or sync function that returns (status, message, metadata)
            critical: If True, failure of this check marks system as unhealthy
        """
        self.checks[name] = {"func": check_func, "critical": critical}
        logger.debug(f"Registered health check: {name} (critical: {critical})")

    def set_cache(self, cache):
        """Set cache instance for health checks"""
        self._cache = cache

    async def run_check(self, name: str) -> HealthCheck:
        """Run a single health check"""
        check_config = self.checks.get(name)
        if not check_config:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
                timestamp=datetime.utcnow(),
                response_time_ms=0.0,
            )

        start_time = datetime.utcnow()

        try:
            func = check_config["func"]

            # Execute check
            if asyncio.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()

            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Parse result
            if isinstance(result, tuple):
                if len(result) == 3:
                    status, message, metadata = result
                else:
                    status, message = result
                    metadata = {}
            else:
                status = result
                message = "OK" if status == HealthStatus.HEALTHY else "Check failed"
                metadata = {}

            return HealthCheck(
                name=name,
                status=status if isinstance(status, HealthStatus) else HealthStatus.HEALTHY,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                metadata=metadata,
            )

        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
            )

    async def run_all_checks(self) -> HealthReport:
        """Run all registered health checks"""
        checks = []

        # Run checks concurrently
        check_names = list(self.checks.keys())
        results = await asyncio.gather(
            *[self.run_check(name) for name in check_names], return_exceptions=True
        )

        overall_status = HealthStatus.HEALTHY

        for name, result in zip(check_names, results):
            if isinstance(result, Exception):
                check = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                    timestamp=datetime.utcnow(),
                    response_time_ms=0.0,
                )
            else:
                check = result

            checks.append(check)
            self.check_results[name] = check

            # Determine overall status
            check_config = self.checks[name]
            if check.status == HealthStatus.UNHEALTHY and check_config["critical"]:
                overall_status = HealthStatus.UNHEALTHY
            elif check.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return HealthReport(
            status=overall_status,
            checks=checks,
            timestamp=datetime.utcnow(),
            version=self.version,
            uptime_seconds=uptime,
        )

    async def check_readiness(self) -> bool:
        """Check if server is ready to accept traffic"""
        if not self.checks:
            return True

        report = await self.run_all_checks()
        return report.status != HealthStatus.UNHEALTHY

    def get_uptime_seconds(self) -> float:
        """Get server uptime in seconds"""
        return (datetime.utcnow() - self._start_time).total_seconds()

    def setup_default_checks(self):
        """Setup default health checks"""

        # Cache health check
        async def check_cache():
            if not self._cache:
                return HealthStatus.DEGRADED, "Cache not initialized", {}

            try:
                health = await self._cache.health_check()
                if health.get("healthy", False):
                    return HealthStatus.HEALTHY, "Cache operational", health
                else:
                    return HealthStatus.DEGRADED, "Cache degraded", health
            except Exception as e:
                return HealthStatus.DEGRADED, f"Cache check failed: {e}", {}

        self.register_check("cache", check_cache, critical=False)

        # Memory check
        def check_memory():
            import psutil

            try:
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    return (
                        HealthStatus.UNHEALTHY,
                        f"Memory usage critical: {memory.percent}%",
                        {"usage_percent": memory.percent},
                    )
                elif memory.percent > 75:
                    return (
                        HealthStatus.DEGRADED,
                        f"Memory usage high: {memory.percent}%",
                        {"usage_percent": memory.percent},
                    )
                else:
                    return (
                        HealthStatus.HEALTHY,
                        f"Memory usage normal: {memory.percent}%",
                        {"usage_percent": memory.percent},
                    )
            except Exception as e:
                return HealthStatus.UNKNOWN, f"Memory check failed: {e}", {}

        self.register_check("memory", check_memory, critical=True)

        # Disk check
        def check_disk():
            import psutil

            try:
                disk = psutil.disk_usage("/")
                if disk.percent > 90:
                    return (
                        HealthStatus.UNHEALTHY,
                        f"Disk usage critical: {disk.percent}%",
                        {"usage_percent": disk.percent},
                    )
                elif disk.percent > 80:
                    return (
                        HealthStatus.DEGRADED,
                        f"Disk usage high: {disk.percent}%",
                        {"usage_percent": disk.percent},
                    )
                else:
                    return (
                        HealthStatus.HEALTHY,
                        f"Disk usage normal: {disk.percent}%",
                        {"usage_percent": disk.percent},
                    )
            except Exception as e:
                return HealthStatus.UNKNOWN, f"Disk check failed: {e}", {}

        self.register_check("disk", check_disk, critical=True)


def create_default_health_manager(version: str = "1.0.0") -> HealthManager:
    """Create a health manager with default checks"""
    manager = HealthManager(version=version)
    manager.setup_default_checks()
    return manager
