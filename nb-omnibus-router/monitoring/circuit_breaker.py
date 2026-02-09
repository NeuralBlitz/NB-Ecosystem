"""
Circuit Breaker Pattern Implementation
For external service calls and resilience
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Optional, Any, TypeVar
from dataclasses import dataclass, field
from functools import wraps
import logging
from monitoring.metrics import (
    CIRCUIT_BREAKER_STATE,
    CIRCUIT_BREAKER_FAILURES,
    CIRCUIT_BREAKER_SUCCESSES,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    expected_exception: type = Exception
    half_open_max_calls: int = 3
    success_threshold: int = 2
    name: str = "default"


class CircuitBreaker:
    """
        Circuit Breaker pattern implementation.

        Protects against cascading failures by stopping requests
    to failing services.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

        # Set initial metric
        CIRCUIT_BREAKER_STATE.labels(service_name=config.name, endpoint="default").state(
            self.state.value
        )

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            await self._update_state()

            if self.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.config.name}' is OPEN")

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.config.name}' HALF_OPEN limit reached"
                    )
                self.half_open_calls += 1

        # Execute the function outside the lock
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self._on_success()
            return result
        except self.config.expected_exception as e:
            await self._on_failure()
            raise

    async def _update_state(self):
        """Update circuit breaker state based on current conditions."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit '{self.config.name}' transitioning to HALF_OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                self._update_metric()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.config.recovery_timeout

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            CIRCUIT_BREAKER_SUCCESSES.labels(
                service_name=self.config.name, endpoint="default"
            ).inc()

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"Circuit '{self.config.name}' transitioning to CLOSED")
                    self._reset()
            else:
                self.failure_count = max(0, self.failure_count - 1)

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            CIRCUIT_BREAKER_FAILURES.labels(service_name=self.config.name, endpoint="default").inc()

            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                logger.warning(
                    f"Circuit '{self.config.name}' failure in HALF_OPEN, transitioning to OPEN"
                )
                self.state = CircuitBreakerState.OPEN
                self._update_metric()
            elif self.failure_count >= self.config.failure_threshold:
                logger.warning(
                    f"Circuit '{self.config.name}' failure threshold reached ({self.failure_count}), "
                    f"transitioning to OPEN"
                )
                self.state = CircuitBreakerState.OPEN
                self._update_metric()

    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
        self._update_metric()

    def _update_metric(self):
        """Update Prometheus metric."""
        CIRCUIT_BREAKER_STATE.labels(service_name=self.config.name, endpoint="default").state(
            self.state.value
        )

    def get_state(self) -> dict:
        """Get current circuit breaker state."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_calls": self.half_open_calls,
            "last_failure_time": self.last_failure_time,
        }


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""

    pass


# Circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name."""
    return _circuit_breakers.get(name)


def create_circuit_breaker(config: CircuitBreakerConfig) -> CircuitBreaker:
    """Create and register a circuit breaker."""
    breaker = CircuitBreaker(config)
    _circuit_breakers[config.name] = breaker
    return breaker


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    expected_exception: type = Exception,
):
    """Decorator to apply circuit breaker to a function."""

    def decorator(func: Callable):
        # Get or create circuit breaker
        breaker = _circuit_breakers.get(name)
        if breaker is None:
            config = CircuitBreakerConfig(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception,
            )
            breaker = create_circuit_breaker(config)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(breaker.call(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class CircuitBreakerManager:
    """Manager for all circuit breakers."""

    @staticmethod
    def get_all_states() -> dict:
        """Get states of all circuit breakers."""
        return {name: breaker.get_state() for name, breaker in _circuit_breakers.items()}

    @staticmethod
    def reset_all():
        """Reset all circuit breakers."""
        for breaker in _circuit_breakers.values():
            breaker._reset()
        logger.info("All circuit breakers reset")

    @staticmethod
    def reset(name: str):
        """Reset specific circuit breaker."""
        breaker = _circuit_breakers.get(name)
        if breaker:
            breaker._reset()
            logger.info(f"Circuit breaker '{name}' reset")


# Predefined circuit breakers for common services
def init_default_circuit_breakers():
    """Initialize default circuit breakers for NeuralBlitz services."""
    services = [
        ("quantum_engine", 3, 60.0),
        ("consciousness_bridge", 5, 30.0),
        ("reality_network", 5, 30.0),
        ("external_api", 10, 60.0),
        ("database", 5, 30.0),
        ("cache", 10, 15.0),
    ]

    for name, threshold, timeout in services:
        config = CircuitBreakerConfig(
            name=name,
            failure_threshold=threshold,
            recovery_timeout=timeout,
        )
        create_circuit_breaker(config)
        logger.info(f"Created circuit breaker: {name}")
