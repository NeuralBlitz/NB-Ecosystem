"""
NeuralBlitz Prometheus Metrics
Custom metrics for the NeuralBlitz ecosystem
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    Enum,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from functools import wraps
import time
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

# Create a custom registry
REGISTRY = CollectorRegistry()

# ============================================================================
# Quantum Metrics
# ============================================================================

QUANTUM_COHERENCE_LEVEL = Gauge(
    "neuralblitz_quantum_coherence_level",
    "Current quantum coherence level (0-1)",
    ["instance_id", "reality_type"],
    registry=REGISTRY,
)

QUANTUM_ENTANGLEMENT_FIDELITY = Gauge(
    "neuralblitz_quantum_entanglement_fidelity",
    "Entanglement fidelity across reality pairs",
    ["source_reality", "target_reality"],
    registry=REGISTRY,
)

QUANTUM_SIMULATION_DURATION = Histogram(
    "neuralblitz_quantum_simulation_duration_seconds",
    "Time spent on quantum simulations",
    ["qubits", "circuit_depth"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

QUANTUM_ERROR_RATE = Counter(
    "neuralblitz_quantum_errors_total",
    "Total quantum computation errors",
    ["error_type", "qubits"],
    registry=REGISTRY,
)

QUANTUM_STATE_EVOLUTION = Gauge(
    "neuralblitz_quantum_state_evolution_rate",
    "Rate of quantum state evolution per second",
    ["reality_layer"],
    registry=REGISTRY,
)

# ============================================================================
# Reality Network Metrics
# ============================================================================

REALITY_NETWORK_STABILITY = Gauge(
    "neuralblitz_reality_network_stability",
    "Reality network stability index (0-1)",
    ["network_id", "reality_type"],
    registry=REGISTRY,
)

REALITY_BRIDGE_LATENCY = Histogram(
    "neuralblitz_reality_bridge_latency_seconds",
    "Latency for cross-reality data transfer",
    ["source_reality", "target_reality"],
    buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=REGISTRY,
)

REALITY_SYNCHRONIZATION_RATE = Gauge(
    "neuralblitz_reality_synchronization_rate",
    "Rate of reality synchronization events per minute",
    ["sync_type"],
    registry=REGISTRY,
)

REALITY_ENTANGlement_PAIRS = Gauge(
    "neuralblitz_reality_entanglement_pairs_active",
    "Number of active cross-reality entanglement pairs",
    ["reality_combination"],
    registry=REGISTRY,
)

REALITY_PHASE_COHERENCE = Gauge(
    "neuralblitz_reality_phase_coherence",
    "Phase coherence between realities (0-1)",
    ["reality_a", "reality_b"],
    registry=REGISTRY,
)

# ============================================================================
# Consciousness Metrics
# ============================================================================

CONSCIOUSNESS_EVOLUTION_RATE = Gauge(
    "neuralblitz_consciousness_evolution_rate",
    "Rate of consciousness evolution (levels per hour)",
    ["entity_id", "dimension"],
    registry=REGISTRY,
)

CONSCIOUSNESS_LEVEL = Gauge(
    "neuralblitz_consciousness_level",
    "Current consciousness level (1-8)",
    ["entity_id"],
    registry=REGISTRY,
)

CONSCIOUSNESS_INTEGRATION = Gauge(
    "neuralblitz_consciousness_integration_percentage",
    "Consciousness integration percentage",
    ["entity_id"],
    registry=REGISTRY,
)

COSMIC_BRIDGE_STRENGTH = Gauge(
    "neuralblitz_cosmic_bridge_strength",
    "Strength of cosmic consciousness bridge (0-1)",
    ["bridge_id"],
    registry=REGISTRY,
)

DIMENSIONAL_ACCESS_COUNT = Gauge(
    "neuralblitz_dimensional_access_count",
    "Number of accessible dimensions",
    ["entity_id"],
    registry=REGISTRY,
)

UNIVERSAL_FIELD_COHERENCE = Gauge(
    "neuralblitz_universal_field_coherence",
    "Coherence with universal information field",
    ["entity_id"],
    registry=REGISTRY,
)

# ============================================================================
# Free Energy Metrics
# ============================================================================

FREE_ENERGY_CALCULATION = Gauge(
    "neuralblitz_free_energy_value",
    "Calculated free energy value",
    ["calculation_type", "system_id"],
    registry=REGISTRY,
)

FREE_ENERGY_EFFICIENCY = Gauge(
    "neuralblitz_free_energy_efficiency",
    "Free energy extraction efficiency (0-1)",
    ["method", "system_id"],
    registry=REGISTRY,
)

FREE_ENERGY_CALCULATION_DURATION = Histogram(
    "neuralblitz_free_energy_calculation_duration_seconds",
    "Time spent on free energy calculations",
    ["calculation_type"],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
    registry=REGISTRY,
)

ENERGY_EXTRACTION_RATE = Gauge(
    "neuralblitz_energy_extraction_rate",
    "Rate of energy extraction (units per second)",
    ["source", "system_id"],
    registry=REGISTRY,
)

# ============================================================================
# API Performance Metrics
# ============================================================================

API_REQUEST_DURATION = Histogram(
    "neuralblitz_api_request_duration_seconds",
    "API request latency by route",
    ["method", "route", "status_code"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY,
)

API_REQUESTS_TOTAL = Counter(
    "neuralblitz_api_requests_total",
    "Total API requests",
    ["method", "route", "status_code", "partner_id"],
    registry=REGISTRY,
)

API_ERRORS_TOTAL = Counter(
    "neuralblitz_api_errors_total",
    "Total API errors",
    ["method", "route", "error_type", "partner_id"],
    registry=REGISTRY,
)

API_RATE_LIMIT_HITS = Counter(
    "neuralblitz_api_rate_limit_hits_total",
    "Total rate limit hits",
    ["route", "partner_id"],
    registry=REGISTRY,
)

API_ACTIVE_CONNECTIONS = Gauge(
    "neuralblitz_api_active_connections",
    "Number of active API connections",
    ["route"],
    registry=REGISTRY,
)

API_PAYLOAD_SIZE = Histogram(
    "neuralblitz_api_payload_size_bytes",
    "Size of API request/response payloads",
    ["direction", "route"],
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
    registry=REGISTRY,
)

# ============================================================================
# System Health Metrics
# ============================================================================

SYSTEM_UPTIME = Gauge(
    "neuralblitz_system_uptime_seconds",
    "System uptime in seconds",
    ["service_name"],
    registry=REGISTRY,
)

SERVICE_HEALTH = Enum(
    "neuralblitz_service_health",
    "Service health status",
    ["service_name"],
    states=["healthy", "degraded", "unhealthy", "unknown"],
    registry=REGISTRY,
)

CIRCUIT_BREAKER_STATE = Enum(
    "neuralblitz_circuit_breaker_state",
    "Circuit breaker state",
    ["service_name", "endpoint"],
    states=["closed", "open", "half_open"],
    registry=REGISTRY,
)

CIRCUIT_BREAKER_FAILURES = Counter(
    "neuralblitz_circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["service_name", "endpoint"],
    registry=REGISTRY,
)

CIRCUIT_BREAKER_SUCCESSES = Counter(
    "neuralblitz_circuit_breaker_successes_total",
    "Total circuit breaker successes",
    ["service_name", "endpoint"],
    registry=REGISTRY,
)

# ============================================================================
# Distributed Tracing Metrics
# ============================================================================

TRACE_SPAN_DURATION = Histogram(
    "neuralblitz_trace_span_duration_seconds",
    "Duration of distributed trace spans",
    ["operation", "service"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

TRACE_ERRORS = Counter(
    "neuralblitz_trace_errors_total",
    "Total trace errors",
    ["operation", "error_type"],
    registry=REGISTRY,
)

# ============================================================================
# Build Info
# ============================================================================

BUILD_INFO = Info(
    "neuralblitz_build",
    "Build information",
    registry=REGISTRY,
)

# ============================================================================
# Decorators and Helpers
# ============================================================================


def timed(metric: Histogram, labels: Optional[dict] = None):
    """Decorator to time function execution."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

        return wrapper

    return decorator


def async_timed(metric: Histogram, labels: Optional[dict] = None):
    """Decorator to time async function execution."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)

        return wrapper

    return decorator


def track_errors(metric: Counter, labels: Optional[dict] = None):
    """Decorator to track function errors."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_labels = labels.copy() if labels else {}
                error_labels["error_type"] = type(e).__name__
                metric.labels(**error_labels).inc()
                raise

        return wrapper

    return decorator


def get_prometheus_metrics():
    """Generate Prometheus metrics output."""
    return generate_latest(REGISTRY)


def set_build_info(version: str, commit: str, branch: str, build_time: str):
    """Set build information."""
    BUILD_INFO.info(
        {
            "version": version,
            "commit": commit,
            "branch": branch,
            "build_time": build_time,
        }
    )
