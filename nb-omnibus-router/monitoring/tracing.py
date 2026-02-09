"""
NeuralBlitz OpenTelemetry Distributed Tracing
Distributed tracing configuration using OpenTelemetry
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.trace import Status, StatusCode
from opentelemetry.propagate import extract, inject, set_global_textmap
from opentelemetry.propagators.b3 import B3Format
from opentelemetry.propagators.jaeger import JaegerPropagator
from functools import wraps
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for distributed tracing."""

    def __init__(
        self,
        service_name: str = "neuralblitz",
        service_version: str = "1.0.0",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        sampling_rate: float = 1.0,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.jaeger_endpoint = jaeger_endpoint or "http://jaeger:14268/api/traces"
        self.otlp_endpoint = otlp_endpoint or "http://otel-collector:4317"
        self.sampling_rate = sampling_rate


class TracingManager:
    """Manager for distributed tracing."""

    def __init__(self, config: TracingConfig):
        self.config = config
        self.tracer = None
        self._setup_tracing()

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        # Create resource
        resource = Resource.create(
            {
                SERVICE_NAME: self.config.service_name,
                SERVICE_VERSION: self.config.service_version,
                "deployment.environment": "production",
            }
        )

        # Create provider
        provider = TracerProvider(resource=resource)

        # Add Jaeger exporter
        try:
            jaeger_exporter = JaegerExporter(
                agent_host_name="jaeger",
                agent_port=6831,
            )
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            logger.info("Jaeger exporter configured")
        except Exception as e:
            logger.warning(f"Failed to configure Jaeger exporter: {e}")

        # Add OTLP exporter
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info("OTLP exporter configured")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")

        # Set global provider
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(self.config.service_name)

        # Set propagators
        set_global_textmap(B3Format())

        logger.info(f"Tracing initialized for {self.config.service_name}")

    def get_tracer(self):
        """Get the tracer instance."""
        return self.tracer


# Global tracing manager
tracing_manager: Optional[TracingManager] = None


def init_tracing(config: Optional[TracingConfig] = None):
    """Initialize global tracing."""
    global tracing_manager
    if config is None:
        config = TracingConfig()
    tracing_manager = TracingManager(config)
    return tracing_manager


def get_tracer():
    """Get the global tracer."""
    if tracing_manager is None:
        init_tracing()
    return tracing_manager.get_tracer()


def traced(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace function execution."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = operation_name or func.__name__

            with tracer.start_as_current_span(name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function info
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def async_traced(
    operation_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator to trace async function execution."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = operation_name or func.__name__

            with tracer.start_as_current_span(name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function info
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                span.set_attribute("function.async", True)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


class SpanContext:
    """Context manager for manual span creation."""

    def __init__(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: Optional[trace.SpanKind] = None,
    ):
        self.name = name
        self.attributes = attributes or {}
        self.kind = kind or trace.SpanKind.INTERNAL
        self.span = None
        self.tracer = get_tracer()

    def __enter__(self):
        self.span = self.tracer.start_span(self.name, kind=self.kind)
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self.span.record_exception(exc_val)
        else:
            self.span.set_status(Status(StatusCode.OK))
        self.span.end()


def add_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Add event to current span."""
    tracer = get_tracer()
    current_span = trace.get_current_span()
    if current_span:
        current_span.add_event(name, attributes or {})


def set_attribute(key: str, value: Any):
    """Set attribute on current span."""
    current_span = trace.get_current_span()
    if current_span:
        current_span.set_attribute(key, value)


def set_error(message: str):
    """Set error status on current span."""
    current_span = trace.get_current_span()
    if current_span:
        current_span.set_status(Status(StatusCode.ERROR, message))


# NeuralBlitz-specific tracing functions
def trace_quantum_operation(operation: str, qubits: int, circuit_depth: int):
    """Trace quantum operations."""
    set_attribute("quantum.operation", operation)
    set_attribute("quantum.qubits", qubits)
    set_attribute("quantum.circuit_depth", circuit_depth)
    add_event(
        "quantum_operation_started",
        {
            "operation": operation,
            "qubits": qubits,
        },
    )


def trace_reality_bridge(source: str, target: str, data_size: int):
    """Trace reality bridge operations."""
    set_attribute("reality.source", source)
    set_attribute("reality.target", target)
    set_attribute("reality.data_size", data_size)
    add_event(
        "reality_bridge_operation",
        {
            "source": source,
            "target": target,
        },
    )


def trace_consciousness_evolution(from_level: int, to_level: int, entity_id: str):
    """Trace consciousness evolution."""
    set_attribute("consciousness.from_level", from_level)
    set_attribute("consciousness.to_level", to_level)
    set_attribute("consciousness.entity_id", entity_id)
    add_event(
        "consciousness_evolution",
        {
            "from_level": from_level,
            "to_level": to_level,
        },
    )
