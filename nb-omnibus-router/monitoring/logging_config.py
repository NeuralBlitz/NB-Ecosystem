"""
NeuralBlitz Structured Logging
Logging with correlation IDs and structured output
"""

import logging
import json
import uuid
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
request_id: ContextVar[str] = ContextVar("request_id", default="")
user_id: ContextVar[str] = ContextVar("user_id", default="")


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get() or "N/A"
        record.request_id = request_id.get() or "N/A"
        record.user_id = user_id.get() or "N/A"
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ):
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["severity"] = record.levelname
        log_record["logger"] = record.name

        # Add correlation IDs
        log_record["correlation_id"] = getattr(record, "correlation_id", "N/A")
        log_record["request_id"] = getattr(record, "request_id", "N/A")
        log_record["user_id"] = getattr(record, "user_id", "N/A")

        # Add source location
        log_record["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add NeuralBlitz-specific fields if available
        if hasattr(record, "quantum_coherence"):
            log_record["quantum_coherence"] = record.quantum_coherence
        if hasattr(record, "reality_network"):
            log_record["reality_network"] = record.reality_network
        if hasattr(record, "consciousness_level"):
            log_record["consciousness_level"] = record.consciousness_level

        # Rename fields for consistency
        if "levelname" in log_record:
            del log_record["levelname"]
        if "name" in log_record:
            del log_record["name"]


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    service_name: str = "neuralblitz",
) -> logging.Logger:
    """Setup structured logging configuration."""

    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Add correlation ID filter
    handler.addFilter(CorrelationIdFilter())

    # Set formatter
    if format_type == "json":
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(severity)s %(name)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def set_request_id(rid: Optional[str] = None) -> str:
    """Set request ID for current context."""
    if rid is None:
        rid = str(uuid.uuid4())
    request_id.set(rid)
    return rid


def set_user_id(uid: str) -> str:
    """Set user ID for current context."""
    user_id.set(uid)
    return uid


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id.get() or "N/A"


def get_request_id() -> str:
    """Get current request ID."""
    return request_id.get() or "N/A"


def get_user_id() -> str:
    """Get current user ID."""
    return user_id.get() or "N/A"


def clear_context():
    """Clear all context variables."""
    correlation_id.set("")
    request_id.set("")
    user_id.set("")


class ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context information."""

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        kwargs.setdefault("extra", {})
        kwargs["extra"].update(
            {
                "correlation_id": get_correlation_id(),
                "request_id": get_request_id(),
                "user_id": get_user_id(),
            }
        )
        return msg, kwargs


def get_logger(name: str) -> logging.Logger:
    """Get a logger with context support."""
    logger = logging.getLogger(name)
    return ContextAdapter(logger, {})


# Convenience functions for logging with context
def log_quantum_event(
    logger: logging.Logger,
    event: str,
    coherence_level: float,
    reality_type: str,
    extra: Optional[Dict[str, Any]] = None,
):
    """Log quantum-specific events."""
    extra = extra or {}
    extra.update(
        {
            "quantum_coherence": coherence_level,
            "reality_type": reality_type,
            "event_type": "quantum",
        }
    )
    logger.info(f"Quantum event: {event}", extra=extra)


def log_reality_event(
    logger: logging.Logger,
    event: str,
    stability_index: float,
    network_id: str,
    extra: Optional[Dict[str, Any]] = None,
):
    """Log reality network events."""
    extra = extra or {}
    extra.update(
        {
            "reality_network": {
                "stability": stability_index,
                "network_id": network_id,
            },
            "event_type": "reality",
        }
    )
    logger.info(f"Reality event: {event}", extra=extra)


def log_consciousness_event(
    logger: logging.Logger,
    event: str,
    level: int,
    entity_id: str,
    extra: Optional[Dict[str, Any]] = None,
):
    """Log consciousness events."""
    extra = extra or {}
    extra.update(
        {
            "consciousness_level": level,
            "entity_id": entity_id,
            "event_type": "consciousness",
        }
    )
    logger.info(f"Consciousness event: {event}", extra=extra)
