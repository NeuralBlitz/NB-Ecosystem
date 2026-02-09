"""
Logging Utilities Module
NeuralBlitz Core - Structured Logging and Log Management

Provides:
- Logger configuration
- Structured logging helpers
- Log formatting utilities
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json


def configure_logger(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        handler: Custom handler

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create handler if not provided
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

    # Set formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create logger with default configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def log_with_context(
    logger: logging.Logger, level: int, message: str, **context
) -> None:
    """Log message with structured context.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields
    """
    extra = {"extra_data": context}
    logger.log(level, message, extra=extra)


def log_operation_start(logger: logging.Logger, operation: str, **context) -> datetime:
    """Log start of operation and return timestamp.

    Args:
        logger: Logger instance
        operation: Operation name
        **context: Additional context

    Returns:
        Start timestamp
    """
    start_time = datetime.utcnow()
    log_with_context(
        logger,
        logging.INFO,
        f"Starting operation: {operation}",
        operation=operation,
        status="started",
        start_time=start_time.isoformat(),
        **context,
    )
    return start_time


def log_operation_end(
    logger: logging.Logger,
    operation: str,
    start_time: datetime,
    success: bool = True,
    **context,
) -> None:
    """Log end of operation with duration.

    Args:
        logger: Logger instance
        operation: Operation name
        start_time: Operation start timestamp
        success: Whether operation succeeded
        **context: Additional context
    """
    end_time = datetime.utcnow()
    duration_ms = (end_time - start_time).total_seconds() * 1000

    log_with_context(
        logger,
        logging.INFO if success else logging.ERROR,
        f"Operation {operation} {'succeeded' if success else 'failed'}",
        operation=operation,
        status="completed" if success else "failed",
        duration_ms=duration_ms,
        end_time=end_time.isoformat(),
        **context,
    )


def log_error(
    logger: logging.Logger, error: Exception, operation: Optional[str] = None, **context
) -> None:
    """Log error with context.

    Args:
        logger: Logger instance
        error: Exception to log
        operation: Optional operation name
        **context: Additional context
    """
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        **context,
    }

    if operation:
        log_data["operation"] = operation

    log_with_context(
        logger,
        logging.ERROR,
        f"Error in {operation or 'operation'}: {error}",
        **log_data,
    )


class LogContext:
    """Context manager for operation logging."""

    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time: Optional[datetime] = None

    def __enter__(self) -> "LogContext":
        self.start_time = log_operation_start(
            self.logger, self.operation, **self.context
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_val is not None:
            log_operation_end(
                self.logger,
                self.operation,
                self.start_time,
                success=False,
                error=str(exc_val),
                error_type=exc_type.__name__,
            )
        else:
            log_operation_end(
                self.logger, self.operation, self.start_time, success=True
            )
        return False  # Don't suppress exceptions


# Create default logger for the module
logger = get_logger(__name__)
