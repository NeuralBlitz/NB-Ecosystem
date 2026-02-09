"""
Error Handling Utilities Module
NeuralBlitz Core - Custom Exceptions and Error Helpers

Provides:
- Base exception classes
- Permission errors
- Authentication errors
- Error formatting utilities
"""

from typing import Optional, Dict, Any, List


class NeuralBlitzError(Exception):
    """Base exception for all NeuralBlitz errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code or "neuralblitz_error"
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class AuthenticationError(NeuralBlitzError):
    """Raised when authentication fails."""

    def __init__(self, message: str, code: str = "authentication_error"):
        super().__init__(message, code=code)


class AuthorizationError(NeuralBlitzError):
    """Raised when authorization fails (insufficient permissions)."""

    def __init__(
        self,
        message: str,
        code: str = "authorization_error",
        required_permission: Optional[str] = None,
    ):
        super().__init__(
            message, code=code, details={"required_permission": required_permission}
        )


class RateLimitError(NeuralBlitzError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, limit: int, retry_after: Optional[int] = None):
        super().__init__(
            message,
            code="rate_limit_exceeded",
            details={"limit": limit, "retry_after": retry_after},
        )
        self.limit = limit
        self.retry_after = retry_after


class QuotaExceededError(NeuralBlitzError):
    """Raised when quota is exceeded."""

    def __init__(self, message: str, quota_remaining: int = 0):
        super().__init__(
            message, code="quota_exceeded", details={"quota_remaining": quota_remaining}
        )
        self.quota_remaining = quota_remaining


class ValidationError(NeuralBlitzError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message,
            code="validation_error",
            details={
                "field": field,
                "value": str(value)[:100] if value is not None else None,
            },
        )
        self.field = field
        self.value = value


class NotFoundError(NeuralBlitzError):
    """Raised when a resource is not found."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        super().__init__(
            message,
            code="not_found",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConflictError(NeuralBlitzError):
    """Raised when there's a resource conflict."""

    def __init__(self, message: str, conflicting_field: Optional[str] = None):
        super().__init__(
            message, code="conflict", details={"conflicting_field": conflicting_field}
        )


class PermissionDeniedError(NeuralBlitzError):
    """Raised when permission is denied."""

    def __init__(
        self,
        message: str,
        required: Optional[List[str]] = None,
        missing: Optional[List[str]] = None,
    ):
        super().__init__(
            message,
            code="permission_denied",
            details={"required": required, "missing": missing},
        )
        self.required = required or []
        self.missing = missing or []


class SelfRewriteBlockedError(PermissionDeniedError):
    """Raised when self-rewrite operation is blocked."""

    def __init__(
        self,
        message: str = "Self-rewrite is blocked. Requires Judex Quorum.",
        missing_proofs: Optional[List[str]] = None,
    ):
        super().__init__(
            message,
            required=["judex_quorum"],
            missing=missing_proofs or ["judex_quorum_approval"],
        )


class TeletopologicalBlockedError(PermissionDeniedError):
    """Raised when teletopological operation is blocked."""

    def __init__(self, message: str = "Teletopological transfers are blocked"):
        super().__init__(message, required=["teletopological_enabled"])


class ServiceUnavailableError(NeuralBlitzError):
    """Raised when a service is temporarily unavailable."""

    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(
            message, code="service_unavailable", details={"retry_after": retry_after}
        )


def format_error_response(
    error: Exception, include_traceback: bool = False
) -> Dict[str, Any]:
    """Format an exception into a standardized error response.

    Args:
        error: Exception to format
        include_traceback: Whether to include traceback (debug only)

    Returns:
        Dictionary with error details
    """
    import traceback

    if isinstance(error, NeuralBlitzError):
        response = error.to_dict()
    else:
        response = {
            "error": True,
            "code": getattr(error, "code", "internal_error"),
            "message": str(error),
            "details": {},
        }

    if include_traceback:
        response["traceback"] = traceback.format_exc()

    return response


def raise_if_none(value: Any, message: str = "Value cannot be None"):
    """Raise NotFoundError if value is None.

    Args:
        value: Value to check
        message: Error message

    Returns:
        The value if not None

    Raises:
        NotFoundError: If value is None
    """
    if value is None:
        raise NotFoundError(message)
    return value


def raise_unless(condition: bool, message: str, error_class: type = ValidationError):
    """Raise error unless condition is True.

    Args:
        condition: Condition to check
        message: Error message
        error_class: Exception class to raise
    """
    if not condition:
        raise error_class(message)


# Compatibility aliases for legacy code
AuthError = AuthenticationError
PermissionError = PermissionDeniedError
NotFound = NotFoundError
ServiceError = ServiceUnavailableError
