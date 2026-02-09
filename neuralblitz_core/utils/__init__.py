"""
NeuralBlitz Core Utilities Package

Provides shared utility functions across the NeuralBlitz codebase:
- Math utilities: numpy operations, vector calculations
- Type utilities: conversion, serialization
- Validation: input validation and data integrity
- Logging: structured logging helpers
- Errors: custom exception classes
- ID generation: unique ID and hash generation
"""

# Math utilities
from .math_utils import (
    clip_value,
    clip,
    normalize_vector,
    normalize,
    safe_mean,
    mean,
    safe_sum,
    sum_values,
    compute_cross_hessian,
    compute_frobenius_norm,
    compute_trace,
    generate_random_beta,
    add_noise_to_value,
    compute_phase_difference,
    apply_phase_shift,
    NUMPY_AVAILABLE,
)

# Type utilities
from .type_utils import (
    to_dict_safe,
    to_dict,
    from_dict_safe,
    from_dict,
    serialize_json,
    serialize,
    deserialize_json,
    deserialize,
    safe_cast,
    ensure_list,
    ensure_dict,
    truncate_string,
    parse_datetime,
)

# Validation utilities
from .validation import (
    ValidationError as ValidationException,
    validate_not_none,
    validate_type,
    validate_range,
    validate_string,
    validate_list,
    validate_dict,
    validate_api_key_format,
    validate_id_format,
    clamp_value,
    is_not_none,
    is_type,
    in_range,
    is_valid_string,
    is_valid_list,
    is_valid_dict,
)

# Error utilities
from .errors import (
    NeuralBlitzError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    QuotaExceededError,
    ValidationError,
    NotFoundError,
    ConflictError,
    PermissionDeniedError,
    SelfRewriteBlockedError,
    TeletopologicalBlockedError,
    ServiceUnavailableError,
    format_error_response,
    raise_if_none,
    raise_unless,
    # Compatibility aliases
    AuthError,
    PermissionError,
    NotFound,
    ServiceError,
)

# Logging utilities
from .logging import (
    configure_logger,
    get_logger,
    StructuredLogFormatter,
    log_with_context,
    log_operation_start,
    log_operation_end,
    log_error,
    LogContext,
    logger,
)

# ID and hash utilities
from .id_utils import (
    generate_uuid,
    generate_id,
    generate_short_uuid,
    generate_timestamp_id,
    generate_hash_id,
    generate_secure_token,
    generate_api_key,
    compute_sha256,
    compute_sha512,
    hash_data,
    hash_sha512,
    compute_content_hash,
    generate_dag_node_hash,
    generate_proof_id,
    generate_trace_id,
    generate_sandbox_id,
    generate_correlate_ref,
    generate_braid_id,
    generate_transfer_id,
)

__all__ = [
    # Math
    "clip_value",
    "clip",
    "normalize_vector",
    "normalize",
    "safe_mean",
    "mean",
    "safe_sum",
    "sum_values",
    "compute_cross_hessian",
    "compute_frobenius_norm",
    "compute_trace",
    "generate_random_beta",
    "add_noise_to_value",
    "compute_phase_difference",
    "apply_phase_shift",
    "NUMPY_AVAILABLE",
    # Type conversion
    "to_dict_safe",
    "to_dict",
    "from_dict_safe",
    "from_dict",
    "serialize_json",
    "serialize",
    "deserialize_json",
    "deserialize",
    "safe_cast",
    "ensure_list",
    "ensure_dict",
    "truncate_string",
    "parse_datetime",
    # Validation
    "ValidationException",
    "validate_not_none",
    "validate_type",
    "validate_range",
    "validate_string",
    "validate_list",
    "validate_dict",
    "validate_api_key_format",
    "validate_id_format",
    "clamp_value",
    "is_not_none",
    "is_type",
    "in_range",
    "is_valid_string",
    "is_valid_list",
    "is_valid_dict",
    # Errors
    "NeuralBlitzError",
    "AuthenticationError",
    "AuthError",
    "AuthorizationError",
    "RateLimitError",
    "QuotaExceededError",
    "ValidationError",
    "NotFoundError",
    "NotFound",
    "ConflictError",
    "PermissionDeniedError",
    "PermissionError",
    "SelfRewriteBlockedError",
    "TeletopologicalBlockedError",
    "ServiceUnavailableError",
    "ServiceError",
    "format_error_response",
    "raise_if_none",
    "raise_unless",
    # Logging
    "configure_logger",
    "get_logger",
    "StructuredLogFormatter",
    "log_with_context",
    "log_operation_start",
    "log_operation_end",
    "log_error",
    "LogContext",
    "logger",
    # ID utilities
    "generate_uuid",
    "generate_id",
    "generate_short_uuid",
    "generate_timestamp_id",
    "generate_hash_id",
    "generate_secure_token",
    "generate_api_key",
    "compute_sha256",
    "compute_sha512",
    "hash_data",
    "hash_sha512",
    "compute_content_hash",
    "generate_dag_node_hash",
    "generate_proof_id",
    "generate_trace_id",
    "generate_sandbox_id",
    "generate_correlate_ref",
    "generate_braid_id",
    "generate_transfer_id",
]
