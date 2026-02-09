"""
Validation Utilities Module
NeuralBlitz Core - Validation and Data Integrity Helpers

Provides:
- Input validation functions
- Schema validation
- Range and type checking
- Dictionary validation
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type
import re


class ValidationError(ValueError):
    """Raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


def validate_not_none(value: Any, field_name: str = "value") -> Any:
    """Validate that value is not None.

    Args:
        value: Value to validate
        field_name: Name of field for error message

    Returns:
        The value if not None

    Raises:
        ValidationError: If value is None
    """
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field=field_name)
    return value


def validate_type(
    value: Any, expected_type: Union[Type, tuple], field_name: str = "value"
) -> Any:
    """Validate that value is of expected type.

    Args:
        value: Value to validate
        expected_type: Expected type or tuple of types
        field_name: Name of field for error message

    Returns:
        The value if type matches

    Raises:
        ValidationError: If type doesn't match
    """
    if not isinstance(value, expected_type):
        type_names = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else " or ".join(t.__name__ for t in expected_type)
        )
        raise ValidationError(
            f"{field_name} must be of type {type_names}, got {type(value).__name__}",
            field=field_name,
            value=value,
        )
    return value


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    field_name: str = "value",
) -> Union[int, float]:
    """Validate that value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        field_name: Name of field for error message

    Returns:
        The value if in range

    Raises:
        ValidationError: If value is out of range
    """
    if min_val is not None and value < min_val:
        raise ValidationError(
            f"{field_name} must be >= {min_val}, got {value}",
            field=field_name,
            value=value,
        )

    if max_val is not None and value > max_val:
        raise ValidationError(
            f"{field_name} must be <= {max_val}, got {value}",
            field=field_name,
            value=value,
        )

    return value


def validate_string(
    value: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    allow_empty: bool = True,
    field_name: str = "value",
) -> str:
    """Validate string value.

    Args:
        value: String to validate
        min_length: Minimum length
        max_length: Maximum length
        pattern: Regex pattern to match
        allow_empty: Whether empty strings are allowed
        field_name: Name of field for error message

    Returns:
        The string if valid

    Raises:
        ValidationError: If validation fails
    """
    validate_type(value, str, field_name)

    if not allow_empty and len(value) == 0:
        raise ValidationError(f"{field_name} cannot be empty", field=field_name)

    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"{field_name} must be at least {min_length} characters", field=field_name
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"{field_name} must be at most {max_length} characters", field=field_name
        )

    if pattern is not None and not re.match(pattern, value):
        raise ValidationError(
            f"{field_name} does not match required pattern",
            field=field_name,
            value=value,
        )

    return value


def validate_list(
    value: List[Any],
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    item_validator: Optional[Callable[[Any], Any]] = None,
    field_name: str = "value",
) -> List[Any]:
    """Validate list value.

    Args:
        value: List to validate
        min_length: Minimum length
        max_length: Maximum length
        item_validator: Optional validator for each item
        field_name: Name of field for error message

    Returns:
        The list if valid

    Raises:
        ValidationError: If validation fails
    """
    validate_type(value, list, field_name)

    if min_length is not None and len(value) < min_length:
        raise ValidationError(
            f"{field_name} must have at least {min_length} items", field=field_name
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"{field_name} must have at most {max_length} items", field=field_name
        )

    if item_validator:
        for i, item in enumerate(value):
            try:
                item_validator(item)
            except ValidationError as e:
                raise ValidationError(
                    f"{field_name}[{i}]: {e}", field=f"{field_name}[{i}]"
                )

    return value


def validate_dict(
    data: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
    allowed_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, Callable[[Any], Any]]] = None,
    path: str = "",
) -> Dict[str, Any]:
    """Validate dictionary structure.

    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        allowed_fields: List of allowed field names (None = all allowed)
        field_validators: Dict of field names to validator functions
        path: Current path for error messages

    Returns:
        The dictionary if valid

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, dict):
        raise ValidationError(
            f"{path}Expected dict, got {type(data).__name__}", field=path
        )

    full_path = f"{path}." if path else ""

    # Check required fields
    if required_fields:
        for field in required_fields:
            if field not in data:
                raise ValidationError(
                    f"{full_path}{field} is required", field=f"{full_path}{field}"
                )

    # Check allowed fields
    if allowed_fields is not None:
        for field in data:
            if field not in allowed_fields:
                raise ValidationError(
                    f"{full_path}{field} is not an allowed field",
                    field=f"{full_path}{field}",
                )

    # Run field validators
    if field_validators:
        for field, validator in field_validators.items():
            if field in data:
                try:
                    validator(data[field])
                except ValidationError as e:
                    raise ValidationError(
                        f"{full_path}{field}: {e}", field=f"{full_path}{field}"
                    )

    return data


def validate_api_key_format(api_key: str, prefix: str = "nb_") -> str:
    """Validate API key format.

    Args:
        api_key: API key to validate
        prefix: Expected prefix

    Returns:
        The API key if valid

    Raises:
        ValidationError: If format is invalid
    """
    validate_string(api_key, min_length=1, field_name="api_key")

    if not api_key.startswith(prefix):
        raise ValidationError(
            f"API key must start with '{prefix}'",
            field="api_key",
            value=api_key[:10] + "...",
        )

    if len(api_key) < 20:
        raise ValidationError("API key is too short", field="api_key")

    return api_key


def validate_id_format(
    value: str,
    prefix: Optional[str] = None,
    min_length: int = 8,
    max_length: int = 64,
    field_name: str = "id",
) -> str:
    """Validate ID format.

    Args:
        value: ID to validate
        prefix: Expected prefix (optional)
        min_length: Minimum length
        max_length: Maximum length
        field_name: Name of field for error message

    Returns:
        The ID if valid

    Raises:
        ValidationError: If format is invalid
    """
    validate_string(
        value, min_length=min_length, max_length=max_length, field_name=field_name
    )

    if prefix and not value.startswith(prefix):
        raise ValidationError(
            f"{field_name} must start with '{prefix}'", field=field_name
        )

    # Check for valid characters (alphanumeric, underscore, hyphen)
    if not re.match(r"^[a-zA-Z0-9_-]+$", value):
        raise ValidationError(
            f"{field_name} contains invalid characters", field=field_name
        )

    return value


def clamp_value(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
) -> Union[int, float]:
    """Clamp value to range (returns clamped value, doesn't raise).

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value


# Compatibility aliases
is_not_none = validate_not_none
is_type = validate_type
in_range = validate_range
is_valid_string = validate_string
is_valid_list = validate_list
is_valid_dict = validate_dict
