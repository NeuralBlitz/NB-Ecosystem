"""
Type Conversion Utilities Module
NeuralBlitz Core - Type Conversion and Serialization Helpers

Provides:
- Dictionary serialization helpers
- Type conversion utilities
- Safe casting functions
"""

from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, date
import json
from dataclasses import is_dataclass, asdict


def to_dict_safe(
    obj: Any, max_depth: int = 3, truncate_strings: int = 100
) -> Dict[str, Any]:
    """Safely convert object to dictionary with depth limit.

    Args:
        obj: Object to convert
        max_depth: Maximum recursion depth
        truncate_strings: Maximum string length before truncation

    Returns:
        Dictionary representation
    """
    if max_depth <= 0:
        return {"__truncated__": str(obj)[:truncate_strings]}

    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, str) and len(obj) > truncate_strings:
            return obj[:truncate_strings] + "..."
        return obj

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, (list, tuple)):
        return [to_dict_safe(item, max_depth - 1, truncate_strings) for item in obj]

    if isinstance(obj, dict):
        return {
            str(k): to_dict_safe(v, max_depth - 1, truncate_strings)
            for k, v in obj.items()
        }

    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for k, v in asdict(obj).items():
            result[k] = to_dict_safe(v, max_depth - 1, truncate_strings)
        return result

    # Handle objects with to_dict method
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return to_dict_safe(obj.to_dict(), max_depth - 1, truncate_strings)
        except:
            return {"__error__": "to_dict failed", "__type__": type(obj).__name__}

    # Handle enums
    if hasattr(obj, "value"):
        return obj.value

    # Default: convert to string
    return {"__type__": type(obj).__name__, "__value__": str(obj)[:truncate_strings]}


def from_dict_safe(
    data: Dict[str, Any], target_class: type, strict: bool = False
) -> Any:
    """Safely convert dictionary to object.

    Args:
        data: Dictionary data
        target_class: Target class to instantiate
        strict: If True, raise error on missing fields

    Returns:
        Instantiated object
    """
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")

    if is_dataclass(target_class):
        import inspect

        sig = inspect.signature(target_class.__init__)
        params = list(sig.parameters.keys())[1:]  # Exclude 'self'

        kwargs = {}
        for param in params:
            if param in data:
                kwargs[param] = data[param]
            elif strict:
                raise ValueError(f"Missing required field: {param}")

        return target_class(**kwargs)

    # For regular classes with from_dict
    if hasattr(target_class, "from_dict") and callable(
        getattr(target_class, "from_dict")
    ):
        return target_class.from_dict(data)

    raise ValueError(f"Cannot deserialize to {target_class}")


def serialize_json(
    data: Any, indent: Optional[int] = None, sort_keys: bool = True
) -> str:
    """Serialize data to JSON string.

    Args:
        data: Data to serialize
        indent: Indentation level (None for compact)
        sort_keys: Whether to sort dictionary keys

    Returns:
        JSON string
    """

    def default_handler(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "value"):  # Enum
            return obj.value
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default_handler, indent=indent, sort_keys=sort_keys)


def deserialize_json(json_str: str) -> Any:
    """Deserialize JSON string to Python object.

    Args:
        json_str: JSON string

    Returns:
        Deserialized Python object
    """
    return json.loads(json_str)


def safe_cast(value: Any, target_type: type, default: Any = None) -> Any:
    """Safely cast a value to target type.

    Args:
        value: Value to cast
        target_type: Target type
        default: Default value if casting fails

    Returns:
        Casted value or default
    """
    if value is None:
        return default

    try:
        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)

        if target_type in (int, float, str):
            return target_type(value)

        if target_type == list and not isinstance(value, list):
            return [value] if value is not None else []

        if target_type == dict and not isinstance(value, dict):
            return {} if value is None else {"value": value}

        return target_type(value)
    except (ValueError, TypeError):
        return default


def ensure_list(value: Any) -> List[Any]:
    """Ensure value is a list.

    Args:
        value: Value to convert

    Returns:
        List containing value, or value if already list
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def ensure_dict(value: Any, key: str = "value") -> Dict[str, Any]:
    """Ensure value is a dictionary.

    Args:
        value: Value to convert
        key: Key to use if value needs wrapping

    Returns:
        Dictionary
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {key: value}


def truncate_string(value: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length.

    Args:
        value: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(value) <= max_length:
        return value
    return value[: max_length - len(suffix)] + suffix


def parse_datetime(
    value: Union[str, datetime, None], default: Optional[datetime] = None
) -> Optional[datetime]:
    """Parse datetime from various formats.

    Args:
        value: Value to parse
        default: Default value if parsing fails

    Returns:
        Parsed datetime or default
    """
    if value is None:
        return default

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

    return default


# Compatibility aliases
serialize = serialize_json
deserialize = deserialize_json
to_dict = to_dict_safe
from_dict = from_dict_safe
