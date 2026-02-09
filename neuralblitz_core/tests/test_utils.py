"""
Comprehensive tests for NeuralBlitz Core Utilities
"""

import unittest
from datetime import datetime, timedelta
import math
import json

# Import all utility modules
from neuralblitz_core.utils import (
    # Math
    clip_value,
    normalize_vector,
    safe_mean,
    safe_sum,
    compute_cross_hessian,
    compute_frobenius_norm,
    compute_trace,
    generate_random_beta,
    add_noise_to_value,
    compute_phase_difference,
    apply_phase_shift,
    # Type conversion
    to_dict_safe,
    from_dict_safe,
    serialize_json,
    deserialize_json,
    safe_cast,
    ensure_list,
    ensure_dict,
    truncate_string,
    parse_datetime,
    # Validation
    ValidationException,
    validate_not_none,
    validate_type,
    validate_range,
    validate_string,
    validate_list,
    validate_dict,
    validate_api_key_format,
    validate_id_format,
    clamp_value,
    # Errors
    NeuralBlitzError,
    AuthenticationError,
    PermissionDeniedError,
    SelfRewriteBlockedError,
    format_error_response,
    raise_if_none,
    raise_unless,
    # Logging
    get_logger,
    log_with_context,
    log_operation_start,
    log_operation_end,
    # ID utilities
    generate_uuid,
    generate_short_uuid,
    generate_hash_id,
    compute_sha256,
    generate_api_key,
    generate_dag_node_hash,
    generate_proof_id,
)


try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class TestMathUtils(unittest.TestCase):
    """Test math utility functions."""

    def test_clip_value(self):
        self.assertEqual(clip_value(0.5, 0.0, 1.0), 0.5)
        self.assertEqual(clip_value(-0.1, 0.0, 1.0), 0.0)
        self.assertEqual(clip_value(1.5, 0.0, 1.0), 1.0)
        self.assertEqual(clip_value(5, 0, 10), 5)

    def test_normalize_vector(self):
        # Test with list
        vec = [3.0, 4.0]
        norm = normalize_vector(vec)
        expected_norm = 1.0
        if NUMPY_AVAILABLE:
            self.assertAlmostEqual(float(np.linalg.norm(norm)), expected_norm)
        else:
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in norm)), expected_norm)

        # Test with zero vector
        zero_vec = [0.0, 0.0]
        result = normalize_vector(zero_vec)
        self.assertEqual(result, zero_vec)

    def test_safe_mean(self):
        self.assertEqual(safe_mean([1, 2, 3, 4, 5]), 3.0)
        self.assertEqual(safe_mean([], default=0.0), 0.0)
        self.assertEqual(safe_mean([], default=100.0), 100.0)

    def test_safe_sum(self):
        self.assertEqual(safe_sum([1, 2, 3, 4, 5]), 15.0)
        self.assertEqual(safe_sum([], default=0.0), 0.0)
        self.assertEqual(safe_sum([], default=50.0), 50.0)

    def test_compute_trace(self):
        matrix = [[1, 2], [3, 4]]
        self.assertEqual(compute_trace(matrix), 5.0)

        if NUMPY_AVAILABLE:
            np_matrix = np.array([[1, 2], [3, 4]])
            self.assertEqual(compute_trace(np_matrix), 5.0)

    def test_compute_frobenius_norm(self):
        matrix = [[1, 2], [3, 4]]
        norm = compute_frobenius_norm(matrix)
        expected = math.sqrt(1 + 4 + 9 + 16)
        self.assertAlmostEqual(norm, expected)

    def test_add_noise_to_value(self):
        value = 10.0
        noisy = add_noise_to_value(value, noise_scale=0.0)
        self.assertAlmostEqual(noisy, value)

    def test_compute_phase_difference(self):
        phase1 = complex(1, 1)  # 45 degrees
        phase2 = complex(1, 0)  # 0 degrees
        diff = compute_phase_difference(phase1, phase2)
        self.assertIsInstance(diff, float)

    def test_apply_phase_shift(self):
        phase = complex(1, 0)
        shifted = apply_phase_shift(phase, math.pi / 2)
        self.assertIsInstance(shifted, complex)


class TestTypeUtils(unittest.TestCase):
    """Test type conversion utility functions."""

    def test_to_dict_safe_basic(self):
        self.assertEqual(to_dict_safe("test"), "test")
        self.assertEqual(to_dict_safe(42), 42)
        self.assertEqual(to_dict_safe(3.14), 3.14)
        self.assertEqual(to_dict_safe(True), True)
        self.assertEqual(to_dict_safe(None), None)

    def test_to_dict_safe_datetime(self):
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = to_dict_safe(dt)
        self.assertEqual(result, "2024-01-15T10:30:00")

    def test_to_dict_safe_list(self):
        data = [1, 2, 3, "test"]
        result = to_dict_safe(data)
        self.assertEqual(result, data)

    def test_to_dict_safe_dict(self):
        data = {"key1": "value1", "key2": 42}
        result = to_dict_safe(data)
        self.assertEqual(result, data)

    def test_to_dict_safe_string_truncation(self):
        long_string = "x" * 200
        result = to_dict_safe(long_string, truncate_strings=100)
        self.assertTrue(result.endswith("..."))
        self.assertEqual(len(result), 103)

    def test_to_dict_safe_max_depth(self):
        data = {"level1": {"level2": {"level3": {"level4": "deep"}}}}
        result = to_dict_safe(data, max_depth=2)
        self.assertIn("__truncated__", str(result))

    def test_serialize_deserialize_json(self):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        json_str = serialize_json(data)
        result = deserialize_json(json_str)
        self.assertEqual(result, data)

    def test_safe_cast(self):
        self.assertEqual(safe_cast("123", int), 123)
        self.assertEqual(safe_cast("123.45", float), 123.45)
        self.assertEqual(safe_cast(123, str), "123")
        self.assertEqual(safe_cast("true", bool), True)
        self.assertEqual(safe_cast("yes", bool), True)
        self.assertEqual(safe_cast(1, bool), True)
        self.assertEqual(safe_cast(0, bool), False)
        self.assertEqual(safe_cast("invalid", int, default=0), 0)

    def test_ensure_list(self):
        self.assertEqual(ensure_list([1, 2, 3]), [1, 2, 3])
        self.assertEqual(ensure_list("single"), ["single"])
        self.assertEqual(ensure_list(None), [])
        self.assertEqual(ensure_list((1, 2, 3)), [1, 2, 3])

    def test_ensure_dict(self):
        self.assertEqual(ensure_dict({"key": "value"}), {"key": "value"})
        self.assertEqual(ensure_dict("value"), {"value": "value"})
        self.assertEqual(ensure_dict(None), {})

    def test_truncate_string(self):
        self.assertEqual(truncate_string("short", 100), "short")
        long_str = "x" * 200
        truncated = truncate_string(long_str, 50)
        self.assertEqual(len(truncated), 50)
        self.assertTrue(truncated.endswith("..."))

    def test_parse_datetime(self):
        dt_str = "2024-01-15T10:30:00"
        result = parse_datetime(dt_str)
        self.assertIsInstance(result, datetime)
        self.assertEqual(result.year, 2024)

        self.assertIsNone(parse_datetime(None))
        self.assertIsNone(parse_datetime("invalid"))

        dt = datetime.now()
        self.assertEqual(parse_datetime(dt), dt)


class TestValidationUtils(unittest.TestCase):
    """Test validation utility functions."""

    def test_validate_not_none(self):
        self.assertEqual(validate_not_none("value"), "value")
        self.assertEqual(validate_not_none(0), 0)
        with self.assertRaises(ValidationException):
            validate_not_none(None)

    def test_validate_type(self):
        self.assertEqual(validate_type("test", str), "test")
        self.assertEqual(validate_type(42, int), 42)
        with self.assertRaises(ValidationException):
            validate_type("test", int)
        with self.assertRaises(ValidationException):
            validate_type(42, str)

    def test_validate_range(self):
        self.assertEqual(validate_range(50, 0, 100), 50)
        self.assertEqual(validate_range(0, 0, 100), 0)
        self.assertEqual(validate_range(100, 0, 100), 100)
        with self.assertRaises(ValidationException):
            validate_range(-1, 0, 100)
        with self.assertRaises(ValidationException):
            validate_range(101, 0, 100)

    def test_validate_string(self):
        self.assertEqual(validate_string("test"), "test")
        with self.assertRaises(ValidationException):
            validate_string("", allow_empty=False)
        with self.assertRaises(ValidationException):
            validate_string("ab", min_length=3)
        with self.assertRaises(ValidationException):
            validate_string("abcdefghij", max_length=5)

    def test_validate_list(self):
        self.assertEqual(validate_list([1, 2, 3]), [1, 2, 3])
        with self.assertRaises(ValidationException):
            validate_list([1], min_length=2)
        with self.assertRaises(ValidationException):
            validate_list([1, 2, 3], max_length=2)

    def test_validate_dict(self):
        data = {"required": "value", "optional": "value2"}
        result = validate_dict(data, required_fields=["required"])
        self.assertEqual(result, data)

        with self.assertRaises(ValidationException):
            validate_dict({}, required_fields=["required"])

        with self.assertRaises(ValidationException):
            validate_dict({"extra": "value"}, allowed_fields=["allowed"])

    def test_validate_api_key_format(self):
        valid_key = "nb_" + "a" * 32
        self.assertEqual(validate_api_key_format(valid_key), valid_key)

        with self.assertRaises(ValidationException):
            validate_api_key_format("invalid_key")

        with self.assertRaises(ValidationException):
            validate_api_key_format("nb_short")

    def test_validate_id_format(self):
        self.assertEqual(validate_id_format("valid_id_123"), "valid_id_123")

        with self.assertRaises(ValidationException):
            validate_id_format("invalid.id")

        with self.assertRaises(ValidationException):
            validate_id_format("ab", min_length=5)

    def test_clamp_value(self):
        self.assertEqual(clamp_value(50, 0, 100), 50)
        self.assertEqual(clamp_value(-10, 0, 100), 0)
        self.assertEqual(clamp_value(150, 0, 100), 100)


class TestErrorUtils(unittest.TestCase):
    """Test error utility functions."""

    def test_neural_blitz_error(self):
        err = NeuralBlitzError("Test error", code="test_code")
        self.assertEqual(err.message, "Test error")
        self.assertEqual(err.code, "test_code")

        dict_repr = err.to_dict()
        self.assertEqual(dict_repr["error"], True)
        self.assertEqual(dict_repr["code"], "test_code")

    def test_authentication_error(self):
        err = AuthenticationError("Auth failed", code="auth_failed")
        self.assertEqual(err.code, "auth_failed")

    def test_permission_denied_error(self):
        err = PermissionDeniedError("Access denied", required=["admin"])
        self.assertEqual(err.required, ["admin"])

    def test_self_rewrite_blocked_error(self):
        err = SelfRewriteBlockedError()
        self.assertEqual(err.code, "permission_denied")
        self.assertIn("judex_quorum", err.required)

    def test_format_error_response(self):
        err = ValueError("Test error")
        response = format_error_response(err)
        self.assertEqual(response["error"], True)
        self.assertIn("message", response)

    def test_raise_if_none(self):
        self.assertEqual(raise_if_none("value"), "value")
        with self.assertRaises(Exception):  # NotFoundError
            raise_if_none(None)

    def test_raise_unless(self):
        raise_unless(True, "Should not raise")
        with self.assertRaises(Exception):
            raise_unless(False, "Should raise")


class TestIDUtils(unittest.TestCase):
    """Test ID and hash utility functions."""

    def test_generate_uuid(self):
        uuid1 = generate_uuid()
        uuid2 = generate_uuid()
        self.assertIsInstance(uuid1, str)
        self.assertNotEqual(uuid1, uuid2)
        self.assertEqual(len(uuid1), 36)  # Standard UUID length

    def test_generate_short_uuid(self):
        short = generate_short_uuid(16)
        self.assertEqual(len(short), 16)

    def test_generate_hash_id(self):
        hash_id = generate_hash_id("test data")
        self.assertIsInstance(hash_id, str)
        self.assertEqual(len(hash_id), 16)  # Default length is 16

        # Test full length
        full_hash = generate_hash_id("test data", length=None)
        self.assertEqual(len(full_hash), 64)  # SHA256 length

    def test_compute_sha256(self):
        hash1 = compute_sha256("test")
        hash2 = compute_sha256("test")
        hash3 = compute_sha256("different")
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertEqual(len(hash1), 64)

    def test_generate_api_key(self):
        key = generate_api_key()
        self.assertTrue(key.startswith("nb_"))
        self.assertEqual(len(key), 67)  # nb_ + 64 hex chars

    def test_generate_dag_node_hash(self):
        hash_val = generate_dag_node_hash(
            index=1,
            timestamp=1234567890.0,
            operation="test",
            actor_id="actor1",
            payload_hash="payload123",
            parent_hash="parent456",
        )
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)

    def test_generate_proof_id(self):
        proof_id = generate_proof_id("TestTheorem")
        self.assertTrue(proof_id.startswith("VPROOF#"))


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple utilities."""

    def test_complex_data_serialization(self):
        """Test complex data structure with datetime and nested dicts."""
        data = {
            "id": generate_uuid(),
            "timestamp": datetime.now(),
            "values": [1, 2, 3, 4, 5],
            "metadata": {"computed": True, "mean": safe_mean([1, 2, 3, 4, 5])},
        }

        # Convert to dict (handling datetime)
        dict_repr = to_dict_safe(data)

        # Serialize to JSON
        json_str = serialize_json(dict_repr)

        # Deserialize
        restored = deserialize_json(json_str)

        self.assertEqual(restored["values"], data["values"])
        self.assertEqual(restored["metadata"]["mean"], 3.0)

    def test_error_handling_flow(self):
        """Test complete error handling flow."""
        try:
            validate_not_none(None, "critical_field")
        except Exception as e:
            response = format_error_response(e)
            self.assertTrue(response["error"])
            self.assertIn("message", response)


if __name__ == "__main__":
    unittest.main()
