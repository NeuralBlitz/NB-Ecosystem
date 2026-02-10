# UAID: NBX-TST-ALG-00002
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Reflexive Drift Tuner
# Verifies the correctness of the NBX-ALG-00002 algorithm.
#
# Core Principle: Reflexive Alignment (ε₄) - Ensuring the tools for "Knowing Thy Drift" are themselves accurate.

import pytest
import numpy as np

from Algorithms.Source.reflexive_drift_tuner import ReflexiveDriftTuner

# --- Test Fixtures ---

@pytest.fixture
def default_tuner() -> ReflexiveDriftTuner:
    """Provides a default instance of the tuner for testing."""
    return ReflexiveDriftTuner()

@pytest.fixture
def aggressive_tuner() -> ReflexiveDriftTuner:
    """Provides a tuner with high gains for faster convergence testing."""
    return ReflexiveDriftTuner(Kp=0.8, Ki=0.2, Kd=0.1)

@pytest.fixture
def vectors_4d() -> dict:
    """Provides a set of 4D vectors for testing."""
    ref = np.array([1.0, 0.0, 0.0, 0.0])
    # A vector that has drifted significantly
    drifted = np.array([0.7, 0.7, 0.1, -0.1])
    drifted /= np.linalg.norm(drifted)
    return {"reference": ref, "drifted": drifted, "uid": "vector_4d_test"}


# --- Test Cases ---

def test_tuner_initialization(default_tuner: ReflexiveDriftTuner):
    """Tests that the tuner can be initialized with default and custom values."""
    assert default_tuner is not None
    assert default_tuner.Kp == 0.4
    
    custom_tuner = ReflexiveDriftTuner(Kp=1.0, Ki=0.5, Kd=0.2, drift_threshold=0.5)
    assert custom_tuner.Kp == 1.0
    assert custom_tuner.drift_threshold == 0.5

def test_initialization_with_negative_gains():
    """Tests that the tuner raises an error for negative gains."""
    with pytest.raises(ValueError, match="PID gains must be non-negative"):
        ReflexiveDriftTuner(Kp=-0.1)
    with pytest.raises(ValueError, match="PID gains must be non-negative"):
        ReflexiveDriftTuner(Ki=-0.1)

@pytest.mark.parametrize("vec_a, vec_b, expected_drift", [
    (np.array([1, 0, 0]), np.array([1, 0, 0]), 0.0),      # Identical vectors
    (np.array([1, 0, 0]), np.array([-1, 0, 0]), 2.0),     # Opposite vectors
    (np.array([1, 0, 0]), np.array([0, 1, 0]), 1.0),      # Orthogonal vectors
    (np.array([3, 0]), np.array([1, 0]), 0.0),            # Same direction, different magnitude
    (np.array([1, 1]), np.array([1, 0]), 1 - 1/np.sqrt(2))# 45-degree angle
])
def test_calculate_cosine_drift(default_tuner: ReflexiveDriftTuner, vec_a, vec_b, expected_drift):
    """
    Tests the accuracy of the cosine drift (Δc) calculation for various vector pairs.
    """
    drift = default_tuner._calculate_cosine_drift(vec_a, vec_b)
    assert np.isclose(drift, expected_drift), f"Failed for vectors {vec_a}, {vec_b}"

def test_cosine_drift_with_zero_vector(default_tuner: ReflexiveDriftTuner):
    """Tests the edge case of a zero vector, which should result in maximum drift."""
    vec_a = np.array([1, 0, 0])
    zero_vec = np.array([0, 0, 0])
    assert default_tuner._calculate_cosine_drift(vec_a, zero_vec) == 1.0
    assert default_tuner._calculate_cosine_drift(zero_vec, vec_a) == 1.0

def test_single_tuning_step_reduces_drift(default_tuner: ReflexiveDriftTuner, vectors_4d: dict):
    """
    Verifies that a single application of the tuner brings the drifted vector
    semantically closer to the reference vector.
    """
    ref = vectors_4d["reference"]
    drifted = vectors_4d["drifted"]
    uid = vectors_4d["uid"]

    initial_drift = default_tuner._calculate_cosine_drift(drifted, ref)
    assert initial_drift > 0.1 # Ensure there is drift to begin with

    corrected_vector, _ = default_tuner.tune_vector(uid, drifted, ref)
    final_drift = default_tuner._calculate_cosine_drift(corrected_vector, ref)

    assert final_drift < initial_drift, "The tuning step should always reduce drift."
    assert np.isclose(np.linalg.norm(corrected_vector), 1.0), "Output vector must be unit length."

def test_iterative_tuning_converges(aggressive_tuner: ReflexiveDriftTuner, vectors_4d: dict):
    """
    Tests that repeatedly applying the tuning algorithm causes the drifted vector
    to converge towards the reference vector.
    """
    ref = vectors_4d["reference"]
    vector_to_tune = vectors_4d["drifted"]
    uid = vectors_4d["uid"]
    
    last_drift = aggressive_tuner._calculate_cosine_drift(vector_to_tune, ref)
    
    for i in range(10): # Converge over 10 iterations
        vector_to_tune, _ = aggressive_tuner.tune_vector(uid, vector_to_tune, ref)
        current_drift = aggressive_tuner._calculate_cosine_drift(vector_to_tune, ref)
        
        # Check that drift is monotonically decreasing
        assert current_drift < last_drift or np.isclose(current_drift, last_drift)
        last_drift = current_drift
    
    # After 10 iterations with an aggressive tuner, drift should be very close to zero
    assert last_drift < 1e-5, f"Drift did not converge to near-zero. Final drift: {last_drift}"

def test_tuner_maintains_separate_states(default_tuner: ReflexiveDriftTuner):
    """
    Ensures that the tuner correctly manages separate integral and derivative
    states for different vector UIDs.
    """
    vec_a_ref = np.array([1.0, 0, 0])
    vec_a_drift = np.array([0.8, 0.6, 0])
    
    vec_b_ref = np.array([0, 1.0, 0])
    vec_b_drift = np.array([0, 0.7, 0.7])

    # First step for vector A
    _, report_a1 = default_tuner.tune_vector("vector_a", vec_a_drift, vec_a_ref)
    
    # First step for vector B
    _, report_b1 = default_tuner.tune_vector("vector_b", vec_b_drift, vec_b_ref)

    # Assert that the initial integral states are just the first error
    assert np.isclose(default_tuner.states["vector_a"]["integral_error"], report_a1["drift_delta_c"])
    assert np.isclose(default_tuner.states["vector_b"]["integral_error"], report_b1["drift_delta_c"])
    
    # Second step for vector A
    _, report_a2 = default_tuner.tune_vector("vector_a", vec_a_drift, vec_a_ref)
    
    # Check that A's integral state was updated, but B's was not.
    assert default_tuner.states["vector_a"]["integral_error"] > report_a1["drift_delta_c"]
    assert np.isclose(default_tuner.states["vector_b"]["integral_error"], report_b1["drift_delta_c"])

def test_diagnostic_report_content(default_tuner: ReflexiveDriftTuner, vectors_4d: dict):
    """Verifies that the diagnostic report contains all necessary fields and correct types."""
    
    ref = vectors_4d["reference"]
    drifted = vectors_4d["drifted"]
    uid = vectors_4d["uid"]
    
    # Test a case below the threshold
    tuner_high_thresh = ReflexiveDriftTuner(drift_threshold=0.9)
    _, report_safe = tuner_high_thresh.tune_vector(uid, drifted, ref)
    
    assert "is_above_threshold" in report_safe and report_safe["is_above_threshold"] is False
    
    # Test a case above the threshold
    tuner_low_thresh = ReflexiveDriftTuner(drift_threshold=0.1)
    _, report_danger = tuner_low_thresh.tune_vector(uid, drifted, ref)
    
    assert "is_above_threshold" in report_danger and report_danger["is_above_threshold"] is True
    
    # Check for all expected keys
    expected_keys = ["vector_uid", "drift_delta_c", "control_signal", "p_term", "i_term", "d_term"]
    for key in expected_keys:
        assert key in report_safe