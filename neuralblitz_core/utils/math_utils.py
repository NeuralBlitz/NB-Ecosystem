"""
Math Utilities Module
NeuralBlitz Core - Mathematical Operations and Helpers

Provides:
- Vector normalization and operations
- Clipping and range validation
- Statistical calculations
- Matrix operations
"""

from typing import List, Union, Optional, Tuple
import math

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


def clip_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clip a value to specified range.

    Args:
        value: Value to clip
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clipped value
    """
    return max(min_val, min(max_val, value))


def normalize_vector(
    vector: Union[List[float], "np.ndarray"],
) -> Union[List[float], "np.ndarray"]:
    """Normalize a vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Normalized vector
    """
    if NUMPY_AVAILABLE and isinstance(vector, np.ndarray):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    else:
        # Pure Python implementation
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]


def safe_mean(values: List[Union[float, int]], default: float = 0.0) -> float:
    """Calculate mean safely, returning default if list is empty.

    Args:
        values: List of numeric values
        default: Default value to return if list is empty

    Returns:
        Mean value or default
    """
    if not values:
        return default
    if NUMPY_AVAILABLE:
        return float(np.mean(values))
    return sum(values) / len(values)


def safe_sum(values: List[Union[float, int]], default: float = 0.0) -> float:
    """Calculate sum safely, returning default if list is empty.

    Args:
        values: List of numeric values
        default: Default value to return if list is empty

    Returns:
        Sum value or default
    """
    if not values:
        return default
    if NUMPY_AVAILABLE:
        return float(np.sum(values))
    return sum(values)


def compute_cross_hessian(
    state: Union[List[float], "np.ndarray"], regularization: float = 0.1
) -> Union[List[List[float]], "np.ndarray"]:
    """Compute cross-Hessian matrix from state vector.

    Args:
        state: Input state vector
        regularization: Regularization parameter

    Returns:
        Cross-Hessian matrix
    """
    if NUMPY_AVAILABLE and isinstance(state, np.ndarray):
        n_dims = len(state)
        hessian = np.outer(state, state)
        hessian += np.eye(n_dims) * regularization
        # Normalize
        norm = np.linalg.norm(hessian)
        if norm > 0:
            hessian = hessian / norm
        return hessian
    else:
        # Pure Python implementation
        n_dims = len(state)
        hessian = [[state[i] * state[j] for j in range(n_dims)] for i in range(n_dims)]
        # Add regularization
        for i in range(n_dims):
            hessian[i][i] += regularization
        # Normalize
        norm = math.sqrt(
            sum(sum(row[i] * row[i] for i in range(n_dims)) for row in hessian)
        )
        if norm > 0:
            hessian = [
                [hessian[i][j] / norm for j in range(n_dims)] for i in range(n_dims)
            ]
        return hessian


def compute_frobenius_norm(matrix: Union[List[List[float]], "np.ndarray"]) -> float:
    """Compute Frobenius norm of a matrix.

    Args:
        matrix: Input matrix

    Returns:
        Frobenius norm
    """
    if NUMPY_AVAILABLE and isinstance(matrix, np.ndarray):
        return float(np.linalg.norm(matrix, "fro"))
    else:
        return math.sqrt(sum(sum(x * x for x in row) for row in matrix))


def compute_trace(matrix: Union[List[List[float]], "np.ndarray"]) -> float:
    """Compute trace of a matrix.

    Args:
        matrix: Input matrix

    Returns:
        Trace value
    """
    if NUMPY_AVAILABLE and isinstance(matrix, np.ndarray):
        return float(np.trace(matrix))
    else:
        return sum(
            matrix[i][i]
            for i in range(min(len(matrix), len(matrix[0]) if matrix else 0))
        )


def generate_random_beta(alpha: float = 2.0, beta: float = 2.0) -> float:
    """Generate random number from Beta distribution.

    Args:
        alpha: Alpha parameter
        beta: Beta parameter

    Returns:
        Random value from Beta distribution
    """
    if NUMPY_AVAILABLE:
        return float(np.random.beta(alpha, beta))
    else:
        # Fallback using uniform random
        import random

        return random.random()


def add_noise_to_value(value: float, noise_scale: float = 0.1) -> float:
    """Add random noise to a value.

    Args:
        value: Base value
        noise_scale: Scale of noise

    Returns:
        Value with noise added
    """
    if NUMPY_AVAILABLE:
        return value + float(np.random.randn() * noise_scale)
    else:
        import random

        return value + random.gauss(0, noise_scale)


def compute_phase_difference(phase1: complex, phase2: complex) -> float:
    """Compute phase difference between two complex numbers.

    Args:
        phase1: First complex number
        phase2: Second complex number

    Returns:
        Phase difference in radians
    """
    if NUMPY_AVAILABLE:
        return float(np.angle(phase1) - np.angle(phase2))
    else:
        import cmath

        return cmath.phase(phase1) - cmath.phase(phase2)


def apply_phase_shift(phase: complex, shift: float) -> complex:
    """Apply phase shift to complex number.

    Args:
        phase: Complex phase
        shift: Phase shift to apply

    Returns:
        Phase-shifted complex number
    """
    if NUMPY_AVAILABLE:
        return phase * np.exp(1j * shift)
    else:
        import cmath

        return phase * cmath.exp(1j * shift)


# Compatibility aliases for legacy code
clip = clip_value
normalize = normalize_vector
mean = safe_mean
sum_values = safe_sum
