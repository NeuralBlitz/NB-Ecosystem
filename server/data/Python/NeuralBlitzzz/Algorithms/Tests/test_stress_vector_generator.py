# UAID: NBX-TST-ALG-00017
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Stress Vector Generator (stress_vector_generator.py, NBX-ALG-00017)
#
# Core Principle: Resilience - validating the tools that generate our chaos tests.

import pytest
import numpy as np
from typing import List

# Import the class we are testing
from Algorithms.Source.stress_vector_generator import StressVectorGenerator

# --- Test Fixtures ---

@pytest.fixture
def base_vector() -> np.ndarray:
    """Provides a simple, consistent base vector for testing."""
    vec = np.array([1.0, 0.0, 0.0, 0.0])
    return vec / np.linalg.norm(vec) # Ensure it's a unit vector

@pytest.fixture
def vector_generator() -> StressVectorGenerator:
    """Provides a default instance of the StressVectorGenerator."""
    return StressVectorGenerator(seed=42) # Use a seed for reproducibility

# --- Test Cases ---

class TestStressVectorGenerator:

    def test_initialization(self):
        """Tests that the generator initializes correctly."""
        gen_no_seed = StressVectorGenerator()
        assert gen_no_seed.rng is not None

        gen_with_seed = StressVectorGenerator(seed=123)
        # Check if the seed was used by generating a number and comparing
        val1 = gen_with_seed.rng.random()
        gen_with_seed_2 = StressVectorGenerator(seed=123)
        val2 = gen_with_seed_2.rng.random()
        assert val1 == val2

    def test_generate_opposite_vector(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the generation of a vector that is perfectly anti-parallel.
        The cosine similarity should be -1.0.
        """
        opposite_vec = vector_generator.generate_opposite_vector(base_vector)
        
        # The opposite vector should just be the negative of the base vector
        np.testing.assert_allclose(opposite_vec, -base_vector)
        
        # Verify cosine similarity is -1
        cosine_similarity = np.dot(base_vector, opposite_vec)
        assert cosine_similarity == pytest.approx(-1.0)
        
        # Verify it's still a unit vector
        assert np.linalg.norm(opposite_vec) == pytest.approx(1.0)

    def test_generate_orthogonal_vector(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the generation of a vector that is perfectly orthogonal.
        The dot product (and cosine similarity) should be 0.
        """
        orthogonal_vec = vector_generator.generate_orthogonal_vector(base_vector)

        # Verify the dot product is zero
        dot_product = np.dot(base_vector, orthogonal_vec)
        assert dot_product == pytest.approx(0.0)

        # Verify it's a unit vector
        assert np.linalg.norm(orthogonal_vec) == pytest.approx(1.0)
        
        # Verify it's not the zero vector
        assert not np.allclose(orthogonal_vec, np.zeros_like(orthogonal_vec))

    def test_generate_noisy_vector(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the generation of a noisy vector. The result should be different from
        the base vector but still highly similar.
        """
        noise_level = 0.1
        noisy_vec = vector_generator.generate_noisy_vector(base_vector, noise_level)

        # The new vector should not be identical to the original
        assert not np.allclose(base_vector, noisy_vec)

        # The cosine similarity should be high (close to 1) but not exactly 1
        cosine_similarity = np.dot(base_vector, noisy_vec)
        assert 0.9 < cosine_similarity < 1.0

        # Verify it's a unit vector
        assert np.linalg.norm(noisy_vec) == pytest.approx(1.0)
        
    def test_generate_noisy_vector_with_max_noise(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the edge case where noise level is very high. The result should be
        a nearly random vector.
        """
        # A noise level of 2.0 means the noise vector's magnitude is twice the signal's
        noise_level = 2.0
        noisy_vec = vector_generator.generate_noisy_vector(base_vector, noise_level)

        # The similarity should be low, indicating it's close to a random vector
        cosine_similarity = np.dot(base_vector, noisy_vec)
        assert cosine_similarity < 0.5

    @pytest.mark.parametrize("invalid_noise_level", [-0.1, -100.0])
    def test_generate_noisy_vector_raises_error_on_invalid_noise(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray, invalid_noise_level):
        """Tests that a ValueError is raised for negative noise levels."""
        with pytest.raises(ValueError, match="ERR-PARAM-002"):
            vector_generator.generate_noisy_vector(base_vector, invalid_noise_level)

    def test_generate_high_frequency_vector(self, vector_generator: StressVectorGenerator):
        """
        Tests the generation of a high-frequency vector. The resulting vector should
        have rapidly alternating signs.
        """
        dim = 10
        hf_vec = vector_generator.generate_high_frequency_vector(dim)
        
        assert hf_vec.shape == (dim,)
        assert np.linalg.norm(hf_vec) == pytest.approx(1.0)
        
        # Check for alternating signs (or close to it)
        # The product of adjacent elements should be negative
        sign_products = hf_vec[:-1] * hf_vec[1:]
        assert np.all(sign_products < 0)

    @pytest.mark.parametrize("invalid_dim", [0, 1, -10])
    def test_generate_high_frequency_vector_raises_error_on_invalid_dim(self, vector_generator: StressVectorGenerator, invalid_dim):
        """Tests that a ValueError is raised for invalid dimensions."""
        with pytest.raises(ValueError, match="ERR-PARAM-003"):
            vector_generator.generate_high_frequency_vector(invalid_dim)

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and numpy are installed:
    #    pip install pytest numpy
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_stress_vector_generator.py
    
    print("This is a test file. Use 'pytest' to execute it.")