# UAID: NBX-TST-ALG-00008
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Persona Fusion Mixer (persona_fusion_mixer.py, NBX-ALG-00008)
#
# Core Principle: Synergy & Emergence - validating the mechanics of conceptual blending.

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict

# Import the class we are testing
from Algorithms.Source.persona_fusion_mixer import PersonaFusionMixer

# --- Test Fixtures ---

@pytest.fixture
def mixer_instance() -> PersonaFusionMixer:
    """Provides a default instance of the PersonaFusionMixer."""
    return PersonaFusionMixer(epsilon=0.05, max_iters=100)

@pytest.fixture
def distinct_logit_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Provides two distinct, representative logit vectors for testing."""
    # Vocab Size = 5
    # Persona A ("Logic") has a strong preference for token at index 1.
    logits_a = np.array([-1.0, 5.0, 0.0, -2.0, -3.0], dtype=np.float32)
    # Persona B ("Creative") has a strong preference for token at index 3.
    logits_b = np.array([-2.0, -1.0, 0.0, 5.0, -1.0], dtype=np.float32)
    return logits_a, logits_b

# --- Helper Function ---

def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """A numerically stable function to convert logits to probabilities."""
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=0)

# --- Test Cases ---

class TestPersonaFusionMixer:

    def test_initialization(self):
        """Tests that the mixer initializes with correct default or specified parameters."""
        mixer_default = PersonaFusionMixer()
        assert mixer_default.epsilon == 0.01
        assert mixer_default.max_iters == 50

        mixer_custom = PersonaFusionMixer(epsilon=0.5, max_iters=200)
        assert mixer_custom.epsilon == 0.5
        assert mixer_custom.max_iters == 200

    @pytest.mark.parametrize("logits, weights, expected_error, match_str", [
        ([], [], ValueError, "logit_vectors cannot be empty"),
        ([np.array([1, 2])], [0.5, 0.5], ValueError, "have the same length"),
        ([np.array([1, 2]), np.array([3, 4])], [0.5], ValueError, "have the same length"),
        ([np.array([1, 2]), np.array([3, 4])], [0.5, 0.6], ValueError, "weights must sum to 1.0"),
    ])
    def test_fuse_logits_raises_errors_on_invalid_input(self, mixer_instance: PersonaFusionMixer, logits, weights, expected_error, match_str):
        """Tests that `fuse_logits` correctly raises ValueErrors for malformed inputs."""
        with pytest.raises(expected_error, match=match_str):
            mixer_instance.fuse_logits(logits, weights)
            
    def test_fuse_logits_handles_mismatched_vocab_size(self, mixer_instance: PersonaFusionMixer):
        """Tests that a clear error is raised if logit vectors have different shapes."""
        logits_a = np.array([1.0, 2.0, 3.0])
        logits_b = np.array([1.0, 2.0]) # Mismatched shape
        weights = [0.5, 0.5]
        
        # This should fail during the numpy array creation or an operation within the method
        with pytest.raises(ValueError):
             mixer_instance.fuse_logits([logits_a, logits_b], weights)


    def test_fusion_with_full_weight_on_one_persona(self, mixer_instance: PersonaFusionMixer, distinct_logit_vectors):
        """
        Tests the case where one persona has 100% weight. The output should be
        nearly identical to that persona's original distribution.
        """
        logits_a, logits_b = distinct_logit_vectors

        # Test with 100% weight on Persona A
        fused_logits_a, _ = mixer_instance.fuse_logits([logits_a, logits_b], [1.0, 0.0])
        probs_a = logits_to_probs(logits_a)
        fused_probs_a = logits_to_probs(fused_logits_a)
        np.testing.assert_allclose(fused_probs_a, probs_a, atol=1e-4)

        # Test with 100% weight on Persona B
        fused_logits_b, _ = mixer_instance.fuse_logits([logits_a, logits_b], [0.0, 1.0])
        probs_b = logits_to_probs(logits_b)
        fused_probs_b = logits_to_probs(fused_logits_b)
        np.testing.assert_allclose(fused_probs_b, probs_b, atol=1e-4)

    def test_fusion_with_equal_weights_blends_distributions(self, mixer_instance: PersonaFusionMixer, distinct_logit_vectors):
        """
        Tests if a 50/50 fusion creates a new, blended distribution where the
        original peaks are dampened and a new peak may emerge.
        """
        logits_a, logits_b = distinct_logit_vectors
        probs_a = logits_to_probs(logits_a)
        probs_b = logits_to_probs(logits_b)

        fused_logits, report = mixer_instance.fuse_logits([logits_a, logits_b], [0.5, 0.5])
        fused_probs = logits_to_probs(fused_logits)

        # Assert that the fusion converged
        assert report["status"] == "converged"
        
        # Assert the output has the correct shape
        assert fused_logits.shape == logits_a.shape

        # The peak of the fused distribution should be less prominent than the original peaks,
        # indicating a true blend rather than just picking one.
        assert np.max(fused_probs) < np.max(probs_a)
        assert np.max(fused_probs) < np.max(probs_b)
        
        # The probability of the original peaks should be reduced in the fused distribution.
        peak_a_index = np.argmax(logits_a) # Index 1
        peak_b_index = np.argmax(logits_b) # Index 3
        assert fused_probs[peak_a_index] < probs_a[peak_a_index]
        assert fused_probs[peak_b_index] < probs_b[peak_b_index]

    def test_fusion_with_multiple_personas(self, mixer_instance: PersonaFusionMixer):
        """Tests fusion with more than two personas."""
        logits_a = np.array([5.0, 0.0, 0.0])
        logits_b = np.array([0.0, 5.0, 0.0])
        logits_c = np.array([0.0, 0.0, 5.0])
        weights = [0.4, 0.4, 0.2]

        fused_logits, report = mixer_instance.fuse_logits([logits_a, logits_b, logits_c], weights)
        
        assert report["status"] == "converged"
        assert fused_logits.shape == logits_a.shape
        
        # The fused logits should show influence from all three, with A and B being stronger
        assert np.argmax(fused_logits) in [0, 1] # The peak should be from the higher-weighted personas.


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and numpy are installed:
    #    pip install pytest numpy
    # 3. (Optional but recommended) Install sentence-transformers if you want to run the
    #    main algorithm file, though it is not needed for this test file.
    # 4. Run the tests:
    #    pytest Algorithms/Tests/test_persona_fusion_mixer.py
    
    print("This is a test file. Use 'pytest' to execute it.")