# UAID: NBX-TST-ALG-00012
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Semantic Persona Diff (semantic_persona_diff.py, NBX-ALG-00012)
#
# Core Principle: Reflexive Alignment (ε₄) - validating the tools that measure our drift.

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict

# Import the class we are testing
from Algorithms.Source.semantic_persona_diff import SemanticPersonaDiff

# --- Mocking Dependencies ---

class MockSentenceTransformer:
    """
    A deterministic mock of the sentence_transformers.SentenceTransformer class.
    It returns pre-defined vectors for specific inputs, avoiding model downloads
    and ensuring test reproducibility.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_dim = 4  # A simple 4D space for testing

        # Pre-defined vocabulary and their corresponding vectors
        # These vectors are designed to be orthogonal or nearly so for clear results.
        self.vocab = {
            "The system architecture must prioritize scalability. We should use gRPC.": np.array([1.0, 0.0, 0.0, 0.0]),
            "System architecture needs resilience and high availability. Use gRPC.": np.array([0.9, 0.1, 0.0, 0.0]), # Similar to above
            "Explore the liminal space between dream and reality using archetypes.": np.array([0.0, 1.0, 0.0, 0.0]),
            "A test case for the identical document comparison.": np.array([0.0, 0.0, 1.0, 0.0]),
        }

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def encode(self, sentences: List[str], normalize_embeddings: bool = False) -> np.ndarray:
        """Returns the pre-defined vector for a known sentence."""
        embeddings = [self.vocab.get(s, np.zeros(self.embedding_dim)) for s in sentences]
        embeddings_arr = np.array(embeddings, dtype=np.float32)
        
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            embeddings_arr /= norms
            
        return embeddings_arr

# --- Test Fixtures ---

@pytest.fixture
def differ_instance(monkeypatch) -> SemanticPersonaDiff:
    """
    Provides a SemanticPersonaDiff instance with the SentenceTransformer
    dependency mocked out.
    """
    # Use monkeypatch to replace the real, heavy SentenceTransformer with our lightweight mock
    monkeypatch.setattr("Algorithms.Source.semantic_persona_diff.SentenceTransformer", MockSentenceTransformer)
    return SemanticPersonaDiff()

# --- Test Cases ---

class TestSemanticPersonaDiff:

    def test_initialization(self, differ_instance: SemanticPersonaDiff):
        """Tests that the differ initializes correctly using the mocked model."""
        assert differ_instance is not None
        assert isinstance(differ_instance.model, MockSentenceTransformer)
        assert differ_instance.embedding_dim == 4

    def test_compute_diff_with_highly_similar_histories(self, differ_instance: SemanticPersonaDiff):
        """
        Tests two histories that are conceptually very close.
        Expects a cosine similarity near 1.0.
        """
        history_a = ["The system architecture must prioritize scalability.", "We should use gRPC."]
        history_b = ["System architecture needs resilience and high availability.", "Use gRPC."]

        report = differ_instance.compute_diff(history_a, history_b)
        
        assert "error" not in report
        assert isinstance(report["cosine_similarity"], float)
        assert report["cosine_similarity"] == pytest.approx(0.995, abs=1e-3) # (0.9*1.0 + 0.1*0.0) / sqrt(0.82)

    def test_compute_diff_with_highly_dissimilar_histories(self, differ_instance: SemanticPersonaDiff):
        """
        Tests two histories that are conceptually unrelated.
        Expects a cosine similarity near 0.0.
        """
        history_a = ["The system architecture must prioritize scalability.", "We should use gRPC."]
        history_b = ["Explore the liminal space between dream and reality using archetypes."]

        report = differ_instance.compute_diff(history_a, history_b)
        
        assert "error" not in report
        assert report["cosine_similarity"] == pytest.approx(0.0)

    def test_compute_diff_with_identical_histories(self, differ_instance: SemanticPersonaDiff):
        """
        Tests two identical histories.
        Expects a cosine similarity of exactly 1.0.
        """
        history_a = ["A test case for the identical document comparison."]
        history_b = ["A test case for the identical document comparison."]

        report = differ_instance.compute_diff(history_a, history_b)
        
        assert "error" not in report
        assert report["cosine_similarity"] == pytest.approx(1.0)
        # The vector difference should be a zero vector.
        assert np.allclose(report["vector_difference"], np.zeros(differ_instance.embedding_dim))

    def test_compute_diff_with_empty_history(self, differ_instance: SemanticPersonaDiff):
        """
        Tests edge cases where one or both input histories are empty lists.
        """
        history_a = ["Some content."]
        
        # Case 1: B is empty
        report1 = differ_instance.compute_diff(history_a, [])
        assert "error" in report1
        assert report1["error"] == "One or both histories are empty."
        assert report1["cosine_similarity"] == 0.0

        # Case 2: A is empty
        report2 = differ_instance.compute_diff([], history_a)
        assert "error" in report2
        
        # Case 3: Both are empty
        report3 = differ_instance.compute_diff([], [])
        assert "error" in report3

    def test_vector_difference_direction(self, differ_instance: SemanticPersonaDiff):
        """
        Verifies the vector difference `A - B` is calculated correctly.
        """
        history_a = ["The system architecture must prioritize scalability.", "We should use gRPC."] # Approx vector [1, 0, 0, 0]
        history_b = ["Explore the liminal space between dream and reality using archetypes."] # Approx vector [0, 1, 0, 0]

        report = differ_instance.compute_diff(history_a, history_b)
        
        # Expected difference is roughly [1, -1, 0, 0] after normalization
        diff_vec = np.array(report["vector_difference"])
        
        # The first component should be positive, the second negative.
        assert diff_vec[0] > 0
        assert diff_vec[1] < 0
        assert np.isclose(diff_vec[2], 0.0)
        assert np.isclose(diff_vec[3], 0.0)

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and sentence-transformers are installed:
    #    pip install pytest sentence-transformers
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_semantic_persona_diff.py
    
    print("This is a test file. Use 'pytest' to execute it.")