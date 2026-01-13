# UAID: NBX-TST-ALG-00005
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Latent Directory Navigator (latent_dir_nav.py, NBX-ALG-00005)
#
# Core Principle: Epistemic Fidelity (ε₃) - ensuring our search tools find what is meant, not just what is named.

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict

# Import the class we are testing
from Algorithms.Source.latent_dir_nav import LatentFileSystem

# Mark all tests in this file as standard pytest tests
# (No asyncio needed for this module)

# --- Mocking Dependencies ---

class MockSentenceTransformer:
    """
    A mock of the sentence_transformers.SentenceTransformer class.
    It returns pre-defined vectors for specific inputs to make tests deterministic
    and avoid downloading a real model.
    """
    def __init__(self, model_name: str):
        # The model name is ignored, but the parameter must exist for compatibility.
        self.model_name = model_name
        self.embedding_dim = 3 # Use a simple 3D space for easy testing

        # Pre-defined vocabulary and their corresponding vectors
        self.vocab = {
            # Queries
            "quantum mechanics": np.array([0.9, 0.1, 0.0]),
            "ancient rome": np.array([0.1, 0.9, 0.0]),
            "baking instructions": np.array([0.0, 0.1, 0.9]),
            # Documents (files)
            "A document about quantum entanglement and superposition.": np.array([1.0, 0.0, 0.0]),
            "The history of the Roman Empire, focusing on Caesar.": np.array([0.0, 1.0, 0.0]),
            "A recipe for baking a chocolate cake.": np.array([0.0, 0.0, 1.0]),
        }

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dim

    def encode(self, sentences: List[str], normalize_embeddings: bool = False) -> np.ndarray:
        """Returns the pre-defined vector for a known sentence."""
        embeddings = []
        for s in sentences:
            # Find the vector from our vocab or return a zero vector for unknown text
            vec = self.vocab.get(s, np.zeros(self.embedding_dim))
            embeddings.append(vec)
        
        embeddings_arr = np.array(embeddings, dtype=np.float32)
        
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1
            embeddings_arr /= norms
            
        return embeddings_arr

# --- Test Fixtures ---

@pytest.fixture
def latent_fs_directory(tmp_path: Path) -> Path:
    """Creates a temporary directory with a few semantically distinct files."""
    test_dir = tmp_path / "latent_fs_root"
    test_dir.mkdir()

    # Create files with the exact content our mock model expects
    (test_dir / "quantum_physics.txt").write_text("A document about quantum entanglement and superposition.")
    (test_dir / "roman_history.md").write_text("The history of the Roman Empire, focusing on Caesar.")
    (test_dir / "cake_recipe.txt").write_text("A recipe for baking a chocolate cake.")
    (test_dir / "empty_file.log").touch()
    
    return test_dir

# --- Test Cases ---

class TestLatentFileSystem:

    def test_initialization(self, latent_fs_directory: Path, monkeypatch):
        """Tests that the LatentFileSystem initializes correctly."""
        # Patch the SentenceTransformer class so it doesn't download a real model
        monkeypatch.setattr("Algorithms.Source.latent_dir_nav.SentenceTransformer", MockSentenceTransformer)

        lfs = LatentFileSystem(str(latent_fs_directory))
        
        assert lfs.root == latent_fs_directory
        assert len(lfs.file_paths) == 3 # Should ignore the empty file
        assert "quantum_physics.txt" in [p.name for p in lfs.file_paths]

    def test_ls_finds_correct_file_semantically(self, latent_fs_directory: Path, monkeypatch):
        """Tests if `ls` returns the most semantically similar file first."""
        monkeypatch.setattr("Algorithms.Source.latent_dir_nav.SentenceTransformer", MockSentenceTransformer)
        lfs = LatentFileSystem(str(latent_fs_directory))

        # Test case 1: Quantum Physics
        results_quantum = lfs.ls("quantum mechanics", k=1)
        assert len(results_quantum) == 1
        assert results_quantum[0].name == "quantum_physics.txt"

        # Test case 2: Roman History
        results_history = lfs.ls("ancient rome", k=1)
        assert len(results_history) == 1
        assert results_history[0].name == "roman_history.md"

        # Test case 3: Baking Recipe
        results_baking = lfs.ls("baking instructions", k=1)
        assert len(results_baking) == 1
        assert results_baking[0].name == "cake_recipe.txt"

    def test_ls_respects_k_parameter(self, latent_fs_directory: Path, monkeypatch):
        """Tests if the `k` parameter correctly limits the number of results."""
        monkeypatch.setattr("Algorithms.Source.latent_dir_nav.SentenceTransformer", MockSentenceTransformer)
        lfs = LatentFileSystem(str(latent_fs_directory))

        results_k1 = lfs.ls("anything", k=1)
        assert len(results_k1) == 1

        results_k2 = lfs.ls("anything", k=2)
        assert len(results_k2) == 2

        results_k_large = lfs.ls("anything", k=10)
        assert len(results_k_large) == 3 # Should be capped at the number of files

    def test_ls_ranking_order(self, latent_fs_directory: Path, monkeypatch):
        """Tests if the ranking of results is correct."""
        monkeypatch.setattr("Algorithms.Source.latent_dir_nav.SentenceTransformer", MockSentenceTransformer)
        lfs = LatentFileSystem(str(latent_fs_directory))
        
        # A query for "quantum mechanics" should rank the physics doc highest,
        # then the other two which are equally dissimilar.
        results = lfs.ls("quantum mechanics", k=3)
        result_names = [p.name for p in results]
        
        assert result_names[0] == "quantum_physics.txt"
        # The order of the other two might vary but they should be last
        assert set(result_names[1:]) == {"roman_history.md", "cake_recipe.txt"}

    def test_ls_with_empty_directory(self, tmp_path: Path, monkeypatch):
        """Tests the behavior with an empty directory."""
        monkeypatch.setattr("Algorithms.Source.latent_dir_nav.SentenceTransformer", MockSentenceTransformer)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        lfs = LatentFileSystem(str(empty_dir))
        results = lfs.ls("any query")
        
        assert lfs.file_paths == []
        assert results == []

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and sentence-transformers are installed:
    #    pip install pytest sentence-transformers
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_latent_dir_nav.py
    
    print("This is a test file. Use 'pytest' to execute it.")