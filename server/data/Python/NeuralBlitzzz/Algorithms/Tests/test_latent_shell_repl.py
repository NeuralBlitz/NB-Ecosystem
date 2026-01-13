# UAID: NBX-TST-ALG-00018
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Latent Shell REPL (latent_shell_repl.py, NBX-ALG-00018)
#
# Core Principle: Radical Transparency (ε₂) - validating our deep introspection and navigation tools.

import pytest
import numpy as np
from typing import Dict, Any, List, Optional

# Import the class we are testing
from Algorithms.Source.latent_shell_repl import LatentShellREPL

# --- Mocking Dependencies ---

class MockDRSClient:
    """A mock of the DRS Engine client for deterministic testing."""
    def __init__(self):
        # A simple 3D vector space for concepts
        self.vectors = {
            "/": np.array([0.0, 0.0, 0.0]),
            "science": np.array([1.0, 0.0, 0.0]),
            "art": np.array([0.0, 1.0, 0.0]),
            "physics": np.array([0.9, 0.1, 0.0]), # Close to science
            "sculpture": np.array([0.1, 0.9, 0.0]), # Close to art
            "philosophy": np.array([0.5, 0.5, 0.0]), # Between science and art
        }
        # Normalize all vectors
        for key, vec in self.vectors.items():
            norm = np.linalg.norm(vec)
            if norm > 0:
                self.vectors[key] = vec / norm

    def concept_exists(self, path: str) -> bool:
        return path in self.vectors

    def get_vector(self, path: str) -> Optional[np.ndarray]:
        return self.vectors.get(path)

    def get_concept_details(self, path: str) -> Optional[Dict[str, Any]]:
        if self.concept_exists(path):
            return {
                "UAID": f"NBX-CONCEPT-{path.upper()}",
                "vector": self.vectors[path].tolist(),
                "description": f"A sample concept node for '{path}'."
            }
        return None

    def find_nearest_neighbors(self, vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Finds k nearest concepts to a given vector based on cosine similarity."""
        similarities = []
        for name, vec in self.vectors.items():
            # Cosine similarity is dot product for normalized vectors
            sim = np.dot(vector, vec)
            similarities.append({"path": name, "similarity": sim})
        
        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:k]

# --- Test Fixtures ---

@pytest.fixture
def repl_instance() -> LatentShellREPL:
    """Provides a LatentShellREPL instance initialized with the mock DRS client."""
    mock_drs = MockDRSClient()
    return LatentShellREPL(drs_client=mock_drs)

# --- Test Cases ---

class TestLatentShellREPL:

    def test_initialization(self, repl_instance: LatentShellREPL):
        """Tests that the REPL starts at the root."""
        assert repl_instance.current_path == "/"
        np.testing.assert_array_equal(repl_instance.current_vector, np.array([0.0, 0.0, 0.0]))

    @pytest.mark.parametrize("line, expected_cmd, expected_args", [
        ("ls", "ls", []),
        ("  cd   science  ", "cd", ["science"]),
        ("cat /art/sculpture", "cat", ["/art/sculpture"]),
        ("help", "help", []),
        ("exit", "exit", []),
    ])
    def test_command_parsing(self, repl_instance: LatentShellREPL, line, expected_cmd, expected_args):
        """Tests the internal `_parse_command` method."""
        cmd, args = repl_instance._parse_command(line)
        assert cmd == expected_cmd
        assert args == expected_args

    def test_pwd_command(self, repl_instance: LatentShellREPL, capsys):
        """Tests the `pwd` (print working directory) command."""
        repl_instance.do_pwd("")
        captured = capsys.readouterr()
        assert captured.out.strip() == "/"

    def test_cd_command_success(self, repl_instance: LatentShellREPL):
        """Tests successfully changing the current conceptual directory."""
        repl_instance.do_cd("science")
        assert repl_instance.current_path == "science"
        # Check if the vector was updated correctly
        np.testing.assert_allclose(repl_instance.current_vector, repl_instance.drs.get_vector("science"))

    def test_cd_command_failure(self, repl_instance: LatentShellREPL, capsys):
        """Tests trying to `cd` to a non-existent concept."""
        initial_path = repl_instance.current_path
        repl_instance.do_cd("non_existent_concept")
        captured = capsys.readouterr()
        
        # Path should not change
        assert repl_instance.current_path == initial_path
        # An error message should be printed
        assert "Concept 'non_existent_concept' not found" in captured.out

    def test_ls_command(self, repl_instance: LatentShellREPL, capsys):
        """Tests the `ls` (list semantic neighbors) command."""
        repl_instance.do_cd("science") # Move to a known concept
        repl_instance.do_ls("")
        captured = capsys.readouterr()
        
        # The output should be a formatted table
        assert "Path" in captured.out
        assert "Similarity" in captured.out
        # 'physics' is the closest neighbor to 'science' in our mock DRS
        assert "physics" in captured.out
        # The order should be correct, with science itself being the most similar
        lines = captured.out.strip().split('\\n')
        assert "science" in lines[2] # Header is 2 lines
        assert "physics" in lines[3]

    def test_cat_command_success(self, repl_instance: LatentShellREPL, capsys):
        """Tests the `cat` (show concept details) command."""
        repl_instance.do_cat("art")
        captured = capsys.readouterr()
        
        assert "Details for concept: art" in captured.out
        assert "UAID" in captured.out
        assert "NBX-CONCEPT-ART" in captured.out

    def test_cat_command_failure(self, repl_instance: LatentShellREPL, capsys):
        """Tests `cat` on a non-existent concept."""
        repl_instance.do_cat("mythology")
        captured = capsys.readouterr()
        assert "Concept 'mythology' not found" in captured.out

    def test_exit_command(self, repl_instance: LatentShellREPL):
        """Tests that the `exit` command signals the loop to terminate."""
        assert repl_instance.do_exit("") is True

    def test_default_handler_for_unknown_command(self, repl_instance: LatentShellREPL, capsys):
        """Tests the fallback for an unknown command."""
        repl_instance.default("unknown_command and args")
        captured = capsys.readouterr()
        assert "Unknown command: 'unknown_command'" in captured.out


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and numpy are installed:
    #    pip install pytest numpy
    # 3. (Optional but recommended) Install sentence-transformers if you want to run the
    #    main algorithm file, though it is not needed for this test file.
    # 4. Run the tests:
    #    pytest Algorithms/Tests/test_latent_shell_repl.py
    
    print("This is a test file. Use 'pytest' to execute it.")