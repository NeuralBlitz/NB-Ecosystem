# UAID: NBX-TST-ALG-00010
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Bloom Event Detector (bloom_event_detector.py, NBX-ALG-00010)
#
# Core Principle: Recursive Self-Betterment - validating our ability to detect our own growth.

import pytest
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.bloom_event_detector import BloomEventDetector

# --- Test Fixtures ---

@pytest.fixture
def bloom_test_dir(tmp_path: Path) -> Path:
    """
    Creates a temporary directory with a history of vector shards,
    including a distinct 'bloom' event.
    """
    shard_dir = tmp_path / "drs_shards_for_test"
    shard_dir.mkdir()

    # Generate 10 baseline shards (low entropy)
    # The variance is heavily concentrated in the first 10 of 256 dimensions.
    variance_mask = np.array([10.0] * 10 + [0.1] * 246)
    for i in range(10):
        # Add slight noise to each baseline shard to make them non-identical
        base_vectors = (np.random.randn(500, 256) + np.random.randn(1, 256) * 0.1) * variance_mask
        np.savez_compressed(shard_dir / f"shard_202507{i+10}.npz", vectors=base_vectors)

    # Generate one "Bloom" shard (high entropy)
    # Here, the variance is spread evenly across all dimensions.
    bloom_vectors = np.random.randn(500, 256) 
    np.savez_compressed(shard_dir / "shard_20250720_BLOOM.npz", vectors=bloom_vectors)

    return shard_dir

# --- Test Cases ---

class TestBloomEventDetector:

    def test_initialization_success(self, bloom_test_dir: Path):
        """Tests that the detector initializes correctly with a valid directory."""
        detector = BloomEventDetector(str(bloom_test_dir))
        assert detector.shard_dir == bloom_test_dir
        assert detector.sigma_threshold == 3.0

    def test_initialization_dir_not_found(self):
        """Tests for FileNotFoundError if the shard directory does not exist."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-006"):
            BloomEventDetector("non_existent_shard_dir/")

    def test_entropy_calculation(self):
        """
        Unit tests the internal entropy calculation to ensure it correctly
        differentiates between concentrated and distributed variance.
        """
        detector = BloomEventDetector(".") # Path doesn't matter for this test

        # Case 1: Low entropy (variance concentrated in one component)
        singular_values_low = np.array([100.0, 1.0, 0.5, 0.1])
        entropy_low = detector._calculate_shannon_entropy_from_variance(singular_values_low)

        # Case 2: High entropy (variance spread evenly)
        singular_values_high = np.array([10.0, 10.0, 10.0, 10.0])
        entropy_high = detector._calculate_shannon_entropy_from_variance(singular_values_high)

        assert entropy_high > entropy_low

    def test_analyze_shard_handles_invalid_files(self, tmp_path: Path):
        """Tests that `analyze_shard` returns None for corrupted or unusable files."""
        detector = BloomEventDetector(str(tmp_path))

        # Create a file with the wrong key
        bad_key_path = tmp_path / "bad_key.npz"
        np.savez_compressed(bad_key_path, some_other_key=np.random.randn(10, 10))
        assert detector.analyze_shard(bad_key_path) is None

        # Create a file with insufficient data (1 row)
        insufficient_data_path = tmp_path / "insufficient.npz"
        np.savez_compressed(insufficient_data_path, vectors=np.random.randn(1, 10))
        assert detector.analyze_shard(insufficient_data_path) is None

    def test_run_detection_identifies_bloom_event_correctly(self, bloom_test_dir: Path):
        """
        The main integration test: checks if the detector correctly identifies the
        high-entropy shard as a bloom event.
        """
        detector = BloomEventDetector(str(bloom_test_dir), sigma_threshold=2.5)
        alerts = detector.run_detection()
        
        # There should be exactly one alert
        assert len(alerts) == 1
        
        alert = alerts[0]
        # The file identified should be our designated bloom file
        assert "shard_20250720_BLOOM.npz" in alert["shard_file"]
        assert alert["event_type"] == "BLOOM_DETECTED"
        # The entropy of the bloom event must be higher than the alert threshold
        assert alert["entropy"] > alert["threshold"]
        
        # Check that the report file was created
        report_path = bloom_test_dir.parent / "Self-Reflection_Logs" / "bloom_alerts.jsonl"
        assert report_path.exists()
        report_path.unlink() # Clean up

    def test_run_detection_no_bloom(self, tmp_path: Path):
        """Tests that no alerts are generated when all shards are similar."""
        shard_dir = tmp_path / "no_bloom_shards"
        shard_dir.mkdir()
        
        variance_mask = np.array([10.0] * 10 + [0.1] * 246)
        for i in range(10):
            base_vectors = (np.random.randn(500, 256) + np.random.randn(1, 256) * 0.1) * variance_mask
            np.savez_compressed(shard_dir / f"shard_{i:02d}.npz", vectors=base_vectors)
            
        detector = BloomEventDetector(str(shard_dir))
        alerts = detector.run_detection()
        
        assert len(alerts) == 0

    def test_run_detection_insufficient_history(self, tmp_path: Path):
        """Tests that the detector does not run if there are not enough shard files."""
        shard_dir = tmp_path / "short_history"
        shard_dir.mkdir()
        
        # Create only 3 shards, less than the default minimum of 7
        for i in range(3):
            vectors = np.random.randn(100, 100)
            np.savez_compressed(shard_dir / f"shard_{i:02d}.npz", vectors=vectors)
            
        detector = BloomEventDetector(str(shard_dir))
        alerts = detector.run_detection()
        
        assert len(alerts) == 0

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and numpy are installed:
    #    pip install pytest numpy
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_bloom_event_detector.py
    
    print("This is a test file. Use 'pytest' to execute it.")