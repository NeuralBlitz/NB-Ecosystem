# UAID: NBX-TST-ALG-00020
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Distributed Shard Compactor (distributed_shard_compactor.py, NBX-ALG-00020)
#
# Core Principle: Sustainability (ε₅) - validating our data lifecycle management tools.

import pytest
import numpy as np
import asyncio
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.distributed_shard_compactor import DistributedShardCompactor

# Mark all tests in this file as asyncio tests
pytestmark = pytest.mark.asyncio

# --- Test Fixtures ---

@pytest.fixture
def mock_shard_directory(tmp_path: Path) -> Dict[str, Path]:
    """Creates a mock DRS shard directory structure for testing."""
    source_dir = tmp_path / "drs_shards_raw"
    target_dir = tmp_path / "drs_shards_compacted"
    source_dir.mkdir()
    target_dir.mkdir()

    # --- Create sample shards for a specific date ---
    # These should be compacted.
    date_str = "20250728"
    # Shard 1 for the date
    vectors1 = np.random.randn(100, 64)
    np.savez_compressed(source_dir / f"shard_{date_str}_part_00.npz", vectors=vectors1)
    # Shard 2 for the date
    vectors2 = np.random.randn(150, 64)
    np.savez_compressed(source_dir / f"shard_{date_str}_part_01.npz", vectors=vectors2)
    
    # --- Create a sample shard for a different date ---
    # This should be ignored by the compactor.
    other_date_str = "20250727"
    vectors3 = np.random.randn(50, 64)
    np.savez_compressed(source_dir / f"shard_{other_date_str}_part_00.npz", vectors=vectors3)
    
    return {"source": source_dir, "target": target_dir}

@pytest.fixture
def corrupted_shard_directory(mock_shard_directory: Dict[str, Path]) -> Dict[str, Path]:
    """Adds a corrupted file to the mock shard directory."""
    source_dir = mock_shard_directory["source"]
    date_str = "20250728"
    # This file is not a valid .npz file
    (source_dir / f"shard_{date_str}_part_02_corrupted.npz").write_text("this is not a numpy file")
    return mock_shard_directory


# --- Test Cases ---

class TestDistributedShardCompactor:

    def test_initialization_success(self, mock_shard_directory: Dict[str, Path]):
        """Tests successful initialization."""
        compactor = DistributedShardCompactor(
            str(mock_shard_directory["source"]),
            str(mock_shard_directory["target"])
        )
        assert compactor.source_dir == mock_shard_directory["source"]
        assert compactor.target_dir == mock_shard_directory["target"]

    def test_initialization_source_not_found(self, tmp_path: Path):
        """Tests for FileNotFoundError if the source directory is missing."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-010"):
            DistributedShardCompactor("non_existent_source", str(tmp_path))

    async def test_discover_shards_by_date(self, mock_shard_directory: Dict[str, Path]):
        """Unit tests the shard discovery logic."""
        compactor = DistributedShardCompactor(
            str(mock_shard_directory["source"]),
            str(mock_shard_directory["target"])
        )
        shards_to_compact = compactor._discover_shards("20250728")
        
        assert len(shards_to_compact) == 2
        shard_names = {p.name for p in shards_to_compact}
        assert "shard_20250728_part_00.npz" in shard_names
        assert "shard_20250728_part_01.npz" in shard_names
        assert "shard_20250727_part_00.npz" not in shard_names

    async def test_run_compaction_successful(self, mock_shard_directory: Dict[str, Path]):
        """The main integration test for a successful compaction run."""
        source_dir = mock_shard_directory["source"]
        target_dir = mock_shard_directory["target"]
        
        compactor = DistributedShardCompactor(str(source_dir), str(target_dir))
        report = await compactor.compact_date("20250728")

        # --- Assertions on the Report ---
        assert report["status"] == "SUCCESS"
        assert report["date_processed"] == "20250728"
        assert len(report["source_shards_processed"]) == 2
        assert report["total_vectors_compacted"] == 250 # 100 + 150
        assert not report["errors"]

        # --- Assertions on the File System ---
        # 1. The new compacted file should exist
        compacted_file_path = Path(report["compacted_shard_path"])
        assert compacted_file_path.exists()
        assert compacted_file_path.parent == target_dir
        
        # 2. The original source files for that date should be gone
        assert not (source_dir / "shard_20250728_part_00.npz").exists()
        assert not (source_dir / "shard_20250728_part_01.npz").exists()

        # 3. The shard from the other date should still be there
        assert (source_dir / "shard_20250727_part_00.npz").exists()

        # 4. Check the content of the compacted file
        with np.load(compacted_file_path) as data:
            compacted_vectors = data['vectors']
            assert compacted_vectors.shape == (250, 64)

    async def test_run_compaction_no_shards_found(self, mock_shard_directory: Dict[str, Path]):
        """Tests the case where no shards match the date glob."""
        compactor = DistributedShardCompactor(
            str(mock_shard_directory["source"]),
            str(mock_shard_directory["target"])
        )
        report = await compactor.compact_date("20250101") # A date with no shards

        assert report["status"] == "NO_SHARDS_FOUND"
        assert report["total_vectors_compacted"] == 0
        assert not report["errors"]
        # Ensure no new file was created
        assert not list(mock_shard_directory["target"].glob("*.npz"))

    async def test_run_compaction_handles_corrupted_shard(self, corrupted_shard_directory: Dict[str, Path]):
        """Tests that the process is resilient to a single corrupted file."""
        source_dir = corrupted_shard_directory["source"]
        target_dir = corrupted_shard_directory["target"]
        
        compactor = DistributedShardCompactor(str(source_dir), str(target_dir))
        report = await compactor.compact_date("20250728")

        # --- Assertions on the Report ---
        assert report["status"] == "PARTIAL_FAILURE"
        assert len(report["errors"]) == 1
        assert "shard_20250728_part_02_corrupted.npz" in report["errors"][0]
        # It should have still processed the 2 valid shards
        assert report["total_vectors_compacted"] == 250

        # --- Assertions on the File System ---
        # 1. The compacted file (from the valid shards) should still be created
        assert Path(report["compacted_shard_path"]).exists()
        
        # 2. CRITICAL: The cleanup should NOT have run because of the error.
        #    The original valid shards should still exist.
        assert (source_dir / "shard_20250728_part_00.npz").exists()
        assert (source_dir / "shard_20250728_part_01.npz").exists()
        
        # 3. The corrupted file should still be there for inspection.
        assert (source_dir / "shard_20250728_part_02_corrupted.npz").exists()

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest, pytest-asyncio, and numpy are installed:
    #    pip install pytest pytest-asyncio numpy
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_distributed_shard_compactor.py
    
    print("This is a test file. Use 'pytest' to execute it.")