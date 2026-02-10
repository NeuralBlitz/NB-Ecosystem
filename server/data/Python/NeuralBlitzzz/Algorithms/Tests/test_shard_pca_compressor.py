# UAID: NBX-TST-ALG-00013
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Shard PCA Compressor (shard_pca_compressor.py, NBX-ALG-00013)
#
# Core Principle: Sustainability (ε₅) - validating our data archival and compression tools.

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.shard_pca_compressor import ShardPCACompressor

# --- Test Fixtures ---

@pytest.fixture
def sample_shard_file(tmp_path: Path) -> Path:
    """Creates a sample vector shard .npz file for testing."""
    # Create a synthetic dataset where variance is concentrated in the first few dimensions.
    # This makes the effect of PCA more predictable and pronounced.
    num_samples = 1000
    original_dim = 128
    
    # Create a random covariance matrix with decaying eigenvalues
    rng = np.random.default_rng(seed=42)
    random_matrix = rng.random((original_dim, original_dim))
    eigvals = np.exp(-np.arange(original_dim) * 0.5)
    covariance_matrix = random_matrix.T @ np.diag(eigvals) @ random_matrix
    
    # Generate data from this distribution
    vectors = rngmultivariate_normal(np.zeros(original_dim), covariance_matrix, size=num_samples)
    
    shard_path = tmp_path / "sample_shard.npz"
    np.savez_compressed(shard_path, vectors=vectors)
    return shard_path

# --- Test Cases ---

class TestShardPCACompressor:

    def test_initialization_success(self, sample_shard_file: Path):
        """Tests that the compressor initializes correctly with a valid file."""
        compressor = ShardPCACompressor(str(sample_shard_file))
        assert compressor.shard_path == sample_shard_file
        assert compressor.vectors.shape == (1000, 128)

    def test_initialization_file_not_found(self):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-008"):
            ShardPCACompressor("non_existent_shard.npz")

    def test_initialization_bad_shard_key(self, tmp_path: Path):
        """Tests for ValueError if the .npz file has the wrong key."""
        bad_key_path = tmp_path / "bad_key.npz"
        np.savez_compressed(bad_key_path, some_other_key=np.random.randn(10, 10))
        with pytest.raises(ValueError, match="ERR-SCHEMA-003"):
            ShardPCACompressor(str(bad_key_path))

    def test_compress_to_target_dimension(self, sample_shard_file: Path):
        """
        Tests compression to a specific target dimension.
        """
        target_dim = 32
        compressor = ShardPCACompressor(str(sample_shard_file))
        compressed_vectors, report = compressor.compress(target_dimension=target_dim)

        assert compressed_vectors.shape == (1000, target_dim)
        assert report["final_dimension"] == target_dim
        assert report["original_dimension"] == 128
        assert "variance_preserved_ratio" in report
        
    def test_compress_to_variance_ratio(self, sample_shard_file: Path):
        """
        Tests compression to a target variance preservation ratio.
        """
        variance_to_preserve = 0.95
        compressor = ShardPCACompressor(str(sample_shard_file))
        compressed_vectors, report = compressor.compress(variance_ratio_to_preserve=variance_to_preserve)

        # The final dimension should be less than the original
        assert report["final_dimension"] < 128
        # The preserved variance should be at least the target
        assert report["variance_preserved_ratio"] >= variance_to_preserve
        assert compressed_vectors.shape == (1000, report["final_dimension"])

    def test_compress_raises_error_if_no_target_specified(self, sample_shard_file: Path):
        """
        Tests that a ValueError is raised if neither target_dimension nor
        variance_ratio_to_preserve is provided.
        """
        compressor = ShardPCACompressor(str(sample_shard_file))
        with pytest.raises(ValueError, match="ERR-PARAM-001"):
            compressor.compress()

    def test_compress_handles_insufficient_data(self, tmp_path: Path):
        """
        Tests the behavior when the number of samples is less than the number of dimensions,
        which limits the number of principal components.
        """
        # 10 samples, 20 dimensions
        vectors = np.random.randn(10, 20)
        shard_path = tmp_path / "insufficient_data.npz"
        np.savez_compressed(shard_path, vectors=vectors)

        compressor = ShardPCACompressor(str(shard_path))
        # Try to compress to 15 dimensions, which is impossible.
        compressed_vectors, report = compressor.compress(target_dimension=15)

        # PCA can only find N-1 components for N samples.
        # The result should be capped at 9 dimensions.
        assert report["final_dimension"] == 9
        assert compressed_vectors.shape == (10, 9)
        assert "Warning: Target dimension was capped" in report["notes"][0]
        
    def test_reconstruction(self, sample_shard_file: Path):
        """
        Tests the reconstruction process to ensure the inverse transform can be applied.
        The reconstruction won't be perfect, but its shape and overall structure
        should be correct.
        """
        target_dim = 16
        compressor = ShardPCACompressor(str(sample_shard_file))
        compressed_vectors, _ = compressor.compress(target_dimension=target_dim)

        # Reconstruct the data back to the original dimension
        reconstructed_vectors = compressor.reconstruct(compressed_vectors)
        
        assert reconstructed_vectors.shape == compressor.vectors.shape
        
        # The reconstructed data should have a higher MSE from the original than from itself
        mse = np.mean((compressor.vectors - reconstructed_vectors)**2)
        assert mse > 0
        
        # Reconstructing already reconstructed data should change it very little
        re_reconstructed = compressor.reconstruct(compressor.compress(target_dimension=target_dim)[0])
        mse_recon = np.mean((reconstructed_vectors - re_reconstructed)**2)
        assert mse_recon < 1e-10

    def test_save_functionality(self, sample_shard_file: Path, tmp_path: Path):
        """Tests that the save method creates the correct files."""
        target_dim = 8
        compressor = ShardPCACompressor(str(sample_shard_file))
        compressed_vectors, _ = compressor.compress(target_dimension=target_dim)
        
        base_name = tmp_path / "compressed_output"
        
        paths = compressor.save(str(base_name))
        
        assert "data_path" in paths
        assert "metadata_path" in paths
        
        data_path = Path(paths["data_path"])
        metadata_path = Path(paths["metadata_path"])
        
        assert data_path.exists()
        assert metadata_path.exists()
        
        # Check metadata content
        metadata = json.loads(metadata_path.read_text())
        assert metadata["original_dimension"] == 128
        assert metadata["compressed_dimension"] == 8
        assert "mean_vector" in metadata
        assert "principal_components" in metadata
        
        # Check if saved data can be loaded
        loaded_data = np.load(data_path)['compressed_vectors']
        assert loaded_data.shape == compressed_vectors.shape

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and numpy are installed:
    #    pip install pytest numpy
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_shard_pca_compressor.py
    
    print("This is a test file. Use 'pytest' to execute it.")