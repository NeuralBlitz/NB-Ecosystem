# UAID: NBX-ALG-00013
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: DRS Shard PCA Compressor
# Part of the DRS_Engine
#
# Core Principle: Sustainability (ε₅) - managing data footprint for long-term viability.

import numpy as np
from pathlib import Path
from typing import Optional, Dict
import sklearn.decomposition as skd
import pyarrow as pa
import pyarrow.parquet as pq
import datetime as dt

class ShardPCACompressor:
    """
    Compresses high-dimensional DRS vector shards into a lower-dimensional
    representation using Incremental PCA, suitable for long-term storage
    and large-scale analytics.
    """

    def __init__(self, target_dimensions: int = 256, batch_size: int = 4096):
        """
        Initializes the compressor.

        Args:
            target_dimensions (int): The number of principal components to keep.
                                     This will be the dimensionality of the compressed vectors.
            batch_size (int): The number of vectors to process at a time.
                              This allows compression of datasets that are too large
                              to fit into memory.
        """
        if target_dimensions < 1:
            raise ValueError("ERR-INIT-002: target_dimensions must be a positive integer.")
        
        self.target_dimensions = target_dimensions
        self.batch_size = batch_size
        self.ipca = skd.IncrementalPCA(n_components=self.target_dimensions, batch_size=self.batch_size)
        print(f"ShardPCACompressor initialized for {self.target_dimensions}-D output.")

    def compress_shard(self, shard_npz_path: str, out_parquet_path: Optional[str] = None) -> Path:
        """
        Loads a .npz vector shard, fits the IPCA model, transforms the data,
        and saves it to a Parquet file.

        Args:
            shard_npz_path (str): The path to the source .npz file. It is expected
                                  to contain an array under the key 'vectors' and
                                  optionally an array of 'uids'.
            out_parquet_path (Optional[str]): The path for the output Parquet file.
                                              If None, it's saved next to the input.

        Returns:
            Path: The path to the created Parquet file.
        """
        shard_path = Path(shard_npz_path)
        if not shard_path.exists():
            raise FileNotFoundError(f"ERR-FS-008: Input shard file not found at '{shard_path}'")
            
        if out_parquet_path is None:
            out_path = shard_path.parent / f"{shard_path.stem}_compressed_{self.target_dimensions}D.parquet"
        else:
            out_path = Path(out_parquet_path)

        try:
            with np.load(shard_path) as data:
                vectors = data['vectors']
                uids = data.get('uids') # Optional UID array
        except Exception as e:
            raise IOError(f"ERR-READ-002: Failed to load data from shard '{shard_path}'. Reason: {e}")

        print(f"Compressing shard '{shard_path.name}' ({vectors.shape[0]} vectors, {vectors.shape[1]}-D)...")
        
        # Fit and transform the data using Incremental PCA
        # This handles large datasets by processing in batches.
        transformed_vectors = self.ipca.fit_transform(vectors)

        # --- Save to Parquet format for efficient storage and analytics ---
        # Create an Arrow Table. Parquet is highly efficient for columnar data.
        columns = [pa.array(transformed_vectors.astype(np.float16))] # Use float16 for further space saving
        column_names = ['vector_compressed']

        if uids is not None and len(uids) == len(vectors):
            columns.insert(0, pa.array(uids))
            column_names.insert(0, 'uid')
            
        table = pa.Table.from_arrays(columns, names=column_names)

        # Add metadata, including a GoldenDAG of the PCA model itself for reproducibility
        pca_model_hash = self._get_model_hash()
        metadata = {
            'source_file': str(shard_path.name),
            'source_dimensions': str(vectors.shape[1]),
            'target_dimensions': str(self.target_dimensions),
            'compression_date': dt.datetime.utcnow().isoformat() + "Z",
            'ipca_model_goldendag': pca_model_hash,
            'neuralblitz_codex_version': 'C-OMNI-ALGO-SHARD_COMPRESSOR-v1.0-SEALED'
        }
        
        # Add metadata to the schema
        final_schema = table.schema.with_metadata(metadata)
        
        pq.write_table(table.cast(final_schema), out_path, compression='ZSTD')

        print(f"Compressed shard saved to: {out_path}")
        return out_path
        
    def _get_model_hash(self) -> str:
        """Generates a deterministic hash of the trained IPCA model components."""
        from hashlib import sha256
        hasher = sha256()
        # Hash the principal components to create a unique fingerprint for the model state
        hasher.update(self.ipca.components_.tobytes())
        return hasher.hexdigest()

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke DRS_Engine.maintenance --task=compress_shard --path="/DRS_Engine/shards/shard_20250728.npz"

    print("--- Initiating NeuralBlitz Shard PCA Compressor Simulation ---")

    # Create a dummy high-dimensional shard file
    sim_shard_dir = Path("./sim_shards_for_pca")
    sim_shard_dir.mkdir(exist_ok=True)
    
    source_dims = 8192
    num_vectors = 5000
    
    dummy_vectors = np.random.randn(num_vectors, source_dims).astype(np.float32)
    # Create some dummy UIDs as well
    dummy_uids = [f"NBX-VEC-{i:05d}" for i in range(num_vectors)]

    source_path = sim_shard_dir / "high_dim_shard_20250728.npz"
    np.savez_compressed(source_path, vectors=dummy_vectors, uids=dummy_uids)

    print(f"\nCreated dummy shard '{source_path.name}' with {num_vectors} vectors of dimension {source_dims}.")
    original_size_mb = source_path.stat().st_size / (1024 * 1024)
    print(f"Original file size: {original_size_mb:.2f} MB")

    try:
        compressor = ShardPCACompressor(target_dimensions=128)
        
        compressed_path = compressor.compress_shard(str(source_path))
        
        compressed_size_mb = compressed_path.stat().st_size / (1024 * 1024)
        print(f"Compressed file size: {compressed_size_mb:.2f} MB")
        
        reduction_factor = original_size_mb / compressed_size_mb
        print(f"\nAchieved a compression factor of ~{reduction_factor:.1f}x")

        # You can inspect the .parquet file with a suitable library like pandas:
        # import pandas as pd
        # df = pd.read_parquet(compressed_path)
        # print(df.head())
        
    except Exception as e:
        print(f"\nAn error occurred during the simulation: {e}")
    finally:
        # Clean up the dummy directory and files
        if 'source_path' in locals() and source_path.exists():
            source_path.unlink()
        if 'compressed_path' in locals() and compressed_path.exists():
            compressed_path.unlink()
        if 'sim_shard_dir' in locals() and sim_shard_dir.exists():
            sim_shard_dir.rmdir()