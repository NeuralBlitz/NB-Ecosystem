# UAID: NBX-ALG-00020
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Distributed DRS Shard Compactor
# Part of the DRS_Engine (Maintenance Daemon)
#
# Core Principle: Sustainability (ε₅) & Scalability - Managing massive data stores efficiently.

import numpy as np
import json
from pathlib import Path
import shutil
import datetime as dt
from typing import List, Dict

# NeuralBlitz uses Ray for distributed computation.
# For this standalone script, we will mock the Ray interface if it's not installed.
try:
    import ray
    import faiss # FAISS is used for efficient K-Means
except ImportError:
    print("Warning: 'ray' or 'faiss-cpu' not installed. Using mock objects for demonstration.")
    # --- Mock Ray and FAISS for standalone execution ---
    class MockRay:
        def init(self, ignore_reinit_error=False): pass
        def shutdown(self): pass
        def remote(self, fn): return fn # The function itself is returned
        def get(self, object_refs): return [ref() for ref in object_refs] # Call functions directly
    ray = MockRay()
    # Mocking FAISS is more complex, we will assume it exists or use a simpler algorithm
    class MockFAISS:
        def kmeans(self, d, x, k, gpu=False):
            print(f"[MockFAISS] Clustering {x.shape[0]} vectors into {k} centroids.")
            from sklearn.cluster import MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(x)
            return kmeans.cluster_centers_, kmeans.predict(x)
    faiss = MockFAISS()
    # --- End Mocks ---

@ray.remote
def load_vectors_from_shard(shard_path: str) -> np.ndarray:
    """A Ray remote task to load vectors from a single .npz shard file."""
    try:
        with np.load(shard_path) as data:
            # Assumes vectors are stored under the 'vectors' key
            return data['vectors']
    except Exception as e:
        print(f"Error loading shard {shard_path}: {e}")
        return np.array([])

@ray.remote
def save_compacted_shard(vectors: np.ndarray, uids: np.ndarray, output_path: str):
    """A Ray remote task to save a new, compacted shard."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, vectors=vectors, uids=uids)
    return output_path

class DistributedShardCompactor:
    """
    Merges and re-partitions multiple small DRS shards into a few large,
    coherent shards using distributed K-Means clustering.
    """
    def __init__(self, source_dir: str, target_dir: str, num_target_shards: int):
        """
        Initializes the compactor.

        Args:
            source_dir (str): Directory of input shards to be compacted.
            target_dir (str): Directory where the new, compacted shards will be saved.
            num_target_shards (int): The number of output shards to create.
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.num_target_shards = num_target_shards
        
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"ERR-FS-012: Source shard directory not found: '{source_dir}'")
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        ray.init(ignore_reinit_error=True)

    def run_compaction(self) -> Dict:
        """
        Executes the full distributed compaction pipeline.
        """
        start_time = dt.datetime.utcnow()
        print(f"--- Starting Distributed Shard Compaction at {start_time.isoformat()}Z ---")

        # --- Phase 1: Discover and Load Shards in Parallel ---
        source_paths = sorted([str(p) for p in self.source_dir.glob("*.npz")])
        if not source_paths:
            print("No shards found to compact. Exiting.")
            return {"status": "NO_OP", "message": "No source shards found."}
        
        print(f"Discovered {len(source_paths)} shards. Loading vectors in parallel...")
        vector_futures = [load_vectors_from_shard.remote(p) for p in source_paths]
        all_vectors_list = ray.get(vector_futures)
        
        # Concatenate all vectors into a single massive array
        all_vectors = np.vstack([v for v in all_vectors_list if v.size > 0])
        total_vectors, dims = all_vectors.shape
        print(f"Loaded a total of {total_vectors} vectors of dimension {dims}.")
        
        # We assume UIDs would be loaded similarly and concatenated
        all_uids = np.arange(total_vectors) # Placeholder for real UIDs

        # --- Phase 2: Distributed K-Means Clustering ---
        # This step re-partitions the vectors into semantically coherent groups.
        print(f"Clustering {total_vectors} vectors into {self.num_target_shards} new shards using K-Means...")
        
        # FAISS is highly optimized for this. The 'gpu' flag would be used in production.
        centroids, assignments = faiss.kmeans(d=dims, x=all_vectors, k=self.num_target_shards, gpu=False)
        print("Clustering complete.")

        # --- Phase 3: Save New Shards in Parallel ---
        print(f"Partitioning and saving {self.num_target_shards} new compacted shards...")
        save_futures = []
        for k in range(self.num_target_shards):
            # Select all vectors and UIDs assigned to this cluster
            indices_k = np.where(assignments == k)[0]
            if len(indices_k) > 0:
                vectors_k = all_vectors[indices_k]
                uids_k = all_uids[indices_k]
                output_path = self.target_dir / f"compacted_shard_{k:03d}.npz"
                save_futures.append(save_compacted_shard.remote(vectors_k, uids_k, str(output_path)))
        
        # Wait for all save operations to complete
        saved_paths = ray.get(save_futures)
        print("All new shards have been saved.")

        end_time = dt.datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # --- Phase 4: Generate Report ---
        report = {
            "UAID": f"NBX-LOG-COMPACT-{end_time.strftime('%Y%m%d%H%M%S')}",
            "status": "SUCCESS",
            "start_time": start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
            "duration_sec": duration,
            "source_shards_processed": len(source_paths),
            "total_vectors_processed": total_vectors,
            "target_shards_created": len(saved_paths),
            "output_directory": str(self.target_dir)
        }
        
        report_path = self.target_dir / "_compaction_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        
        ray.shutdown()
        return report

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke DRS_Engine.maintenance --task=compact_shards 
    #                --source_dir="/DRS_Engine/shards/daily" 
    #                --target_dir="/DRS_Engine/shards/weekly_compacted" --k=4

    print("--- Initiating NeuralBlitz Distributed Shard Compactor Simulation ---")

    # --- Setup a dummy environment ---
    source_dir_path = Path("./sim_shards_to_compact")
    target_dir_path = Path("./sim_shards_compacted")
    source_dir_path.mkdir(exist_ok=True)
    
    print("Generating 10 small, fragmented source shards...")
    for i in range(10):
        # Create shards with random data
        num_vecs = np.random.randint(500, 1000)
        dims = 128
        vectors = np.random.randn(num_vecs, dims).astype(np.float32)
        np.savez_compressed(source_dir_path / f"source_shard_{i:02d}.npz", vectors=vectors)

    try:
        # Initialize and run the compactor
        compactor = DistributedShardCompactor(
            source_dir=str(source_dir_path),
            target_dir=str(target_dir_path),
            num_target_shards=2 # Consolidate 10 shards into 2
        )
        final_report = compactor.run_compaction()

        print("\n--- Compaction Complete ---")
        print(json.dumps(final_report, indent=2))
        
    except Exception as e:
        print(f"\nAn error occurred during the simulation: {e}")
    finally:
        # --- Clean up dummy directories ---
        shutil.rmtree(source_dir_path, ignore_errors=True)
        shutil.rmtree(target_dir_path, ignore_errors=True)