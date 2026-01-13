# UAID: NBX-ALG-00010
# GoldenDAG: c3d4e<seg_34>7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Bloom Event Detector
# Part of the MetaMind and Self-Reflection_Logs Subsystems
#
# Core Principle: Recursive Self-Betterment - understanding the system's own creative expansion.

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
import datetime as dt

class BloomEventDetector:
    """
    Analyzes sequences of DRS vector shards to detect 'Bloom' or 'Hyperbloom'
    events. A Bloom is identified as a statistically significant increase in the
    effective dimensionality or variance of the latent space usage.
    """

    def __init__(self, shard_dir: str, sigma_threshold: float = 3.0, min_shard_history: int = 7):
        """
        Initializes the detector.

        Args:
            shard_dir (str): The directory containing DRS vector shard files (e.g., .npz).
            sigma_threshold (float): The number of standard deviations above the mean
                                     entropy required to trigger a Bloom alert.
            min_shard_history (int): The minimum number of historical shards to analyze
                                     to establish a baseline entropy.
        """
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.is_dir():
            raise FileNotFoundError(f"ERR-FS-006: Shard directory not found at '{self.shard_dir}'")

        self.sigma_threshold = sigma_threshold
        self.min_shard_history = min_shard_history

    def _calculate_shannon_entropy_from_variance(self, principal_components: np.ndarray) -> float:
        """
        Calculates the Shannon entropy of the variance distribution. A higher entropy
        means the variance is spread across more dimensions, indicating a 'bloom'.
        
        Args:
            principal_components (np.ndarray): The singular values from an SVD.

        Returns:
            float: The calculated Shannon entropy.
        """
        # Normalize the variance explained by each component to get a probability distribution
        variance_explained = principal_components**2 / np.sum(principal_components**2)
        
        # Filter out zero probabilities to avoid log(0)
        variance_explained = variance_explained[variance_explained > 0]
        
        # Calculate Shannon entropy
        entropy = -np.sum(variance_explained * np.log2(variance_explained))
        return float(entropy)

    def analyze_shard(self, shard_path: Path) -> Optional[float]:
        """
        Analyzes a single vector shard file and calculates its entropy.

        Args:
            shard_path (Path): Path to the .npz shard file. Expected to contain
                               an array under the key 'vectors'.

        Returns:
            Optional[float]: The entropy of the shard, or None if the shard is invalid.
        """
        try:
            with np.load(shard_path) as data:
                vectors = data['vectors']
            
            if vectors.ndim != 2 or vectors.shape[0] < 2:
                print(f"Warning: Skipping shard '{shard_path.name}' with insufficient data.")
                return None
            
            # Center the data before SVD
            centered_vectors = vectors - np.mean(vectors, axis=0)
            
            # Use SVD to find principal components. We only need the singular values 's'.
            # Truncating to min(shape) for performance on very wide/tall matrices.
            num_components = min(centered_vectors.shape)
            s = np.linalg.svd(centered_vectors, full_matrices=False, compute_uv=False)[:num_components]
            
            return self._calculate_shannon_entropy_from_variance(s)
        except Exception as e:
            print(f"Warning: Failed to process shard '{shard_path.name}'. Reason: {e}")
            return None

    def run_detection(self) -> List[Dict]:
        """
        Runs the detection process over all shards in the directory and returns
        a list of detected bloom events.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents
                        a detected bloom event.
        """
        shard_files = sorted(self.shard_dir.glob("*.npz"))
        if len(shard_files) < self.min_shard_history:
            print(f"Info: Insufficient history ({len(shard_files)} shards). "
                  f"Need at least {self.min_shard_history} to run detection.")
            return []

        print(f"Analyzing {len(shard_files)} DRS vector shards...")
        
        entropies = []
        for shard_file in shard_files:
            entropy = self.analyze_shard(shard_file)
            if entropy is not None:
                entropies.append({"file": str(shard_file), "entropy": entropy})

        if not entropies:
            return []
            
        entropy_values = np.array([e['entropy'] for e in entropies])
        
        # --- Statistical Anomaly Detection ---
        mean_entropy = np.mean(entropy_values)
        std_entropy = np.std(entropy_values)
        alert_threshold = mean_entropy + self.sigma_threshold * std_entropy

        alerts = []
        for entry in entropies:
            if entry['entropy'] > alert_threshold:
                alert = {
                    "event_type": "BLOOM_DETECTED",
                    "UAID": f"NBX-LOG-BLM-{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    "shard_file": entry['file'],
                    "entropy": entry['entropy'],
                    "mean_entropy": mean_entropy,
                    "std_entropy": std_entropy,
                    "threshold": alert_threshold,
                    "sigma_level": (entry['entropy'] - mean_entropy) / std_entropy,
                    "timestamp": dt.datetime.utcnow().isoformat() + "Z"
                }
                alerts.append(alert)
        
        print(f"Detection complete. Found {len(alerts)} potential bloom events.")
        
        if alerts:
            log_path = self.shard_dir.parent / "Self-Reflection_Logs" / "bloom_alerts.jsonl"
            log_path.parent.mkdir(exist_ok=True)
            with log_path.open('a') as f:
                for alert in alerts:
                    f.write(json.dumps(alert) + '\n')
            print(f"Alerts logged to: {log_path}")

        return alerts


if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke MetaMind --run_bloom_detection --shard_dir="/DRS_Engine/shards/"

    print("--- Initiating NeuralBlitz Bloom Event Detector Simulation ---")
    
    # Create a dummy shard directory and populate it with sample data
    sim_shard_dir = Path("./sim_shards")
    sim_shard_dir.mkdir(exist_ok=True)
    
    # Generate some baseline shards (low entropy)
    print("Generating baseline vector shards...")
    for i in range(10):
        # Data concentrated in the first few dimensions
        base_vectors = np.random.randn(1000, 512) * np.array([5.0] * 10 + [0.1] * 502)
        np.savez_compressed(sim_shard_dir / f"shard_{i:02d}.npz", vectors=base_vectors)

    # Generate a "Bloom" shard (high entropy)
    print("Generating a BLOOM vector shard...")
    bloom_vectors = np.random.randn(1000, 512) # Uniform variance across all dimensions
    np.savez_compressed(sim_shard_dir / "shard_10_BLOOM.npz", vectors=bloom_vectors)

    try:
        # Initialize and run the detector
        detector = BloomEventDetector(str(sim_shard_dir), sigma_threshold=2.5)
        detected_events = detector.run_detection()
        
        print("\n--- Detection Report ---")
        if detected_events:
            print(json.dumps(detected_events, indent=2))
        else:
            print("No significant bloom events detected.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the dummy directory and files
        for f in sim_shard_dir.glob("*.npz"):
            f.unlink()
        sim_shard_dir.rmdir()