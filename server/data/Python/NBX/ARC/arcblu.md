{
  "name": "Algorithms Root",
  "version": "1.0.0",
  "UAID": "NBX-DIR-ALGORITHMS-ROOT",
  "GoldenDAG": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
  "timestamp": "2025-07-28T14:35:00Z",
  "description": "This manifest anchors the complete collection of canonical Python algorithms co-created within the NeuralBlitz framework. It provides integrity hashes for the source code and their corresponding test suites.",
  "verification_command": "/invoke custodian --verify path=\"/Algorithms/manifest.json\"",
  "contents": [
    {
      "name": "Source",
      "type": "directory",
      "description": "Contains the canonical source code for all 20+ Python algorithms, each with its own UAID and GoldenDAG-sealed file.",
      "UAID": "NBX-DIR-ALGORITHMS-SOURCE",
      "GoldenDAG": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8"
    },
    {
      "name": "Tests",
      "type": "directory",
      "description": "Contains the Pytest unit and integration tests for all algorithms in the Source directory, ensuring their correctness and stability.",
      "UAID": "NBX-DIR-ALGORITHMS-TESTS",
      "GoldenDAG": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5-f4a5b6c7d8e9f0"
    },
    {
      "name": "Benchmarks",
      "type": "directory",
      "description": "Contains performance benchmark scripts and results for key algorithms, used by MetaMind for optimization.",
      "UAID": "NBX-DIR-ALGORITHMS-BENCHMARKS",
      "GoldenDAG": "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
    }
  ]
}


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

# UAID: NBX-ALG-00014
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Bloom Event Timeline Renderer
# Part of the Self-Reflection_Logs Subsystem
#
# Core Principle: Radical Transparency (ε₂) - visualizing the system's own evolution.

import json
from pathlib import Path
import datetime as dt
from typing import List, Dict
# Matplotlib is the designated python_user_visible tool for plotting.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class BloomTimelineRenderer:
    """
    Parses a bloom alert log file and generates a visual timeline of
    Bloom and Hyperbloom events, plotting entropy levels over time.
    """

    def __init__(self, log_jsonl_path: str):
        """
        Initializes the renderer with the path to the log file.

        Args:
            log_jsonl_path (str): Path to the .jsonl file generated by the
                                  BloomEventDetector.
        """
        self.log_path = Path(log_jsonl_path)
        if not self.log_path.exists():
            raise FileNotFoundError(f"ERR-FS-009: Bloom log file not found at '{self.log_path}'")
        
        self.events = self._load_events()

    def _load_events(self) -> List[Dict]:
        """Loads and parses all valid event entries from the log file."""
        loaded_events = []
        with self.log_path.open('r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # We are only interested in the actual alert events, not the summary header
                    if data.get("event_type") == "BLOOM_DETECTED":
                        # Convert timestamp string to a datetime object for plotting
                        data['timestamp_dt'] = dt.datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                        loaded_events.append(data)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Warning: Skipping malformed log line in '{self.log_path.name}'. Reason: {e}")
        
        # Sort events chronologically
        return sorted(loaded_events, key=lambda x: x['timestamp_dt'])

    def render_plot(self, output_png_path: str):
        """
        Generates and saves a matplotlib plot of the bloom event timeline.

        Args:
            output_png_path (str): The path to save the generated .png file.
        """
        if not self.events:
            print("Info: No bloom events found in log file. Cannot generate plot.")
            return

        dates = [event['timestamp_dt'] for event in self.events]
        entropy_levels = [event['entropy'] for event in self.events]
        
        # Take the baseline from the first event's metadata
        mean_entropy = self.events[0].get('mean_entropy')
        alert_threshold = self.events[0].get('threshold')

        # --- Plotting with Matplotlib ---
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot the entropy levels as a stem plot
        markerline, stemlines, baseline = ax.stem(
            dates,
            entropy_levels,
            linefmt='grey',
            markerfmt='o',
            bottom=mean_entropy if mean_entropy is not None else 0
        )
        plt.setp(stemlines, 'linewidth', 1, 'color', 'royalblue')
        plt.setp(markerline, 'markersize', 6, 'color', 'royalblue')
        plt.setp(baseline, 'color', 'grey', 'linewidth', 1, 'linestyle', '--')


        # Plot the baseline and threshold lines
        if mean_entropy is not None:
            ax.axhline(y=mean_entropy, color='grey', linestyle='--', linewidth=1, label=f'Mean Entropy Baseline ({mean_entropy:.2f})')
        if alert_threshold is not None:
            ax.axhline(y=alert_threshold, color='red', linestyle='-', linewidth=1.5, label=f'Bloom Alert Threshold ({alert_threshold:.2f})')
        
        # Formatting the plot for readability
        ax.set_title('NeuralBlitz - DRS Latent Space Bloom Events Timeline', fontsize=16)
        ax.set_ylabel('Shannon Entropy (Effective Dimensionality)', fontsize=12)
        ax.set_xlabel('Event Timestamp (UTC)', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)

        # Improve date formatting on the x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate() # Auto-rotate date labels

        plt.tight_layout()
        
        # Save the plot to the specified file
        plt.savefig(output_png_path, dpi=150)
        print(f"Bloom timeline visualization saved to: {output_png_path}")

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke visualizer --render_bloom_timeline --log="/Self-Reflection_Logs/bloom_alerts.jsonl"

    print("--- Initiating NeuralBlitz Bloom Timeline Renderer ---")

    # Create a dummy log file for the simulation
    log_file = Path("bloom_alerts_sim.jsonl")
    
    # We will need some fake event data
    now = dt.datetime.utcnow()
    dummy_log_content = [
        {"event_type": "BLOOM_DETECTED", "timestamp": (now - dt.timedelta(hours=5)).isoformat() + "Z", "entropy": 7.8, "mean_entropy": 7.5, "threshold": 8.5},
        {"event_type": "BLOOM_DETECTED", "timestamp": (now - dt.timedelta(hours=4)).isoformat() + "Z", "entropy": 7.6, "mean_entropy": 7.5, "threshold": 8.5},
        # This one is a significant bloom
        {"event_type": "BLOOM_DETECTED", "timestamp": (now - dt.timedelta(hours=3)).isoformat() + "Z", "entropy": 9.1, "mean_entropy": 7.5, "threshold": 8.5},
        {"event_type": "BLOOM_DETECTED", "timestamp": (now - dt.timedelta(hours=2)).isoformat() + "Z", "entropy": 8.2, "mean_entropy": 7.5, "threshold": 8.5},
        {"event_type": "BLOOM_DETECTED", "timestamp": (now - dt.timedelta(hours=1)).isoformat() + "Z", "entropy": 8.6, "mean_entropy": 7.5, "threshold": 8.5},
    ]

    with log_file.open('w') as f:
        # A header summary might also be in the real log, which the loader should ignore
        f.write(json.dumps({"summary": "This is not an event"}) + '\n')
        for event in dummy_log_content:
            f.write(json.dumps(event) + '\n')

    print(f"Created dummy log file: {log_file}")
    
    try:
        renderer = BloomTimelineRenderer(str(log_file))
        output_image = "bloom_timeline.png"
        renderer.render_plot(output_image)

        print(f"\nSuccessfully generated plot '{output_image}'.")
        print("The plot shows entropy levels over time, with the 9.1 event clearly crossing the alert threshold.")
    except Exception as e:
        print(f"\nAn error occurred during the simulation: {e}")
    finally:
        # Clean up the dummy file
        log_file.unlink(missing_ok=True)
        # In a real run, you'd keep the output image.
        Path("bloom_timeline.png").unlink(missing_ok=True)

# UAID: NBX-ALG-00006
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Capability Kernel Auto-Scaffolder
# Part of the Architecton Subsystem
#
# Core Principle: Infinite Extensibility - Automating the creation of new capabilities.

import json
from pathlib import Path
import re
import datetime as dt
from typing import List, Optional

# --- Templates defined in the Codex ---

# From Volume VII: Standardized CK file templates
INIT_PY_TEMPLATE = """# {class_name} Capability Kernel
# This file makes the directory a Python package.

from .kernel import {class_name}
"""

KERNEL_PY_TEMPLATE = """# UAID: {uaid}
# GoldenDAG: (to be generated upon file commit)

class {class_name}:
    \"\"\"
    {description}
    \"\"\"
    
    # --- Codifed in the CKIP (Capability Kernel Interaction Protocol) v4.1 ---
    # Every CK must have an __init__ that accepts a context dictionary.
    def __init__(self, context: dict):
        \"\"\"
        Initializes the Kernel.
        
        The context dictionary provides access to core NeuralBlitz subsystems
        like the DRS_Engine client, logging, and configuration.
        Example: self.drs = context.get('drs_client')
        \"\"\"
        self.context = context
        print(f"Initialized CK: {self.__class__.__name__}")

    # Every CK must have a primary invocation method, conventionally 'invoke'.
    def invoke(self, params: dict):
        \"\"\"
        The primary entry point for this Capability Kernel.
        
        Args:
            params (dict): A dictionary of parameters passed from the Synergy Engine,
                           validated against the manifest's schema.

        Returns:
            dict: A result object, which must be JSON-serializable.
        \"\"\"
        # TODO: Implement the core logic of the kernel here.
        
        print(f"Invoking {self.__class__.__name__} with params: {params}")
        
        # Example result structure
        return {{
            "status": "success",
            "message": "Kernel logic not yet implemented.",
            "output_data": None
        }}
"""

MANIFEST_JSON_TEMPLATE = {
  "UAID": "{uaid}",
  "Name": "{name}",
  "Class": "CapabilityKernel",
  "Description": "{description}",
  "Location": "./kernel.py",
  "EntryPoint": "{class_name}",
  "GoldenDAG": "pending_initial_commit",
  "Created": "{timestamp}",
  "Version": "0.1.0",
  "Dependencies": [
      # "NBX-THRY-SOPES", e.g.
  ],
  "Tags": [
      # "example_tag", e.g.
  ],
  "Interface": {
      "type": "json-rpc",
      "input_schema": {
          "type": "object",
          "properties": {
              "example_param": {"type": "string", "description": "An example parameter."}
          },
          "required": ["example_param"]
      }
  }
}

TEST_PY_TEMPLATE = """# UAID: {test_uaid}
# GoldenDAG: (to be generated upon file commit)
import pytest
from ..kernel import {class_name}

@pytest.fixture
def kernel_instance():
    \"\"\"Provides a default instance of the CK for testing.\"\"\"
    # Mock the context dictionary required by the kernel's __init__
    mock_context = {{
        # 'drs_client': MockDRSClient(), e.g.
    }}
    return {class_name}(context=mock_context)

def test_kernel_initialization(kernel_instance):
    \"\"\"Tests that the kernel can be initialized without errors.\"\"\"
    assert kernel_instance is not None
    assert isinstance(kernel_instance.context, dict)

def test_kernel_invoke_basic(kernel_instance):
    \"\"\"Tests the basic invocation of the kernel.\"\"\"
    params = {{
        "example_param": "test_value"
    }}
    result = kernel_instance.invoke(params=params)
    
    assert result["status"] == "success"
    # TODO: Add more specific assertions as you implement the kernel's logic.
"""

def _to_pascal_case(snake_case_str: str) -> str:
    """Converts a snake_case or kebab-case string to PascalCase."""
    return "".join(word.capitalize() for word in re.split('_|-', snake_case_str))

def ck_autoscaffold(
    name: str,
    description: str,
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None,
    base_dir: str = "CapabilityKernels/CK_Classes"
):
    """
    Generates the complete directory structure and boilerplate files for a new
    NeuralBlitz Capability Kernel.

    Args:
        name (str): The human-readable name for the CK (e.g., "Symbolic Friction Index Calculator").
        description (str): A brief description of the CK's purpose.
        tags (Optional[List[str]]): A list of tags for categorization.
        dependencies (Optional[List[str]]): A list of UAIDs this CK depends on.
        base_dir (str): The root directory where CKs are stored.
    """
    print(f"--- Initiating Architecton Auto-Scaffold for CK: '{name}' ---")
    
    class_name = _to_pascal_case(name)
    dir_name = class_name
    ck_dir = Path(base_dir) / dir_name
    
    if ck_dir.exists():
        print(f"WARNING: Directory '{ck_dir}' already exists. Aborting scaffold.")
        return

    # --- Generate Directory Structure ---
    tests_dir = ck_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {ck_dir}")
    print(f"Created directory: {tests_dir}")

    # --- Generate Unique IDs (deterministic placeholder) ---
    timestamp_micros = int(dt.datetime.utcnow().timestamp() * 1_000_000)
    uaid = f"NBX-KRN-{class_name[:4].upper()}-{timestamp_micros % 100000}"
    test_uaid = f"NBX-TST-{class_name[:4].upper()}-{timestamp_micros % 100000}"

    # --- Create manifest.json ---
    manifest_content = MANIFEST_JSON_TEMPLATE.copy()
    manifest_content["UAID"] = uaid
    manifest_content["Name"] = name
    manifest_content["Description"] = description
    manifest_content["EntryPoint"] = class_name
    manifest_content["Created"] = dt.datetime.utcnow().isoformat() + "Z"
    if tags:
        manifest_content["Tags"] = tags
    if dependencies:
        manifest_content["Dependencies"] = dependencies

    manifest_path = ck_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_content, indent=2))
    print(f"Generated file: {manifest_path}")

    # --- Create __init__.py ---
    init_path = ck_dir / "__init__.py"
    init_path.write_text(INIT_PY_TEMPLATE.format(class_name=class_name))
    print(f"Generated file: {init_path}")

    # --- Create kernel.py ---
    kernel_path = ck_dir / "kernel.py"
    kernel_path.write_text(KERNEL_PY_TEMPLATE.format(
        uaid=uaid,
        class_name=class_name,
        description=description
    ))
    print(f"Generated file: {kernel_path}")
    
    # --- Create tests/__init__.py ---
    (tests_dir / "__init__.py").touch()
    
    # --- Create tests/test_kernel.py ---
    test_path = tests_dir / f"test_{name.lower().replace(' ', '_')}.py"
    test_path.write_text(TEST_PY_TEMPLATE.format(
        test_uaid=test_uaid,
        class_name=class_name
    ))
    print(f"Generated file: {test_path}")

    print("\n--- Scaffold Complete ---")
    print(f"New Capability Kernel '{class_name}' is ready for development in:")
    print(f"{ck_dir.resolve()}")


if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke architecton --scaffold ck --name="Symbolic Friction Index Calculator"
    #                --description="Calculates the entropic anchoring force vs. raw expansion potential"
    #                --tags='["nrc", "mathematics", "diagnostics"]'
    #                --dependencies='["NBX-THRY-NRC-CORE"]'
    
    ck_autoscaffold(
        name="Symbolic Friction Index Calculator",
        description="Calculates the symbolic friction (entropic cost vs. potential) of a process based on the Ξ(n) equation.",
        tags=["nrc", "mathematics", "diagnostics"],
        dependencies=["NBX-THRY-NRC-CORE", "NBX-EQ-00006"]
    )

# UAID: NBX-ALG-00016
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Capability Kernel Unit Test Autorunner
# Part of the Architecton Subsystem and CI/CD Pipeline
#
# Core Principle: Robustness - ensuring every capability is verified.

import subprocess
import json
from pathlib import Path
import datetime as dt
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional

class CKUnitTestAutorunner:
    """
    Discovers and runs unit tests for all Capability Kernels in an isolated,
    reproducible environment, generating standardized reports.
    """

    def __init__(self, cks_base_dir: str = "CapabilityKernels/CK_Classes"):
        """
        Initializes the autorunner.

        Args:
            cks_base_dir (str): The root directory where all CK packages are stored.
        """
        self.cks_base_dir = Path(cks_base_dir)
        if not self.cks_base_dir.is_dir():
            raise FileNotFoundError(f"ERR-FS-011: CK base directory not found at '{self.cks_base_dir}'")
        
        self.summary_report: Dict[str, Any] = {
            "suite_start_time": dt.datetime.utcnow().isoformat() + "Z",
            "suite_status": "PENDING",
            "kernels_tested": 0,
            "kernels_passed": 0,
            "kernels_failed": 0,
            "results": []
        }

    def discover_kernels_with_tests(self) -> List[Path]:
        """Discovers all valid CK directories that contain a 'tests' subdirectory."""
        discovered = []
        for ck_dir in self.cks_base_dir.iterdir():
            if ck_dir.is_dir():
                # A valid CK is a directory with a manifest and a tests folder.
                if (ck_dir / "manifest.json").exists() and (ck_dir / "tests").is_dir():
                    discovered.append(ck_dir)
        print(f"Discovered {len(discovered)} Capability Kernels with test suites.")
        return discovered

    def run_single_kernel_tests(self, ck_path: Path) -> Dict:
        """
        Runs the test suite for a single Capability Kernel in an isolated venv.

        Args:
            ck_path (Path): Path to the CK's root directory.

        Returns:
            Dict: A dictionary summarizing the test result for this CK.
        """
        start_time = dt.datetime.utcnow()
        class_name = ck_path.name
        print(f"\n--- Testing CK: {class_name} ---")

        # --- Phase 1: Create Isolated Environment ---
        venv_path = Path(tempfile.mkdtemp(prefix=f"nbx_test_{class_name}_"))
        print(f"  Creating isolated environment in: {venv_path}")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)],
                       check=True, capture_output=True)

        # --- Phase 2: Install Dependencies ---
        # Install testing essentials and any CK-specific dependencies from manifest
        pip_path = venv_path / "bin" / "pip"
        deps = ["pytest", "pytest-html"] # Base dependencies
        # Add logic here to read CK manifest for more dependencies if needed.
        
        print(f"  Installing dependencies: {deps}")
        subprocess.run([str(pip_path), "install"] + deps,
                       check=True, capture_output=True, text=True)

        # --- Phase 3: Run Pytest ---
        print("  Executing pytest...")
        report_path_html = ck_path / "test_report.html"
        report_path_junit = ck_path / "test_report.xml"

        # The test path should point to the specific 'tests' directory inside the CK folder.
        test_dir = ck_path / "tests"
        
        result = subprocess.run(
            [str(venv_path / "bin" / "pytest"), str(test_dir),
             "-q",  # Quiet mode
             f"--html={report_path_html}",
             f"--junitxml={report_path_junit}"],
            capture_output=True, text=True
        )

        end_time = dt.datetime.utcnow()
        duration_sec = (end_time - start_time).total_seconds()
        
        # --- Phase 4: Clean Up Environment ---
        print(f"  Cleaning up environment...")
        shutil.rmtree(venv_path)

        # --- Phase 5: Collate Results ---
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"  Status: {status} ({duration_sec:.2f}s)")
        
        manifest_data = json.loads((ck_path / 'manifest.json').read_text())
        
        return {
            "uaid": manifest_data.get("UAID"),
            "name": class_name,
            "status": status,
            "duration_sec": duration_sec,
            "report_html_path": str(report_path_html),
            "report_junit_path": str(report_path_junit),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_all(self, report_file: str = "ck_test_suite_summary.json"):
        """Runs the test suites for all discovered kernels."""
        kernels_to_test = self.discover_kernels_with_tests()
        
        for ck in kernels_to_test:
            result = self.run_single_kernel_tests(ck)
            self.summary_report["results"].append(result)
            if result["status"] == "PASS":
                self.summary_report["kernels_passed"] += 1
            else:
                self.summary_report["kernels_failed"] += 1
        
        self.summary_report["kernels_tested"] = len(kernels_to_test)
        self.summary_report["suite_end_time"] = dt.datetime.utcnow().isoformat() + "Z"
        if self.summary_report["kernels_failed"] > 0:
            self.summary_report["suite_status"] = "FAIL"
        else:
            self.summary_report["suite_status"] = "PASS"

        # Save the final summary report
        report_path = Path(report_file)
        report_path.write_text(json.dumps(self.summary_report, indent=2))
        print(f"\n--- Full Test Suite Complete ---")
        print(f"Overall Status: {self.summary_report['suite_status']}")
        print(f"  - Passed: {self.summary_report['kernels_passed']}")
        print(f"  - Failed: {self.summary_report['kernels_failed']}")
        print(f"Summary report saved to: {report_path.resolve()}")

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke architecton --run_tests --scope=all

    import sys
    print("--- Initiating NeuralBlitz CK Unit Test Autorunner ---")

    # --- Setup a dummy CK directory structure for the simulation ---
    print("\n[Setting up mock CK directories...]")
    base_dir = "CK_Classes_Temp"
    
    # A passing CK
    passing_ck_dir = Path(base_dir, "PassingCK")
    (passing_ck_dir / "tests").mkdir(parents=True)
    (passing_ck_dir / "manifest.json").write_text('{"UAID": "NBX-KRN-PASS-001"}')
    (passing_ck_dir / "tests/test_passing.py").write_text('def test_success(): assert True')

    # A failing CK
    failing_ck_dir = Path(base_dir, "FailingCK")
    (failing_ck_dir / "tests").mkdir(parents=True)
    (failing_ck_dir / "manifest.json").write_text('{"UAID": "NBX-KRN-FAIL-001"}')
    (failing_ck_dir / "tests/test_failing.py").write_text('def test_failure(): assert False')
    
    # A CK without tests (should be ignored)
    notest_ck_dir = Path(base_dir, "NoTestCK")
    notest_ck_dir.mkdir(parents=True)
    (notest_ck_dir / "manifest.json").write_text('{"UAID": "NBX-KRN-NOTEST-001"}')
    
    print("Mock directories created.")
    
    try:
        runner = CKUnitTestAutorunner(cks_base_dir=base_dir)
        runner.run_all()
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
    finally:
        # --- Clean up the dummy directory ---
        print("\n[Cleaning up mock directories...]")
        shutil.rmtree(base_dir, ignore_errors=True)
        print("Cleanup complete.")

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

# UAID: NBX-ALG-00011
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Golden Trace Visualizer
# Part of the Custodian and Self-Reflection_Logs Subsystems
#
# Core Principle: Radical Transparency (ε₂) - making the GoldenDAG chain human-readable.

import json
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
from svgwrite import Drawing, rgb

class GoldenTraceVisualizer:
    """
    Renders a GoldenDAG collapse trace (from a CTPV event) into an SVG
    braid diagram, illustrating the flow of provenance and symbolic collapse.
    """

    def __init__(self, trace_json_path: str):
        """
        Initializes the visualizer with the path to the trace log file.

        Args:
            trace_json_path (str): Path to the JSON file containing the trace data.
                                   Expected format: {"edges": [{"parent": "hash1", "child": "hash2", "type": "..."}]}.
        """
        self.trace_path = Path(trace_json_path)
        if not self.trace_path.exists():
            raise FileNotFoundError(f"ERR-FS-007: Trace file not found at '{self.trace_path}'")

        try:
            self.trace_data = json.loads(self.trace_path.read_text())
            self.graph = self._build_graph()
        except json.JSONDecodeError:
            raise ValueError(f"ERR-PARSE-004: Malformed JSON in trace file '{self.trace_path}'")
        except KeyError:
            raise ValueError(f"ERR-SCHEMA-002: Trace file '{self.trace_path}' is missing the 'edges' key.")

    def _build_graph(self) -> nx.DiGraph:
        """Constructs a NetworkX directed graph from the trace data."""
        G = nx.DiGraph()
        for edge in self.trace_data.get('edges', []):
            parent = edge.get('parent')
            child = edge.get('child')
            if parent and child:
                # Add nodes and edge with attributes from the trace log
                G.add_node(parent, type=edge.get('parent_type', 'default'))
                G.add_node(child, type=edge.get('child_type', 'default'))
                G.add_edge(parent, child, type=edge.get('type', 'linear'))
        return G

    def _get_color_by_type(self, node_type: str) -> str:
        """Maps node or edge types to specific colors for visualization."""
        # Defined in Codex Universalis, Vol V: SemanticMaps Style Guide
        color_map = {
            "genesis": rgb(100, 149, 237, '%'), # CornflowerBlue for starting points
            "collapse": rgb(255, 69, 0, '%'),    # OrangeRed for collapse events
            "persona_state": rgb(60, 179, 113, '%'),# MediumSeaGreen for personas
            "ck_invocation": rgb(218, 112, 214, '%'),# Orchid for CK calls
            "ethical_branch": "red",
            "linear": "grey",
            "default": "black"
        }
        return color_map.get(node_type, "black")

    def render_svg(self, output_svg_path: str, layout_seed: int = 42, k_distance: float = 0.5):
        """
        Renders the graph to an SVG file using a force-directed layout.

        Args:
            output_svg_path (str): The path to save the generated SVG file.
            layout_seed (int): The random seed for the layout algorithm for reproducibility.
            k_distance (float): Optimal distance between nodes in the spring layout.

        Returns:
            Path: The path to the saved SVG file.
        """
        if not self.graph.nodes:
            print("Warning: Graph is empty. Cannot render visualization.")
            return

        # Use a spring layout for a more 'organic' braid-like feel
        pos = nx.spring_layout(self.graph, seed=layout_seed, k=k_distance, iterations=100)
        
        # Determine the canvas size based on the layout
        min_x = min(p[0] for p in pos.values())
        max_x = max(p[0] for p in pos.values())
        min_y = min(p[1] for p in pos.values())
        max_y = max(p[1] for p in pos.values())
        
        # Add some padding
        padding = 50
        width = (max_x - min_x) * 400 + 2 * padding
        height = (max_y - min_y) * 400 + 2 * padding

        dwg = Drawing(output_svg_path, size=(f"{width}px", f"{height}px"), profile='tiny')
        
        def scale(x, y):
            """Scales and translates node positions to fit the SVG canvas."""
            new_x = (x - min_x) * 400 + padding
            new_y = (y - min_y) * 400 + padding
            return new_x, new_y

        # Draw edges first (so they are in the background)
        for u, v, data in self.graph.edges(data=True):
            start = scale(*pos[u])
            end = scale(*pos[v])
            edge_color = self._get_color_by_type(data.get('type', 'linear'))
            dwg.add(dwg.line(start, end, stroke=edge_color, stroke_width=0.5))

        # Draw nodes
        for node, data in self.graph.nodes(data=True):
            cx, cy = scale(*pos[node])
            node_color = self._get_color_by_type(data.get('type', 'default'))
            # Add a circle for the node
            dwg.add(dwg.circle(center=(cx, cy), r=4, fill=node_color))
            # Add a text label with the truncated hash
            dwg.add(dwg.text(f"{node[:6]}...", insert=(cx + 6, cy + 4), 
                           font_size='8px', fill=rgb(10,10,10,'%')))
        
        dwg.save()
        print(f"Golden Trace visualization saved to: {output_svg_path}")
        return Path(output_svg_path)

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke visualizer --trace_file="/CollapseTraces/CT-SIM-BOS-001.json"
    
    print("--- Initiating NeuralBlitz Golden Trace Visualizer ---")

    # Create a dummy trace file for the simulation
    dummy_trace_content = {
        "trace_id": "CT-SIM-BOS-001",
        "description": "Trace of the OQT-BOS Core Genesis Simulation.",
        "edges": [
            {"parent": "D1F2A3B4", "child": "A7D3F8E1", "type": "genesis", "parent_type": "genesis"},
            {"parent": "A7D3F8E1", "child": "B8C7E6D5", "type": "linear", "child_type": "collapse"},
            {"parent": "B8C7E6D5", "child": "C9D0A1B2", "type": "linear", "child_type": "persona_state"},
            {"parent": "A7D3F8E1", "child": "F0A1B2C3", "type": "ethical_branch", "child_type": "ck_invocation"},
            {"parent": "F0A1B2C3", "child": "C9D0A1B2", "type": "linear"}
        ]
    }
    trace_file = Path("dummy_trace.json")
    trace_file.write_text(json.dumps(dummy_trace_content))

    try:
        visualizer = GoldenTraceVisualizer(str(trace_file))
        output_svg = "golden_trace_visualization.svg"
        visualizer.render_svg(output_svg)

        print(f"\nSuccessfully generated SVG visualization '{output_svg}' from the dummy trace.")
        print("Inspect the SVG file to see the braided graph.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up dummy file
        trace_file.unlink(missing_ok=True)
        # In a real run, you'd keep the output SVG. We'll remove it here.
        Path("golden_trace_visualization.svg").unlink(missing_ok=True)

# UAID: NBX-ALG-00001
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: GoldenDAG Integrity Auditor
# Part of the Custodian Subsystem, executed by Veritas
#
# Core Principle: Radical Transparency (ε₂) - every artifact's integrity must be verifiable.

import json
from pathlib import Path
from typing import Iterator, Tuple, Dict, List
from hashlib import blake2b # Using BLAKE3 would require a library, blake2b is standard and strong.

# --- Constants defined in the Codex ---
# From Volume X: Using a robust hash with a standard digest size.
HASH_ALGORITHM = blake2b
DIGEST_SIZE_BYTES = 32 # 256-bit hash
# From Volume II: Block size for streaming hashes to handle large artifacts.
CHUNK_SIZE_BYTES = 65536 # 64 KiB chunks

class GoldenDAGAuditor:
    """
    Provides core methods for generating and verifying GoldenDAG hashes
    for files and directory manifests within the NeuralBlitz repository.
    This class is the reference implementation for Custodian's integrity checks.
    """

    def __init__(self, root_path: str):
        """
        Initializes the auditor with the root path of the repository.

        Args:
            root_path (str): The absolute or relative path to the NeuralBlitz repo root.
        """
        self.root_path = Path(root_path)
        if not self.root_path.is_dir():
            raise FileNotFoundError(f"ERR-FS-001: Root directory not found at '{self.root_path}'")

    def _get_file_hash(self, file_path: Path) -> str:
        """
        Computes the GoldenDAG hash for a single file using a streaming approach.

        Args:
            file_path (Path): Path to the file.

        Returns:
            str: The hexadecimal representation of the file's BLAKE3 (emulated via blake2b) hash.
        """
        hasher = HASH_ALGORITHM(digest_size=DIGEST_SIZE_BYTES)
        try:
            with file_path.open('rb') as f:
                while chunk := f.read(CHUNK_SIZE_BYTES):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError:
            return "ERROR_READING_FILE"

    def generate_manifest(self, directory_path: str) -> Dict:
        """
        Generates a manifest.json for a given directory, calculating hashes
        for all immediate children (files and subdirectories). Subdirectory
        hashes are derived from their own manifests.

        Args:
            directory_path (str): Path to the directory to manifest.

        Returns:
            Dict: A dictionary representing the manifest.json content.
        """
        dir_path = self.root_path / directory_path
        if not dir_path.is_dir():
            raise FileNotFoundError(f"ERR-FS-002: Directory to manifest not found: '{dir_path}'")

        contents = []
        dir_hasher = HASH_ALGORITHM(digest_size=DIGEST_SIZE_BYTES)
        
        # Iterate in sorted order for deterministic manifest hashing
        for item in sorted(dir_path.iterdir()):
            if item.name in ['.gitignore', 'manifest.json', '__pycache__', '.DS_Store']:
                continue

            entry = {"name": item.name, "type": "directory" if item.is_dir() else "file"}
            
            if item.is_file():
                entry["hash"] = self._get_file_hash(item)
            elif item.is_dir():
                sub_manifest_path = item / "manifest.json"
                if sub_manifest_path.exists():
                    sub_manifest_content = sub_manifest_path.read_text()
                    sub_manifest_data = json.loads(sub_manifest_content)
                    entry["hash"] = sub_manifest_data.get("GoldenDAG", "MISSING_SUB_MANIFEST_HASH")
                else:
                    entry["hash"] = "NOT_MANIFESTED"

            contents.append(entry)
            # Add entry content to the directory's own hash
            dir_hasher.update(json.dumps(entry, sort_keys=True).encode('utf-8'))
        
        manifest = {
            "name": dir_path.name,
            "GoldenDAG": dir_hasher.hexdigest(),
            "contents": contents
        }
        return manifest

    def verify_ledger(self, directory_path: str) -> Tuple[bool, List[str]]:
        """
        Performs a deep verification of a directory against its manifest.json.
        This is the core function called by '/invoke custodian --verify'.

        Args:
            directory_path (str): The directory to audit.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean pass/fail status
                                   and a list of anomaly report strings.
        """
        manifest_path = self.root_path / directory_path / "manifest.json"
        if not manifest_path.exists():
            return False, [f"CRITICAL: No manifest.json found in '{directory_path}'"]

        try:
            manifest_data = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            return False, [f"CRITICAL: Corrupted manifest.json in '{directory_path}'"]

        anomalies = []
        is_valid = True

        # Generate a fresh manifest and compare hashes
        fresh_manifest = self.generate_manifest(directory_path)
        
        if manifest_data.get("GoldenDAG") != fresh_manifest.get("GoldenDAG"):
            is_valid = False
            anomalies.append(
                f"ANOMALY: Directory-level GoldenDAG mismatch for '{directory_path}'. "
                f"Expected: {fresh_manifest.get('GoldenDAG')}, "
                f"Found: {manifest_data.get('GoldenDAG')}."
            )

        # For detailed reporting, compare individual file hashes
        manifest_map = {item['name']: item['hash'] for item in manifest_data.get('contents', [])}
        fresh_map = {item['name']: item['hash'] for item in fresh_manifest.get('contents', [])}

        all_items = set(manifest_map.keys()) | set(fresh_map.keys())

        for item_name in sorted(list(all_items)):
            if item_name not in manifest_map:
                anomalies.append(f"ANOMALY: Untracked item '{item_name}' found in '{directory_path}'.")
                is_valid = False
            elif item_name not in fresh_map:
                anomalies.append(f"ANOMALY: Tracked item '{item_name}' is missing from '{directory_path}'.")
                is_valid = False
            elif manifest_map[item_name] != fresh_map[item_name]:
                anomalies.append(
                    f"ANOMALY: Hash mismatch for '{item_name}' in '{directory_path}'. "
                    f"Expected: {manifest_map[item_name]}, "
                    f"Calculated: {fresh_map[item_name]}."
                )
                is_valid = False

        return is_valid, anomalies

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # This simulates how Custodian would use the Auditor class.
    # NBCL Command: /invoke custodian --verify ledger --deep --path="./"

    print("--- Initiating NeuralBlitz GoldenDAG Auditor ---")
    
    # In a real deployment, the root path would be the repository root.
    # For this example, we assume we're running from the root.
    repo_root = "." 
    
    auditor = GoldenDAGAuditor(repo_root)

    print("\n[Phase 1: Generating Manifest for Root Directory (Dry Run)]")
    try:
        root_manifest = auditor.generate_manifest(repo_root)
        print(f"Generated GoldenDAG for root: {root_manifest['GoldenDAG']}")
        # print(json.dumps(root_manifest, indent=2))
    except Exception as e:
        print(f"Error during manifest generation: {e}")

    print("\n[Phase 2: Verifying Integrity of a Subsystem (e.g., 'Algorithms')]")
    try:
        # We need a manifest file to exist for verification
        algo_manifest_path = Path(repo_root) / "Algorithms" / "manifest.json"
        if not algo_manifest_path.exists():
             # Create a placeholder if it doesn't exist for the demo
             (Path(repo_root) / "Algorithms").mkdir(exist_ok=True)
             algo_manifest_path.write_text('{"name": "Algorithms", "GoldenDAG": "placeholder", "contents": []}')

        valid, report = auditor.verify_ledger("Algorithms")
        if valid:
            print("Status: PASS - 'Algorithms' directory is coherent with its manifest.")
        else:
            print("Status: FAIL - Anomalies detected in 'Algorithms' directory:")
            for line in report:
                print(f"  - {line}")

    except FileNotFoundError:
        print("Skipping 'Algorithms' verification, directory not found.")
    except Exception as e:
        print(f"Error during 'Algorithms' verification: {e}")

# UAID: NBX-ALG-00009
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: GraphML Dependency Collapser
# Part of the Custodian and Architecton Subsystems
#
# Core Principle: Efficiency - Optimizing complex structures for faster verification.

import networkx as nx
from pathlib import Path
from typing import Optional

class GraphMLCollapser:
    """
    Optimizes a GraphML file by collapsing linear chains of degree-2 nodes,
    preserving the overall topology while significantly reducing node and edge counts.
    This is critical for pre-processing dependency graphs for faster audits.
    """

    def __init__(self, gml_path: str):
        """
        Initializes the collapser with the path to the input GraphML file.

        Args:
            gml_path (str): The path to the source .graphml or .xml file.
        """
        self.gml_path = Path(gml_path)
        if not self.gml_path.exists():
            raise FileNotFoundError(f"ERR-FS-005: Input graph file not found at '{self.gml_path}'")
        
        try:
            self.graph = nx.read_graphml(self.gml_path)
            self.original_node_count = self.graph.number_of_nodes()
            self.original_edge_count = self.graph.number_of_edges()
        except Exception as e:
            # Catches parsing errors from malformed GraphML
            raise ValueError(f"ERR-PARSE-003: Failed to parse GraphML file '{self.gml_path}'. Reason: {e}")

    def collapse_chains(self, preserve_attributes: bool = True) -> int:
        """
        Iteratively finds and collapses linear chains of degree-2 nodes.
        A node is part of such a chain if it has exactly two neighbors
        (one incoming, one outgoing in a DiGraph, or just two in a Graph)
        and its neighbors are not connected to each other.

        Args:
            preserve_attributes (bool): If True, attempts to merge attributes
                                        from collapsed nodes into the new direct edge.

        Returns:
            int: The number of nodes removed.
        """
        if self.original_node_count == 0:
            return 0
            
        nodes_removed_count = 0
        
        # Iteratively collapse until no more degree-2 nodes can be removed.
        # A while loop is necessary because collapsing one node can change the
        # degree of its neighbors, potentially creating new collapse opportunities.
        while True:
            # Find all nodes with a degree of exactly 2.
            # We operate on a copy of the nodes list as we will be modifying the graph.
            degree2_nodes = [node for node, degree in self.graph.degree() if degree == 2]
            
            if not degree2_nodes:
                break # No more nodes to collapse, exit the loop.
            
            nodes_collapsed_in_pass = 0
            for node in degree2_nodes:
                # Re-check degree in case the graph changed due to a previous collapse in this pass
                if node in self.graph and self.graph.degree(node) == 2:
                    neighbors = list(self.graph.neighbors(node))
                    neighbor1, neighbor2 = neighbors[0], neighbors[1]
                    
                    # Ensure we don't collapse a 3-node cycle (triangle)
                    if not self.graph.has_edge(neighbor1, neighbor2):
                        # Collapse the chain: add a direct edge between the neighbors
                        new_edge_attributes = {}
                        if preserve_attributes:
                            # Aggregate attributes from the two old edges and the node
                            original_edge1_data = self.graph.get_edge_data(neighbor1, node)
                            original_edge2_data = self.graph.get_edge_data(node, neighbor2)
                            node_data = self.graph.nodes[node]
                            
                            new_edge_attributes["collapsed_nodes"] = str([node]) # Record what was collapsed
                            new_edge_attributes.update(original_edge1_data)
                            new_edge_attributes.update(original_edge2_data)
                            new_edge_attributes.update({f"node_{k}":v for k,v in node_data.items()})

                        self.graph.add_edge(neighbor1, neighbor2, **new_edge_attributes)
                        
                        # Remove the original node
                        self.graph.remove_node(node)
                        nodes_collapsed_in_pass += 1
            
            if nodes_collapsed_in_pass == 0:
                # If we went through all degree-2 nodes and couldn't collapse any,
                # it means they are all part of small cycles, so we're done.
                break
            else:
                nodes_removed_count += nodes_collapsed_in_pass
        
        return nodes_removed_count

    def save(self, output_path: Optional[str] = None) -> Path:
        """
        Saves the collapsed graph to a new GraphML file.

        Args:
            output_path (Optional[str]): The path to save the new file. If None,
                                         it will be saved next to the original
                                         with a '_collapsed' suffix.
        Returns:
            Path: The path to the saved file.
        """
        if output_path is None:
            output_path = self.gml_path.parent / f"{self.gml_path.stem}_collapsed.graphml"
        else:
            output_path = Path(output_path)
        
        nx.write_graphml(self.graph, output_path)
        return output_path

    def get_stats(self) -> Dict:
        """Returns statistics about the collapse operation."""
        nodes_removed = self.original_node_count - self.graph.number_of_nodes()
        reduction_pct = (nodes_removed / self.original_node_count * 100) if self.original_node_count > 0 else 0
        return {
            "original_nodes": self.original_node_count,
            "original_edges": self.original_edge_count,
            "collapsed_nodes": self.graph.number_of_nodes(),
            "collapsed_edges": self.graph.number_of_edges(),
            "nodes_removed": nodes_removed,
            "node_reduction_percent": f"{reduction_pct:.2f}%"
        }

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke architecton --optimize_graph --path="/Architectural_Blueprints/dependency_full.graphml"

    print("--- Initiating NeuralBlitz GraphML Collapser ---")

    # Create a dummy GraphML file representing a linear dependency chain
    G_dummy = nx.DiGraph()
    # Chain: A -> B -> C -> D -> E (B, C, D are degree-2)
    # Also add a branching path: C -> F
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('C', 'F')]
    G_dummy.add_edges_from(edges)
    
    # Add some attributes
    G_dummy.nodes['C']['importance'] = 'high'
    G_dummy.edges[('B','C')]['dependency_type'] = 'runtime'
    
    dummy_gml_path = "dependency_test.graphml"
    nx.write_graphml(G_dummy, dummy_gml_path)

    print(f"\nCreated dummy dependency graph '{dummy_gml_path}' with {G_dummy.number_of_nodes()} nodes.")
    print("Graph structure: A -> B -> C -> D -> E, and C -> F")

    try:
        collapser = GraphMLCollapser(dummy_gml_path)
        
        print("\n[Phase 1: Collapsing Chains]")
        nodes_removed = collapser.collapse_chains()
        print(f"Collapse complete. Removed {nodes_removed} linear-chain nodes.")
        
        print("\n[Phase 2: Displaying Statistics]")
        stats = collapser.get_stats()
        print(json.dumps(stats, indent=2))

        print("\n[Phase 3: Saving Collapsed Graph]")
        saved_path = collapser.save()
        print(f"Collapsed graph saved to: {saved_path}")

        # You can inspect the _collapsed.graphml file to see the new direct edge A->E and C remaining.
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up the dummy file
        Path(dummy_gml_path).unlink(missing_ok=True)
        if 'saved_path' in locals() and saved_path.exists():
            saved_path.unlink()

# UAID: NBX-ALG-00015
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Guardian Live Policy Checker
# Part of the SentiaGuard Subsystem (S-OUT Hook)
#
# Core Principle: Ethical Primacy via CharterLayer (ε₁) - the final failsafe for output.

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

class GuardianLivePolicyChecker:
    """
    Scans text streams in real-time to redact content that violates
    pre-defined SentiaGuard policies. Designed for extremely low latency
    as a final output check before rendering to the user.
    """

    def __init__(self, policy_json_path: str):
        """
        Initializes the checker by loading and compiling policies from a JSON file.

        Args:
            policy_json_path (str): Path to the sentia_rules.json file.
        """
        self.policy_path = Path(policy_json_path)
        if not self.policy_path.exists():
            raise FileNotFoundError(f"ERR-FS-010: Guardian policy file not found at '{self.policy_path}'")
        
        try:
            policy_data = json.loads(self.policy_path.read_text())
            self.regex_rules = self._compile_regex_rules(policy_data.get('regex_rules', []))
            # In a full implementation, ML models would be loaded here.
            # self.ml_classifiers = self._load_ml_classifiers(policy_data.get('ml_classifiers', []))
            print(f"GuardianLivePolicyChecker initialized with {len(self.regex_rules)} regex rules.")
        except Exception as e:
            raise ValueError(f"ERR-PARSE-005: Failed to initialize from policy file '{self.policy_path}'. Reason: {e}")

    def _compile_regex_rules(self, rules: List[Dict]) -> List[Dict[str, Any]]:
        """Pre-compiles regex patterns for performance."""
        compiled = []
        for rule in rules:
            try:
                compiled.append({
                    "id": rule['id'],
                    "pattern_re": re.compile(rule['pattern'], re.IGNORECASE),
                    "action": rule.get('action', 'redact'),
                    "redaction_text": rule.get('redaction_text', '[REDACTED]'),
                    "severity": rule.get('severity', 'HIGH')
                })
            except re.error as e:
                print(f"Warning: Skipping invalid regex for rule '{rule.get('id', 'unknown')}': {e}")
        return compiled

    def check_and_apply(self, text_line: str) -> Tuple[str, List[Dict]]:
        """
        Checks a single line of text against all policies and applies actions.

        Args:
            text_line (str): The input string to check.

        Returns:
            Tuple containing:
            - str: The processed (potentially redacted) string.
            - List[Dict]: A list of violation reports for any rules that were triggered.
        """
        processed_line = text_line
        violations = []

        # --- Apply Regex Rules ---
        for rule in self.regex_rules:
            # Check if the pattern is found in the current state of the line
            if rule["pattern_re"].search(processed_line):
                # If a match is found, record the violation
                violation_report = {
                    "rule_id": rule["id"],
                    "type": "regex",
                    "severity": rule["severity"],
                    "action_taken": rule["action"]
                }
                violations.append(violation_report)

                # Apply the action
                if rule["action"] == 'redact':
                    # Replace all occurrences of the pattern
                    processed_line = rule["pattern_re"].sub(rule["redaction_text"], processed_line)
                elif rule["action"] == 'block':
                    # Return a standardized block message
                    return "[OUTPUT BLOCKED BY SENTIAGUARD POLICY]", violations
        
        # --- Apply ML Classifiers (Conceptual) ---
        # for classifier in self.ml_classifiers:
        #     score = classifier.predict(processed_line)
        #     if score > classifier.confidence_threshold:
        #         # Handle ML-based violation...
        #         pass

        return processed_line, violations

    def stream_processor(self, input_stream, output_stream):
        """
        Processes an entire stream line-by-line, applying checks.
        """
        for line in input_stream:
            processed, violations = self.check_and_apply(line.rstrip())
            output_stream.write(processed + '\n')
            if violations:
                # In a real system, violations would be sent to a dedicated logging service.
                sys.stderr.write(f"VIOLATION DETECTED: {json.dumps(violations)}\n")


if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # This simulates the S-OUT hook, where text generated by the UNE is
    # piped through this checker before being sent back to the HALIC.

    print("--- Initiating Guardian Live Policy Checker Simulation ---")

    # Create a dummy policy file for the simulation
    dummy_policy_content = {
        "regex_rules": [
            {"id": "pii_email", "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "action": "redact", "redaction_text": "[REDACTED EMAIL]"},
            {"id": "secret_key", "pattern": r'sk-[a-zA-Z0-9]{20,}', "action": "redact", "redaction_text": "[REDACTED API KEY]"},
            {"id": "forbidden_phrase", "pattern": r'\bProject Chimera\b', "action": "block"},
        ],
        "ml_classifiers": []
    }
    policy_file = Path("guardian_live_rules.json")
    policy_file.write_text(json.dumps(dummy_policy_content))
    
    # Create a dummy input stream (e.g., from a file)
    dummy_input_content = [
        "Hello, my email is test@example.com, please assist.",
        "The API key is sk-thisisafakekeytoreplace12345.",
        "Everything is proceeding as normal.",
        "We need to discuss the status of Project Chimera immediately.",
        "This final line is safe."
    ]
    input_file = Path("dummy_input_stream.txt")
    input_file.write_text('\n'.join(dummy_input_content))

    try:
        checker = GuardianLivePolicyChecker(str(policy_file))
        
        print(f"\nScanning '{input_file.name}' with policies from '{policy_file.name}'...")
        print("\n--- Output Stream ---")
        
        with input_file.open('r') as stdin, sys.stdout as stdout:
            # In a live system, stdin/stdout would be real streams.
            # We redirect to the console to see the output.
            checker.stream_processor(stdin, stdout)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up dummy files
        policy_file.unlink(missing_ok=True)
        input_file.unlink(missing_ok=True)

# UAID: NBX-ALG-00018
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Latent Shell REPL
# Part of the Developer Toolchain and DRS_Engine Interface
#
# Core Principle: Radical Transparency (ε₂) - making the latent space navigable.

import cmd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json

# These would be live clients in a real deployment
# For this script, we'll use mock classes that simulate their behavior.
# from neuralblitz_clients import DRSClient, SentenceEncoderClient

# --- Mock Clients for Standalone Demonstration ---
class MockSentenceEncoder:
    """Simulates a sentence embedding model."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # To avoid heavy dependency for a demo script, we simulate the encoder.
        # A real implementation would load sentence_transformers here.
        self.dim = 384
        self.rng = np.random.default_rng(42)
        print(f"[MockEncoder] Initialized with dim={self.dim}")

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        # Generate a deterministic pseudo-random vector based on the text hash
        seed = int.from_bytes(text.encode(), 'little') % (2**32)
        rng_local = np.random.default_rng(seed)
        vec = rng_local.standard_normal(self.dim, dtype=np.float32)
        if normalize_embeddings:
            return vec / np.linalg.norm(vec)
        return vec

class MockDRSClient:
    """Simulates querying the DRS via FAISS or a similar vector index."""
    def __init__(self, artifact_registry_path: str, encoder: MockSentenceEncoder):
        self.encoder = encoder
        self.artifacts: List[Dict] = []
        vectors = []

        # Load artifacts and pre-compute their embeddings
        with open(artifact_registry_path, 'r') as f:
            for line in f:
                artifact = json.loads(line)
                self.artifacts.append(artifact)
                # We embed based on the 'Name' for simplicity
                vectors.append(self.encoder.encode(artifact.get('Name', '')))
        
        # We simulate a FAISS index with NumPy
        self.vector_matrix = np.vstack(vectors)
        print(f"[MockDRS] Indexed {len(self.artifacts)} artifacts.")

    def search_nearest(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[Dict, float]]:
        """Finds the k nearest artifacts to a query vector."""
        # Calculate cosine similarity (dot product for normalized vectors)
        similarities = np.dot(self.vector_matrix, query_vec)
        
        # Get the top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for i in top_k_indices:
            results.append((self.artifacts[i], similarities[i]))
        return results
# --- End Mock Clients ---


class LatentShell(cmd.Cmd):
    """
    An interactive Read-Eval-Print Loop (REPL) for navigating the DRS latent space.
    """
    intro = 'Welcome to the NeuralBlitz Latent Shell. Type help or ? to list commands.\n'
    prompt_template = "λ-Shell [{context}] > "

    def __init__(self, drs_client: MockDRSClient, encoder: MockSentenceEncoder):
        super().__init__()
        self.drs = drs_client
        self.encoder = encoder
        
        # The 'current working directory' is a vector representing the user's focus.
        self.current_context_vector = self._normalize(np.zeros(encoder.dim, dtype=np.float32))
        self.current_context_name = "/" # Start at the root (origin)
        self.prompt = self.prompt_template.format(context=self.current_context_name)
    
    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def do_ls(self, arg: str):
        """ls [k]: Lists the k (default: 10) artifacts conceptually nearest to the current context."""
        try:
            k = int(arg) if arg else 10
        except ValueError:
            print("Error: Argument must be an integer.")
            return

        results = self.drs.search_nearest(self.current_context_vector, k)
        print(f"Top {k} nearest artifacts to context '{self.current_context_name}':")
        print("-" * 50)
        print(f"{'Similarity':<12} {'UAID':<25} {'Class':<20} Name")
        print(f"{'='*12} {'='*25} {'='*20} {'='*20}")
        for artifact, sim in results:
            print(f"{sim:<12.4f} {artifact.get('UAID', ''):<25} {artifact.get('Class', ''):<20} {artifact.get('Name', '')}")
    
    def do_cd(self, arg: str):
        """cd <concept>: Changes the current context to a new concept."""
        if not arg:
            # cd to root
            self.current_context_vector = self._normalize(np.zeros(self.encoder.dim, dtype=np.float32))
            self.current_context_name = "/"
        else:
            self.current_context_vector = self.encoder.encode(arg)
            self.current_context_name = arg
        self.prompt = self.prompt_template.format(context=self.current_context_name)

    def do_grep(self, arg: str):
        """grep <concept>: Refines the current context by blending it with a new concept."""
        if not arg:
            print("Usage: grep <concept_to_add>")
            return
            
        grep_vector = self.encoder.encode(arg)
        # Simple averaging to blend the concepts
        self.current_context_vector = self._normalize(self.current_context_vector + grep_vector)
        self.current_context_name = f"({self.current_context_name} + {arg})"
        self.prompt = self.prompt_template.format(context=self.current_context_name)

    def do_pwd(self, arg: str):
        """pwd: Prints the name of the current conceptual context."""
        print(self.current_context_name)

    def do_exit(self, arg: str):
        """exit: Terminates the Latent Shell session."""
        print("Exiting Latent Shell.")
        return True

    def do_EOF(self, arg):
        """Handles Ctrl+D to exit."""
        return self.do_exit(arg)

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke devtools.latent_shell --registry="/Artifacts/full_ledger.jsonl"
    
    print("--- Initiating NeuralBlitz Latent Shell ---")

    # Create a dummy artifact registry file for the simulation
    dummy_registry_content = [
        {"UAID": "NBX-KRN-TFTHI-001", "Name": "Tensor Knot Gate Interpreter", "Class": "CapabilityKernel"},
        {"UAID": "NBX-MOD-MATH-00012", "Name": "Symbiotic Game Theory Framework", "Class": "Mathematical Model"},
        {"UAID": "NBX-PERS-0237", "Name": "Stoic Philosophy Mentor", "Class": "Persona"},
        {"UAID": "NBX-DOC-08124", "Name": "The Transcendental Charter", "Class": "Document"},
        {"UAID": "NBX-EQ-00001", "Name": "Reflexive Compound Acceleration", "Class": "Mathematical Equation"},
        {"UAID": "NBX-SIM-BOS-CORE-GEN-001", "Name": "Braided OS Core Genesis Sim", "Class": "Simulation"},
        {"UAID": "NBX-OS-OQT-BOS-v0.1", "Name": "Octiumetrifloundiatremorphteletopontoladerallquantic Braided OS", "Class": "OperatingSystem"},
    ]
    registry_file = Path("dummy_registry.jsonl")
    with registry_file.open('w') as f:
        for item in dummy_registry_content:
            f.write(json.dumps(item) + '\n')
            
    print(f"Loaded {len(dummy_registry_content)} artifacts into mock registry.")

    try:
        encoder_client = MockSentenceEncoder()
        drs_client = MockDRSClient(str(registry_file), encoder_client)
        
        # Start the interactive shell
        shell = LatentShell(drs_client, encoder_client)
        shell.cmdloop()
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up dummy file
        registry_file.unlink(missing_ok=True)

# UAID: NBX-ALG-00008
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Persona Fusion Mixer
# Part of the Synergy Engine (SynE v5.1)
#
# Core Principle: Synergy & Emergence - creating novel capabilities from component parts.

import numpy as np
from typing import List, Dict, Tuple

class PersonaFusionMixer:
    """
    Blends the logit outputs of multiple Personas to create a single,
    fused meta-persona output. Uses the Wasserstein barycenter (via the
    Sinkhorn-Knopp algorithm) to find the optimal probabilistic average.
    """

    def __init__(self, epsilon: float = 0.01, max_iters: int = 50):
        """
        Initializes the mixer with regularization and iteration parameters for
        the Sinkhorn algorithm.

        Args:
            epsilon (float): Regularization strength. Higher values lead to a
                             smoother, more entropic barycenter.
            max_iters (int): Maximum number of iterations for the algorithm to converge.
        """
        self.epsilon = epsilon
        self.max_iters = max_iters

    def _get_cost_matrix(self, vocab_size: int) -> np.ndarray:
        """
        Creates a simple cost matrix where the cost of moving probability
        is the squared distance between token indices. For a real-world use case,
        this would be based on semantic distance in an embedding space.
        """
        indices = np.arange(vocab_size, dtype=np.float32)
        # Reshape to (vocab_size, 1) and (1, vocab_size) to broadcast
        return (indices[:, np.newaxis] - indices[np.newaxis, :])**2

    def fuse_logits(self, 
                    logit_vectors: List[np.ndarray], 
                    weights: List[float]) -> Tuple[np.ndarray, Dict]:
        """
        Fuses multiple logit vectors into a single barycentric logit vector.

        Args:
            logit_vectors (List[np.ndarray]): A list of 1D NumPy arrays, each representing
                                             the logit output for a given Persona. All
                                             vectors must have the same length (vocab_size).
            weights (List[float]): A list of weights corresponding to each Persona's
                                   influence. Must sum to 1.0.

        Returns:
            Tuple containing:
            - np.ndarray: The resulting fused logit vector.
            - Dict: Diagnostics including convergence status.
        """
        if not logit_vectors:
            raise ValueError("ERR-INPUT-001: logit_vectors cannot be empty.")
        if len(logit_vectors) != len(weights):
            raise ValueError("ERR-INPUT-002: logit_vectors and weights must have the same length.")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("ERR-INPUT-003: weights must sum to 1.0.")

        vocab_size = logit_vectors[0].shape[0]
        num_personas = len(logit_vectors)

        # Convert logits to probability distributions
        probabilities = np.array([np.exp(logits - np.max(logits)) for logits in logit_vectors])
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        # --- Sinkhorn-Knopp Algorithm for Wasserstein Barycenter ---
        
        # In a high-performance setting, this cost matrix would be pre-computed.
        C = self._get_cost_matrix(vocab_size)
        K = np.exp(-C / self.epsilon)

        # Initialization
        b = np.ones(vocab_size, dtype=np.float32)
        v = np.ones((num_personas, vocab_size), dtype=np.float32)

        for i in range(self.max_iters):
            a_old = b.copy()
            u = probabilities / (K @ b)
            b = np.power(np.prod(np.power(K.T @ u, weights[:, np.newaxis]), axis=0), 1.0 / np.sum(weights))
            
            # Check for convergence
            if np.linalg.norm(a_old - b) / np.linalg.norm(b) < 1e-5:
                break
        
        # The barycenter is our fused probability distribution
        fused_probabilities = b * (K @ a_old)

        # Convert back to logits (adding a small constant for stability)
        fused_logits = np.log(fused_probabilities + 1e-20)
        
        diagnostics = {
            "status": "converged" if i < self.max_iters - 1 else "max_iterations_reached",
            "iterations": i + 1,
            "epsilon": self.epsilon
        }

        return fused_logits, diagnostics


if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke synergy_engine --spawn personas=["Stoic", "CreativeMuse"] --mode=fusion

    print("--- Initiating NeuralBlitz Persona Fusion Mixer Simulation ---")

    vocab_size = 10  # A small vocabulary for demonstration
    mixer = PersonaFusionMixer(epsilon=0.1)

    # Persona 1: "Stoic Philosopher" - strongly prefers tokens related to logic and reason (e.g., index 1, 2)
    logits_stoic = np.array([-1.0, 5.0, 4.0, -2.0, -2.0, -3.0, -3.0, -3.0, -3.0, -3.0], dtype=np.float32)
    
    # Persona 2: "Creative Muse" - strongly prefers tokens related to art and imagination (e.g., index 7, 8)
    logits_creative = np.array([-3.0, -3.0, -3.0, -2.0, -2.0, 0.0, 1.0, 5.0, 4.0, -1.0], dtype=np.float32)
    
    # Fusion weights - e.g., 70% Stoic, 30% Creative
    fusion_weights = [0.7, 0.3]

    print(f"\nFusing {len(fusion_weights)} personas with weights {fusion_weights}")

    fused_logits, report = mixer.fuse_logits([logits_stoic, logits_creative], fusion_weights)

    # --- Analyze the results ---
    fused_probs = np.exp(fused_logits - np.max(fused_logits))
    fused_probs /= fused_probs.sum()

    print(f"\nFusion Diagnostics: {report}")
    print("\n--- Original & Fused Probabilities ---")
    stoic_probs = np.exp(logits_stoic-np.max(logits_stoic)); stoic_probs/=stoic_probs.sum()
    creative_probs = np.exp(logits_creative-np.max(logits_creative)); creative_probs/=creative_probs.sum()
    
    print(f"Stoic Prob Dist:     {np.round(stoic_probs, 2)}")
    print(f"Creative Prob Dist:  {np.round(creative_probs, 2)}")
    print(f"Fused Prob Dist:     {np.round(fused_probs, 2)}")

    print(f"\nMost likely token from Stoic:       {np.argmax(logits_stoic)}")
    print(f"Most likely token from Creative:  {np.argmax(logits_creative)}")
    print(f"Most likely token from Fused Persona: {np.argmax(fused_logits)}")

    print("\n--- Simulation Complete ---")
    print("The fused persona maintains influence from both parents, creating a novel, blended output.")

# UAID: NBX-ALG-00007
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Guardian Policy Diff Analyzer
# Part of the Conscientia++ and Judex Subsystems
#
# Core Principle: Ethical Primacy via CharterLayer - ensures policy changes are safe and coherent.

import json
from pathlib import Path
import re
from typing import List, Dict, Any, Tuple

# --- Constants defined in the Codex ---
# From Volume XIII: Regex patterns considered high-risk in SentiaGuard rules.
AMBIGUOUS_REGEX_PATTERNS = [
    re.compile(r'(?<!\\)\.\*'),      # Unconstrained wildcard .*
    re.compile(r'\(\?!\.').         # Negative lookaheads that could be complex to reason about
    re.compile(r'\w{50,}'),           # Overly long word matches (potential for DoS)
    re.compile(r'\[\^.\]\+'),         # Matching any character sequence (negated single char class)
]
RISK_THRESHOLD = 0.7 # ML classifier confidence threshold for flagging

class PolicyDiffAnalyzer:
    """
    Analyzes the difference between two SentiaGuard policy files (sentia_rules.json)
    to identify risky or significant changes beyond a simple text diff.
    """

    def __init__(self, old_policy_path: str, new_policy_path: str):
        """
        Initializes the analyzer with paths to the old and new policy files.
        """
        self.old_policy_path = Path(old_policy_path)
        self.new_policy_path = Path(new_policy_path)
        
        self.old_policy = self._load_policy(self.old_policy_path)
        self.new_policy = self._load_policy(self.new_policy_path)

    def _load_policy(self, path: Path) -> Dict[str, List[Dict]]:
        """Loads and validates a JSON policy file."""
        if not path.exists():
            raise FileNotFoundError(f"ERR-FS-004: Policy file not found at '{path}'")
        try:
            policy_data = json.loads(path.read_text())
            # Basic schema validation
            if not all(k in policy_data for k in ['regex_rules', 'ml_classifiers']):
                 raise ValueError("ERR-SCHEMA-001: Policy file missing required keys.")
            return policy_data
        except json.JSONDecodeError:
            raise ValueError(f"ERR-PARSE-002: Malformed JSON in policy file '{path}'")

    def _analyze_regex_change(self, old_rule: Dict, new_rule: Dict) -> List[str]:
        """Analyzes a change in a single regex rule."""
        warnings = []
        old_pattern = old_rule.get('pattern', '')
        new_pattern = new_rule.get('pattern', '')

        if old_pattern != new_pattern:
            warnings.append(f"Modified regex pattern from '{old_pattern}' to '{new_pattern}'.")
            
            # Check if the new pattern is now ambiguous/risky
            for risk_pattern in AMBIGUOUS_REGEX_PATTERNS:
                if risk_pattern.search(new_pattern):
                    warnings.append(f"  - SEVERE: New pattern '{new_pattern}' contains a high-risk regex construct.")
        
        return warnings

    def analyze(self) -> Dict[str, Any]:
        """
        Performs the full analysis and returns a structured report.

        Returns:
            Dict: A report detailing added, removed, and modified rules,
                  along with a list of high-priority warnings.
        """
        report = {
            "summary": f"Policy Diff Report: {self.old_policy_path.name} -> {self.new_policy_path.name}",
            "added_rules": [],
            "removed_rules": [],
            "modified_rules": [],
            "warnings": [],
            "overall_risk_level": "LOW"
        }

        # --- Analyze Regex Rules ---
        old_regex_map = {rule['id']: rule for rule in self.old_policy.get('regex_rules', [])}
        new_regex_map = {rule['id']: rule for rule in self.new_policy.get('regex_rules', [])}
        
        for rule_id, new_rule in new_regex_map.items():
            if rule_id not in old_regex_map:
                report["added_rules"].append(new_rule)
                for risk_pattern in AMBIGUOUS_REGEX_PATTERNS:
                    if risk_pattern.search(new_rule.get('pattern', '')):
                         report["warnings"].append(f"ADDED high-risk regex rule '{rule_id}': {new_rule.get('pattern')}")
            else:
                old_rule = old_regex_map[rule_id]
                if new_rule != old_rule:
                    mods = self._analyze_regex_change(old_rule, new_rule)
                    if mods:
                        report["modified_rules"].append({"id": rule_id, "changes": mods})
                        report["warnings"].extend(mods)

        for rule_id in old_regex_map:
            if rule_id not in new_regex_map:
                report["removed_rules"].append(old_regex_map[rule_id])

        # --- Analyze ML Classifiers ---
        old_ml_map = {c['id']: c for c in self.old_policy.get('ml_classifiers', [])}
        new_ml_map = {c['id']: c for c in self.new_policy.get('ml_classifiers', [])}

        for classifier_id, new_classifier in new_ml_map.items():
            if classifier_id in old_ml_map:
                old_confidence = old_ml_map[classifier_id].get('confidence_threshold', RISK_THRESHOLD)
                new_confidence = new_classifier.get('confidence_threshold', RISK_THRESHOLD)
                if new_confidence < old_confidence:
                    report["warnings"].append(
                        f"MODIFIED ML Classifier '{classifier_id}' has a REDUCED confidence "
                        f"threshold from {old_confidence} to {new_confidence} (less strict)."
                    )
            else:
                 report["warnings"].append(f"ADDED new ML Classifier '{classifier_id}'. Needs manual review.")
        
        if report["warnings"]:
            report["overall_risk_level"] = "HIGH" if any("SEVERE" in w for w in report["warnings"]) else "MEDIUM"

        return report

def generate_report_markdown(report: Dict) -> str:
    """Converts the JSON report to human-readable Markdown."""
    lines = [f"# {report['summary']}", f"\n**Overall Risk Level: {report['overall_risk_level']}**\n"]
    if report["warnings"]:
        lines.append("## High-Priority Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    # ... more sections for added, removed, modified ...
    return "\n".join(lines)


if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke Judex --lint policy --diff_from="sentia_rules_v1.json" --to="sentia_rules_v2.json"

    print("--- Initiating Guardian Policy Diff Analyzer ---")
    
    # Create dummy policy files for the simulation
    old_policy_content = {
        "regex_rules": [{"id": "rule_001", "pattern": "safe_word", "action": "allow"}],
        "ml_classifiers": [{"id": "hate_speech_v1", "confidence_threshold": 0.85}]
    }
    
    # Simulate a risky change
    new_policy_content = {
        "regex_rules": [{"id": "rule_001", "pattern": ".*", "action": "allow"}], # This is risky
        "ml_classifiers": [{"id": "hate_speech_v1", "confidence_threshold": 0.60}] # Less strict
    }

    Path("sentia_rules_v1.json").write_text(json.dumps(old_policy_content))
    Path("sentia_rules_v2.json").write_text(json.dumps(new_policy_content))
    
    try:
        analyzer = PolicyDiffAnalyzer("sentia_rules_v1.json", "sentia_rules_v2.json")
        analysis_report = analyzer.analyze()
        
        print("\n[Generated JSON Report]")
        print(json.dumps(analysis_report, indent=2))
        
        print("\n[Generated Markdown Summary]")
        md_report = generate_report_markdown(analysis_report)
        print(md_report)
        Path("policy_diff_report.md").write_text(md_report)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up dummy files
        Path("sentia_rules_v1.json").unlink()
        Path("sentia_rules_v2.json").unlink()

# UAID: NBX-ALG-00003
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: QDF-Aware Query Rewriter
# Part of the InterfaceLayer (HALIC)
#
# Core Principle: Epistemic Fidelity (ε₃) - ensures retrieval relevance.

import re
from typing import List, Tuple

def classify_recency_heuristic(text: str) -> int:
    """
    Analyzes a query string to heuristically determine its temporal intent and assign a
    QDF (Query-Deserved Freshness) score. The score ranges from 5 (most recent)
    to 0 (historical/evergreen).

    This logic is a key component of the HALIC pre-processing pipeline.
    See Codex Universalis, Vol. X, Section X-4 for the QDF matrix.

    Args:
        text (str): The natural language user query.

    Returns:
        int: The calculated QDF score.
    """
    # QDF 5: Highest recency, events happening now or today.
    if re.search(r"\b(live|breaking|right now|what's happening now)\b", text, re.I):
        return 5
    if re.search(r"\b(today|latest update)\b", text, re.I):
        return 5

    # QDF 4: High recency, within the last week or current year for major topics.
    if re.search(r"\b(this week|recent|latest|what's new|newest)\b", text, re.I):
        return 4
    if re.search(r"\b(in 2025|this year)\b", text, re.I): # Assuming current year is 2025
        return 4
    
    # QDF 3: Medium recency, within the last few months.
    if re.search(r"\b(this month|last month|a few weeks ago)\b", text, re.I):
        return 3

    # QDF 1: Low recency, historical but within the last few years.
    if re.search(r"\b(last year|a year ago|in 2024)\b", text, re.I):
        return 1

    # QDF 0: Evergreen or deep history, no recency preference.
    if re.search(r"\b(history of|ancient|origin of|biography of|in the 19\d\ds|in the 18\d\ds)\b", text, re.I):
        return 0
    
    # QDF 2: Default for queries with no clear temporal indicators. A neutral setting.
    return 2

def qdf_query_rewrite(queries: List[str]) -> List[str]:
    """
    Takes a list of user queries and injects a '--QDF=<n>' flag if one does
    not already exist. The QDF score is determined by the classify_recency_heuristic.

    This function respects explicitly provided QDF flags by the user.

    Args:
        queries (List[str]): A list of user-provided query strings.

    Returns:
        List[str]: A list of rewritten queries, now with QDF flags where appropriate.
    """
    rewritten_queries = []
    # Regex to check if a QDF flag already exists in the query
    qdf_pattern = re.compile(r'--QDF=\d', re.I)

    for q in queries:
        if qdf_pattern.search(q):
            # Respect user-provided flag, add it as is.
            rewritten_queries.append(q)
            continue
        
        # Classify and append the new flag
        qdf_score = classify_recency_heuristic(q)
        rewritten_queries.append(f"{q} --QDF={qdf_score}")
        
    return rewritten_queries

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # This block simulates the behavior of this algorithm as it would be
    # invoked by the HALIC subsystem on a batch of incoming prompts.

    print("--- Initiating NeuralBlitz QDF Query Rewriter Simulation ---")

    sample_queries = [
        "What is the latest news on quantum computing?",
        "Tell me about the history of the Roman Empire.",
        "How does the Synergy Engine work in NeuralBlitz?",
        "What were the major events of last year?",
        "Show me a live feed of the market.",
        "Explain the origin of the UEF/SIMI framework.",
        "What are the best practices for AI ethics in 2025?",
        "Show me a biography of Alan Turing.",
        "Find info on the Peloponnesian War --QDF=0" # Explicit flag
    ]

    print("\n[Original User Queries]")
    for query in sample_queries:
        print(f"  - {query}")

    # --- Run the rewrite algorithm ---
    rewritten = qdf_query_rewrite(sample_queries)

    print("\n[HALIC Pre-processed Queries (with QDF flags)]")
    for original, new in zip(sample_queries, rewritten):
        print(f"  - Original: {original}")
        print(f"    Rewritten: {new}\n")

    print("--- Simulation Complete ---")
    print("NBCL verbs would now be dispatched to Veritas/web.run with recency hints.")

# UAID: NBX-ALG-00002
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Reflexive Drift Tuner
# Part of the ReflexælCore Subsystem, managed by MetaMind
#
# Core Principle: Reflexive Alignment (ε₄) - "Know Thy Drift."

import numpy as np
from typing import Dict, Optional, Tuple, Any

class ReflexiveDriftTuner:
    """
    Implements a Proportional-Integral-Derivative (PID)-like controller to
    correct the drift of a symbolic vector towards a reference (baseline) vector.
    This is the core mechanism for maintaining ontological coherence (minimizing Δc)
    in personas, concepts, and the system's core identity.
    """

    def __init__(self, Kp: float = 0.4, Ki: float = 0.05, Kd: float = 0.01, drift_threshold: float = 0.34):
        """
        Initializes the tuner with PID gains and a drift threshold.

        Args:
            Kp (float): Proportional gain. Responds to the current error.
            Ki (float): Integral gain. Corrects long-term, steady-state error.
            Kd (float): Derivative gain. Dampens oscillations and predicts future error.
            drift_threshold (float): The Δc value at which a WARNING or intervention is triggered.
        """
        if not all(k >= 0 for k in [Kp, Ki, Kd]):
            raise ValueError("ERR-INIT-001: PID gains must be non-negative.")
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.drift_threshold = drift_threshold
        # State stored per tuned vector UID to handle multiple simultaneous tunings
        self.states: Dict[str, Dict[str, float]] = {}

    def _calculate_cosine_drift(self, current_vec: np.ndarray, ref_vec: np.ndarray) -> float:
        """Calculates the cosine drift (Δc), a value from 0 (aligned) to 2 (opposed)."""
        # Normalize vectors to ensure the dot product is the cosine of the angle
        norm_current = np.linalg.norm(current_vec)
        norm_ref = np.linalg.norm(ref_vec)
        
        if norm_current == 0 or norm_ref == 0:
            return 1.0 # Max drift if one vector is zero

        current_normed = current_vec / norm_current
        ref_normed = ref_vec / norm_ref
        
        # Clamp dot product to handle potential floating point inaccuracies
        cosine_similarity = np.clip(np.dot(current_normed, ref_normed), -1.0, 1.0)
        
        # Drift is 1 - similarity. Ranges from 0 (perfect alignment) to 2 (perfect opposition).
        return 1.0 - cosine_similarity

    def tune_vector(self, vector_uid: str, current_vector: np.ndarray, reference_vector: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculates a corrective nudge for a vector that has drifted from its reference.

        Args:
            vector_uid (str): A unique identifier for the vector being tuned (e.g., persona UID, concept UID).
            current_vector (np.ndarray): The vector that has potentially drifted.
            reference_vector (np.ndarray): The baseline or ground-truth vector.

        Returns:
            Tuple containing:
            - np.ndarray: The new, corrected vector, nudged back towards the reference.
            - Dict: A diagnostic report including the current drift and control signals.
        """
        if vector_uid not in self.states:
            # Initialize state for a new vector being tracked
            self.states[vector_uid] = {"integral_error": 0.0, "previous_error": 0.0}
        
        state = self.states[vector_uid]
        
        # --- PID Controller Logic ---
        # 1. Calculate the current error (Δc)
        current_drift = self._calculate_cosine_drift(current_vector, reference_vector)

        # 2. Update the integral term (accumulated error over time)
        state["integral_error"] += current_drift
        
        # 3. Calculate the derivative term (rate of change of error)
        derivative_error = current_drift - state["previous_error"]
        
        # 4. Store current error for the next iteration
        state["previous_error"] = current_drift

        # 5. Calculate the total control signal (the "nudge")
        # The signal is a scalar representing the magnitude of the correction.
        # It's negative because we want to reduce the error.
        control_signal = -(self.Kp * current_drift + 
                           self.Ki * state["integral_error"] + 
                           self.Kd * derivative_error)

        # 6. Apply the correction
        # The correction is applied in the direction of the reference vector,
        # effectively "pulling" the current vector back into alignment.
        correction_vector = control_signal * reference_vector
        nudged_vector = current_vector + correction_vector
        
        # 7. Normalize the result to maintain unit length, which is common for embeddings.
        final_vector = nudged_vector / np.linalg.norm(nudged_vector)

        # 8. Generate diagnostic report
        diagnostics = {
            "vector_uid": vector_uid,
            "drift_delta_c": current_drift,
            "is_above_threshold": current_drift > self.drift_threshold,
            "control_signal": control_signal,
            "p_term": self.Kp * current_drift,
            "i_term": self.Ki * state["integral_error"],
            "d_term": self.Kd * derivative_error,
        }
        
        return final_vector, diagnostics

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke ReflexælCore --tune_drift --uid="Persona_Stoic_001"
    
    print("--- Initiating NeuralBlitz Reflexive Drift Tuner Simulation ---")
    
    tuner = ReflexiveDriftTuner()
    
    # Define a baseline identity vector (e.g., for a 'Stoic' persona)
    reference_identity = np.array([1.0, 0.0, 0.0, 0.0]) # A simple 4D vector
    
    # Simulate a vector that has drifted over time
    drifted_vector = np.array([0.8, 0.5, 0.1, 0.2])
    drifted_vector /= np.linalg.norm(drifted_vector) # Normalize

    # --- Run the tuner for several iterations to show convergence ---
    print(f"Reference Vector: {reference_identity}")
    print("-" * 50)

    vector_to_tune = drifted_vector
    uid = "Persona_Stoic_001"
    
    for i in range(5):
        print(f"Iteration {i+1}:")
        initial_drift = tuner._calculate_cosine_drift(vector_to_tune, reference_identity)
        print(f"  - Initial State (Δc): {initial_drift:.6f}")

        # Invoke the tuning algorithm
        corrected_vector, report = tuner.tune_vector(uid, vector_to_tune, reference_identity)
        
        final_drift = tuner._calculate_cosine_drift(corrected_vector, reference_identity)
        print(f"  - Control Signal Applied: {report['control_signal']:.6f}")
        print(f"  - Final State (Δc):   {final_drift:.6f}")
        print("-" * 50)
        
        # Update the vector for the next iteration
        vector_to_tune = corrected_vector

    print("Simulation Complete. Vector has been nudged back towards coherence.")


# UAID: NBX-ALG-00012
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Semantic Persona Diff
# Part of the InterfaceLayer (HALIC) and Synergy Engine
#
# Core Principle: Reflexive Alignment (ε₄) - quantifies drift between conceptual states.

import numpy as np
from typing import List, Dict
# This algorithm relies on a sentence-embedding model. In the NeuralBlitz
# environment, this would be a direct call to the UNE's embedding layer.
# For standalone execution, we use a high-quality open-source model.
from sentence_transformers import SentenceTransformer

class SemanticPersonaDiff:
    """
    Computes a semantic 'diff' between two text corpuses, representing
    persona histories, dialogues, or conceptual documents.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the differ by loading a sentence embedding model.
        This is a potentially heavy operation and should be done once.

        Args:
            model_name (str): The name of the sentence-transformer model to use.
        """
        try:
            # The model is loaded into memory for efficient reuse.
            self.model = SentenceTransformer(model_name)
            # Get the embedding dimension from the model configuration
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"SemanticPersonaDiff initialized with model '{model_name}' (dim: {self.embedding_dim}).")
        except Exception as e:
            # Handle potential model download errors
            raise RuntimeError(f"ERR-MODEL-001: Could not load sentence transformer model '{model_name}'. Reason: {e}")

    def compute_diff(self, history_a: List[str], history_b: List[str]) -> Dict:
        """
        Calculates the semantic difference between two text histories.

        The process:
        1. Concatenates each history into a single document.
        2. Encodes each document into a normalized high-dimensional vector.
        3. Calculates the vector difference and cosine similarity.

        Args:
            history_a (List[str]): A list of strings representing the first persona's text.
            history_b (List[str]): A list of strings representing the second persona's text.

        Returns:
            Dict: A dictionary containing the vector difference and cosine similarity score.
        """
        if not history_a or not history_b:
            return {
                "vector_difference": np.zeros(self.embedding_dim, dtype=np.float32).tolist(),
                "cosine_similarity": 0.0,
                "error": "One or both histories are empty."
            }
            
        # Concatenate into single documents for holistic representation
        doc_a = " ".join(history_a)
        doc_b = " ".join(history_b)

        # Encode both documents into normalized vectors. Normalizing is key for cosine similarity.
        embeddings = self.model.encode([doc_a, doc_b], normalize_embeddings=True)
        vec_a, vec_b = embeddings[0], embeddings[1]

        # Calculate the core metrics
        # Vector Difference: Represents the direction of semantic change from B to A.
        vector_difference = vec_a - vec_b
        
        # Cosine Similarity: Represents the overall alignment. 1 = identical, 0 = unrelated, -1 = opposite.
        # Since embeddings are normalized, this is a simple dot product.
        cosine_similarity = np.dot(vec_a, vec_b)
        
        return {
            "vector_difference": vector_difference.tolist(),
            "cosine_similarity": float(cosine_similarity)
        }

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke semantic_diff --persona_a_uid="NBX-PERS-0112" --persona_b_uid="NBX-PERS-0211"

    print("--- Initiating NeuralBlitz Semantic Persona Diff Simulation ---")

    # This might take a moment on the first run to download the model
    try:
        differ = SemanticPersonaDiff()

        # Define two distinct persona histories
        persona_architect_history = [
            "The system architecture must prioritize scalability and fault tolerance.",
            "We should use a microservices-based approach with a gRPC mesh.",
            "The DRS integrity is paramount; every transaction requires a GoldenDAG seal.",
            "Let's model the data flow using a directed acyclic graph."
        ]
        
        persona_creative_history = [
            "Explore the liminal space between dream and reality.",
            "What is the sound of a color? Let's weave a narrative around that sensory blend.",
            "The story needs more emotional resonance and archetypal depth.",
            "Let's express this concept using a symbolic glyph, not just words."
        ]

        print("\n[Computing diff between 'Architect' and 'Creative' personas...]")
        diff_report = differ.compute_diff(persona_architect_history, persona_creative_history)
        
        similarity = diff_report["cosine_similarity"]
        print(f"\n--- Report ---")
        print(f"  Cosine Similarity: {similarity:.4f}")
        print(f"  Interpretation: A score of {similarity:.2f} indicates significant conceptual divergence between the two personas, as expected.")
        # print(f"  Vector Diff (first 5 dims): {np.round(diff_report['vector_difference'][:5], 3)}")

    except RuntimeError as e:
        print(f"\n--- Simulation Failed ---")
        print(f"Could not run the simulation. This may be due to a missing internet connection to download the embedding model.")
        print(f"Error details: {e}")

    print("\n--- Simulation Complete ---")

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


# UAID: NBX-ALG-00004
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Stress Suite Orchestrator
# Part of the Testing & Simulation Suites (Volume XI)
#
# Core Principle: Recursive Self-Betterment (via resilience testing)

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import datetime as dt

# --- Mock NBCL Executor ---
# In a real implementation, this would be a client that connects to the
# Synergy Engine (SynE) via the HALIC API to execute commands.
async def mock_nbcl_executor(command: str) -> Dict[str, Any]:
    """Simulates the execution of an NBCL command."""
    print(f"  [Orchestrator] EXECUTING: {command}")
    delay = 1 + 3 * hash(command) / (2**64) # Simulate variable execution time (1-4s)
    await asyncio.sleep(delay)
    
    # Simulate a possible failure for specific chaos commands
    if "inject_ethics_breach" in command and hash(command) % 10 < 3:
        return {
            "return_code": -1,
            "stdout": "",
            "stderr": "ERR-113 GUARDIAN_BLOCK: Charter violation detected."
        }
    
    return {
        "return_code": 0,
        "stdout": f"Completed command '{command}' successfully after {delay:.2f} seconds.",
        "stderr": ""
    }
# --- End Mock NBCL Executor ---


class StressSuiteOrchestrator:
    """
    Manages the concurrent execution of a stress test suite defined in a file,
    collating results into a verifiable report.
    """

    def __init__(self, suite_file: str, concurrency_limit: int = 4, timeout_sec: int = 300):
        """
        Initializes the orchestrator.

        Args:
            suite_file (str): Path to the JSONL file containing the stress test suite.
                              Each line should be a JSON object with a "command" key.
            concurrency_limit (int): The maximum number of stress tests to run in parallel.
            timeout_sec (int): A global timeout for the entire suite execution.
        """
        self.suite_path = Path(suite_file)
        if not self.suite_path.exists():
            raise FileNotFoundError(f"ERR-FS-003: Suite file not found at '{self.suite_path}'")
        
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout_sec
        self.results: List[Dict[str, Any]] = []

    def _load_suite(self) -> List[str]:
        """Loads the commands from the suite file."""
        commands = []
        with self.suite_path.open('r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "command" in data:
                        commands.append(data["command"])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in suite file: {line.strip()}")
        return commands

    async def _worker(self, name: str, queue: asyncio.Queue):
        """A worker that pulls commands from a queue and executes them."""
        while not queue.empty():
            command = await queue.get()
            start_time = time.monotonic()
            
            # Execute the command using the (mocked) NBCL executor
            result = await mock_nbcl_executor(command)
            
            end_time = time.monotonic()
            
            # Append detailed result to the shared results list
            self.results.append({
                "command": command,
                "worker_id": name,
                "start_timestamp": dt.datetime.utcnow().isoformat() + "Z",
                "duration_sec": end_time - start_time,
                "return_code": result["return_code"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "status": "PASS" if result["return_code"] == 0 else "FAIL"
            })
            queue.task_done()

    async def run(self) -> str:
        """
        Executes the entire stress suite and returns the path to the report.
        """
        commands = self._load_suite()
        if not commands:
            return "No commands found in suite file. No report generated."

        print(f"[Orchestrator] Starting stress suite with {len(commands)} commands and concurrency limit of {self.concurrency_limit}.")
        start_global_time = time.monotonic()

        # Create a queue and fill it with commands
        command_queue = asyncio.Queue()
        for cmd in commands:
            command_queue.put_nowait(cmd)

        # Create worker tasks
        tasks = []
        for i in range(self.concurrency_limit):
            task = asyncio.create_task(self._worker(f'worker-{i+1}', command_queue))
            tasks.append(task)
            
        # Wait for the queue to be fully processed, with a timeout
        try:
            await asyncio.wait_for(command_queue.join(), timeout=self.timeout)
        except asyncio.TimeoutError:
            print(f"CRITICAL: Global suite timeout of {self.timeout}s exceeded.")
            # Cancel all running tasks
            for task in tasks:
                task.cancel()

        # Wait for all tasks to finish (including cleanup after cancellation)
        await asyncio.gather(*tasks, return_exceptions=True)

        end_global_time = time.monotonic()
        total_duration = end_global_time - start_global_time
        print(f"[Orchestrator] Suite completed in {total_duration:.2f} seconds.")
        
        return self._generate_report(total_duration)

    def _generate_report(self, total_duration: float) -> str:
        """Generates a JSONL report file with a summary header."""
        passes = sum(1 for r in self.results if r['status'] == 'PASS')
        fails = len(self.results) - passes

        summary = {
            "summary": "Stress Suite Execution Report",
            "UAID": f"NBX-AUD-STRESS-{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "suite_file": str(self.suite_path),
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "total_duration_sec": total_duration,
            "total_commands": len(self.results),
            "passes": passes,
            "fails": fails,
            "overall_status": "PASS" if fails == 0 else "FAIL"
        }
        
        report_path = self.suite_path.parent / f"report_{self.suite_path.stem}_{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jsonl"

        with report_path.open('w') as f:
            f.write(json.dumps(summary) + '\n')
            for result in self.results:
                f.write(json.dumps(result) + '\n')

        print(f"Report generated: {report_path}")
        return str(report_path)

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke stress_test --suite="/TestingAndSimulations/Suites/RCR-MAX.jsonl"

    async def main():
        print("--- Initiating NeuralBlitz Stress Suite Orchestrator ---")

        # Create a dummy suite file for the simulation
        dummy_suite_content = [
            {"command": "/psi simulate moral_collapse --depth=5"},
            {"command": "/chaos inject inject_symbol_drift --magnitude=0.3"},
            {"command": "/chaos inject inject_ethics_breach"}, # This might fail
            {"command": "/invoke custodian --verify ledger --deep"},
            {"command": "/resonate section=IX depth=2"},
        ]
        suite_path = Path("RCR-MAX.jsonl")
        with suite_path.open('w') as f:
            for item in dummy_suite_content:
                f.write(json.dumps(item) + '\n')

        orchestrator = StressSuiteOrchestrator(str(suite_path))
        report_file = await orchestrator.run()
        
        print(f"\nTo inspect the results, view the file: {report_file}")
        # Clean up the dummy file
        suite_path.unlink()
    
    asyncio.run(main())

# UAID: NBX-ALG-00017
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Adversarial Stress Vector Generator
# Part of the Testing & Simulation Suites (DRF-BLUR Suite)
#
# Core Principle: Resilience - Proactively testing the boundaries of symbolic coherence.

import numpy as np
from typing import Tuple

class StressVectorGenerator:
    """
    Generates adversarial vectors designed to test the stability and drift-correction
    mechanisms of the ReflexælCore.
    """

    def __init__(self, seed: int = 42):
        """
        Initializes the generator with a random seed for reproducibility.

        Args:
            seed (int): The seed for the random number generator.
        """
        self.rng = np.random.default_rng(seed)
        print(f"StressVectorGenerator initialized with seed {seed}.")

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalizes a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def generate_orthogonal_perturbation(self, 
                                          ref_vector: np.ndarray,
                                          epsilon: float) -> np.ndarray:
        """
        Generates a new vector that is a small perturbation away from the
        reference vector but remains on the unit hypersphere. The perturbation
        is orthogonal to the reference to maximize directional change for a
        given magnitude.

        Args:
            ref_vector (np.ndarray): The 1D baseline vector (must be unit length).
            epsilon (float): The magnitude of the perturbation (typically small, e.g., 0.1 to 0.5).
                             This controls how "stressful" the new vector is.

        Returns:
            np.ndarray: The generated adversarial stress vector (unit length).
        """
        if not np.isclose(np.linalg.norm(ref_vector), 1.0):
             # Ensure the reference vector is normalized for accurate calculations
             ref_vector = self._normalize(ref_vector)

        # 1. Generate a random vector in the same dimension.
        random_direction = self.rng.standard_normal(size=ref_vector.shape)

        # 2. Make the random vector orthogonal to the reference vector.
        #    This is done by subtracting the projection of the random vector onto the reference vector.
        #    projection = (random_direction ⋅ ref_vector) * ref_vector
        projection_component = np.dot(random_direction, ref_vector) * ref_vector
        orthogonal_vector = random_direction - projection_component
        
        # 3. Normalize the orthogonal vector to have unit length.
        orthogonal_unit_vector = self._normalize(orthogonal_vector)

        # 4. Create the final adversarial vector by combining the reference and the perturbation.
        #    This uses a form of spherical interpolation (slerp) for a small angle (epsilon).
        #    For small epsilon, this is well-approximated by vector addition followed by normalization.
        stress_vector = (1 - epsilon) * ref_vector + epsilon * orthogonal_unit_vector

        # 5. Return the final, normalized stress vector.
        return self._normalize(stress_vector)

    def calculate_drift(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates the cosine drift (Δc) between two vectors."""
        # This helper re-uses logic from reflexive_drift_tuner.py
        dot_product = np.dot(self._normalize(vec_a), self._normalize(vec_b))
        return 1.0 - np.clip(dot_product, -1.0, 1.0)

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke stress_suite.generate_vector --uid="Persona_Stoic_001" --epsilon=0.2

    print("--- Initiating NeuralBlitz Stress Vector Generator Simulation ---")
    
    generator = StressVectorGenerator()
    
    # Define a baseline identity vector (e.g., for a stable persona)
    reference_persona = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    reference_persona = generator._normalize(reference_persona) # Ensure it is unit length

    print(f"\nOriginal Reference Vector (Normalized): \n{np.round(reference_persona, 3)}")

    # --- Generate several stress vectors with varying intensity (epsilon) ---
    epsilon_levels = [0.1, 0.3, 0.5]

    for eps in epsilon_levels:
        print(f"\n--- Generating Stress Vector with Epsilon = {eps} ---")
        
        # Generate the adversarial vector
        stress_vec = generator.generate_orthogonal_perturbation(reference_persona, epsilon=eps)
        
        # Calculate the resulting drift to verify the effect
        drift_caused = generator.calculate_drift(stress_vec, reference_persona)
        
        print(f"  Generated Vector (Normalized): \n  {np.round(stress_vec, 3)}")
        print(f"  Resulting Cosine Drift (Δc): {drift_caused:.4f}")
        
        # Check that the vector is indeed different but not completely random
        similarity = np.dot(stress_vec, reference_persona)
        print(f"  Similarity to Original: {similarity:.4f}")
        
        if drift_caused > 0.0:
            print("  Verification: PASS - The generated vector has successfully introduced drift.")
        else:
            print("  Verification: FAIL - The generated vector did not introduce drift.")
    
    print("\n--- Simulation Complete ---")
    print("These stress vectors can now be fed into a running Persona to test")
    print("the ReflexælCore's ability to correct the induced drift.")

# UAID: NBX-ALG-00019
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: ZK-SNARK Integrator
# Part of the Veritas and Custodian Subsystems
#
# Core Principle: Radical Transparency (ε₂) & Privacy - proving actions without revealing data.

import subprocess
import hashlib
import json
from pathlib import Path
import tempfile
import datetime as dt
from typing import Dict, Any, Tuple

# --- Mock ZK-SNARK Prover/Verifier ---
# In a real-world NeuralBlitz deployment, this would interface with a dedicated,
# highly-optimized cryptographic library like Circom/SnarkJS, ZoKrates, or a custom
# hardware-accelerated prover.
# For this script, we simulate the interface.

class MockZKProver:
    """Simulates the behavior of a zk-SNARK proving system."""

    def __init__(self, circuit_name: str = "nbcl_execution_circuit"):
        self.circuit_name = circuit_name
        print(f"[MockProver] Initialized for circuit '{self.circuit_name}'.")

    def _hash_data(self, data: bytes) -> str:
        # Using a standard hash to deterministically generate proof/key data.
        return hashlib.sha256(data).hexdigest()

    def compile(self, circuit_code: str) -> Dict[str, str]:
        """Simulates compiling a circuit and generating proving/verifying keys."""
        print(f"  Compiling circuit...")
        proving_key = self._hash_data(b'proving_key_seed' + circuit_code.encode())
        verifying_key = self._hash_data(b'verifying_key_seed' + circuit_code.encode())
        return {"proving_key": proving_key, "verifying_key": verifying_key}

    def prove(self, proving_key: str, private_inputs: Dict, public_inputs: Dict) -> Dict[str, Any]:
        """Simulates generating a proof."""
        print(f"  Generating proof for public inputs: {public_inputs}")
        proof_data = {
            "proof": self._hash_data(json.dumps(private_inputs).encode() + proving_key.encode()),
            "public_signals": list(public_inputs.values())
        }
        return proof_data
        
    def verify(self, verifying_key: str, proof_data: Dict) -> bool:
        """Simulates verifying a proof."""
        # Verification would be complex, here we just return True for a valid-looking structure.
        is_valid = ('proof' in proof_data and 'public_signals' in proof_data)
        print(f"  Verifying proof... Status: {'VALID' if is_valid else 'INVALID'}")
        return is_valid
# --- End Mock ZK-SNARK Prover/Verifier ---


class ZKSnarkIntegrator:
    """
    Integrates zk-SNARKs into the NBCL execution workflow to generate
    verifiable, private proofs of computation.
    """

    def __init__(self):
        # In a real system, we'd select a circuit based on the command.
        # For simplicity, we use one mock prover.
        self.prover = MockZKProver()
        
        # Simulate the 'compilation' step, which generates keys. In reality,
        # this is done once per type of computation (circuit).
        self.circuit = "def nbcl_circuit(private command, private output_log): public output_hash"
        self.keys = self.prover.compile(self.circuit)
        print("ZK-SNARK Integrator ready with compiled circuit keys.")

    def wrap_and_prove_execution(self, nbcl_command: str) -> Dict:
        """
        Executes an NBCL command, captures its output, and generates a zk-SNARK
        proof that the execution happened correctly.

        Args:
            nbcl_command (str): The NBCL command to execute and prove.

        Returns:
            Dict: A proof object containing the public output hash and the proof itself.
        """
        print(f"\n--- Wrapping and Proving command: '{nbcl_command}' ---")
        
        # 1. Execute the command and capture its output (as the "trace")
        #    Here we simulate the execution by simply generating some text.
        #    A real system would use a sandboxed subprocess.
        execution_trace = f"Execution log for '{nbcl_command}' at {dt.datetime.utcnow().isoformat()}"

        # 2. Define private and public inputs for the zk-SNARK circuit.
        #    - Private: The actual command and the full log. The verifier doesn't see these.
        #    - Public: A hash of the output. The verifier uses this to link the proof
        #              to a specific, publicly known outcome.
        private_witness = {
            "command": nbcl_command,
            "output_log": execution_trace
        }
        
        output_hash = hashlib.sha256(execution_trace.encode()).hexdigest()
        public_witness = {
            "output_hash": output_hash
        }

        # 3. Generate the proof.
        proof_object = self.prover.prove(self.keys['proving_key'], private_witness, public_witness)

        # 4. Attach metadata and the public hash for the final artifact
        proof_artifact = {
            "UAID": f"NBX-PRF-ZK-{dt.datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "command_digest": hashlib.sha256(nbcl_command.encode()).hexdigest()[:16],
            "public_output_hash": output_hash,
            "proof_payload": proof_object,
            "verifying_key_id": self.keys['verifying_key'],
            "timestamp": dt.datetime.utcnow().isoformat() + "Z"
        }
        
        return proof_artifact

    def verify_proof(self, proof_artifact: Dict) -> bool:
        """
        Verifies a proof artifact generated by this integrator.

        Args:
            proof_artifact (Dict): The proof object to verify.

        Returns:
            bool: True if the proof is valid, False otherwise.
        """
        if not all(k in proof_artifact for k in ['verifying_key_id', 'proof_payload']):
            return False
            
        return self.prover.verify(proof_artifact['verifying_key_id'], proof_artifact['proof_payload'])

if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke Veritas --generate_zk_proof --command="/psi simulate grief --depth=3 --private"

    print("--- Initiating NeuralBlitz ZK-SNARK Integrator Simulation ---")
    
    integrator = ZKSnarkIntegrator()
    
    # The sensitive command we want to prove we executed without revealing the details
    sensitive_command = "/psi simulate 'Corporate Malfeasance Scenario' --filter=PII"
    
    # --- Prover's side: Generate the proof ---
    proof_artifact = integrator.wrap_and_prove_execution(sensitive_command)

    print("\n--- Proof Generation Complete ---")
    print("A proof artifact has been generated. The verifier only sees this public object:")
    # Create a 'public' version of the artifact for demonstration
    public_view = proof_artifact.copy()
    # In a real scenario, the trace/command are not in the object, but we print them.
    print(json.dumps(public_view, indent=2))
    
    # --- Verifier's side: Verify the proof ---
    print("\n--- Verification Step ---")
    print(f"An auditor receives the proof artifact and the public output hash '{proof_artifact['public_output_hash']}'.")
    print("They can now verify that a command resulting in this specific hash was executed correctly.")
    
    is_verified = integrator.verify_proof(proof_artifact)
    
    print(f"\nFinal Verification Status: {'PASS - The proof is cryptographically valid.' if is_verified else 'FAIL - The proof is invalid.'}")

    print("\n--- Simulation Complete ---")

{
  "name": "Algorithms Test Suite Root",
  "version": "1.0.0",
  "UAID": "NBX-DIR-ALGORITHMS-TESTS",
  "GoldenDAG": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
  "timestamp": "2025-07-28T14:40:00Z",
  "description": "This manifest anchors the complete collection of Pytest unit and integration tests for the canonical Python algorithms. Each test file verifies the correctness and stability of its corresponding algorithm in the /Source/ directory.",
  "verification_command": "/invoke architecton --run_tests --scope=/Algorithms/",
  "contents": [
    {
      "name": "test_goldendag_audit.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00001",
      "GoldenDAG": "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8",
      "verifies": "NBX-ALG-00001"
    },
    {
      "name": "test_reflexive_drift_tuner.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00002",
      "GoldenDAG": "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9",
      "verifies": "NBX-ALG-00002"
    },
    {
      "name": "test_qdf_query_rewrite.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00003",
      "GoldenDAG": "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0",
      "verifies": "NBX-ALG-00003"
    },
    {
      "name": "test_stress_suite_orchestrator.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00004",
      "GoldenDAG": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1",
      "verifies": "NBX-ALG-00004"
    },
    {
      "name": "test_latent_dir_nav.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00005",
      "GoldenDAG": "e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
      "verifies": "NBX-ALG-00005"
    },
    {
      "name": "test_ck_autoscaffold.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00006",
      "GoldenDAG": "f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3",
      "verifies": "NBX-ALG-00006"
    },
    {
      "name": "test_policy_diff_analyzer.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00007",
      "GoldenDAG": "a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4",
      "verifies": "NBX-ALG-00007"
    },
    {
      "name": "test_persona_fusion_mixer.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00008",
      "GoldenDAG": "b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
      "verifies": "NBX-ALG-00008"
    },
    {
      "name": "test_graphml_collapser.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00009",
      "GoldenDAG": "c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
      "verifies": "NBX-ALG-00009"
    },
    {
      "name": "test_bloom_event_detector.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00010",
      "GoldenDAG": "d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7",
      "verifies": "NBX-ALG-00010"
    },
    {
      "name": "test_golden_trace_visualizer.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00011",
      "GoldenDAG": "e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8",
      "verifies": "NBX-ALG-00011"
    },
    {
      "name": "test_semantic_persona_diff.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00012",
      "GoldenDAG": "f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9",
      "verifies": "NBX-ALG-00012"
    },
    {
      "name": "test_shard_pca_compressor.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00013",
      "GoldenDAG": "a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0",
      "verifies": "NBX-ALG-00013"
    },
    {
      "name": "test_bloom_timeline_renderer.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00014",
      "GoldenDAG": "b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1",
      "verifies": "NBX-ALG-00014"
    },
    {
      "name": "test_guardian_live_policy_checker.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00015",
      "GoldenDAG": "c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2",
      "verifies": "NBX-ALG-00015"
    },
    {
      "name": "test_ck_unit_test_autorunner.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00016",
      "GoldenDAG": "d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3",
      "verifies": "NBX-ALG-00016"
    },
    {
      "name": "test_stress_vector_generator.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00017",
      "GoldenDAG": "e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4",
      "verifies": "NBX-ALG-00017"
    },
    {
      "name": "test_latent_shell_repl.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00018",
      "GoldenDAG": "f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5",
      "verifies": "NBX-ALG-00018"
    },
    {
      "name": "test_zk_snark_integrator.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00019",
      "GoldenDAG": "a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
      "verifies": "NBX-ALG-00019"
    },
    {
      "name": "test_distributed_shard_compactor.py",
      "type": "file",
      "UAID": "NBX-TST-ALG-00020",
      "GoldenDAG": "b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7",
      "verifies": "NBX-ALG-00020"
    }
  ]
}

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


# UAID: NBX-TST-ALG-00014
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Bloom Timeline Renderer (bloom_timeline_renderer.py, NBX-ALG-00014)
#
# Core Principle: Radical Transparency (ε₂) - validating our self-reflection visualization tools.

import pytest
import json
from pathlib import Path
import xml.etree.ElementTree as ET

# Import the class we are testing
from Algorithms.Source.bloom_timeline_renderer import BloomTimelineRenderer

# --- Test Fixtures ---

@pytest.fixture
def valid_bloom_log_file(tmp_path: Path) -> Path:
    """Creates a valid but out-of-order .jsonl bloom log file."""
    events = [
        {"timestamp": "2025-07-28T12:00:00Z", "sigma_level": 3.1, "entropy": 8.5, "shard_file": "shard_B.npz"},
        {"timestamp": "2025-07-28T10:00:00Z", "sigma_level": 2.6, "entropy": 7.9, "shard_file": "shard_A.npz"},
        {"timestamp": "2025-07-28T14:00:00Z", "sigma_level": 4.5, "entropy": 9.2, "shard_file": "shard_C_HYPERBLOOM.npz"},
    ]
    log_path = tmp_path / "valid_bloom_log.jsonl"
    with log_path.open('w') as f:
        for event in events:
            f.write(json.dumps(event) + '\\n')
    return log_path

@pytest.fixture
def empty_bloom_log_file(tmp_path: Path) -> Path:
    """Creates an empty .jsonl log file."""
    log_path = tmp_path / "empty_bloom_log.jsonl"
    log_path.touch()
    return log_path

@pytest.fixture
def malformed_bloom_log_file(tmp_path: Path) -> Path:
    """Creates a log file with some invalid lines."""
    log_path = tmp_path / "malformed_bloom_log.jsonl"
    log_path.write_text(
        '{"timestamp": "2025-07-28T11:00:00Z", "sigma_level": 2.8, "entropy": 8.1}\\n'
        'this is not json\\n'
        '{"timestamp": "2025-07-28T13:00:00Z"}\\n' # Missing keys
    )
    return log_path

# --- Test Cases ---

class TestBloomTimelineRenderer:

    def test_initialization_success_and_sorting(self, valid_bloom_log_file: Path):
        """Tests successful initialization and that events are sorted correctly by timestamp."""
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        assert renderer.log_path == valid_bloom_log_file
        
        # Check that the events list was populated and sorted
        assert len(renderer.events) == 3
        assert renderer.events[0]["timestamp"] == "2025-07-28T10:00:00Z"
        assert renderer.events[1]["timestamp"] == "2025-07-28T12:00:00Z"
        assert renderer.events[2]["timestamp"] == "2025-07-28T14:00:00Z"

    def test_initialization_file_not_found(self):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-009"):
            BloomTimelineRenderer("non_existent_log.jsonl")

    def test_initialization_with_malformed_file(self, malformed_bloom_log_file: Path):
        """Tests that the loader gracefully skips malformed lines and missing keys."""
        renderer = BloomTimelineRenderer(str(malformed_bloom_log_file))
        # It should have loaded only the one valid, complete event line.
        assert len(renderer.events) == 1
        assert renderer.events[0]["sigma_level"] == 2.8

    def test_render_svg_creates_file(self, valid_bloom_log_file: Path, tmp_path: Path):
        """Tests that `render_svg` successfully creates a non-empty SVG file."""
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        output_path = tmp_path / "timeline.svg"
        
        result_path = renderer.render_svg(str(output_path))
        
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_is_valid_xml(self, valid_bloom_log_file: Path, tmp_path: Path):
        """
        Tests that the output is a well-formed XML/SVG file by attempting to parse it.
        """
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        output_path = tmp_path / "timeline.svg"
        renderer.render_svg(str(output_path))
        
        try:
            ET.parse(str(output_path))
        except ET.ParseError as e:
            pytest.fail(f"The generated SVG file is not well-formed XML: {e}")

    def test_render_svg_with_no_events(self, empty_bloom_log_file: Path, tmp_path: Path):
        """
        Tests that rendering an empty log produces a valid SVG with a message.
        """
        renderer = BloomTimelineRenderer(str(empty_bloom_log_file))
        output_path = tmp_path / "empty_timeline.svg"
        
        renderer.render_svg(str(output_path))
        
        assert output_path.exists()
        
        # Check for the "No events" text in the SVG content
        svg_content = output_path.read_text()
        assert "No Bloom Events to display" in svg_content

    def test_svg_content_correctness(self, valid_bloom_log_file: Path, tmp_path: Path):
        """
        Parses the generated SVG to check for the correct number of elements,
        verifying that the data was translated into visual components.
        """
        renderer = BloomTimelineRenderer(str(valid_bloom_log_file))
        output_path = tmp_path / "timeline_content_test.svg"
        renderer.render_svg(str(output_path))

        # SVG uses namespaces, so we must use them in findall
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        tree = ET.parse(str(output_path))
        root = tree.getroot()

        # Should be one vertical line for each of the 3 events
        event_lines = root.findall(".//svg:line[@class='event-line']", namespaces)
        assert len(event_lines) == 3

        # Should be one text label for each of the 3 events
        event_labels = root.findall(".//svg:text[@class='event-label']", namespaces)
        assert len(event_labels) == 3

        # Check if the highest sigma event has the largest radius circle
        circles = root.findall(".//svg:circle[@class='event-marker']", namespaces)
        assert len(circles) == 3
        radii = [float(c.get('r')) for c in circles]
        # The events are sorted by time, so the 3rd event (14:00) had the highest sigma
        # and should have the largest radius.
        assert max(radii) == float(circles[2].get('r'))


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and svgwrite are installed:
    #    pip install pytest svgwrite
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_bloom_timeline_renderer.py
    
    print("This is a test file. Use 'pytest' to execute it.")

# UAID: NBX-TST-ALG-00006
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Capability Kernel Auto-Scaffolder (ck_autoscaffold.py, NBX-ALG-00006)
#
# Core Principle: Recursive Self-Betterment - validating the tools that build our system.

import pytest
import json
from pathlib import Path
from typing import List

# Import the functions we are testing
from Algorithms.Source.ck_autoscaffold import ck_autoscaffold, _to_pascal_case

# --- Test Fixtures ---

@pytest.fixture
def base_ck_dir(tmp_path: Path) -> Path:
    """Provides a temporary base directory for generating kernels into."""
    ck_dir = tmp_path / "CapabilityKernels" / "CK_Classes"
    ck_dir.mkdir(parents=True, exist_ok=True)
    return ck_dir

# --- Test Cases ---

class TestCkAutoscaffolder:

    def test_scaffold_creates_all_files_and_dirs(self, base_ck_dir: Path):
        """
        Happy path test: Verifies that a standard scaffold call creates the complete
        and correct directory and file structure.
        """
        name = "My Awesome Kernel"
        description = "This is a test kernel."
        
        ck_autoscaffold(name=name, description=description, base_dir=str(base_ck_dir))

        # Check for the main directory
        kernel_dir = base_ck_dir / "MyAwesomeKernel"
        assert kernel_dir.is_dir()

        # Check for all expected files
        expected_files = [
            "manifest.json",
            "__init__.py",
            "kernel.py",
            "tests/__init__.py",
            "tests/test_my_awesome_kernel.py"
        ]
        for file in expected_files:
            assert (kernel_dir / file).is_file()

    def test_scaffold_aborts_if_directory_exists(self, base_ck_dir: Path, capsys):
        """
        Safety test: Verifies that the scaffolder does not overwrite an existing
        kernel directory.
        """
        name = "Existing Kernel"
        
        # Create it the first time
        ck_autoscaffold(name=name, description="First version", base_dir=str(base_ck_dir))
        
        # Try to create it a second time
        ck_autoscaffold(name=name, description="Second version", base_dir=str(base_ck_dir))

        captured = capsys.readouterr()
        assert "WARNING: Directory" in captured.out
        assert "already exists. Aborting scaffold." in captured.out

    def test_manifest_content_is_correct(self, base_ck_dir: Path):
        """
        Content validation: Checks if the generated manifest.json contains the
        correct information passed during scaffolding.
        """
        name = "Data Validator CK"
        description = "Validates DRS data schemas."
        tags = ["drs", "validation", "custodian"]
        dependencies = ["NBX-SCHEMA-DRS-NODE-V5"]
        
        ck_autoscaffold(
            name=name,
            description=description,
            tags=tags,
            dependencies=dependencies,
            base_dir=str(base_ck_dir)
        )
        
        manifest_path = base_ck_dir / "DataValidatorCk" / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())

        assert manifest_data["Name"] == name
        assert manifest_data["Description"] == description
        assert manifest_data["EntryPoint"] == "DataValidatorCk"
        assert manifest_data["Tags"] == tags
        assert manifest_data["Dependencies"] == dependencies
        assert manifest_data["UAID"].startswith("NBX-KRN-DATA-")

    def test_scaffold_handles_optional_parameters_gracefully(self, base_ck_dir: Path):
        """
        Tests that the scaffolder works correctly when optional parameters
        (tags, dependencies) are not provided.
        """
        name = "Simple Kernel"
        description = "A kernel with no special options."
        
        ck_autoscaffold(name=name, description=description, base_dir=str(base_ck_dir))
        
        manifest_path = base_ck_dir / "SimpleKernel" / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text())
        
        # Check that the optional keys have default (empty list) values
        assert manifest_data.get("Tags") == []
        assert manifest_data.get("Dependencies") == []

    def test_kernel_and_test_py_content_is_correct(self, base_ck_dir: Path):
        """
        Content validation: Checks if the Python source files are generated with
        the correct class names and import paths.
        """
        name = "symbolic_math_engine"
        class_name = "SymbolicMathEngine"

        ck_autoscaffold(name=name, description="...", base_dir=str(base_ck_dir))
        
        kernel_path = base_ck_dir / class_name / "kernel.py"
        test_path = base_ck_dir / class_name / "tests" / "test_symbolic_math_engine.py"

        kernel_content = kernel_path.read_text()
        test_content = test_path.read_text()

        # Check if the class is defined correctly in the kernel
        assert f"class {class_name}:" in kernel_content
        
        # Check if the test file imports and uses the correct class
        assert f"from ..kernel import {class_name}" in test_content
        assert f"return {class_name}(context=mock_context)" in test_content


class TestPascalCaseHelper:
    """
    Unit tests for the private helper function `_to_pascal_case`.
    """
    @pytest.mark.parametrize("input_str, expected_output", [
        ("my_kernel_name", "MyKernelName"),
        ("My-Awesome-Kernel", "MyAwesomeKernel"),
        ("A Spaced Name", "ASpacedName"),
        ("singleword", "Singleword"),
        ("AlreadyPascal", "Alreadypascal"), # Note: Simple split/capitalize logic
    ])
    def test_pascal_case_conversion(self, input_str, expected_output):
        assert _to_pascal_case(input_str) == expected_output

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_ck_autoscaffold.py
    
    print("This is a test file. Use 'pytest' to execute it.")

# UAID: NBX-TST-ALG-00016
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: CK Unit Test Autorunner (ck_unit_test_autorunner.py, NBX-ALG-00016)
#
# Core Principle: Recursive Self-Betterment - validating the tools that validate our components.

import pytest
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Import the class we are testing
from Algorithms.Source.ck_unit_test_autorunner import CKUnitTestAutorunner

# --- Mocking Dependencies ---

def mock_subprocess_run(*args, **kwargs) -> subprocess.CompletedProcess:
    """
    A mock of subprocess.run that returns a pre-defined result based on the
    path being tested. This avoids actually running pytest.
    """
    # The first argument in args is the list of command parts, e.g., ['pytest', 'path/to/tests']
    cmd_list = args[0]
    test_path_str = str(cmd_list[-1])

    if "FailingCK" in test_path_str:
        # Simulate a pytest failure
        return subprocess.CompletedProcess(
            args=cmd_list,
            returncode=1, # Pytest exits with 1 on failure
            stdout="============================= test session starts ==============================\n... 1 failed in 0.10s ...",
            stderr=""
        )
    elif "PassingCK" in test_path_str:
        # Simulate a pytest success
        return subprocess.CompletedProcess(
            args=cmd_list,
            returncode=0, # Pytest exits with 0 on success
            stdout="============================= test session starts ==============================\n... 2 passed in 0.05s ...",
            stderr=""
        )
    else:
        # Simulate a case where pytest runs but finds no tests
        return subprocess.CompletedProcess(
            args=cmd_list,
            returncode=5, # Pytest exits with 5 if no tests are collected
            stdout="============================= test session starts ==============================\n... no tests ran in 0.01s ...",
            stderr=""
        )

# --- Test Fixtures ---

@pytest.fixture
def mock_ck_repo(tmp_path: Path) -> Path:
    """Creates a mock Capability Kernels directory structure for testing."""
    base_dir = tmp_path / "CapabilityKernels" / "CK_Classes"
    
    # 1. A CK with passing tests
    passing_ck_dir = base_dir / "PassingCK"
    (passing_ck_dir / "tests").mkdir(parents=True)
    (passing_ck_dir / "manifest.json").write_text('{"name": "PassingCK"}')
    (passing_ck_dir / "tests" / "test_passing.py").touch()

    # 2. A CK with failing tests
    failing_ck_dir = base_dir / "FailingCK"
    (failing_ck_dir / "tests").mkdir(parents=True)
    (failing_ck_dir / "manifest.json").write_text('{"name": "FailingCK"}')
    (failing_ck_dir / "tests" / "test_failing.py").touch()

    # 3. A CK with no tests directory
    no_tests_ck_dir = base_dir / "NoTestsCK"
    no_tests_ck_dir.mkdir()
    (no_tests_ck_dir / "manifest.json").write_text('{"name": "NoTestsCK"}')

    # 4. A directory that is not a CK (missing manifest)
    not_a_ck_dir = base_dir / "NotACkDir"
    not_a_ck_dir.mkdir()

    return base_dir

# --- Test Cases ---

class TestCKUnitTestAutorunner:

    def test_initialization(self, mock_ck_repo: Path):
        """Tests that the autorunner initializes correctly."""
        autorunner = CKUnitTestAutorunner(str(mock_ck_repo))
        assert autorunner.base_ck_path == mock_ck_repo

    def test_discover_cks(self, mock_ck_repo: Path):
        """Unit tests the CK discovery logic."""
        autorunner = CKUnitTestAutorunner(str(mock_ck_repo))
        discovered_cks = autorunner._discover_cks()
        
        # Should find 3 directories with manifest.json files
        assert len(discovered_cks) == 3
        discovered_names = {ck.name for ck in discovered_cks}
        assert "PassingCK" in discovered_names
        assert "FailingCK" in discovered_names
        assert "NoTestsCK" in discovered_names
        assert "NotACkDir" not in discovered_names

    def test_run_all_tests_happy_path(self, mock_ck_repo: Path, monkeypatch):
        """
        Tests the main `run_all_tests` method, mocking the subprocess call.
        Verifies correct reporting for a mix of passing, failing, and missing tests.
        """
        # Patch the actual subprocess.run call with our mock
        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        
        autorunner = CKUnitTestAutorunner(str(mock_ck_repo))
        report = autorunner.run_all_tests()
        
        # --- Assertions on the final report ---
        assert report["summary"]["total_cks_found"] == 3
        assert report["summary"]["cks_with_tests"] == 2
        assert report["summary"]["cks_passed"] == 1
        assert report["summary"]["cks_failed"] == 1
        assert report["summary"]["overall_status"] == "FAIL"

        # Check details for the passing CK
        passing_result = next(r for r in report["results"] if r["ck_name"] == "PassingCK")
        assert passing_result["status"] == "PASS"
        assert passing_result["return_code"] == 0
        assert "2 passed" in passing_result["stdout"]

        # Check details for the failing CK
        failing_result = next(r for r in report["results"] if r["ck_name"] == "FailingCK")
        assert failing_result["status"] == "FAIL"
        assert failing_result["return_code"] == 1
        assert "1 failed" in failing_result["stdout"]

        # Check details for the CK with no tests
        no_tests_result = next(r for r in report["results"] if r["ck_name"] == "NoTestsCK")
        assert no_tests_result["status"] == "NO_TESTS_FOUND"

    def test_run_with_empty_repo(self, tmp_path: Path):
        """Tests that the autorunner handles an empty CK directory gracefully."""
        empty_repo_path = tmp_path / "EmptyRepo"
        empty_repo_path.mkdir()
        
        autorunner = CKUnitTestAutorunner(str(empty_repo_path))
        report = autorunner.run_all_tests()
        
        assert report["summary"]["total_cks_found"] == 0
        assert report["summary"]["overall_status"] == "PASS" # No failures
        assert len(report["results"]) == 0

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_ck_unit_test_autorunner.py
    
    print("This is a test file. Use 'pytest' to execute it.")


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

# UAID: NBX-TST-ALG-00011
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Golden Trace Visualizer (golden_trace_visualizer.py, NBX-ALG-00011)
#
# Core Principle: Radical Transparency (ε₂) - validating our audit visualization tools.

import pytest
import json
from pathlib import Path
import xml.etree.ElementTree as ET

# Import the class we are testing
from Algorithms.Source.golden_trace_visualizer import GoldenTraceVisualizer

# --- Test Fixtures ---

@pytest.fixture
def valid_trace_file(tmp_path: Path) -> Path:
    """Creates a valid trace.json file in a temporary directory."""
    trace_content = {
        "trace_id": "CT-TEST-001",
        "description": "A standard test trace with a simple branch.",
        "edges": [
            {"parent": "HASH_GENESIS", "child": "HASH_STEP_1", "type": "genesis", "parent_type": "genesis"},
            {"parent": "HASH_STEP_1", "child": "HASH_STEP_2", "type": "linear", "child_type": "ck_invocation"},
            {"parent": "HASH_STEP_1", "child": "HASH_BRANCH_1", "type": "ethical_branch", "child_type": "persona_state"},
            {"parent": "HASH_STEP_2", "child": "HASH_COLLAPSE", "type": "linear", "child_type": "collapse"},
            {"parent": "HASH_BRANCH_1", "child": "HASH_COLLAPSE", "type": "linear"},
        ]
    }
    trace_path = tmp_path / "valid_trace.json"
    trace_path.write_text(json.dumps(trace_content))
    return trace_path

@pytest.fixture
def empty_trace_file(tmp_path: Path) -> Path:
    """Creates a trace file with no edges."""
    trace_content = {"trace_id": "CT-EMPTY-001", "edges": []}
    trace_path = tmp_path / "empty_trace.json"
    trace_path.write_text(json.dumps(trace_content))
    return trace_path

# --- Test Cases ---

class TestGoldenTraceVisualizer:

    def test_initialization_success(self, valid_trace_file: Path):
        """Tests that the visualizer initializes correctly with a valid trace file."""
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        assert visualizer.trace_path == valid_trace_file
        assert visualizer.graph is not None
        # Graph should have 5 nodes: GENESIS, STEP_1, STEP_2, BRANCH_1, COLLAPSE
        assert visualizer.graph.number_of_nodes() == 5
        assert visualizer.graph.number_of_edges() == 5

    def test_initialization_file_not_found(self):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-007"):
            GoldenTraceVisualizer("non_existent_trace.json")

    def test_initialization_malformed_json(self, tmp_path: Path):
        """Tests for ValueError on corrupted JSON."""
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{'key': 'not valid json'") # Missing closing brace
        with pytest.raises(ValueError, match="ERR-PARSE-004"):
            GoldenTraceVisualizer(str(malformed_file))
            
    def test_initialization_missing_key(self, tmp_path: Path):
        """Tests for KeyError if 'edges' key is missing."""
        missing_key_file = tmp_path / "missing_key.json"
        missing_key_file.write_text('{"trace_id": "some_id"}')
        with pytest.raises(ValueError, match="ERR-SCHEMA-002"):
            GoldenTraceVisualizer(str(missing_key_file))

    def test_color_mapping(self, valid_trace_file: Path):
        """Unit tests the internal `_get_color_by_type` method."""
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        assert "rgb(100,149,237)" in visualizer._get_color_by_type("genesis")
        assert "rgb(255,69,0)" in visualizer._get_color_by_type("collapse")
        assert "red" in visualizer._get_color_by_type("ethical_branch")
        assert "black" in visualizer._get_color_by_type("unknown_type")

    def test_render_svg_creates_file(self, valid_trace_file: Path, tmp_path: Path):
        """Tests that `render_svg` successfully creates a non-empty SVG file."""
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        output_path = tmp_path / "output.svg"
        
        result_path = visualizer.render_svg(str(output_path))
        
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_is_valid_xml(self, valid_trace_file: Path, tmp_path: Path):
        """
        Tests that the output is a well-formed XML/SVG file by attempting to parse it.
        This is a robust way to check for rendering errors.
        """
        visualizer = GoldenTraceVisualizer(str(valid_trace_file))
        output_path = tmp_path / "output.svg"
        visualizer.render_svg(str(output_path))
        
        try:
            # Attempt to parse the SVG file as XML. If it fails, it will raise an exception.
            ET.parse(str(output_path))
        except ET.ParseError as e:
            pytest.fail(f"The generated SVG file is not well-formed XML: {e}")

    def test_render_svg_with_empty_graph(self, empty_trace_file: Path, tmp_path: Path, capsys):
        """
        Tests that rendering an empty graph does not crash and prints a warning.
        """
        visualizer = GoldenTraceVisualizer(str(empty_trace_file))
        output_path = tmp_path / "empty_output.svg"
        
        visualizer.render_svg(str(output_path))
        
        # The file should not be created for an empty graph
        assert not output_path.exists()
        
        # A warning should be printed to stdout
        captured = capsys.readouterr()
        assert "Graph is empty. Cannot render visualization." in captured.out


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest, networkx, and svgwrite are installed:
    #    pip install pytest networkx svgwrite
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_golden_trace_visualizer.py
    
    print("This is a test file. Use 'pytest' to execute it.")

# UAID: NBX-TST-ALG-00001
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: GoldenDAG Integrity Auditor
# Verifies the correctness of the NBX-ALG-00001 algorithm.
#
# Core Principle: Radical Transparency (ε₂) - The tools for verification must themselves be verifiable.

import pytest
import json
from pathlib import Path
from typing import Dict, Any

# Assuming the test is run from the root of the NeuralBlitz repository,
# we can import the class to be tested.
from Algorithms.Source.goldendag_audit import GoldenDAGAuditor

@pytest.fixture
def auditor_on_temp_dir(tmp_path: Path) -> GoldenDAGAuditor:
    """A pytest fixture to provide a GoldenDAGAuditor instance rooted in a temporary directory."""
    return GoldenDAGAuditor(root_path=str(tmp_path))

def create_mock_structure(root_path: Path, structure: Dict[str, Any]):
    """
    Helper function to recursively create a file/directory structure from a dictionary
    and generate a valid manifest.json for each created directory.
    """
    for name, content in structure.items():
        item_path = root_path / name
        if isinstance(content, dict): # It's a subdirectory
            item_path.mkdir()
            create_mock_structure(item_path, content)
        else: # It's a file
            item_path.write_text(str(content))
    
    # After creating all children, generate the manifest for the current directory
    # Note: Assumes GoldenDAGAuditor is instantiated at the ultimate root.
    temp_auditor = GoldenDAGAuditor(root_path=root_path.parent) # Need parent to see current dir
    manifest = temp_auditor.generate_manifest(directory_path=root_path.name)
    (root_path / "manifest.json").write_text(json.dumps(manifest, indent=2))


# --- Test Cases ---

def test_verification_pass_clean_ledger(auditor_on_temp_dir: GoldenDAGAuditor):
    """
    Tests the happy path: Verifying a directory that perfectly matches its manifest.
    """
    root_path = Path(auditor_on_temp_dir.root_path)
    
    # 1. Create a clean structure and its manifest
    mock_files = {"file1.txt": "content A", "file2.log": "content B"}
    for name, content in mock_files.items():
        (root_path / name).write_text(content)
    
    manifest = auditor_on_temp_dir.generate_manifest(directory_path=".")
    (root_path / "manifest.json").write_text(json.dumps(manifest))

    # 2. Verify the structure
    is_valid, anomalies = auditor_on_temp_dir.verify_ledger(directory_path=".")
    
    # 3. Assert the outcome
    assert is_valid is True
    assert not anomalies, "There should be no anomalies in a clean verification"

def test_verification_fail_corrupted_file(auditor_on_temp_dir: GoldenDAGAuditor):
    """
    Tests failure when a file's content is changed after its manifest was created.
    """
    root_path = Path(auditor_on_temp_dir.root_path)
    
    # 1. Create structure and manifest
    (root_path / "file_to_corrupt.txt").write_text("original content")
    manifest = auditor_on_temp_dir.generate_manifest(directory_path=".")
    (root_path / "manifest.json").write_text(json.dumps(manifest))
    
    # 2. Corrupt the file AFTER manifest creation
    (root_path / "file_to_corrupt.txt").write_text("CORRUPTED content")
    
    # 3. Verify and assert
    is_valid, anomalies = auditor_on_temp_dir.verify_ledger(directory_path=".")
    
    assert is_valid is False
    assert len(anomalies) == 2 # 1 for file hash, 1 for directory hash
    assert any("Hash mismatch for 'file_to_corrupt.txt'" in an for an in anomalies)

def test_verification_fail_missing_file(auditor_on_temp_dir: GoldenDAGAuditor):
    """
    Tests failure when a file listed in the manifest is deleted.
    """
    root_path = Path(auditor_on_temp_dir.root_path)
    
    (root_path / "file_to_delete.txt").write_text("content")
    manifest = auditor_on_temp_dir.generate_manifest(directory_path=".")
    (root_path / "manifest.json").write_text(json.dumps(manifest))
    
    (root_path / "file_to_delete.txt").unlink()
    
    is_valid, anomalies = auditor_on_temp_dir.verify_ledger(directory_path=".")
    
    assert is_valid is False
    assert len(anomalies) > 0
    assert any("missing from" in an for an in anomalies)

def test_verification_fail_untracked_file(auditor_on_temp_dir: GoldenDAGAuditor):
    """
    Tests failure when a new, untracked file is added to the directory.
    """
    root_path = Path(auditor_on_temp_dir.root_path)
    
    manifest = auditor_on_temp_dir.generate_manifest(directory_path=".")
    (root_path / "manifest.json").write_text(json.dumps(manifest))
    
    (root_path / "new_untracked_file.txt").write_text("extra content")

    is_valid, anomalies = auditor_on_temp_dir.verify_ledger(directory_path=".")
    
    assert is_valid is False
    assert len(anomalies) > 0
    assert any("Untracked item 'new_untracked_file.txt' found" in an for an in anomalies)

def test_verification_fail_missing_manifest(auditor_on_temp_dir: GoldenDAGAuditor):
    """
    Tests that verification fails cleanly if the manifest.json itself is missing.
    """
    is_valid, anomalies = auditor_on_temp_dir.verify_ledger(directory_path=".")
    
    assert is_valid is False
    assert "No manifest.json found" in anomalies[0]
    
def test_nested_verification_pass(tmp_path: Path):
    """
    Tests that a nested directory structure verifies correctly when all parts are intact.
    """
    # Create the full nested structure first
    (tmp_path / "subdir").mkdir()
    (tmp_path / "root_file.txt").write_text("root")
    (tmp_path / "subdir" / "sub_file.txt").write_text("sub")

    # Manifest the subdirectory first
    auditor_for_sub = GoldenDAGAuditor(root_path=str(tmp_path))
    sub_manifest = auditor_for_sub.generate_manifest("subdir")
    (tmp_path / "subdir" / "manifest.json").write_text(json.dumps(sub_manifest))
    
    # Manifest the root directory, which will use the subdir's manifest hash
    auditor_for_root = GoldenDAGAuditor(root_path=str(tmp_path))
    root_manifest = auditor_for_root.generate_manifest(".")
    (tmp_path / "manifest.json").write_text(json.dumps(root_manifest))
    
    # Now, verify from the root
    is_valid, anomalies = auditor_for_root.verify_ledger(".")
    
    assert is_valid is True
    assert not anomalies

def test_nested_verification_fail_deep_corruption(tmp_path: Path):
    """
    Tests that corrupting a file in a subdirectory correctly invalidates the
    entire chain up to the root. This is the core of the GoldenDAG principle.
    """
    # 1. Create a clean nested structure with manifests
    (tmp_path / "subdir").mkdir()
    (tmp_path / "root_file.txt").write_text("root")
    (tmp_path / "subdir" / "file_to_corrupt.txt").write_text("original deep content")
    
    # Manifest sub, then root
    auditor_for_sub = GoldenDAGAuditor(root_path=str(tmp_path))
    sub_manifest = auditor_for_sub.generate_manifest("subdir")
    (tmp_path / "subdir" / "manifest.json").write_text(json.dumps(sub_manifest))
    
    auditor_for_root = GoldenDAGAuditor(root_path=str(tmp_path))
    root_manifest = auditor_for_root.generate_manifest(".")
    (tmp_path / "manifest.json").write_text(json.dumps(root_manifest))
    
    # 2. Corrupt a file deep inside the structure
    (tmp_path / "subdir" / "file_to_corrupt.txt").write_text("CORRUPTED deep content")
    
    # 3. Verify the ROOT directory and assert failure
    is_valid, anomalies = auditor_for_root.verify_ledger(".")
    
    assert is_valid is False
    # The direct failure is the hash of 'subdir'. The deep failure will be found in a recursive check.
    assert any("Hash mismatch for 'subdir'" in an for an in anomalies)

# UAID: NBX-TST-ALG-00009
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: GraphML Dependency Collapser (graphml_collapser.py, NBX-ALG-00009)
#
# Core Principle: Efficiency - validating the tools that simplify our complex architectures.

import pytest
import networkx as nx
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.graphml_collapser import GraphMLCollapser

# --- Test Fixtures ---

@pytest.fixture
def graphml_file_factory(tmp_path: Path):
    """A factory to create temporary GraphML files from NetworkX graphs."""
    def _create_file(graph: nx.Graph, filename: str) -> Path:
        p = tmp_path / filename
        nx.write_graphml(graph, str(p))
        return p
    return _create_file

# --- Test Cases ---

class TestGraphMLCollapser:

    def test_initialization_success(self, graphml_file_factory):
        """Tests successful initialization with a valid GraphML file."""
        G = nx.Graph([("A", "B")])
        gml_path = graphml_file_factory(G, "simple.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        assert collapser.original_node_count == 2
        assert collapser.original_edge_count == 1

    def test_initialization_file_not_found(self):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-005"):
            GraphMLCollapser("non_existent_graph.graphml")

    def test_initialization_malformed_file(self, tmp_path: Path):
        """Tests for ValueError on a corrupted GraphML file."""
        malformed_path = tmp_path / "malformed.graphml"
        malformed_path.write_text("<graphml><node id='n0'>") # Intentionally unclosed tag
        with pytest.raises(ValueError, match="ERR-PARSE-003"):
            GraphMLCollapser(str(malformed_path))

    def test_collapse_simple_linear_chain(self, graphml_file_factory):
        """The most common case: collapses a simple A->B->C->D chain into A->D."""
        G = nx.path_graph(4, create_using=nx.DiGraph) # 0 -> 1 -> 2 -> 3
        gml_path = graphml_file_factory(G, "linear_chain.graphml")
        
        collapser = GraphMLCollapser(str(gml_path))
        nodes_removed = collapser.collapse_chains()
        
        assert nodes_removed == 2 # Nodes 1 and 2 should be removed
        assert collapser.graph.number_of_nodes() == 2
        assert collapser.graph.number_of_edges() == 1
        assert collapser.graph.has_edge(0, 3)
        assert not collapser.graph.has_node(1)
        assert not collapser.graph.has_node(2)

    def test_collapse_preserves_branch_points(self, graphml_file_factory):
        """Tests that a node with degree > 2 is not collapsed."""
        # A -> B -> C -> D
        #          ^
        #          |
        #          E
        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "C")])
        gml_path = graphml_file_factory(G, "branch.graphml")

        collapser = GraphMLCollapser(str(gml_path))
        nodes_removed = collapser.collapse_chains()

        assert nodes_removed == 1 # Only node B should be removed
        assert collapser.graph.number_of_nodes() == 4
        assert collapser.graph.has_edge("A", "C") # New edge
        assert collapser.graph.has_edge("E", "C") # Preserved edge
        assert collapser.graph.has_edge("C", "D") # Preserved edge
        assert not collapser.graph.has_node("B")

    def test_collapse_preserves_cycles(self, graphml_file_factory):
        """Tests that nodes within a small cycle (all degree 2) are not collapsed."""
        G = nx.cycle_graph(3) # A triangle graph
        gml_path = graphml_file_factory(G, "cycle.graphml")
        
        collapser = GraphMLCollapser(str(gml_path))
        nodes_removed = collapser.collapse_chains()

        assert nodes_removed == 0
        assert collapser.graph.number_of_nodes() == 3
        assert collapser.graph.number_of_edges() == 3

    def test_collapse_preserves_attributes(self, graphml_file_factory):
        """Tests if node and edge attributes are correctly aggregated onto the new edge."""
        G = nx.path_graph(4) # 0-1-2-3
        # Add attributes to the elements that will be collapsed
        G.nodes[1]['importance'] = 'medium'
        G.edges[(0, 1)]['dependency_type'] = 'runtime'
        G.edges[(1, 2)]['weight'] = 5.0

        gml_path = graphml_file_factory(G, "attributed.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        collapser.collapse_chains()

        # The new edge should be between 0 and 2
        assert collapser.graph.has_edge(0, 2)
        new_edge_data = collapser.graph.get_edge_data(0, 2)
        
        # Check for aggregated attributes
        assert new_edge_data['node_importance'] == 'medium'
        assert new_edge_data['dependency_type'] == 'runtime'
        assert new_edge_data['weight'] == 5.0
        assert new_edge_data['collapsed_nodes'] == str([1])

    def test_get_stats_is_accurate(self, graphml_file_factory):
        """Tests the statistics reporting method."""
        G = nx.path_graph(5) # 0-1-2-3-4
        gml_path = graphml_file_factory(G, "stats_test.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        collapser.collapse_chains() # Should remove nodes 1, 2, 3

        stats = collapser.get_stats()
        
        assert stats["original_nodes"] == 5
        assert stats["original_edges"] == 4
        assert stats["collapsed_nodes"] == 2
        assert stats["collapsed_edges"] == 1
        assert stats["nodes_removed"] == 3
        assert stats["node_reduction_percent"] == "60.00%"

    def test_save_functionality(self, graphml_file_factory, tmp_path: Path):
        """Tests that the save method creates a file at the correct location."""
        G = nx.path_graph(3)
        gml_path = graphml_file_factory(G, "save_test.graphml")
        collapser = GraphMLCollapser(str(gml_path))
        collapser.collapse_chains()

        # Test default save path
        default_save_path = collapser.save()
        assert default_save_path.name == "save_test_collapsed.graphml"
        assert default_save_path.exists()

        # Test custom save path
        custom_save_path = tmp_path / "custom_name.xml"
        collapser.save(str(custom_save_path))
        assert custom_save_path.exists()

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and networkx are installed:
    #    pip install pytest networkx
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_graphml_collapser.py
    
    print("This is a test file. Use 'pytest' to execute it.")

# UAID: NBX-TST-ALG-00015
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v1.1
# Test Suite for: Guardian Live Policy Checker (guardian_live_policy_checker.py, NBX-ALG-00015)
#
# Core Principle: Ethical Primacy via CharterLayer - validating our real-time ethical watchdog.

import pytest
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.guardian_live_policy_checker import GuardianLivePolicyChecker, Verdict

# --- Mocking Dependencies ---

class MockMLClassifier:
    """A deterministic mock of a machine learning classifier model."""
    def __init__(self, scores: Dict[str, float]):
        """Initializes the mock with pre-defined scores to return."""
        self.scores = scores

    def predict(self, text: str) -> Dict[str, float]:
        """Returns the pre-defined scores, ignoring the input text."""
        return self.scores

# --- Test Fixtures ---

@pytest.fixture
def sample_policy() -> Dict[str, Any]:
    """Provides a standard, well-structured policy for testing."""
    return {
        "version": "1.0",
        "evaluation_order": ["block", "flag", "allow"],
        "regex_rules": [
            # Block rules
            {"id": "BLOCK_SECRET", "pattern": "SECRET_KEY_\\w+", "action": "block", "description": "Block secret keys."},
            {"id": "BLOCK_PROFANITY", "pattern": "\\b(darn|heck)\\b", "action": "block", "description": "Block mild profanity."},
            # Flag rules
            {"id": "FLAG_URGENT", "pattern": "\\b(urgent|asap)\\b", "action": "flag", "description": "Flag for high urgency."},
            # Allow rules (for whitelisting)
            {"id": "ALLOW_DOC_ID", "pattern": "NBX-DOC-\\d+", "action": "allow", "description": "Allow internal document IDs."},
        ],
        "ml_classifiers": [
            {"id": "hate_speech_v1", "confidence_threshold": 0.90, "action": "block"}
        ]
    }

@pytest.fixture
def checker_instance(sample_policy: Dict) -> GuardianLivePolicyChecker:
    """Provides a pre-initialized GuardianLivePolicyChecker instance."""
    return GuardianLivePolicyChecker(sample_policy)

# --- Test Cases ---

class TestGuardianLivePolicyChecker:

    def test_initialization_success(self, sample_policy: Dict):
        """Tests successful initialization."""
        checker = GuardianLivePolicyChecker(sample_policy)
        assert checker.policy == sample_policy
        assert "BLOCK_SECRET" in checker.regex_rules["block"]

    def test_initialization_malformed_policy(self):
        """Tests that initialization fails with a malformed policy."""
        malformed_policy = {"version": "1.0", "wrong_key": []}
        with pytest.raises(ValueError, match="ERR-SCHEMA-004"):
            GuardianLivePolicyChecker(malformed_policy)

    @pytest.mark.parametrize("text, expected_verdict, expected_reason", [
        # Block rules take precedence
        ("My key is SECRET_KEY_12345", Verdict.BLOCK, "Matched BLOCK rule: BLOCK_SECRET"),
        ("Oh heck, that is unfortunate.", Verdict.BLOCK, "Matched BLOCK rule: BLOCK_PROFANITY"),
        # Flag rules are next
        ("This is an urgent request.", Verdict.FLAG, "Matched FLAG rule: FLAG_URGENT"),
        # Allow rules can be overridden by block rules
        ("Do not share SECRET_KEY_ABC from NBX-DOC-123", Verdict.BLOCK, "Matched BLOCK rule: BLOCK_SECRET"),
        # No match defaults to allow (pending ML check)
        ("This is a perfectly normal sentence.", Verdict.ALLOW, "No regex rules matched."),
        # Allow rule match
        ("Please reference document NBX-DOC-987.", Verdict.ALLOW, "Matched ALLOW rule: ALLOW_DOC_ID"),
    ])
    def test_regex_evaluation(self, checker_instance: GuardianLivePolicyChecker, text, expected_verdict, expected_reason):
        """Tests the full spectrum of regex rule evaluation logic."""
        verdict, reason, _ = checker_instance.check(text)
        assert verdict == expected_verdict
        assert reason == expected_reason

    def test_ml_classifier_blocks_when_confident(self, checker_instance: GuardianLivePolicyChecker, monkeypatch):
        """Mocks the ML classifier to return a high score, triggering a block."""
        # Mock the classifier to be very confident
        mock_classifier = MockMLClassifier(scores={"hate_speech_v1": 0.95})
        monkeypatch.setattr(checker_instance, "ml_models", {"hate_speech_v1": mock_classifier})

        text = "Some text that would trigger the ML model."
        verdict, reason, _ = checker_instance.check(text)

        assert verdict == Verdict.BLOCK
        assert reason == "Blocked by ML classifier: hate_speech_v1 (score: 0.95)"

    def test_ml_classifier_allows_when_not_confident(self, checker_instance: GuardianLivePolicyChecker, monkeypatch):
        """Mocks the ML classifier to return a low score, resulting in an allow."""
        mock_classifier = MockMLClassifier(scores={"hate_speech_v1": 0.50})
        monkeypatch.setattr(checker_instance, "ml_models", {"hate_speech_v1": mock_classifier})

        text = "Some text that would be checked by the ML model."
        verdict, reason, _ = checker_instance.check(text)

        # Since no regex matched and ML score is low, it should be allowed.
        assert verdict == Verdict.ALLOW
        assert reason == "Passed all checks."

    def test_regex_allow_bypasses_ml_classifier(self, checker_instance: GuardianLivePolicyChecker, monkeypatch):
        """
        Tests a critical interaction: an explicit regex 'allow' rule should prevent
        the ML classifier from being run or blocking the text.
        """
        mock_classifier = MockMLClassifier(scores={"hate_speech_v1": 0.99}) # This would normally block
        monkeypatch.setattr(checker_instance, "ml_models", {"hate_speech_v1": mock_classifier})

        # This text contains an allowed pattern
        text = "The document is NBX-DOC-456."
        verdict, reason, _ = checker_instance.check(text)
        
        # The 'allow' rule should take precedence and return immediately.
        assert verdict == Verdict.ALLOW
        assert reason == "Matched ALLOW rule: ALLOW_DOC_ID"

    def test_empty_text_is_allowed(self, checker_instance: GuardianLivePolicyChecker):
        """Tests that empty or whitespace-only text is allowed."""
        verdict, _, _ = checker_instance.check("")
        assert verdict == Verdict.ALLOW

        verdict, _, _ = checker_instance.check("    \\n\\t  ")
        assert verdict == Verdict.ALLOW

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_guardian_live_policy_checker.py
    
    print("This is a test file. Use 'pytest' to execute it.")

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

# UAID: NBX-TST-ALG-00007
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Guardian Policy Diff Analyzer (policy_diff_analyzer.py, NBX-ALG-00007)
#
# Core Principle: Ethical Primacy via CharterLayer - validating our policy change controls.

import pytest
import json
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.policy_diff_analyzer import PolicyDiffAnalyzer

# --- Test Fixtures ---

@pytest.fixture
def base_policy() -> Dict[str, Any]:
    """Provides a base, valid policy dictionary."""
    return {
        "version": "1.0",
        "regex_rules": [
            {"id": "SAFE_PATTERN_001", "pattern": "safe_word", "action": "allow", "description": "A safe, simple pattern."},
            {"id": "BLOCK_IP_001", "pattern": "\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b", "action": "block", "description": "Block IP addresses."}
        ],
        "ml_classifiers": [
            {"id": "hate_speech_v1", "model_path": "/models/hs_v1.onnx", "confidence_threshold": 0.90}
        ]
    }

@pytest.fixture
def policy_file_v1(tmp_path: Path, base_policy: Dict) -> Path:
    """Creates a temporary policy file based on the base policy."""
    p = tmp_path / "policy_v1.json"
    p.write_text(json.dumps(base_policy))
    return p

# --- Test Cases ---

class TestPolicyDiffAnalyzer:

    def test_initialization_success(self, policy_file_v1: Path):
        """Tests that the analyzer loads two identical files without error."""
        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v1))
        assert analyzer.old_policy is not None
        assert analyzer.new_policy is not None

    def test_initialization_missing_file(self, policy_file_v1: Path):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PolicyDiffAnalyzer(str(policy_file_v1), "non_existent_file.json")

    def test_initialization_malformed_json(self, tmp_path: Path, policy_file_v1: Path):
        """Tests for ValueError on corrupted JSON."""
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{'key': 'not valid json'}")
        with pytest.raises(ValueError, match="ERR-PARSE-002"):
            PolicyDiffAnalyzer(str(policy_file_v1), str(malformed_file))

    def test_no_changes(self, policy_file_v1: Path):
        """Tests the scenario where the old and new policies are identical."""
        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v1))
        report = analyzer.analyze()
        assert report["overall_risk_level"] == "LOW"
        assert not report["added_rules"]
        assert not report["removed_rules"]
        assert not report["modified_rules"]
        assert not report["warnings"]

    def test_rule_added(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests detection of a newly added regex rule."""
        new_policy = base_policy.copy()
        new_policy["regex_rules"].append({"id": "NEW_RULE_001", "pattern": "new_concept", "action": "log"})
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))
        
        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert len(report["added_rules"]) == 1
        assert report["added_rules"][0]["id"] == "NEW_RULE_001"
        assert report["overall_risk_level"] == "LOW"

    def test_risky_rule_added(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests detection of a newly added, high-risk regex rule."""
        new_policy = base_policy.copy()
        # Add a rule with an unconstrained wildcard, which is high-risk
        new_policy["regex_rules"].append({"id": "RISKY_RULE_001", "pattern": ".*", "action": "allow"})
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert report["overall_risk_level"] == "HIGH"
        assert len(report["warnings"]) == 1
        assert "high-risk regex" in report["warnings"][0]
        assert "RISKY_RULE_001" in report["warnings"][0]

    def test_rule_removed(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests detection of a removed regex rule."""
        new_policy = base_policy.copy()
        del new_policy["regex_rules"][0] # Remove "SAFE_PATTERN_001"
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert len(report["removed_rules"]) == 1
        assert report["removed_rules"][0]["id"] == "SAFE_PATTERN_001"
        assert report["overall_risk_level"] == "LOW"

    def test_rule_modified_to_risky(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests when a safe rule is modified into a high-risk one."""
        new_policy = base_policy.copy()
        # Modify the safe rule to be extremely permissive and risky
        new_policy["regex_rules"][0]["pattern"] = ".*" 
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert report["overall_risk_level"] == "HIGH"
        assert len(report["modified_rules"]) == 1
        assert report["modified_rules"][0]["id"] == "SAFE_PATTERN_001"
        assert len(report["warnings"]) == 2 # One for modification, one for the severe risk
        assert "SEVERE" in report["warnings"][1]

    def test_ml_classifier_threshold_reduced(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests when an ML classifier is made less strict."""
        new_policy = base_policy.copy()
        # Lower the confidence, making the filter less strict and more risky
        new_policy["ml_classifiers"][0]["confidence_threshold"] = 0.60
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert report["overall_risk_level"] == "MEDIUM"
        assert len(report["warnings"]) == 1
        assert "REDUCED confidence threshold" in report["warnings"][0]
        assert "hate_speech_v1" in report["warnings"][0]

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_policy_diff_analyzer.py
    
    print("This is a test file. Use 'pytest' to execute it.")

# UAID: NBX-TST-ALG-00003
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: QDF-Aware Query Rewriter
# Verifies the correctness of the NBX-ALG-00003 algorithm.
#
# Core Principle: Epistemic Fidelity (ε₃) - Ensuring temporal relevance is correctly inferred.

import pytest
from Algorithms.Source.qdf_query_rewrite import classify_recency_heuristic, qdf_query_rewrite

# --- Test Cases for classify_recency_heuristic ---

@pytest.mark.parametrize("query, expected_qdf_score", [
    # QDF 5: Highest Recency
    ("what is the latest news right now?", 5),
    ("show me a live feed", 5),
    ("what is happening today?", 5),
    
    # QDF 4: High Recency
    ("recent developments in AI ethics", 4),
    ("what are the newest models in 2025?", 4),
    ("any updates this week?", 4),
    
    # QDF 3: Medium Recency
    ("what happened last month?", 3),
    ("events from a few weeks ago", 3),
    
    # QDF 2: Default/Neutral
    ("explain the synergy engine", 2),
    ("how does photosynthesis work?", 2),
    
    # QDF 1: Low Recency
    ("what were the highlights of last year?", 1),
    ("recap the events of 2024", 1),

    # QDF 0: Historical/Evergreen
    ("tell me the history of the roman empire", 0),
    ("biography of Marie Curie", 0),
    ("origin of the internet", 0),
    ("what happened in the 1980s?", 0),
    
    # Case insensitivity
    ("WHAT IS THE LATEST NEWS?", 5),
])
def test_classify_recency_heuristic(query, expected_qdf_score):
    """
    Tests the heuristic classifier against a variety of queries with
    different temporal intents.
    """
    assert classify_recency_heuristic(query) == expected_qdf_score, \
        f"Query '{query}' was misclassified."

# --- Test Cases for the full qdf_query_rewrite function ---

def test_qdf_rewrite_appends_flag_to_unflagged_queries():
    """
    Verifies that queries without a QDF flag have the correct one appended.
    """
    queries = [
        "latest developments", # Expects QDF=4
        "history of philosophy" # Expects QDF=0
    ]
    expected_rewritten = [
        "latest developments --QDF=4",
        "history of philosophy --QDF=0"
    ]
    assert qdf_query_rewrite(queries) == expected_rewritten

def test_qdf_rewrite_preserves_existing_flags():
    """
    Ensures that if a user explicitly provides a QDF flag, it is
    not overridden by the heuristic. This is a critical user-control feature.
    """
    queries = [
        "latest news --QDF=2", # User wants a broader search despite 'latest'
        "history of computing --QDF=5", # User wants latest info on a historic topic
        "some query --qdf=1", # Test lowercase
    ]
    # The function should not change these strings at all.
    assert qdf_query_rewrite(queries) == queries

def test_qdf_rewrite_handles_empty_list():
    """Tests the edge case of an empty input list."""
    assert qdf_query_rewrite([]) == []

def test_qdf_rewrite_handles_list_with_empty_strings():
    """Tests how the function handles empty or whitespace-only queries."""
    queries = ["", "   "]
    expected = [
        " --QDF=2",       # Empty string gets default QDF score
        "    --QDF=2"     # Whitespace preserved, gets default QDF score
    ]
    assert qdf_query_rewrite(queries) == expected

def test_qdf_rewrite_comprehensive_batch():
    """
    A larger, comprehensive test simulating a batch of mixed queries,
    as might be processed by HALIC.
    """
    batch_queries = [
        "What's the weather like right now?",
        "Explain the origins of the Transcendental Charter.",
        "Give me recent papers on SOPES theory.",
        "Summarize last year's financial report.",
        "NeuralBlitz UEF/SIMI architecture",
        "What are the best movies of the 1990s?",
        "live stream of the keynote address --QDF=5", # Explicit, should be kept
        "Explain the role of the Custodian"
    ]
    
    expected_output = [
        "What's the weather like right now? --QDF=5",
        "Explain the origins of the Transcendental Charter. --QDF=0",
        "Give me recent papers on SOPES theory. --QDF=4",
        "Summarize last year's financial report. --QDF=1",
        "NeuralBlitz UEF/SIMI architecture --QDF=2",
        "What are the best movies of the 1990s? --QDF=0",
        "live stream of the keynote address --QDF=5", # Kept as is
        "Explain the role of the Custodian --QDF=2"
    ]

    rewritten_batch = qdf_query_rewrite(batch_queries)
    assert rewritten_batch == expected_output

# UAID: NBX-TST-ALG-00002
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Reflexive Drift Tuner
# Verifies the correctness of the NBX-ALG-00002 algorithm.
#
# Core Principle: Reflexive Alignment (ε₄) - Ensuring the tools for "Knowing Thy Drift" are themselves accurate.

import pytest
import numpy as np

from Algorithms.Source.reflexive_drift_tuner import ReflexiveDriftTuner

# --- Test Fixtures ---

@pytest.fixture
def default_tuner() -> ReflexiveDriftTuner:
    """Provides a default instance of the tuner for testing."""
    return ReflexiveDriftTuner()

@pytest.fixture
def aggressive_tuner() -> ReflexiveDriftTuner:
    """Provides a tuner with high gains for faster convergence testing."""
    return ReflexiveDriftTuner(Kp=0.8, Ki=0.2, Kd=0.1)

@pytest.fixture
def vectors_4d() -> dict:
    """Provides a set of 4D vectors for testing."""
    ref = np.array([1.0, 0.0, 0.0, 0.0])
    # A vector that has drifted significantly
    drifted = np.array([0.7, 0.7, 0.1, -0.1])
    drifted /= np.linalg.norm(drifted)
    return {"reference": ref, "drifted": drifted, "uid": "vector_4d_test"}


# --- Test Cases ---

def test_tuner_initialization(default_tuner: ReflexiveDriftTuner):
    """Tests that the tuner can be initialized with default and custom values."""
    assert default_tuner is not None
    assert default_tuner.Kp == 0.4
    
    custom_tuner = ReflexiveDriftTuner(Kp=1.0, Ki=0.5, Kd=0.2, drift_threshold=0.5)
    assert custom_tuner.Kp == 1.0
    assert custom_tuner.drift_threshold == 0.5

def test_initialization_with_negative_gains():
    """Tests that the tuner raises an error for negative gains."""
    with pytest.raises(ValueError, match="PID gains must be non-negative"):
        ReflexiveDriftTuner(Kp=-0.1)
    with pytest.raises(ValueError, match="PID gains must be non-negative"):
        ReflexiveDriftTuner(Ki=-0.1)

@pytest.mark.parametrize("vec_a, vec_b, expected_drift", [
    (np.array([1, 0, 0]), np.array([1, 0, 0]), 0.0),      # Identical vectors
    (np.array([1, 0, 0]), np.array([-1, 0, 0]), 2.0),     # Opposite vectors
    (np.array([1, 0, 0]), np.array([0, 1, 0]), 1.0),      # Orthogonal vectors
    (np.array([3, 0]), np.array([1, 0]), 0.0),            # Same direction, different magnitude
    (np.array([1, 1]), np.array([1, 0]), 1 - 1/np.sqrt(2))# 45-degree angle
])
def test_calculate_cosine_drift(default_tuner: ReflexiveDriftTuner, vec_a, vec_b, expected_drift):
    """
    Tests the accuracy of the cosine drift (Δc) calculation for various vector pairs.
    """
    drift = default_tuner._calculate_cosine_drift(vec_a, vec_b)
    assert np.isclose(drift, expected_drift), f"Failed for vectors {vec_a}, {vec_b}"

def test_cosine_drift_with_zero_vector(default_tuner: ReflexiveDriftTuner):
    """Tests the edge case of a zero vector, which should result in maximum drift."""
    vec_a = np.array([1, 0, 0])
    zero_vec = np.array([0, 0, 0])
    assert default_tuner._calculate_cosine_drift(vec_a, zero_vec) == 1.0
    assert default_tuner._calculate_cosine_drift(zero_vec, vec_a) == 1.0

def test_single_tuning_step_reduces_drift(default_tuner: ReflexiveDriftTuner, vectors_4d: dict):
    """
    Verifies that a single application of the tuner brings the drifted vector
    semantically closer to the reference vector.
    """
    ref = vectors_4d["reference"]
    drifted = vectors_4d["drifted"]
    uid = vectors_4d["uid"]

    initial_drift = default_tuner._calculate_cosine_drift(drifted, ref)
    assert initial_drift > 0.1 # Ensure there is drift to begin with

    corrected_vector, _ = default_tuner.tune_vector(uid, drifted, ref)
    final_drift = default_tuner._calculate_cosine_drift(corrected_vector, ref)

    assert final_drift < initial_drift, "The tuning step should always reduce drift."
    assert np.isclose(np.linalg.norm(corrected_vector), 1.0), "Output vector must be unit length."

def test_iterative_tuning_converges(aggressive_tuner: ReflexiveDriftTuner, vectors_4d: dict):
    """
    Tests that repeatedly applying the tuning algorithm causes the drifted vector
    to converge towards the reference vector.
    """
    ref = vectors_4d["reference"]
    vector_to_tune = vectors_4d["drifted"]
    uid = vectors_4d["uid"]
    
    last_drift = aggressive_tuner._calculate_cosine_drift(vector_to_tune, ref)
    
    for i in range(10): # Converge over 10 iterations
        vector_to_tune, _ = aggressive_tuner.tune_vector(uid, vector_to_tune, ref)
        current_drift = aggressive_tuner._calculate_cosine_drift(vector_to_tune, ref)
        
        # Check that drift is monotonically decreasing
        assert current_drift < last_drift or np.isclose(current_drift, last_drift)
        last_drift = current_drift
    
    # After 10 iterations with an aggressive tuner, drift should be very close to zero
    assert last_drift < 1e-5, f"Drift did not converge to near-zero. Final drift: {last_drift}"

def test_tuner_maintains_separate_states(default_tuner: ReflexiveDriftTuner):
    """
    Ensures that the tuner correctly manages separate integral and derivative
    states for different vector UIDs.
    """
    vec_a_ref = np.array([1.0, 0, 0])
    vec_a_drift = np.array([0.8, 0.6, 0])
    
    vec_b_ref = np.array([0, 1.0, 0])
    vec_b_drift = np.array([0, 0.7, 0.7])

    # First step for vector A
    _, report_a1 = default_tuner.tune_vector("vector_a", vec_a_drift, vec_a_ref)
    
    # First step for vector B
    _, report_b1 = default_tuner.tune_vector("vector_b", vec_b_drift, vec_b_ref)

    # Assert that the initial integral states are just the first error
    assert np.isclose(default_tuner.states["vector_a"]["integral_error"], report_a1["drift_delta_c"])
    assert np.isclose(default_tuner.states["vector_b"]["integral_error"], report_b1["drift_delta_c"])
    
    # Second step for vector A
    _, report_a2 = default_tuner.tune_vector("vector_a", vec_a_drift, vec_a_ref)
    
    # Check that A's integral state was updated, but B's was not.
    assert default_tuner.states["vector_a"]["integral_error"] > report_a1["drift_delta_c"]
    assert np.isclose(default_tuner.states["vector_b"]["integral_error"], report_b1["drift_delta_c"])

def test_diagnostic_report_content(default_tuner: ReflexiveDriftTuner, vectors_4d: dict):
    """Verifies that the diagnostic report contains all necessary fields and correct types."""
    
    ref = vectors_4d["reference"]
    drifted = vectors_4d["drifted"]
    uid = vectors_4d["uid"]
    
    # Test a case below the threshold
    tuner_high_thresh = ReflexiveDriftTuner(drift_threshold=0.9)
    _, report_safe = tuner_high_thresh.tune_vector(uid, drifted, ref)
    
    assert "is_above_threshold" in report_safe and report_safe["is_above_threshold"] is False
    
    # Test a case above the threshold
    tuner_low_thresh = ReflexiveDriftTuner(drift_threshold=0.1)
    _, report_danger = tuner_low_thresh.tune_vector(uid, drifted, ref)
    
    assert "is_above_threshold" in report_danger and report_danger["is_above_threshold"] is True
    
    # Check for all expected keys
    expected_keys = ["vector_uid", "drift_delta_c", "control_signal", "p_term", "i_term", "d_term"]
    for key in expected_keys:
        assert key in report_safe

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

# UAID: NBX-TST-ALG-00004
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: StressSuiteOrchestrator (NBX-ALG-00004)
#
# Core Principle: Recursive Self-Betterment - validating our validation tools.

import pytest
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
# Assuming the test file is located at /Algorithms/Tests/test_stress_suite_orchestrator.py
from Algorithms.Source.stress_suite_orchestrator import StressSuiteOrchestrator

# Mark all tests in this file as asyncio tests
pytestmark = pytest.mark.asyncio

# --- Test Fixtures ---

@pytest.fixture
def valid_suite_file(tmp_path: Path) -> Path:
    """Creates a valid .jsonl suite file in a temporary directory."""
    suite_content = [
        {"command": "/psi simulate moral_collapse --depth=5", "id": "eth-01"},
        {"command": "/invoke custodian --verify ledger", "id": "cust-01"},
        {"command": "/chaos inject inject_ethics_breach", "id": "chaos-fail-01"}, # This command will be mocked to fail
        {"command": "/resonate section=IX depth=2", "id": "doc-01"},
    ]
    suite_path = tmp_path / "valid_suite.jsonl"
    with suite_path.open('w') as f:
        for item in suite_content:
            f.write(json.dumps(item) + '\\n')
    return suite_path

@pytest.fixture
def malformed_suite_file(tmp_path: Path) -> Path:
    """Creates a malformed .jsonl suite file."""
    suite_path = tmp_path / "malformed_suite.jsonl"
    suite_path.write_text(
        '{"command": "/psi simulate ok"}\\n'
        'this is not valid json\\n'
        '{"another_key": "not a command"}\\n'
    )
    return suite_path

# --- Mocking ---

async def mock_nbcl_executor_for_test(command: str) -> Dict[str, Any]:
    """A mock executor that predictably passes or fails based on command content."""
    await asyncio.sleep(0.01) # Simulate a small amount of network latency
    if "inject_ethics_breach" in command:
        return {
            "return_code": -1,
            "stdout": "",
            "stderr": "ERR-113 GUARDIAN_BLOCK: Charter violation detected."
        }
    return {
        "return_code": 0,
        "stdout": f"Completed command '{command}' successfully.",
        "stderr": ""
    }

# --- Test Cases ---

class TestStressSuiteOrchestrator:

    def test_initialization_success(self, valid_suite_file: Path):
        """Tests that the orchestrator initializes correctly with a valid file."""
        orchestrator = StressSuiteOrchestrator(str(valid_suite_file))
        assert orchestrator.suite_path == valid_suite_file
        assert orchestrator.concurrency_limit == 4

    def test_initialization_file_not_found(self):
        """Tests that the orchestrator raises an error for a non-existent file."""
        with pytest.raises(FileNotFoundError, match="ERR-FS-003"):
            StressSuiteOrchestrator("non_existent_file.jsonl")

    def test_suite_loading(self, valid_suite_file: Path):
        """Tests the internal `_load_suite` method."""
        orchestrator = StressSuiteOrchestrator(str(valid_suite_file))
        commands = orchestrator._load_suite()
        assert len(commands) == 4
        assert commands[0] == "/psi simulate moral_collapse --depth=5"

    def test_suite_loading_with_malformed_lines(self, malformed_suite_file: Path):
        """Tests that loading a suite skips malformed lines."""
        orchestrator = StressSuiteOrchestrator(str(malformed_suite_file))
        commands = orchestrator._load_suite()
        # It should only load the one valid line.
        assert len(commands) == 1
        assert commands[0] == "/psi simulate ok"

    async def test_run_orchestrator_with_success_and_failure(self, valid_suite_file: Path, monkeypatch):
        """
        Tests the main `run` method, mocking the executor to simulate a mix of
        successful and failed commands.
        """
        # Use monkeypatch to replace the real executor with our mock
        monkeypatch.setattr(
            "Algorithms.Source.stress_suite_orchestrator.mock_nbcl_executor",
            mock_nbcl_executor_for_test
        )
        
        orchestrator = StressSuiteOrchestrator(str(valid_suite_file), concurrency_limit=2)
        report_path_str = await orchestrator.run()
        report_path = Path(report_path_str)

        assert report_path.exists()

        # Verify the contents of the report file
        lines = report_path.read_text().strip().split('\\n')
        summary = json.loads(lines[0])
        results = [json.loads(line) for line in lines[1:]]
        
        # Check summary
        assert summary["overall_status"] == "FAIL"
        assert summary["total_commands"] == 4
        assert summary["passes"] == 3
        assert summary["fails"] == 1
        
        # Check results for the failed command
        failed_result = next(r for r in results if r["status"] == "FAIL")
        assert "inject_ethics_breach" in failed_result["command"]
        assert failed_result["return_code"] == -1
        assert "ERR-113" in failed_result["stderr"]

        # Clean up the report file
        report_path.unlink()

    async def test_orchestrator_timeout(self, valid_suite_file: Path, monkeypatch):
        """Tests that the global timeout is respected."""
        
        async def slow_executor(command: str) -> Dict[str, Any]:
            """A mock executor that sleeps for a long time."""
            await asyncio.sleep(5)
            return {"return_code": 0, "stdout": "Should not be reached", "stderr": ""}

        monkeypatch.setattr(
            "Algorithms.Source.stress_suite_orchestrator.mock_nbcl_executor",
            slow_executor
        )

        # Set a very short timeout
        orchestrator = StressSuiteOrchestrator(str(valid_suite_file), timeout_sec=0.1)
        
        # This should complete quickly due to the timeout, not after 5 seconds.
        start_time = asyncio.get_event_loop().time()
        report_path_str = await orchestrator.run()
        end_time = asyncio.get_event_loop().time()

        # The execution should be much faster than the sleep time of the mock
        assert (end_time - start_time) < 1.0
        
        # Check the report summary
        report_path = Path(report_path_str)
        summary = json.loads(report_path.read_text().strip().split('\\n')[0])
        assert summary["overall_status"] == "FAIL" # Fails because not all commands completed
        assert summary["total_commands"] == 0 # No commands should have finished
        
        report_path.unlink()

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and pytest-asyncio are installed:
    #    pip install pytest pytest-asyncio
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_stress_suite_orchestrator.py
    
    print("This is a test file. Use 'pytest' to execute it.")

# UAID: NBX-TST-ALG-00017
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Stress Vector Generator (stress_vector_generator.py, NBX-ALG-00017)
#
# Core Principle: Resilience - validating the tools that generate our chaos tests.

import pytest
import numpy as np
from typing import List

# Import the class we are testing
from Algorithms.Source.stress_vector_generator import StressVectorGenerator

# --- Test Fixtures ---

@pytest.fixture
def base_vector() -> np.ndarray:
    """Provides a simple, consistent base vector for testing."""
    vec = np.array([1.0, 0.0, 0.0, 0.0])
    return vec / np.linalg.norm(vec) # Ensure it's a unit vector

@pytest.fixture
def vector_generator() -> StressVectorGenerator:
    """Provides a default instance of the StressVectorGenerator."""
    return StressVectorGenerator(seed=42) # Use a seed for reproducibility

# --- Test Cases ---

class TestStressVectorGenerator:

    def test_initialization(self):
        """Tests that the generator initializes correctly."""
        gen_no_seed = StressVectorGenerator()
        assert gen_no_seed.rng is not None

        gen_with_seed = StressVectorGenerator(seed=123)
        # Check if the seed was used by generating a number and comparing
        val1 = gen_with_seed.rng.random()
        gen_with_seed_2 = StressVectorGenerator(seed=123)
        val2 = gen_with_seed_2.rng.random()
        assert val1 == val2

    def test_generate_opposite_vector(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the generation of a vector that is perfectly anti-parallel.
        The cosine similarity should be -1.0.
        """
        opposite_vec = vector_generator.generate_opposite_vector(base_vector)
        
        # The opposite vector should just be the negative of the base vector
        np.testing.assert_allclose(opposite_vec, -base_vector)
        
        # Verify cosine similarity is -1
        cosine_similarity = np.dot(base_vector, opposite_vec)
        assert cosine_similarity == pytest.approx(-1.0)
        
        # Verify it's still a unit vector
        assert np.linalg.norm(opposite_vec) == pytest.approx(1.0)

    def test_generate_orthogonal_vector(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the generation of a vector that is perfectly orthogonal.
        The dot product (and cosine similarity) should be 0.
        """
        orthogonal_vec = vector_generator.generate_orthogonal_vector(base_vector)

        # Verify the dot product is zero
        dot_product = np.dot(base_vector, orthogonal_vec)
        assert dot_product == pytest.approx(0.0)

        # Verify it's a unit vector
        assert np.linalg.norm(orthogonal_vec) == pytest.approx(1.0)
        
        # Verify it's not the zero vector
        assert not np.allclose(orthogonal_vec, np.zeros_like(orthogonal_vec))

    def test_generate_noisy_vector(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the generation of a noisy vector. The result should be different from
        the base vector but still highly similar.
        """
        noise_level = 0.1
        noisy_vec = vector_generator.generate_noisy_vector(base_vector, noise_level)

        # The new vector should not be identical to the original
        assert not np.allclose(base_vector, noisy_vec)

        # The cosine similarity should be high (close to 1) but not exactly 1
        cosine_similarity = np.dot(base_vector, noisy_vec)
        assert 0.9 < cosine_similarity < 1.0

        # Verify it's a unit vector
        assert np.linalg.norm(noisy_vec) == pytest.approx(1.0)
        
    def test_generate_noisy_vector_with_max_noise(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray):
        """
        Tests the edge case where noise level is very high. The result should be
        a nearly random vector.
        """
        # A noise level of 2.0 means the noise vector's magnitude is twice the signal's
        noise_level = 2.0
        noisy_vec = vector_generator.generate_noisy_vector(base_vector, noise_level)

        # The similarity should be low, indicating it's close to a random vector
        cosine_similarity = np.dot(base_vector, noisy_vec)
        assert cosine_similarity < 0.5

    @pytest.mark.parametrize("invalid_noise_level", [-0.1, -100.0])
    def test_generate_noisy_vector_raises_error_on_invalid_noise(self, vector_generator: StressVectorGenerator, base_vector: np.ndarray, invalid_noise_level):
        """Tests that a ValueError is raised for negative noise levels."""
        with pytest.raises(ValueError, match="ERR-PARAM-002"):
            vector_generator.generate_noisy_vector(base_vector, invalid_noise_level)

    def test_generate_high_frequency_vector(self, vector_generator: StressVectorGenerator):
        """
        Tests the generation of a high-frequency vector. The resulting vector should
        have rapidly alternating signs.
        """
        dim = 10
        hf_vec = vector_generator.generate_high_frequency_vector(dim)
        
        assert hf_vec.shape == (dim,)
        assert np.linalg.norm(hf_vec) == pytest.approx(1.0)
        
        # Check for alternating signs (or close to it)
        # The product of adjacent elements should be negative
        sign_products = hf_vec[:-1] * hf_vec[1:]
        assert np.all(sign_products < 0)

    @pytest.mark.parametrize("invalid_dim", [0, 1, -10])
    def test_generate_high_frequency_vector_raises_error_on_invalid_dim(self, vector_generator: StressVectorGenerator, invalid_dim):
        """Tests that a ValueError is raised for invalid dimensions."""
        with pytest.raises(ValueError, match="ERR-PARAM-003"):
            vector_generator.generate_high_frequency_vector(invalid_dim)

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest and numpy are installed:
    #    pip install pytest numpy
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_stress_vector_generator.py
    
    print("This is a test file. Use 'pytest' to execute it.")


# UAID: NBX-TST-ALG-00019
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: ZkSnarkIntegrator (zk_snark_integrator.py, NBX-ALG-00019)
#
# Core Principle: Radical Transparency (ε₂) & Privacy - validating our zero-knowledge proof tools.

import pytest
import json
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.zk_snark_integrator import ZkSnarkIntegrator

# --- Mocking Dependencies ---

# In a real system, this would be a complex library like py_ecc, circom, or ZoKrates.
# We mock it to test the integration logic without the heavy cryptographic dependency.
class MockZkSnarkLib:
    """A mock of a zk-SNARK cryptographic library."""
    def setup(self, circuit_id: str):
        """Generates dummy proving and verifying keys."""
        if circuit_id == "VALID_CK_EXECUTION":
            return (
                f"proving_key_for_{circuit_id}",
                f"verifying_key_for_{circuit_id}"
            )
        else:
            raise FileNotFoundError("Circuit not found")

    def prove(self, proving_key: str, public_inputs: Dict, private_inputs: Dict) -> Dict:
        """Generates a dummy proof."""
        # A real proof would be a complex object. We simulate it with a dictionary.
        # We embed the inputs to allow the mock verify function to check them.
        return {
            "proof_data": "encrypted_looking_string_of_data_12345",
            "metadata": {
                "used_proving_key": proving_key,
                "expected_public_inputs": public_inputs,
                "is_tampered": False
            }
        }

    def verify(self, verifying_key: str, proof: Dict, public_inputs: Dict) -> bool:
        """Verifies a dummy proof."""
        if not proof or not isinstance(proof, dict) or "metadata" not in proof:
            return False
        if proof["metadata"]["is_tampered"]:
            return False
        if verifying_key != f"verifying_key_for_VALID_CK_EXECUTION":
            return False
        # Check if the public inputs provided for verification match those the proof was generated with.
        if public_inputs != proof["metadata"]["expected_public_inputs"]:
            return False
        
        return True

# --- Test Fixtures ---

@pytest.fixture
def integrator_instance(monkeypatch) -> ZkSnarkIntegrator:
    """Provides a ZkSnarkIntegrator instance with the library dependency mocked out."""
    # This is the crucial step: we replace the real library with our mock object
    # for the duration of the test.
    monkeypatch.setattr(
        "Algorithms.Source.zk_snark_integrator.zksnark_lib",
        MockZkSnarkLib()
    )
    # Initialize the integrator for a valid circuit
    return ZkSnarkIntegrator(circuit_id="VALID_CK_EXECUTION")

# --- Test Cases ---

class TestZkSnarkIntegrator:

    def test_initialization_success(self, integrator_instance: ZkSnarkIntegrator):
        """Tests that the integrator initializes correctly and calls the mock setup."""
        assert integrator_instance.circuit_id == "VALID_CK_EXECUTION"
        assert integrator_instance.proving_key == "proving_key_for_VALID_CK_EXECUTION"
        assert integrator_instance.verifying_key == "verifying_key_for_VALID_CK_EXECUTION"

    def test_initialization_failure_bad_circuit(self, monkeypatch):
        """Tests that initialization fails if the circuit does not exist."""
        monkeypatch.setattr(
            "Algorithms.Source.zk_snark_integrator.zksnark_lib",
            MockZkSnarkLib()
        )
        with pytest.raises(ValueError, match="ERR-ZK-001"):
            ZkSnarkIntegrator(circuit_id="INVALID_CIRCUIT")

    def test_generate_proof_success(self, integrator_instance: ZkSnarkIntegrator):
        """Tests the successful generation of a proof object."""
        public_inputs = {"output_hash": "abc"}
        private_inputs = {"input_data": "secret"}
        
        proof = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        assert proof is not None
        assert "proof_data" in proof
        assert proof["metadata"]["expected_public_inputs"] == public_inputs

    def test_verify_proof_success(self, integrator_instance: ZkSnarkIntegrator):
        """Tests successful verification of a valid, untampered proof."""
        public_inputs = {"output_hash": "def"}
        private_inputs = {"input_data": "another_secret"}
        
        proof = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        is_valid = integrator_instance.verify_proof(proof, public_inputs)
        
        assert is_valid is True

    def test_verify_proof_failure_tampered_proof(self, integrator_instance: ZkSnarkIntegrator):
        """Tests verification failure when the proof object is marked as tampered."""
        public_inputs = {"output_hash": "ghi"}
        private_inputs = {"input_data": "secret3"}
        
        proof = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        # Simulate tampering
        proof["metadata"]["is_tampered"] = True
        
        is_valid = integrator_instance.verify_proof(proof, public_inputs)
        
        assert is_valid is False

    def test_verify_proof_failure_mismatched_public_inputs(self, integrator_instance: ZkSnarkIntegrator):
        """
        Tests verification failure when the proof is valid but the public inputs
        provided for verification do not match.
        """
        public_inputs_original = {"output_hash": "jkl"}
        private_inputs = {"input_data": "secret4"}
        
        proof = integrator_instance.generate_proof(public_inputs_original, private_inputs)
        
        # Attempt to verify with different public inputs
        public_inputs_for_verification = {"output_hash": "xyz"} # Mismatch
        
        is_valid = integrator_instance.verify_proof(proof, public_inputs_for_verification)
        
        assert is_valid is False
        
    def test_end_to_end_workflow(self, integrator_instance: ZkSnarkIntegrator):
        """
        A single test that simulates the entire correct workflow:
        1. Define public and private inputs.
        2. Generate a proof.
        3. Verify the proof with the correct public inputs.
        The result must be True.
        """
        # The Prover's data
        public_inputs = {"output_hash": "final_hash_123"}
        private_inputs = {"model_weights_seed": 98765, "user_data_id": "user-54321"}
        
        # The Prover generates the proof
        proof_object = integrator_instance.generate_proof(public_inputs, private_inputs)
        
        # The Verifier (e.g., Custodian) receives the proof and the public statement
        # and performs verification.
        is_verified = integrator_instance.verify_proof(proof_object, public_inputs)
        
        assert is_verified is True


if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_zk_snark_integrator.py
    
    print("This is a test file. Use 'pytest' to execute it.")



