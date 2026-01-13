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