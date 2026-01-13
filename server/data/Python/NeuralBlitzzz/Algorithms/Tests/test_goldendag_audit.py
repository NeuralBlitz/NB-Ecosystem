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