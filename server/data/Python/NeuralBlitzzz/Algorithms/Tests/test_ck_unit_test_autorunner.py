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