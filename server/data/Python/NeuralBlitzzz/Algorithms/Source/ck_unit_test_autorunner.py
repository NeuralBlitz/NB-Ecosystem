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