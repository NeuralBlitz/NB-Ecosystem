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
