"""
Unit tests for the Human-AI Language Interface Core (HALIC).

These tests validate the end-to-end interaction flow managed by HALIC,
ensuring prompts are processed, the core engine is called, and responses
are formatted and scanned correctly.
"""

import pytest
from interface_layer.halic.halic_core import HALICEngine
from core_engine.une.une_core import UniversalNeuralEngine
from subsystems.sentiaguard.sentiaguard_core import SentiaGuardEngine

# --- Mock Dependencies ---

class MockUNE:
    """A mock of the UniversalNeuralEngine."""
    def direct_focus(self, intent: str, context: dict):
        # Return a predictable plan based on the intent
        return {
            "key_concepts": intent.split(),
            "logical_path": "mocked_path",
            "status": "PLAN_GENERATED"
        }

class MockSentiaGuard:
    """A mock of the SentiaGuardEngine that always allows."""
    def scan_output(self, response_text: str):
        # Simply pass the text through with an "ALLOW" status.
        return {
            "status": "ALLOW",
            "reason": "Mock scan passed.",
            "final_text": response_text
        }

# --- Test Fixture ---

@pytest.fixture
def halic_instance():
    """Provides a configured instance of the HALICEngine with mock dependencies."""
    config = {}
    une = MockUNE()
    sentiaguard = MockSentiaGuard()
    return HALICEngine(config, une, sentiaguard)

# --- Test Cases ---

def test_halic_initialization(halic_instance):
    """
    Tests that the HALICEngine initializes correctly.
    """
    assert halic_instance is not None
    assert isinstance(halic_instance.une, MockUNE)
    assert isinstance(halic_instance.sentiaguard, MockSentiaGuard)

def test_process_interaction_returns_valid_verdict(halic_instance):
    """
    Tests the full process_interaction flow, checking the final output.
    """
    raw_prompt = "Test prompt"
    final_verdict = halic_instance.process_interaction(raw_prompt)

    # Check the structure of the verdict from SentiaGuard
    assert isinstance(final_verdict, dict)
    assert final_verdict["status"] == "ALLOW"
    assert "final_text" in final_verdict

    # Check the content of the formatted response
    final_text = final_verdict["final_text"]
    assert "This is a helpful response" in final_text
    assert "Test" in final_text # Checks that the core output was integrated
    assert "GoldenDAG:" in final_text
    assert "Trace ID:" in final_text
    assert "Codex ID:" in final_text
