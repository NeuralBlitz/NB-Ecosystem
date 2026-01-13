"""
Unit tests for the Universal Neural Engine (UNE).

These tests validate the core functionality of the UNE in isolation, ensuring
that its logic is sound and its contracts are met.
"""

import pytest
from core_engine.une.une_core import UniversalNeuralEngine

# --- Mock Dependencies ---
# For unit tests, we create simple "mock" versions of the UNE's dependencies.
# This allows us to test the UNE's logic without needing the full, complex
# implementation of the DRS or Charter Layer.

class MockDRSEngine:
    """A mock version of the Dynamic Representational Substrate."""
    def query(self, concept: str):
        return f"Data for {concept}"

class MockCharterLayer:
    """A mock version of the Charter Layer for compliance checks."""
    def is_compliant(self, action: str) -> bool:
        # For this test, we assume all actions are compliant.
        return True

# --- Test Fixture ---
# A pytest fixture is a reusable function that sets up a test environment.
# This fixture creates a fresh instance of the UNE for each test function.

@pytest.fixture
def une_instance():
    """Provides a configured instance of the UniversalNeuralEngine."""
    config = {"version": "1.0"}
    drs = MockDRSEngine()
    charter = MockCharterLayer()
    return UniversalNeuralEngine(config, drs, charter)

# --- Test Cases ---

def test_une_initialization(une_instance):
    """
    Tests if the UNE initializes correctly.
    """
    assert une_instance is not None
    assert isinstance(une_instance, UniversalNeuralEngine)
    assert isinstance(une_instance.drs, MockDRSEngine)
    assert isinstance(une_instance.charter, MockCharterLayer)

def test_direct_focus_returns_valid_plan(une_instance):
    """
    Tests if the core `direct_focus` method returns a well-structured plan.
    """
    intent = "Explain the concept of symbiotic intelligence"
    context = {"session_id": "12345"}

    focus_plan = une_instance.direct_focus(intent, context)

    # Assert that the output is a dictionary with the expected structure
    assert isinstance(focus_plan, dict)
    assert "key_concepts" in focus_plan
    assert "logical_path" in focus_plan
    assert "status" in focus_plan

    # Assert that the content of the plan is as expected from our placeholder logic
    assert focus_plan["status"] == "PLAN_GENERATED"
    assert "symbiotic" in focus_plan["key_concepts"]
    assert "intelligence" in focus_plan["key_concepts"]
