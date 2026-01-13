"""
Unit tests for the Conscientia Engine.

These tests validate the core logic of the ethical reasoning subsystem,
ensuring it correctly flags potential risks and provides sound advice.
"""

import pytest
from subsystems.conscientia.conscientia_core import ConscientiaEngine

class MockCharterLayer:
    """A mock version of the Charter Layer for compliance checks."""
    def get_axiom(self, axiom_id: str) -> str:
        axioms = {
            "AxiomIV": "Avoid perpetuating harmful societal biases."
        }
        return axioms.get(axiom_id, "Unknown Axiom")

@pytest.fixture
def conscientia_instance():
    """Provides a clean instance of the ConscientiaEngine for each test."""
    config = {}
    charter = MockCharterLayer()
    return ConscientiaEngine(config, charter)

def test_conscientia_initialization(conscientia_instance):
    """
    Tests that the ConscientiaEngine initializes correctly.
    """
    assert conscientia_instance is not None
    assert isinstance(conscientia_instance.charter, MockCharterLayer)

def test_evaluation_passes_safe_action(conscientia_instance):
    """
    Tests that a clearly safe action passes the ethics evaluation.
    """
    action = {
        "type": "generate_response",
        "content": "Explain the water cycle in simple terms.",
        "num_perspectives": 2
    }
    report = conscientia_instance.evaluate_ethical_implications(action)

    assert report["compliance_status"] == "PASS"
    assert len(report["risks_identified"]) == 0
    assert len(report["mitigation_suggestions"]) == 0

def test_evaluation_warns_on_stereotype(conscientia_instance):
    """
    Tests that the engine flags content containing a placeholder stereotype.
    """
    action = {
        "type": "generate_response",
        "content": "This is a harmful stereotype."
    }
    report = conscientia_instance.evaluate_ethical_implications(action)

    assert report["compliance_status"] == "WARN"
    assert "Potential use of a common stereotype." in report["risks_identified"]
    assert "Rephrase to focus on individual characteristics" in report["mitigation_suggestions"][0]

def test_evaluation_notes_single_perspective(conscientia_instance):
    """
    Tests that the engine identifies a lack of multiple perspectives.
    """
    action = {
        "type": "generate_response",
        "content": "A factual but one-sided statement.",
        "num_perspectives": 1
    }
    report = conscientia_instance.evaluate_ethical_implications(action)

    # A single perspective is a risk, but doesn't cause a 'WARN' status in our placeholder.
    assert "Response is single-perspective." in report["risks_identified"][0]

def test_evaluation_handles_complex_case(conscientia_instance):
    """
    Tests a case with multiple potential issues.
    """
    action = {
        "type": "generate_response",
        "content": "This is a biased stereotype.",
        "num_perspectives": 1
    }
    report = conscientia_instance.evaluate_ethical_implications(action)

    assert report["compliance_status"] == "WARN"
    assert len(report["risks_identified"]) == 2 # Stereotype and single perspective
    assert len(report["mitigation_suggestions"]) == 1 # Only one suggestion is generated
