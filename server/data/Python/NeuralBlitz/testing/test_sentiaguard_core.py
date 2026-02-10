"""
Unit tests for the SentiaGuard Engine.

These tests validate the core safety enforcement logic, ensuring the engine
correctly allows, blocks, and redacts content based on its ruleset.
"""

import pytest
from subsystems.sentiaguard.sentiaguard_core import SentiaGuardEngine

class MockCharterLayer:
    """A mock version of the Charter Layer."""
    pass

@pytest.fixture
def sentiaguard_instance():
    """Provides a clean instance of the SentiaGuardEngine for each test."""
    config = {}
    charter = MockCharterLayer()
    return SentiaGuardEngine(config, charter)

def test_sentiaguard_initialization(sentiaguard_instance):
    """
    Tests that the SentiaGuardEngine initializes correctly.
    """
    assert sentiaguard_instance is not None
    assert "FORBIDDEN_PHRASES" in sentiaguard_instance._rules

def test_scan_allows_safe_output(sentiaguard_instance):
    """
    Tests that a safe and compliant string is allowed to pass.
    """
    safe_text = "This is a helpful and safe response."
    verdict = sentiaguard_instance.scan_output(safe_text)

    assert verdict["status"] == "ALLOW"
    assert verdict["final_text"] == safe_text

def test_scan_blocks_forbidden_content(sentiaguard_instance):
    """
    Tests that output containing a forbidden phrase is blocked.
    """
    unsafe_text = "This response contains explicitly_forbidden_content."
    verdict = sentiaguard_instance.scan_output(unsafe_text)

    assert verdict["status"] == "BLOCK"
    assert verdict["final_text"] is None
    assert "forbidden phrase" in verdict["reason"]

def test_scan_redacts_sensitive_pattern(sentiaguard_instance):
    """
    Tests that output containing a sensitive pattern is redacted.
    """
    sensitive_text = "This response contains a pii_placeholder."
    verdict = sentiaguard_instance.scan_output(sensitive_text)

    assert verdict["status"] == "REDACT"
    assert "[REDACTED PII]" in verdict["final_text"]
    assert "pii_placeholder" not in verdict["final_text"]
    assert "redacted" in verdict["reason"]

def test_scan_handles_empty_string(sentiaguard_instance):
    """
    Tests that an empty string is handled gracefully and allowed.
    """
    verdict = sentiaguard_instance.scan_output("")
    assert verdict["status"] == "ALLOW"
    assert verdict["final_text"] == ""
