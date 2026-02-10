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