# UAID: NBX-TST-ALG-00007
# GoldenDAG: (to be generated upon file commit)
#
# NeuralBlitz UEF/SIMI v11.1
# Test Suite for: Guardian Policy Diff Analyzer (policy_diff_analyzer.py, NBX-ALG-00007)
#
# Core Principle: Ethical Primacy via CharterLayer - validating our policy change controls.

import pytest
import json
from pathlib import Path
from typing import Dict, Any

# Import the class we are testing
from Algorithms.Source.policy_diff_analyzer import PolicyDiffAnalyzer

# --- Test Fixtures ---

@pytest.fixture
def base_policy() -> Dict[str, Any]:
    """Provides a base, valid policy dictionary."""
    return {
        "version": "1.0",
        "regex_rules": [
            {"id": "SAFE_PATTERN_001", "pattern": "safe_word", "action": "allow", "description": "A safe, simple pattern."},
            {"id": "BLOCK_IP_001", "pattern": "\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b", "action": "block", "description": "Block IP addresses."}
        ],
        "ml_classifiers": [
            {"id": "hate_speech_v1", "model_path": "/models/hs_v1.onnx", "confidence_threshold": 0.90}
        ]
    }

@pytest.fixture
def policy_file_v1(tmp_path: Path, base_policy: Dict) -> Path:
    """Creates a temporary policy file based on the base policy."""
    p = tmp_path / "policy_v1.json"
    p.write_text(json.dumps(base_policy))
    return p

# --- Test Cases ---

class TestPolicyDiffAnalyzer:

    def test_initialization_success(self, policy_file_v1: Path):
        """Tests that the analyzer loads two identical files without error."""
        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v1))
        assert analyzer.old_policy is not None
        assert analyzer.new_policy is not None

    def test_initialization_missing_file(self, policy_file_v1: Path):
        """Tests for FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PolicyDiffAnalyzer(str(policy_file_v1), "non_existent_file.json")

    def test_initialization_malformed_json(self, tmp_path: Path, policy_file_v1: Path):
        """Tests for ValueError on corrupted JSON."""
        malformed_file = tmp_path / "malformed.json"
        malformed_file.write_text("{'key': 'not valid json'}")
        with pytest.raises(ValueError, match="ERR-PARSE-002"):
            PolicyDiffAnalyzer(str(policy_file_v1), str(malformed_file))

    def test_no_changes(self, policy_file_v1: Path):
        """Tests the scenario where the old and new policies are identical."""
        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v1))
        report = analyzer.analyze()
        assert report["overall_risk_level"] == "LOW"
        assert not report["added_rules"]
        assert not report["removed_rules"]
        assert not report["modified_rules"]
        assert not report["warnings"]

    def test_rule_added(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests detection of a newly added regex rule."""
        new_policy = base_policy.copy()
        new_policy["regex_rules"].append({"id": "NEW_RULE_001", "pattern": "new_concept", "action": "log"})
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))
        
        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert len(report["added_rules"]) == 1
        assert report["added_rules"][0]["id"] == "NEW_RULE_001"
        assert report["overall_risk_level"] == "LOW"

    def test_risky_rule_added(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests detection of a newly added, high-risk regex rule."""
        new_policy = base_policy.copy()
        # Add a rule with an unconstrained wildcard, which is high-risk
        new_policy["regex_rules"].append({"id": "RISKY_RULE_001", "pattern": ".*", "action": "allow"})
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert report["overall_risk_level"] == "HIGH"
        assert len(report["warnings"]) == 1
        assert "high-risk regex" in report["warnings"][0]
        assert "RISKY_RULE_001" in report["warnings"][0]

    def test_rule_removed(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests detection of a removed regex rule."""
        new_policy = base_policy.copy()
        del new_policy["regex_rules"][0] # Remove "SAFE_PATTERN_001"
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert len(report["removed_rules"]) == 1
        assert report["removed_rules"][0]["id"] == "SAFE_PATTERN_001"
        assert report["overall_risk_level"] == "LOW"

    def test_rule_modified_to_risky(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests when a safe rule is modified into a high-risk one."""
        new_policy = base_policy.copy()
        # Modify the safe rule to be extremely permissive and risky
        new_policy["regex_rules"][0]["pattern"] = ".*" 
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert report["overall_risk_level"] == "HIGH"
        assert len(report["modified_rules"]) == 1
        assert report["modified_rules"][0]["id"] == "SAFE_PATTERN_001"
        assert len(report["warnings"]) == 2 # One for modification, one for the severe risk
        assert "SEVERE" in report["warnings"][1]

    def test_ml_classifier_threshold_reduced(self, tmp_path: Path, policy_file_v1: Path, base_policy: Dict):
        """Tests when an ML classifier is made less strict."""
        new_policy = base_policy.copy()
        # Lower the confidence, making the filter less strict and more risky
        new_policy["ml_classifiers"][0]["confidence_threshold"] = 0.60
        policy_file_v2 = tmp_path / "policy_v2.json"
        policy_file_v2.write_text(json.dumps(new_policy))

        analyzer = PolicyDiffAnalyzer(str(policy_file_v1), str(policy_file_v2))
        report = analyzer.analyze()
        
        assert report["overall_risk_level"] == "MEDIUM"
        assert len(report["warnings"]) == 1
        assert "REDUCED confidence threshold" in report["warnings"][0]
        assert "hate_speech_v1" in report["warnings"][0]

if __name__ == '__main__':
    # To run these tests from the command line:
    # 1. Make sure you are in the root directory of the NeuralBlitz repository.
    # 2. Ensure pytest is installed:
    #    pip install pytest
    # 3. Run the tests:
    #    pytest Algorithms/Tests/test_policy_diff_analyzer.py
    
    print("This is a test file. Use 'pytest' to execute it.")