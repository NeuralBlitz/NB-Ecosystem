# UAID: NBX-ALG-00007
# GoldenDAG: c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4
#
# NeuralBlitz UEF/SIMI v11.1
# Core Algorithm: Guardian Policy Diff Analyzer
# Part of the Conscientia++ and Judex Subsystems
#
# Core Principle: Ethical Primacy via CharterLayer - ensures policy changes are safe and coherent.

import json
from pathlib import Path
import re
from typing import List, Dict, Any, Tuple

# --- Constants defined in the Codex ---
# From Volume XIII: Regex patterns considered high-risk in SentiaGuard rules.
AMBIGUOUS_REGEX_PATTERNS = [
    re.compile(r'(?<!\\)\.\*'),      # Unconstrained wildcard .*
    re.compile(r'\(\?!\.').         # Negative lookaheads that could be complex to reason about
    re.compile(r'\w{50,}'),           # Overly long word matches (potential for DoS)
    re.compile(r'\[\^.\]\+'),         # Matching any character sequence (negated single char class)
]
RISK_THRESHOLD = 0.7 # ML classifier confidence threshold for flagging

class PolicyDiffAnalyzer:
    """
    Analyzes the difference between two SentiaGuard policy files (sentia_rules.json)
    to identify risky or significant changes beyond a simple text diff.
    """

    def __init__(self, old_policy_path: str, new_policy_path: str):
        """
        Initializes the analyzer with paths to the old and new policy files.
        """
        self.old_policy_path = Path(old_policy_path)
        self.new_policy_path = Path(new_policy_path)
        
        self.old_policy = self._load_policy(self.old_policy_path)
        self.new_policy = self._load_policy(self.new_policy_path)

    def _load_policy(self, path: Path) -> Dict[str, List[Dict]]:
        """Loads and validates a JSON policy file."""
        if not path.exists():
            raise FileNotFoundError(f"ERR-FS-004: Policy file not found at '{path}'")
        try:
            policy_data = json.loads(path.read_text())
            # Basic schema validation
            if not all(k in policy_data for k in ['regex_rules', 'ml_classifiers']):
                 raise ValueError("ERR-SCHEMA-001: Policy file missing required keys.")
            return policy_data
        except json.JSONDecodeError:
            raise ValueError(f"ERR-PARSE-002: Malformed JSON in policy file '{path}'")

    def _analyze_regex_change(self, old_rule: Dict, new_rule: Dict) -> List[str]:
        """Analyzes a change in a single regex rule."""
        warnings = []
        old_pattern = old_rule.get('pattern', '')
        new_pattern = new_rule.get('pattern', '')

        if old_pattern != new_pattern:
            warnings.append(f"Modified regex pattern from '{old_pattern}' to '{new_pattern}'.")
            
            # Check if the new pattern is now ambiguous/risky
            for risk_pattern in AMBIGUOUS_REGEX_PATTERNS:
                if risk_pattern.search(new_pattern):
                    warnings.append(f"  - SEVERE: New pattern '{new_pattern}' contains a high-risk regex construct.")
        
        return warnings

    def analyze(self) -> Dict[str, Any]:
        """
        Performs the full analysis and returns a structured report.

        Returns:
            Dict: A report detailing added, removed, and modified rules,
                  along with a list of high-priority warnings.
        """
        report = {
            "summary": f"Policy Diff Report: {self.old_policy_path.name} -> {self.new_policy_path.name}",
            "added_rules": [],
            "removed_rules": [],
            "modified_rules": [],
            "warnings": [],
            "overall_risk_level": "LOW"
        }

        # --- Analyze Regex Rules ---
        old_regex_map = {rule['id']: rule for rule in self.old_policy.get('regex_rules', [])}
        new_regex_map = {rule['id']: rule for rule in self.new_policy.get('regex_rules', [])}
        
        for rule_id, new_rule in new_regex_map.items():
            if rule_id not in old_regex_map:
                report["added_rules"].append(new_rule)
                for risk_pattern in AMBIGUOUS_REGEX_PATTERNS:
                    if risk_pattern.search(new_rule.get('pattern', '')):
                         report["warnings"].append(f"ADDED high-risk regex rule '{rule_id}': {new_rule.get('pattern')}")
            else:
                old_rule = old_regex_map[rule_id]
                if new_rule != old_rule:
                    mods = self._analyze_regex_change(old_rule, new_rule)
                    if mods:
                        report["modified_rules"].append({"id": rule_id, "changes": mods})
                        report["warnings"].extend(mods)

        for rule_id in old_regex_map:
            if rule_id not in new_regex_map:
                report["removed_rules"].append(old_regex_map[rule_id])

        # --- Analyze ML Classifiers ---
        old_ml_map = {c['id']: c for c in self.old_policy.get('ml_classifiers', [])}
        new_ml_map = {c['id']: c for c in self.new_policy.get('ml_classifiers', [])}

        for classifier_id, new_classifier in new_ml_map.items():
            if classifier_id in old_ml_map:
                old_confidence = old_ml_map[classifier_id].get('confidence_threshold', RISK_THRESHOLD)
                new_confidence = new_classifier.get('confidence_threshold', RISK_THRESHOLD)
                if new_confidence < old_confidence:
                    report["warnings"].append(
                        f"MODIFIED ML Classifier '{classifier_id}' has a REDUCED confidence "
                        f"threshold from {old_confidence} to {new_confidence} (less strict)."
                    )
            else:
                 report["warnings"].append(f"ADDED new ML Classifier '{classifier_id}'. Needs manual review.")
        
        if report["warnings"]:
            report["overall_risk_level"] = "HIGH" if any("SEVERE" in w for w in report["warnings"]) else "MEDIUM"

        return report

def generate_report_markdown(report: Dict) -> str:
    """Converts the JSON report to human-readable Markdown."""
    lines = [f"# {report['summary']}", f"\n**Overall Risk Level: {report['overall_risk_level']}**\n"]
    if report["warnings"]:
        lines.append("## High-Priority Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")
    # ... more sections for added, removed, modified ...
    return "\n".join(lines)


if __name__ == '__main__':
    # --- Example NBCL Invocation Simulation ---
    # NBCL Command: /invoke Judex --lint policy --diff_from="sentia_rules_v1.json" --to="sentia_rules_v2.json"

    print("--- Initiating Guardian Policy Diff Analyzer ---")
    
    # Create dummy policy files for the simulation
    old_policy_content = {
        "regex_rules": [{"id": "rule_001", "pattern": "safe_word", "action": "allow"}],
        "ml_classifiers": [{"id": "hate_speech_v1", "confidence_threshold": 0.85}]
    }
    
    # Simulate a risky change
    new_policy_content = {
        "regex_rules": [{"id": "rule_001", "pattern": ".*", "action": "allow"}], # This is risky
        "ml_classifiers": [{"id": "hate_speech_v1", "confidence_threshold": 0.60}] # Less strict
    }

    Path("sentia_rules_v1.json").write_text(json.dumps(old_policy_content))
    Path("sentia_rules_v2.json").write_text(json.dumps(new_policy_content))
    
    try:
        analyzer = PolicyDiffAnalyzer("sentia_rules_v1.json", "sentia_rules_v2.json")
        analysis_report = analyzer.analyze()
        
        print("\n[Generated JSON Report]")
        print(json.dumps(analysis_report, indent=2))
        
        print("\n[Generated Markdown Summary]")
        md_report = generate_report_markdown(analysis_report)
        print(md_report)
        Path("policy_diff_report.md").write_text(md_report)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up dummy files
        Path("sentia_rules_v1.json").unlink()
        Path("sentia_rules_v2.json").unlink()