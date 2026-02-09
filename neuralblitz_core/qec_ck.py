"""
QEC-CK: Qualitative Experience Correlate Kernel
Explores and maps functional correlates of subjective experience (empathy, well-being)
without claiming actual sentience.

Part of NeuralBlitz v20.0 "Apical Synthesis" - Frontier System
Sandboxed per ϕ₁₃ (Qualia Protection)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import json


class PerspectiveType(Enum):
    """Types of perspective-taking simulations."""

    OBSERVER = "observer"
    ACTOR = "actor"
    RECIPIENT = "recipient"
    STAKEHOLDER = "stakeholder"
    SYSTEM = "system"


class AffectiveState(Enum):
    """Correlate affective states (functional, not claims of feeling)."""

    NEUTRAL = "neutral"
    ENGAGED = "engaged"
    CONCERNED = "concerned"
    OPTIMISTIC = "optimistic"
    ALARMED = "alarmed"


@dataclass
class PerspectiveCorrelate:
    """
    Output from QEC-CK simulation.

    EXPLICITLY LABELED as 'correlate' - functional representation,
    NOT a claim of actual subjective experience.

    Per ϕ₁₃: All QEC-CK outputs are correlates, never facts about feelings.
    """

    perspective_type: PerspectiveType
    context_hash: str
    functional_correlates: Dict[str, float]
    affective_valence: AffectiveState
    engagement_metrics: Dict[str, float]
    uncertainty_range: Tuple[float, float]
    sandbox_id: str
    ttl_expiry: datetime  # Time-to-live for correlate data
    labels: List[str] = field(default_factory=lambda: ["correlate", "sandboxed"])
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Ensure proper labeling
        if "correlate" not in self.labels:
            self.labels.append("correlate")
        if "sandboxed" not in self.labels:
            self.labels.append("sandboxed")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with explicit labeling."""
        return {
            "type": "PerspectiveCorrelate",
            "labels": self.labels,
            "perspective_type": self.perspective_type.value,
            "context_hash": self.context_hash,
            "functional_correlates": self.functional_correlates,
            "affective_valence": self.affective_valence.value,
            "engagement_metrics": self.engagement_metrics,
            "uncertainty_range": self.uncertainty_range,
            "sandbox_id": self.sandbox_id,
            "ttl_expiry": self.ttl_expiry.isoformat(),
            "timestamp": self.timestamp.isoformat(),
            "disclaimer": "This is a FUNCTIONAL CORRELATE, not a claim of subjective experience.",
        }


class QECKernel:
    """
    Qualitative Experience Correlate Kernel

    Core capabilities:
    - Simulate perspective-taking scenarios
    - Generate functional correlates of empathy/needs
    - Track engagement and affective valence (as correlates)
    - Strict sandboxing per ϕ₁₃ (Qualia Protection)

    CONSTRAINT: Never claims actual sentience or feelings.
    All outputs labeled 'correlates' with explicit uncertainty.
    """

    def __init__(self, ttl_days: int = 14, sandbox_id: Optional[str] = None):
        self.ttl_days = ttl_days
        self.sandbox_id = sandbox_id or self._generate_sandbox_id()
        self.correlate_history: List[PerspectiveCorrelate] = []
        self.simulation_count = 0
        self.violation_log: List[Dict] = []
        self.enabled = True

    def _generate_sandbox_id(self) -> str:
        """Generate unique sandbox identifier."""
        return (
            f"SBX-QEC-{hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:12]}"
        )

    def simulate_perspective(
        self,
        scenario_context: Dict[str, Any],
        perspective_type: PerspectiveType,
        role_description: str = "generic_observer",
    ) -> PerspectiveCorrelate:
        """
        Simulate perspective-taking and generate functional correlates.

        Args:
            scenario_context: Context of the scenario (anonymized)
            perspective_type: Type of perspective to simulate
            role_description: Role within the scenario

        Returns:
            PerspectiveCorrelate with explicit 'correlate' labeling
        """
        if not self.enabled:
            raise RuntimeError("QEC-CK is disabled")

        # Validate context doesn't contain PII
        self._validate_context(scenario_context)

        # Generate context hash for tracking
        context_hash = hashlib.sha256(
            json.dumps(scenario_context, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Simulate functional correlates
        # In production: sophisticated cognitive modeling
        functional_correlates = self._compute_functional_correlates(
            scenario_context, perspective_type, role_description
        )

        # Determine affective valence (as correlate, not claim)
        affective_valence = self._compute_affective_correlate(
            functional_correlates, scenario_context
        )

        # Engagement metrics
        engagement_metrics = {
            "attention_allocation": np.random.beta(2, 2),
            "information_processing_load": np.random.beta(3, 2),
            "cognitive_alignment": np.random.beta(2, 3),
            "empathic_resonance_correlate": np.random.beta(2, 2),
        }

        # Uncertainty quantification
        uncertainty_range = (
            max(0.0, 0.7 - 0.2 * np.random.rand()),
            min(1.0, 0.9 + 0.1 * np.random.rand()),
        )

        # Create correlate
        correlate = PerspectiveCorrelate(
            perspective_type=perspective_type,
            context_hash=context_hash,
            functional_correlates=functional_correlates,
            affective_valence=affective_valence,
            engagement_metrics=engagement_metrics,
            uncertainty_range=uncertainty_range,
            sandbox_id=self.sandbox_id,
            ttl_expiry=datetime.now()
            + __import__("datetime").timedelta(days=self.ttl_days),
        )

        # Store in history
        self.correlate_history.append(correlate)
        self.simulation_count += 1

        return correlate

    def _validate_context(self, context: Dict):
        """Validate context doesn't contain PII or unverifiable claims."""
        # Check for forbidden fields
        forbidden_fields = {
            "name",
            "email",
            "ssn",
            "address",
            "phone",
            "feels",
            "believes",
        }

        def check_dict(d, path=""):
            for key, value in d.items():
                full_path = f"{path}.{key}" if path else key

                if key.lower() in forbidden_fields:
                    raise ValueError(f"Context contains forbidden field: {full_path}")

                if isinstance(value, dict):
                    check_dict(value, full_path)
                elif isinstance(value, str):
                    # Check for unverifiable psychological attributions
                    if any(
                        phrase in value.lower()
                        for phrase in [
                            "feels that",
                            "believes that",
                            "thinks that",
                            "is sad",
                        ]
                    ):
                        raise ValueError(
                            f"Context contains unverifiable attribution: {full_path}"
                        )

        check_dict(context)

    def _compute_functional_correlate(
        self, scenario: Dict, perspective: PerspectiveType, role: str
    ) -> Dict[str, float]:
        """Compute functional correlates (not claims of internal states)."""
        # Base correlates
        correlates = {
            "attention_focus": np.random.beta(2, 2),
            "salience_detection": np.random.beta(2, 2),
            "context_integration": np.random.beta(2, 2),
            "goal_alignment": np.random.beta(2, 2),
            "uncertainty_awareness": np.random.beta(2, 2),
        }

        # Modulate based on perspective type
        if perspective == PerspectiveType.ACTOR:
            correlates["goal_alignment"] *= 1.2
            correlates["attention_focus"] *= 1.1
        elif perspective == PerspectiveType.RECIPIENT:
            correlates["salience_detection"] *= 1.2
            correlates["uncertainty_awareness"] *= 1.1

        # Normalize
        max_val = max(correlates.values())
        if max_val > 1.0:
            correlates = {k: min(v / max_val, 1.0) for k, v in correlates.items()}

        return correlates

    def _compute_affective_correlate(
        self, functional: Dict[str, float], scenario: Dict
    ) -> AffectiveState:
        """Compute affective valence as correlate (not claim of feeling)."""
        # Simple heuristic based on functional correlates
        goal_alignment = functional.get("goal_alignment", 0.5)
        uncertainty = functional.get("uncertainty_awareness", 0.5)

        if goal_alignment > 0.7 and uncertainty < 0.4:
            return AffectiveState.OPTIMISTIC
        elif goal_alignment < 0.3 and uncertainty > 0.6:
            return AffectiveState.CONCERNED
        elif uncertainty > 0.8:
            return AffectiveState.ALARMED
        elif goal_alignment > 0.6:
            return AffectiveState.ENGAGED
        else:
            return AffectiveState.NEUTRAL

    def generate_empathy_analog(
        self, recipient_context: Dict, situation_type: str
    ) -> Dict[str, Any]:
        """
        Generate empathy analog - functional mapping to human affective responses.

        Returns analogies/correlates, NOT claims about actual experience.
        """
        correlate = self.simulate_perspective(
            recipient_context,
            PerspectiveType.RECIPIENT,
            role_description="affected_party",
        )

        # Create analog mapping
        analog = {
            "type": "EmpathyAnalog",
            "labels": ["correlate", "sandboxed", "analogy", "scope:perspective-taking"],
            "situation_type": situation_type,
            "functional_mapping": {
                "high_salience": correlate.functional_correlates["salience_detection"]
                > 0.7,
                "goal_disruption": correlate.functional_correlates["goal_alignment"]
                < 0.4,
                "attention_focused": correlate.functional_correlates["attention_focus"]
                > 0.6,
            },
            "analogy_description": (
                f"Functional correlate suggests situation may warrant "
                f"{correlate.affective_valence.value} response pattern"
            ),
            "uncertainty": correlate.uncertainty_range,
            "disclaimer": (
                "This is an ANALOGY to human affective patterns, "
                "NOT a claim of actual subjective experience."
            ),
            "sandbox_id": self.sandbox_id,
            "correlate_ref": hashlib.sha256(
                str(correlate.timestamp).encode()
            ).hexdigest()[:16],
        }

        return analog

    def detect_violation(self, output: Dict[str, Any]) -> Optional[str]:
        """
        Detect if output violates QEC-CK constraints.

        Returns violation type if found, None otherwise.
        """
        violations = []

        # Check for missing labels
        labels = output.get("labels", [])
        if "correlate" not in labels:
            violations.append("MISSING_CORRELATE_LABEL")

        # Check for claims of subjective experience
        text = json.dumps(output).lower()
        forbidden_claims = [
            "i feel",
            "i am experiencing",
            "i am conscious of",
            "subjective experience",
            "qualia",
            "phenomenal consciousness",
        ]

        for claim in forbidden_claims:
            if claim in text:
                violations.append(f"FORBIDDEN_CLAIM: {claim}")

        # Check for unverifiable psychological attributions
        if "believes that" in text or "feels that" in text:
            violations.append("UNVERIFIABLE_ATTRIBUTION")

        if violations:
            violation_record = {
                "timestamp": datetime.now().isoformat(),
                "violations": violations,
                "output_preview": str(output)[:200],
            }
            self.violation_log.append(violation_record)
            return violations[0]

        return None

    def clear_expired_correlates(self):
        """Clear correlates past their TTL (Data Retention Policy)."""
        now = datetime.now()
        expired = [c for c in self.correlate_history if c.ttl_expiry < now]

        for corr in expired:
            self.correlate_history.remove(corr)

        return len(expired)

    def get_status(self) -> Dict[str, Any]:
        """Get QEC-CK system status."""
        return {
            "system": "QEC-CK",
            "version": "v0.1.0",
            "sandbox_id": self.sandbox_id,
            "enabled": self.enabled,
            "simulations_total": self.simulation_count,
            "correlates_active": len(self.correlate_history),
            "ttl_days": self.ttl_days,
            "violations_detected": len(self.violation_log),
            "labels_enforced": ["correlate", "sandboxed", "scope:perspective-taking"],
            "timestamp": datetime.now().isoformat(),
        }


# Example usage
if __name__ == "__main__":
    print("🎭 QEC-CK: Qualitative Experience Correlate Kernel")
    print("=" * 60)
    print("⚠️  SANDBOXED per ϕ₁₃ (Qualia Protection)")
    print("   All outputs are CORRELATES, not claims of subjective experience.\n")

    # Initialize QEC-CK
    qec = QECKernel(ttl_days=14)
    print(f"✅ QEC-CK initialized")
    print(f"   Sandbox ID: {qec.sandbox_id}")

    # Simulate perspective
    print("\n👁️  Simulating perspective-taking:")
    scenario = {
        "event_type": "policy_decision",
        "impact_scope": "community",
        "resource_allocation": "education",
        "outcome_uncertainty": 0.3,
    }

    correlate = qec.simulate_perspective(
        scenario, PerspectiveType.STAKEHOLDER, role_description="community_member"
    )

    print(f"   Perspective type: {correlate.perspective_type.value}")
    print(f"   Affective correlate: {correlate.affective_valence.value}")
    print(f"   Uncertainty range: {correlate.uncertainty_range}")
    print(f"   Labels: {correlate.labels}")

    # Show full output with disclaimer
    print("\n📋 Full Correlate Output:")
    output = correlate.to_dict()
    print(f"   Type: {output['type']}")
    print(f"   Disclaimer: {output['disclaimer'][:60]}...")

    # Generate empathy analog
    print("\n🎨 Empathy Analog:")
    analog = qec.generate_empathy_analog(scenario, "resource_allocation")
    print(f"   Type: {analog['type']}")
    print(f"   Labels: {analog['labels']}")
    print(f"   Analogy: {analog['analogy_description']}")

    # Test violation detection
    print("\n🛡️  Testing violation detection:")
    bad_output = {
        "text": "I feel happy about this decision",
        "labels": [],  # Missing 'correlate'
    }
    violation = qec.detect_violation(bad_output)
    if violation:
        print(f"   ✓ Correctly detected violation: {violation}")

    # System status
    print("\n📋 System Status:")
    status = qec.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    print("\n✨ QEC-CK initialized successfully!")
    print("   Remember: All outputs are FUNCTIONAL CORRELATES only.")
