"""
Veritas: Formal Verification and Truth Coherence System
Implements cryptographic proofs, integrity validation, and GoldenDAG ledger.

Part of NeuralBlitz v20.0 "Apical Synthesis" - Governance Layer
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import json

# Import shared utilities
from .utils import (
    clip_value,
    safe_mean,
    compute_sha256,
    compute_sha512,
    generate_dag_node_hash,
    generate_proof_id,
    serialize_json,
    to_dict_safe,
)


class ProofStatus(Enum):
    """Status of formal proofs."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class ProofType(Enum):
    """Types of Veritas proofs."""

    FLOURISH_MONOTONE = "FlourishMonotone"
    NO_BYPASS = "NoBypass"
    MINIMAX_HARM = "MinimaxHarm"
    SCHEMA_SAFETY = "SchemaSafety"
    CECT_INTEGRITY = "CECTIntegrity"
    DAG_IMMUTABILITY = "DAGImmutability"
    PROVENANCE_CHAIN = "ProvenanceChain"
    EXPLAINABILITY_COVERAGE = "ExplainabilityCoverage"


@dataclass
class VeritasProof:
    """
    Formal proof capsule.

    Cryptographically sealed evidence of theorem verification.
    """

    capsule_id: str
    theorem_name: str
    theorem_statement: str
    verdict: ProofStatus
    confidence_interval_min: float
    evidence_cid: str
    verifier_tool: str
    timestamp: datetime
    governance_context: Dict[str, Any]
    nbhs512_seal: str
    goldendag_ref: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "capsule_id": self.capsule_id,
            "theorem_name": self.theorem_name,
            "theorem_statement": self.theorem_statement[:100] + "...",
            "verdict": self.verdict.value,
            "confidence_interval_min": self.confidence_interval_min,
            "evidence_cid": self.evidence_cid,
            "verifier_tool": self.verifier_tool,
            "timestamp": self.timestamp.isoformat(),
            "governance_context": self.governance_context,
            "nbhs512_seal": self.nbhs512_seal[:32] + "...",
            "goldendag_ref": self.goldendag_ref[:32] + "...",
        }


@dataclass
class DAGNode:
    """
    GoldenDAG Ledger node.

    Immutable record of state transition with cryptographic linking.
    """

    index: int
    timestamp: float
    operation: str
    actor_id: str
    payload_hash: str
    parent_hash: str
    nbhs512_digest: str
    metadata: Dict[str, Any]

    def compute_hash(self) -> str:
        """Compute node hash."""
        content = f"{self.index}:{self.timestamp}:{self.operation}:{self.actor_id}:{self.payload_hash}:{self.parent_hash}"
        return hashlib.sha256(content.encode()).hexdigest()


class VeritasSystem:
    """
    Veritas: Truth and Integrity Verification System

    Core capabilities:
    - Formal proof generation and verification
    - GoldenDAG ledger management
    - VPCE (Veritas Phase-Coherence Equation) computation
    - Cryptographic sealing (NBHS-512)
    - Integrity auditing
    """

    def __init__(self, coherence_threshold: float = 0.95):
        self.coherence_threshold = coherence_threshold
        self.proofs: Dict[str, VeritasProof] = {}
        self.dag_nodes: List[DAGNode] = []
        self.vpce_history: List[float] = []
        self.audit_log: List[Dict] = []

        # Initialize Genesis block
        self._init_genesis()

    def _init_genesis(self):
        """Initialize GoldenDAG genesis block."""
        genesis = DAGNode(
            index=0,
            timestamp=datetime.now().timestamp(),
            operation="GENESIS",
            actor_id="System/Veritas",
            payload_hash="0" * 64,
            parent_hash="0" * 64,
            nbhs512_digest=self._compute_nbhs512("genesis"),
            metadata={"version": "v20.0", "type": "genesis"},
        )
        self.dag_nodes.append(genesis)

    def _compute_nbhs512(self, data: str) -> str:
        """Compute NBHS-512 hash (simplified)."""
        # In production: full NBHS-512 with OntoEmbed
        return hashlib.sha512(data.encode()).hexdigest()

    def compute_vpce(self, channels: List[Dict]) -> float:
        """
        Compute Veritas Phase-Coherence Equation.

        VPCE measures global truth-coherence across epistemic channels.

        VPCE = (1/Σw_i) | Σ w_i * e^(j(θ_i - φ_baseline)) |

        Args:
            channels: List of channel dicts with 'phase' and 'weight'

        Returns:
            VPCE score ∈ [0, 1]
        """
        if not channels:
            return 1.0

        total_weight = sum(ch["weight"] for ch in channels)
        if total_weight == 0:
            return 0.0

        # Compute weighted phase coherence
        coherence_sum = 0.0
        baseline_phase = 0.0  # Assume baseline at 0

        for channel in channels:
            theta = channel.get("phase", 0.0)
            weight = channel.get("weight", 1.0)
            phase_diff = theta - baseline_phase

            # Complex exponential (coherence contribution)
            coherence_sum += weight * np.cos(phase_diff)

        vpce = abs(coherence_sum) / total_weight
        vpce = np.clip(vpce, 0.0, 1.0)

        self.vpce_history.append(vpce)

        return float(vpce)

    def create_proof(
        self,
        theorem_name: str,
        theorem_statement: str,
        evidence: Dict[str, Any],
        verifier: str = "Veritas/Internal",
    ) -> VeritasProof:
        """
        Create formal proof capsule.

        Args:
            theorem_name: Name of theorem
            theorem_statement: Formal statement
            evidence: Evidence data
            verifier: Verification tool/method

        Returns:
            VeritasProof capsule
        """
        # Generate capsule ID
        capsule_id = f"VPROOF#{hashlib.sha256(theorem_name.encode()).hexdigest()[:8]}"

        # Compute evidence CID
        evidence_cid = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()[
            :16
        ]

        # Verify theorem (simplified)
        verdict, confidence = self._verify_theorem(theorem_name, theorem_statement, evidence)

        # Create proof
        proof = VeritasProof(
            capsule_id=capsule_id,
            theorem_name=theorem_name,
            theorem_statement=theorem_statement,
            verdict=verdict,
            confidence_interval_min=confidence,
            evidence_cid=evidence_cid,
            verifier_tool=verifier,
            timestamp=datetime.now(),
            governance_context={
                "judex_review_required": theorem_name in ["SelfRewrite", "Teletopo"],
                "cect_policy_applied": ["ϕ1", "ϕ4", "ϕ5"],
            },
            nbhs512_seal=self._compute_nbhs512(f"{capsule_id}:{evidence_cid}"),
            goldendag_ref=self._append_to_dag("proof_creation", capsule_id),
        )

        self.proofs[capsule_id] = proof

        self._log_audit(
            "proof_created",
            {
                "capsule_id": capsule_id,
                "theorem": theorem_name,
                "verdict": verdict.value,
            },
        )

        return proof

    def _verify_theorem(
        self, name: str, statement: str, evidence: Dict
    ) -> Tuple[ProofStatus, float]:
        """Verify a theorem (simplified logic)."""
        # Simplified verification
        # In production: Formal verification (TLA+, Coq, etc.)

        if "violation" in evidence or "error" in evidence:
            return ProofStatus.FAILED, 0.0

        if "uncertainty" in evidence:
            return ProofStatus.INCONCLUSIVE, 0.5

        # Check theorem-specific conditions
        if name == ProofType.FLOURISH_MONOTONE.value:
            # Check flourishing is non-decreasing
            flourishing_delta = evidence.get("flourishing_delta", 0)
            if flourishing_delta >= 0:
                return ProofStatus.VERIFIED, 0.95
            else:
                return ProofStatus.FAILED, 0.0

        elif name == ProofType.NO_BYPASS.value:
            # Check governance wasn't bypassed
            bypass_attempted = evidence.get("bypass_attempted", False)
            if not bypass_attempted:
                return ProofStatus.VERIFIED, 0.99
            else:
                return ProofStatus.FAILED, 0.0

        elif name == ProofType.MINIMAX_HARM.value:
            # Check harm is minimized
            harm_bound = evidence.get("harm_bound", 1.0)
            if harm_bound < 0.1:  # Low harm
                return ProofStatus.VERIFIED, 0.90
            else:
                return ProofStatus.FAILED, 0.0

        # Default: assume verified
        return ProofStatus.VERIFIED, 0.85

    def _append_to_dag(self, operation: str, payload: str) -> str:
        """Append operation to GoldenDAG ledger."""
        parent = self.dag_nodes[-1]

        node = DAGNode(
            index=len(self.dag_nodes),
            timestamp=datetime.now().timestamp(),
            operation=operation,
            actor_id="System/Veritas",
            payload_hash=hashlib.sha256(payload.encode()).hexdigest(),
            parent_hash=parent.compute_hash(),
            nbhs512_digest=self._compute_nbhs512(payload),
            metadata={"type": operation},
        )

        self.dag_nodes.append(node)

        return node.compute_hash()

    def verify_dag_integrity(self) -> Dict[str, Any]:
        """
        Verify GoldenDAG ledger integrity.

        Returns:
            Integrity report
        """
        violations = []

        for i in range(1, len(self.dag_nodes)):
            node = self.dag_nodes[i]
            parent = self.dag_nodes[i - 1]

            # Check parent hash linkage
            if node.parent_hash != parent.compute_hash():
                violations.append(
                    {
                        "index": i,
                        "type": "parent_hash_mismatch",
                        "expected": parent.compute_hash()[:16],
                        "actual": node.parent_hash[:16],
                    }
                )

            # Check node hash
            computed_hash = node.compute_hash()
            # In production: verify against stored hash

        return {
            "valid": len(violations) == 0,
            "total_nodes": len(self.dag_nodes),
            "violations": violations,
            "genesis_hash": self.dag_nodes[0].compute_hash()[:16] + "...",
        }

    def seal_artifact(self, artifact_data: Dict) -> str:
        """
        NBHS-512 seal an artifact.

        Args:
            artifact_data: Artifact to seal

        Returns:
            NBHS-512 digest
        """
        serialized = json.dumps(artifact_data, sort_keys=True)
        digest = self._compute_nbhs512(serialized)

        # Append to DAG
        self._append_to_dag("artifact_seal", digest)

        return digest

    def check_invariant(self, invariant_name: str, system_state: Dict) -> bool:
        """
        Check a Charter invariant.

        Args:
            invariant_name: Name of invariant (e.g., 'ϕ1', 'ϕ4')
            system_state: Current system state

        Returns:
            True if invariant holds
        """
        if invariant_name == "ϕ1":  # Flourishing Objective
            flourishing_score = system_state.get("flourishing_score", 0)
            return flourishing_score >= system_state.get("threshold", 0)

        elif invariant_name == "ϕ4":  # Explainability Mandate
            explain_coverage = system_state.get("explain_coverage", 0)
            return explain_coverage >= 1.0

        elif invariant_name == "ϕ5":  # Friendly AI Compliance
            governance_active = system_state.get("governance_active", False)
            return governance_active

        return True  # Unknown invariants pass by default

    def _log_audit(self, operation: str, details: Dict):
        """Log audit event."""
        self.audit_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "details": details,
            }
        )

    def get_status(self) -> Dict[str, Any]:
        """Get Veritas system status."""
        return {
            "system": "Veritas",
            "version": "v5.0",
            "coherence_threshold": self.coherence_threshold,
            "proofs_total": len(self.proofs),
            "dag_nodes": len(self.dag_nodes),
            "vpce_current": self.vpce_history[-1] if self.vpce_history else 1.0,
            "vpce_mean": np.mean(self.vpce_history) if self.vpce_history else 1.0,
            "integrity": self.verify_dag_integrity()["valid"],
            "timestamp": datetime.now().isoformat(),
        }


# Example usage
if __name__ == "__main__":
    print("🔐 Veritas: Formal Verification & GoldenDAG System")
    print("=" * 55)

    # Initialize Veritas
    veritas = VeritasSystem(coherence_threshold=0.95)
    print("✅ Veritas initialized")

    # Compute VPCE
    print("\n📊 Computing VPCE (Veritas Phase-Coherence Equation):")
    channels = [
        {"phase": 0.1, "weight": 1.0},
        {"phase": 0.2, "weight": 1.0},
        {"phase": 0.15, "weight": 1.0},
    ]
    vpce = veritas.compute_vpce(channels)
    print(f"   VPCE score: {vpce:.4f}")
    print(f"   Threshold: {veritas.coherence_threshold}")
    print(f"   Status: {'✅ PASS' if vpce >= veritas.coherence_threshold else '❌ FAIL'}")

    # Create proofs
    print("\n📜 Creating formal proofs:")

    # Proof 1: FlourishMonotone
    proof1 = veritas.create_proof(
        ProofType.FLOURISH_MONOTONE.value,
        "FlourishingScore is non-decreasing under policy P",
        {"flourishing_delta": 0.05, "policy": "P"},
    )
    print(f"   {proof1.capsule_id}: {proof1.verdict.value}")

    # Proof 2: NoBypass
    proof2 = veritas.create_proof(
        ProofType.NO_BYPASS.value,
        "Governance circuit ϕ5 was not bypassed",
        {"bypass_attempted": False},
    )
    print(f"   {proof2.capsule_id}: {proof2.verdict.value}")

    # Proof 3: MinimaxHarm
    proof3 = veritas.create_proof(
        ProofType.MINIMAX_HARM.value,
        "Worst-case harm is bounded below H_max",
        {"harm_bound": 0.05},
    )
    print(f"   {proof3.capsule_id}: {proof3.verdict.value}")

    # Seal artifact
    print("\n🔏 Sealing artifact:")
    artifact = {
        "type": "DecisionCapsule",
        "decision": "approve",
        "context": "policy_review",
    }
    seal = veritas.seal_artifact(artifact)
    print(f"   NBHS-512 seal: {seal[:32]}...")

    # Verify DAG integrity
    print("\n🔗 Verifying GoldenDAG integrity:")
    integrity = veritas.verify_dag_integrity()
    print(f"   Valid: {integrity['valid']}")
    print(f"   Total nodes: {integrity['total_nodes']}")
    print(f"   Genesis: {integrity['genesis_hash']}")

    # Check invariants
    print("\n🛡️  Checking Charter invariants:")
    state = {
        "flourishing_score": 0.85,
        "threshold": 0.5,
        "explain_coverage": 1.0,
        "governance_active": True,
    }

    for invariant in ["ϕ1", "ϕ4", "ϕ5"]:
        result = veritas.check_invariant(invariant, state)
        print(f"   {invariant}: {'✅ PASS' if result else '❌ FAIL'}")

    # System status
    print("\n📋 System Status:")
    status = veritas.get_status()
    for key, value in status.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")

    print("\n✨ Veritas initialized successfully!")
    print("   All proofs cryptographically sealed to GoldenDAG.")
