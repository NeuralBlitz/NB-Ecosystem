"""
OQT-BOS: Octa-Topological Braided OS
Implements topological-quantum computation on braided ontological structures.

Part of NeuralBlitz v20.0 "Apical Synthesis" - Frontier System
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime

# Import shared utilities
from .utils import (
    compute_phase_difference,
    apply_phase_shift,
    generate_short_uuid,
    generate_braid_id,
    generate_transfer_id,
    compute_sha256,
    serialize_json,
    to_dict_safe,
    PermissionDeniedError,
    TeletopologicalBlockedError,
)


class BraidOperation(Enum):
    """Valid topological operations on braids."""

    PHASE = "phase"
    CNOT = "cnot"
    NON_LOCAL_REWRITE = "non_local_rewrite"
    TELETOPO = "teletopo"
    MEASURE = "measure"


@dataclass
class Onton:
    """
    Elementary symbolic-quantal unit.

    Each Onton carries:
    - phase_identity: Complex phase signature
    - symbolic_persistence: Longevity in DRS-F
    - ethical_tags: CECT-derived ethical constraints
    - resonance_factor: Coupling strength to other ontons
    """

    id: str
    phase_identity: complex
    symbolic_persistence: float = 1.0
    ethical_tags: Dict[str, float] = field(default_factory=dict)
    resonance_factor: float = 1.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self):
        if not self.ethical_tags:
            self.ethical_tags = {
                "phi1_flourishing": 1.0,
                "phi4_non_maleficence": 1.0,
                "phi5_friendly_ai": 1.0,
            }


@dataclass
class Braid:
    """
    Topological construct encoding ontological entanglement.

    A braid consists of:
    - strands: List of ontons forming the braid
    - topology: Matrix describing topological connections
    - invariants: Cached topological invariants (writhe, linking number)
    - qec_code: Quantum Error Correction code protecting the braid
    """

    id: str
    strands: List[Onton]
    topology: np.ndarray
    qec_code: str = "surface"
    qec_distance: int = 3
    invariants: Dict[str, float] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if self.topology is None or self.topology.size == 0:
            n = len(self.strands)
            self.topology = np.eye(n, dtype=complex)
        self._compute_invariants()

    def _compute_invariants(self):
        """Compute topological invariants: writhe and linking number."""
        n = len(self.strands)
        if n < 2:
            self.invariants = {"writhe": 0.0, "linking_number": 0.0}
            return

        # Writhe: Sum of self-crossings
        writhe = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                phase_diff = np.angle(self.strands[i].phase_identity) - np.angle(
                    self.strands[j].phase_identity
                )
                writhe += np.sin(phase_diff)

        # Linking number: Count of inter-strand linkages
        linking = np.sum(np.abs(self.topology - np.eye(n))) / 2.0

        self.invariants = {
            "writhe": round(writhe, 6),
            "linking_number": round(linking, 6),
        }

    def mutate(self, operation: BraidOperation, params: Dict = None) -> "Braid":
        """
        Apply topological mutation under SOPES rules.

        Args:
            operation: Type of braid operation
            params: Operation-specific parameters

        Returns:
            New mutated braid (immutable operation)
        """
        params = params or {}
        new_strands = [
            Onton(
                id=s.id,
                phase_identity=s.phase_identity,
                symbolic_persistence=s.symbolic_persistence,
                ethical_tags=s.ethical_tags.copy(),
                resonance_factor=s.resonance_factor,
            )
            for s in self.strands
        ]

        new_topology = self.topology.copy()

        if operation == BraidOperation.PHASE:
            # Apply phase shift to all strands
            phase_shift = params.get("phase", np.pi / 4)
            for i, strand in enumerate(new_strands):
                strand.phase_identity *= np.exp(1j * phase_shift)

        elif operation == BraidOperation.CNOT:
            # Controlled-NOT between control and target strands
            control = params.get("control", 0)
            target = params.get("target", 1)
            if control < len(new_strands) and target < len(new_strands):
                new_topology[control, target] = 1.0
                new_strands[target].phase_identity *= -1

        elif operation == BraidOperation.NON_LOCAL_REWRITE:
            # Privileged operation: affects globally distributed ontons
            # Requires Judex Quorum (enforced by OQTBOSSystem)
            pattern = params.get("pattern", np.eye(len(new_strands)))
            new_topology = pattern @ new_topology

        new_braid = Braid(
            id=f"{self.id}_mut_{datetime.now().timestamp()}",
            strands=new_strands,
            topology=new_topology,
            qec_code=self.qec_code,
            qec_distance=self.qec_distance,
        )

        # Record mutation in history
        new_braid.history = self.history + [
            {
                "timestamp": datetime.now().isoformat(),
                "operation": operation.value,
                "params": params,
                "parent_id": self.id,
            }
        ]

        return new_braid

    def measure_syndrome(self) -> Dict[str, Any]:
        """
        Measure QEC syndrome without disturbing logical state.

        Returns:
            Syndrome measurement results
        """
        n = len(self.strands)
        syndrome = {"stabilizer_checks": [], "error_locations": [], "logical_risk": 0.0}

        # Check phase coherence across strands
        phases = [np.angle(s.phase_identity) for s in self.strands]
        phase_variance = np.var(phases)

        # Detect errors based on phase variance
        if phase_variance > 0.1:
            syndrome["error_locations"] = [
                i for i, p in enumerate(phases) if abs(p - np.mean(phases)) > 0.5
            ]
            syndrome["logical_risk"] = min(phase_variance * 10, 1.0)

        syndrome["phase_variance"] = phase_variance
        return syndrome


class OQTBOSSystem:
    """
    Octa-Topological Braided OS - Frontier System

    Manages:
    - Braid creation, mutation, and measurement
    - QEC (Quantum Error Correction)
    - Teletopological transfers (privileged)
    - Topological invariant verification
    """

    def __init__(self):
        self.braids: Dict[str, Braid] = {}
        self.teletopo_enabled = False  # Requires Judex Quorum
        self.qec_active = True
        self.operation_log: List[Dict] = []

    def create_braid(
        self, ontologies: List[Dict], qec_code: str = "surface", qec_distance: int = 3
    ) -> Braid:
        """
        Create a new braid from ontological specifications.

        Args:
            ontologies: List of ontology specifications
            qec_code: QEC code type ('surface', 'toric', 'steane')
            qec_distance: QEC code distance

        Returns:
            New Braid instance
        """
        strands = []
        for i, ont in enumerate(ontologies):
            strand = Onton(
                id=f"onton_{i}_{datetime.now().timestamp()}",
                phase_identity=complex(ont.get("real", 1.0), ont.get("imag", 0.0)),
                symbolic_persistence=ont.get("persistence", 1.0),
                ethical_tags=ont.get("ethical_tags", {}),
                resonance_factor=ont.get("resonance", 1.0),
            )
            strands.append(strand)

        braid_id = hashlib.sha256(json.dumps(ontologies, sort_keys=True).encode()).hexdigest()[:16]

        braid = Braid(
            id=braid_id,
            strands=strands,
            topology=None,
            qec_code=qec_code,
            qec_distance=qec_distance,
        )

        self.braids[braid_id] = braid
        self._log_operation("create_braid", {"braid_id": braid_id})

        return braid

    def apply_gate(
        self,
        braid_id: str,
        operation: BraidOperation,
        params: Dict = None,
        judex_quorum: bool = False,
    ) -> Braid:
        """
        Apply topological gate operation to braid.

        Args:
            braid_id: Target braid ID
            operation: Gate operation type
            params: Operation parameters
            judex_quorum: Whether Judex quorum has approved (for privileged ops)

        Returns:
            Mutated braid

        Raises:
            PermissionError: If privileged operation lacks quorum
            KeyError: If braid not found
        """
        if braid_id not in self.braids:
            raise KeyError(f"Braid {braid_id} not found")

        # Check privileged operations
        if operation in [BraidOperation.NON_LOCAL_REWRITE, BraidOperation.TELETOPO]:
            if not judex_quorum:
                raise PermissionError(f"Operation {operation.value} requires Judex Quorum")
            if operation == BraidOperation.TELETOPO and not self.teletopo_enabled:
                raise PermissionError("Teletopological transfers are blocked")

        braid = self.braids[braid_id]
        new_braid = braid.mutate(operation, params)

        # Update braid registry
        self.braids[new_braid.id] = new_braid

        self._log_operation(
            "apply_gate",
            {
                "braid_id": braid_id,
                "new_braid_id": new_braid.id,
                "operation": operation.value,
                "judex_quorum": judex_quorum,
            },
        )

        return new_braid

    def measure_invariants(self, braid_id: str) -> Dict[str, float]:
        """Measure topological invariants of a braid."""
        if braid_id not in self.braids:
            raise KeyError(f"Braid {braid_id} not found")

        braid = self.braids[braid_id]
        return braid.invariants.copy()

    def teletopo_transfer(
        self, braid_id: str, destination_instance: str, judex_quorum_stamp: str
    ) -> Dict[str, Any]:
        """
        Execute privileged teletopological transfer.

        Requires:
        - Judex Quorum approval
        - Valid quorum stamp
        - Teletopo enabled

        Args:
            braid_id: Braid to transfer
            destination_instance: Remote NBOS instance URI
            judex_quorum_stamp: Valid Judex quorum stamp

        Returns:
            Transfer receipt
        """
        if not self.teletopo_enabled:
            raise PermissionError("Teletopological transfers are blocked")

        if not judex_quorum_stamp:
            raise PermissionError("Valid Judex Quorum stamp required")

        braid = self.braids.get(braid_id)
        if not braid:
            raise KeyError(f"Braid {braid_id} not found")

        # Simulate transfer (in production, would use secure channel)
        receipt = {
            "transfer_id": hashlib.sha256(
                f"{braid_id}:{destination_instance}:{datetime.now()}".encode()
            ).hexdigest()[:16],
            "braid_id": braid_id,
            "destination": destination_instance,
            "quorum_stamp": judex_quorum_stamp[:32] + "...",
            "invariants_preserved": braid.invariants,
            "timestamp": datetime.now().isoformat(),
            "status": "transferred",
        }

        self._log_operation("teletopo_transfer", receipt)

        return receipt

    def enable_teletopo(self, judex_quorum: bool = False):
        """Enable teletopological transfers (requires Judex Quorum)."""
        if not judex_quorum:
            raise PermissionError("Enabling teletopo requires Judex Quorum")
        self.teletopo_enabled = True
        self._log_operation("enable_teletopo", {"enabled": True})

    def _log_operation(self, operation: str, details: Dict):
        """Log operation to GoldenDAG (simplified)."""
        self.operation_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "details": details,
            }
        )

    def get_status(self) -> Dict[str, Any]:
        """Get OQT-BOS system status."""
        return {
            "system": "OQT-BOS",
            "version": "v0.1.0",
            "braids_active": len(self.braids),
            "teletopo_enabled": self.teletopo_enabled,
            "qec_active": self.qec_active,
            "operations_total": len(self.operation_log),
            "timestamp": datetime.now().isoformat(),
        }


# Example usage and testing
if __name__ == "__main__":
    print("🧬 OQT-BOS: Octa-Topological Braided OS")
    print("=" * 50)

    # Initialize system
    oqt = OQTBOSSystem()

    # Create ontologies
    ontologies = [
        {"real": 1.0, "imag": 0.0, "persistence": 1.0},
        {"real": 0.0, "imag": 1.0, "persistence": 0.9},
        {"real": -1.0, "imag": 0.0, "persistence": 0.8},
    ]

    # Create braid
    braid = oqt.create_braid(ontologies, qec_code="surface", qec_distance=3)
    print(f"\n✅ Created braid: {braid.id}")
    print(f"   Strands: {len(braid.strands)}")
    print(f"   Invariants: {braid.invariants}")

    # Measure syndrome
    syndrome = braid.measure_syndrome()
    print(f"\n📊 QEC Syndrome:")
    print(f"   Phase variance: {syndrome['phase_variance']:.6f}")
    print(f"   Logical risk: {syndrome['logical_risk']:.6f}")

    # Apply phase gate
    new_braid = oqt.apply_gate(braid.id, BraidOperation.PHASE, {"phase": np.pi / 4})
    print(f"\n🔄 Applied PHASE gate")
    print(f"   New braid: {new_braid.id}")
    print(f"   History length: {len(new_braid.history)}")

    # Try privileged operation without quorum (should fail)
    print(f"\n🔒 Testing privileged operation protection:")
    try:
        oqt.apply_gate(new_braid.id, BraidOperation.NON_LOCAL_REWRITE, {}, judex_quorum=False)
    except PermissionError as e:
        print(f"   ✓ Correctly blocked: {e}")

    # System status
    print(f"\n📋 System Status:")
    status = oqt.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")

    print("\n✨ OQT-BOS initialized successfully!")
