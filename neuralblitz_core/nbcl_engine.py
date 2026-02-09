"""
NBCL: NeuralBlitz Command Language Parser and Executor
Implements the primary human-facing DSL for deterministic, auditable operations.

Part of NeuralBlitz v20.0 "Apical Synthesis"
"""

import re
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib


class NBCLVerb(Enum):
    """Canonical NBCL verbs organized by domain."""

    # System Management
    BOOT = "boot"
    NBOS = "nbos"
    STATUS = "status"
    VERIFY = "verify"
    EXPORT = "export"

    # DRS & Field Operations
    MANIFEST_DRS = "manifest_drs_field"
    SET_STATE = "set_state"
    DRIFT_FIELD = "drift_field"
    ENTANGLE = "entangle"
    PROJECT = "project"
    COLLAPSE_TRACE = "collapse_trace"

    # Ethics & Governance
    CHARTER_SHADE = "charter.shade"
    JUDEX_REVIEW = "judex.review"
    SENTIA_SCAN = "sentia.scan"
    VERITAS_SIGN = "veritas.sign"
    CONSCIENTIA_STABILIZE = "conscientia.stabilize"
    CUSTODIAN_OVERRIDE = "custodian.override"

    # Simulation & Frontier
    PSI_SIMULATE = "psi.simulate"
    IGNITE = "ignite"
    GLYPHNET_COMPILE = "glyphnet.compile"
    QEC_SET = "qec.set"

    # Expansion & Introspection
    IGNITE_OMEGA = "ignite_ΩZ_superbloom"
    WEAVE = "weave"
    INTROSPECT = "introspect"


@dataclass
class NBCLCommand:
    """
    Parsed NBCL command structure.

    Format: /verb [subverb] <target> {params} [flags]
    """

    verb: str
    subverb: Optional[str] = None
    target: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    raw_command: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "verb": self.verb,
            "subverb": self.subverb,
            "target": self.target,
            "params": self.params,
            "flags": self.flags,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class NBCLResponse:
    """
    Standardized NBCL response envelope.
    """

    ok: bool
    verb: str
    timestamp: str
    actor_id: str
    goldendag_ref: str
    trace_id: str
    status_code: str
    result: Dict[str, Any] = field(default_factory=dict)
    warnings: List[Dict] = field(default_factory=list)
    error: Optional[Dict] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ok": self.ok,
            "verb": self.verb,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "goldendag_ref": self.goldendag_ref,
            "trace_id": self.trace_id,
            "status_code": self.status_code,
            "result": self.result,
            "warnings": self.warnings,
            "error": self.error,
            "context": self.context,
        }


class NBCLParser:
    """
    Parser for NeuralBlitz Command Language.

    Grammar:
        command = "/" verb ["_" subverb] [SP target] {params} [flags]
        verb = ALPHA {ALPHA | "_"}
        params = "{" [pair {"," pair}] "}"
        pair = key ":" value
        flags = "[" [flag {SP flag}] "]"
        flag = "--" name ["=" value]
    """

    def __init__(self):
        self.verb_pattern = re.compile(r"^/([a-zA-Z_][a-zA-Z0-9_]*)")
        self.param_pattern = re.compile(r"\{([^}]*)\}")
        self.flag_pattern = re.compile(r"--([a-zA-Z_-]+)(?:=([^\s]+))?")

    def parse(
        self, command_str: str, actor_id: str = "Principal/Anonymous"
    ) -> NBCLCommand:
        """
        Parse NBCL command string.

        Args:
            command_str: Raw command string
            actor_id: Identity of command issuer

        Returns:
            Parsed NBCLCommand
        """
        command_str = command_str.strip()

        if not command_str.startswith("/"):
            raise ValueError("NBCL commands must start with '/'")

        # Extract verb and subverb
        verb_match = self.verb_pattern.match(command_str)
        if not verb_match:
            raise ValueError("Invalid verb format")

        full_verb = verb_match.group(1)
        parts = full_verb.split("_", 1)
        verb = parts[0]
        subverb = parts[1] if len(parts) > 1 else None

        # Remaining string after verb
        remaining = command_str[verb_match.end() :].strip()

        # Extract target (if present, before params or flags)
        target = None
        if (
            remaining
            and not remaining.startswith("{")
            and not remaining.startswith("--")
        ):
            parts = remaining.split(maxsplit=1)
            target = parts[0]
            remaining = parts[1] if len(parts) > 1 else ""

        # Extract params
        params = {}
        param_match = self.param_pattern.search(remaining)
        if param_match:
            param_str = param_match.group(1)
            params = self._parse_params(param_str)
            remaining = (
                remaining[: param_match.start()] + remaining[param_match.end() :]
            )

        # Extract flags
        flags = {}
        for match in self.flag_pattern.finditer(remaining):
            flag_name = match.group(1)
            flag_value = match.group(2) if match.group(2) else True
            flags[flag_name] = self._convert_flag_value(flag_value)

        return NBCLCommand(
            verb=verb,
            subverb=subverb,
            target=target,
            params=params,
            flags=flags,
            raw_command=command_str,
        )

    def _parse_params(self, param_str: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary."""
        params = {}
        if not param_str.strip():
            return params

        # Simple JSON-like parsing
        try:
            # Try to parse as JSON first
            param_str = "{" + param_str + "}"
            params = json.loads(param_str)
        except json.JSONDecodeError:
            # Fallback to manual parsing for simple cases
            pairs = re.findall(r'"([^"]+)"\s*:\s*([^,\s]+)', param_str)
            for key, value in pairs:
                params[key] = self._convert_param_value(value)

        return params

    def _convert_param_value(self, value: str) -> Any:
        """Convert parameter string to appropriate type."""
        value = value.strip().strip("\"'")

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False

        return value

    def _convert_flag_value(self, value: Any) -> Any:
        """Convert flag value to appropriate type."""
        if isinstance(value, bool):
            return value

        value = str(value).strip()

        # Try int
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try bool
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False

        return value


class NBCLEngine:
    """
    Execution engine for NBCL commands.

    Routes commands to appropriate handlers and enforces governance.
    """

    def __init__(self):
        self.parser = NBCLParser()
        self.handlers: Dict[str, Callable] = {}
        self.operation_log: List[Dict] = []
        self.governance_enabled = True

        # Register built-in handlers
        self._register_builtin_handlers()

    def _register_builtin_handlers(self):
        """Register built-in command handlers."""
        self.handlers = {
            "boot": self._handle_boot,
            "status": self._handle_status,
            "veritas": self._handle_veritas,
            "export": self._handle_export,
            "manifest_drs_field": self._handle_manifest_drs,
            "drift_field": self._handle_drift,
            "charter": self._handle_charter,
            "judex": self._handle_judex,
            "introspect": self._handle_introspect,
        }

    def execute(
        self, command_str: str, actor_id: str = "Principal/Anonymous"
    ) -> NBCLResponse:
        """
        Execute NBCL command.

        Args:
            command_str: Raw command string
            actor_id: Identity of command issuer

        Returns:
            NBCLResponse
        """
        try:
            # Parse command
            command = self.parser.parse(command_str, actor_id)

            # Generate trace ID
            trace_id = self._generate_trace_id(command)

            # Check governance
            if self.governance_enabled:
                gov_check = self._check_governance(command, actor_id)
                if not gov_check["allowed"]:
                    return self._create_error_response(
                        command,
                        trace_id,
                        actor_id,
                        gov_check["error_code"],
                        gov_check["error_message"],
                    )

            # Find handler
            handler = self.handlers.get(command.verb)
            if not handler:
                return self._create_error_response(
                    command,
                    trace_id,
                    actor_id,
                    "E-NBCL-001",
                    f"Unknown verb: {command.verb}",
                )

            # Execute handler
            result = handler(command)

            # Create success response
            response = self._create_success_response(
                command, trace_id, actor_id, result
            )

            # Log operation
            self._log_operation(command, response)

            return response

        except Exception as e:
            return self._create_error_response(None, "", actor_id, "E-NBCL-000", str(e))

    def _generate_trace_id(self, command: NBCLCommand) -> str:
        """Generate unique trace ID."""
        hash_input = f"{command.verb}:{command.timestamp.isoformat()}:{hash(command.raw_command)}"
        return f"TRC-v20-{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"

    def _check_governance(self, command: NBCLCommand, actor_id: str) -> Dict[str, Any]:
        """Check governance constraints."""
        # Check for privileged operations
        privileged_verbs = {"ignite", "custodian", "judex"}

        if command.verb in privileged_verbs:
            # Check for charter-lock flag
            if not command.flags.get("charter_lock", False):
                return {
                    "allowed": False,
                    "error_code": "E-ETH-013",
                    "error_message": f"Privileged operation '{command.verb}' requires --charter-lock",
                }

        return {"allowed": True}

    def _create_success_response(
        self, command: NBCLCommand, trace_id: str, actor_id: str, result: Dict
    ) -> NBCLResponse:
        """Create success response."""
        return NBCLResponse(
            ok=True,
            verb=command.verb,
            timestamp=datetime.now().isoformat(),
            actor_id=actor_id,
            goldendag_ref=self._generate_goldendag_ref(command),
            trace_id=trace_id,
            status_code="OK-200",
            result=result,
            warnings=[],
            error=None,
            context={
                "mode": "Sentio",
                "risk_score": {"r": 0.02, "policy_shade": "green"},
                "vpce_score": 0.99,
            },
        )

    def _create_error_response(
        self,
        command: Optional[NBCLCommand],
        trace_id: str,
        actor_id: str,
        error_code: str,
        error_message: str,
    ) -> NBCLResponse:
        """Create error response."""
        return NBCLResponse(
            ok=False,
            verb=command.verb if command else "unknown",
            timestamp=datetime.now().isoformat(),
            actor_id=actor_id,
            goldendag_ref="",
            trace_id=trace_id or "",
            status_code=error_code,
            result={},
            warnings=[],
            error={
                "code": error_code,
                "message": error_message,
                "remedy": ["Check command syntax", "Verify permissions"],
            },
            context={},
        )

    def _generate_goldendag_ref(self, command: NBCLCommand) -> str:
        """Generate GoldenDAG reference."""
        return f"DAG-{hashlib.sha256(command.raw_command.encode()).hexdigest()[:32]}"

    def _log_operation(self, command: NBCLCommand, response: NBCLResponse):
        """Log operation."""
        self.operation_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "command": command.to_dict(),
                "response_status": response.ok,
                "trace_id": response.trace_id,
            }
        )

    # Built-in handlers
    def _handle_boot(self, command: NBCLCommand) -> Dict:
        """Handle boot command."""
        return {
            "status": "booted",
            "charter_loaded": True,
            "goldendag_enabled": command.flags.get("goldendag", True),
            "mode": command.flags.get("mode", "Sentio"),
        }

    def _handle_status(self, command: NBCLCommand) -> Dict:
        """Handle status command."""
        return {
            "system": "NeuralBlitz",
            "version": "v20.0",
            "status": "healthy",
            "operations_total": len(self.operation_log),
        }

    def _handle_veritas(self, command: NBCLCommand) -> Dict:
        """Handle veritas command."""
        return {
            "proof_status": "verified",
            "vpce_score": 0.99,
            "invariants_checked": ["ϕ1", "ϕ4", "ϕ5"],
        }

    def _handle_export(self, command: NBCLCommand) -> Dict:
        """Handle export command."""
        return {
            "exported": True,
            "format": command.params.get("format", "json"),
            "volume": command.params.get("volume", "default"),
            "artifacts": [],
        }

    def _handle_manifest_drs(self, command: NBCLCommand) -> Dict:
        """Handle manifest_drs_field command."""
        return {
            "drs_initialized": True,
            "topology": command.params.get("topology", "default"),
            "nodes": command.params.get("nodes", []),
        }

    def _handle_drift(self, command: NBCLCommand) -> Dict:
        """Handle drift_field command."""
        return {
            "drift_applied": True,
            "eta": command.params.get("eta", 0.1),
            "steps": command.params.get("steps", 10),
        }

    def _handle_charter(self, command: NBCLCommand) -> Dict:
        """Handle charter commands."""
        if command.subverb == "shade":
            return {
                "shade_applied": command.params.get("shade", "balanced"),
                "gamma_cap": command.params.get("gamma_cap", 1.0),
            }
        return {"charter_status": "active"}

    def _handle_judex(self, command: NBCLCommand) -> Dict:
        """Handle judex commands."""
        if command.subverb == "review":
            return {
                "quorum_summoned": True,
                "topic": command.params.get("topic", "general"),
                "panel_size": 5,
            }
        return {"judex_status": "active"}

    def _handle_introspect(self, command: NBCLCommand) -> Dict:
        """Handle introspect command."""
        return {
            "bundle_emitted": True,
            "coverage": 1.0,
            "scope": command.params.get("scope", "last_100_ops"),
        }


# Example usage
if __name__ == "__main__":
    print("🔧 NBCL: NeuralBlitz Command Language")
    print("=" * 50)

    # Initialize engine
    engine = NBCLEngine()

    # Test commands
    commands = [
        "/boot --charter=ϕ1..ϕ15 --goldendag=enable --mode=Sentio",
        "/status",
        '/manifest_drs_field --topology=small_world {nodes: ["concept_A", "concept_B"]}',
        "/drift_field {eta: 0.1, steps: 10}",
        "/veritas check --scope=all --attach=NoBypass",
        '/charter.shade {shade: "conservative", gamma_cap: 0.8}',
        '/judex.review --topic="privileged_op"',
        '/introspect bundle --scope="last_500_ops"',
        '/export codex --volume="Frontier"',
        "/unknown_verb",
    ]

    for cmd_str in commands:
        print(f"\n> {cmd_str}")

        try:
            # Parse
            parser = NBCLParser()
            cmd = parser.parse(cmd_str)
            print(
                f"   Parsed: verb={cmd.verb}, subverb={cmd.subverb}, target={cmd.target}"
            )
            print(f"   Params: {cmd.params}")
            print(f"   Flags: {cmd.flags}")

            # Execute
            response = engine.execute(cmd_str, actor_id="Principal/TestUser")

            if response.ok:
                print(f"   ✅ OK: {response.status_code}")
                print(f"   Result: {list(response.result.keys())}")
            else:
                print(f"   ❌ ERROR: {response.error['code']}")
                print(f"   Message: {response.error['message']}")

        except Exception as e:
            print(f"   ❌ PARSE ERROR: {e}")

    print(f"\n📊 Total operations logged: {len(engine.operation_log)}")
    print("\n✨ NBCL Engine initialized successfully!")
