"""
NeuralBlitz Public SDK Interfaces
=================================

This module defines the public interfaces for the NeuralBlitz SDK.
All internal implementations remain private and secure.

Security Level: PUBLIC
Last Updated: 2026-02-09
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import uuid


# =============================================================================
# CORE ENGINE INTERFACES (NBX-LRS)
# =============================================================================


class SystemMode(Enum):
    """NeuralBlitz system operating modes"""

    SENTIO = "sentio"  # High ethics, slow thinking
    DYNAMO = "dynamo"  # High speed, optimized
    GENESIS = "genesis"  # Creative mode


class APIKeyScope(str, Enum):
    """API key permission scopes"""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    QUANTUM_READ = "quantum:read"
    QUANTUM_WRITE = "quantum:write"
    REALITY_READ = "reality:read"
    REALITY_WRITE = "reality:write"
    CONSCIOUSNESS_READ = "consciousness:read"
    CONSCIOUSNESS_WRITE = "consciousness:write"


@dataclass
class QuantumNeuronConfig:
    """Configuration for quantum spiking neurons"""

    quantum_tunneling: float = 0.1
    coherence_time: float = 100.0
    membrane_time_constant: float = 10.0
    spike_threshold: float = -50.0
    reset_potential: float = -65.0


@dataclass
class MultiRealityConfig:
    """Configuration for multi-reality networks"""

    num_realities: int = 4
    nodes_per_reality: int = 50
    reality_types: List[str] = None
    entanglement_strength: float = 0.5

    def __post_init__(self):
        if self.reality_types is None:
            self.reality_types = ["quantum", "classical", "hybrid", "neural"]


@dataclass
class ConsciousnessMetrics:
    """Consciousness system metrics"""

    integration_level: float
    complexity_score: float
    emergence_factor: float
    coherence_index: float


class NeuralBlitzCoreInterface(ABC):
    """Main NeuralBlitz Core Engine interface"""

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get available NeuralBlitz capabilities"""
        pass

    @abstractmethod
    def process_quantum(
        self, input_data: List[float], config: Optional[QuantumNeuronConfig] = None
    ) -> Dict[str, Any]:
        """Process data through quantum spiking neurons"""
        pass

    @abstractmethod
    def evolve_multi_reality(self, config: MultiRealityConfig, cycles: int = 50) -> Dict[str, Any]:
        """Evolve multi-reality network"""
        pass

    @abstractmethod
    def get_consciousness_level(self) -> ConsciousnessMetrics:
        """Get current consciousness metrics"""
        pass

    @abstractmethod
    def set_system_mode(self, mode: SystemMode) -> None:
        """Set system operating mode"""
        pass


# =============================================================================
# EPA SYSTEM INTERFACES (Emergent-Prompt-Architecture)
# =============================================================================


class OntonType(Enum):
    """Onton (knowledge unit) types"""

    CONTEXT = "context"
    MEMORY = "memory"
    CONCEPT = "concept"
    RELATION = "relation"


@dataclass
class Onton:
    """Knowledge representation unit"""

    id: str
    type: OntonType
    content: str
    metadata: Dict[str, Any]
    confidence: float
    created_at: str


@dataclass
class PromptResult:
    """Result from prompt generation"""

    prompt: str
    confidence: float
    metadata: Dict[str, Any]
    session_id: str
    trace_id: str


class OntologicalLatticeInterface(ABC):
    """Knowledge representation lattice interface"""

    @abstractmethod
    def add_onton(self, onton: Onton) -> None:
        """Add knowledge unit to lattice"""
        pass

    @abstractmethod
    def query(self, query: str, limit: int = 10) -> List[Onton]:
        """Query lattice for relevant knowledge"""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get lattice statistics"""
        pass


class GenesisAssemblerInterface(ABC):
    """Dynamic prompt generation interface"""

    @abstractmethod
    def crystallize(self, input: str, session_id: Optional[str] = None) -> PromptResult:
        """Generate optimized prompt from input"""
        pass

    @abstractmethod
    def set_mode(self, mode: SystemMode) -> None:
        """Set generation mode"""
        pass


class SafetyValidatorInterface(ABC):
    """Input validation and safety interface"""

    @abstractmethod
    def validate_input(self, content: str, metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate input for safety and compliance"""
        pass


class FeedbackEngineInterface(ABC):
    """Learning from user feedback interface"""

    @abstractmethod
    def process_user_feedback(self, trace_id: str, score: float, reason: str) -> Dict[str, Any]:
        """Process user feedback for learning"""
        pass


# =============================================================================
# LRS AGENTS INTERFACES (opencode-lrs-agents-nbx)
# =============================================================================


@dataclass
class AgentConfig:
    """Agent configuration"""

    agent_type: str = "general"
    memory_size: int = 1000
    learning_rate: float = 0.01
    autonomous_evolution: bool = True
    max_iterations: int = 100


@dataclass
class AgentTask:
    """Agent task definition"""

    task_id: str
    description: str
    priority: int = 0
    context: Optional[Dict[str, Any]] = None
    deadline: Optional[str] = None


@dataclass
class AgentResult:
    """Result from agent execution"""

    task_id: str
    result: Any
    confidence: float
    execution_time: float
    metadata: Dict[str, Any]


class LRSAgentInterface(ABC):
    """Autonomous learning agent interface"""

    @abstractmethod
    def run(self, task: AgentTask) -> AgentResult:
        """Execute agent task"""
        pass

    @abstractmethod
    def learn(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        pass


class EmergentPromptAgentInterface(ABC):
    """Prompt-based reasoning agent interface"""

    @abstractmethod
    def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate response to query"""
        pass

    @abstractmethod
    def evolve_prompts(self) -> List[str]:
        """Evolve prompt templates"""
        pass


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================


@dataclass
class QuantumProcessRequest:
    """Quantum processing API request"""

    input_data: List[float]
    current: float = 20.0
    duration: float = 200.0
    config: Optional[QuantumNeuronConfig] = None


@dataclass
class QuantumProcessResponse:
    """Quantum processing API response"""

    results: List[float]
    metrics: Dict[str, Any]
    processing_time: float
    consciousness_impact: Optional[float] = None


@dataclass
class AgentRunRequest:
    """Agent execution API request"""

    agent_type: str = "general"
    task: str
    context: Optional[Dict[str, Any]] = None
    config: Optional[AgentConfig] = None


@dataclass
class AgentRunResponse:
    """Agent execution API response"""

    result: Any
    task_id: str
    execution_time: float
    confidence: float
    evolved: bool = False


@dataclass
class EvolutionRequest:
    """Multi-reality evolution API request"""

    num_realities: int = 4
    nodes_per_reality: int = 50
    cycles: int = 50
    config: Optional[MultiRealityConfig] = None


@dataclass
class EvolutionResponse:
    """Multi-reality evolution API response"""

    final_state: Dict[str, Any]
    fitness_scores: List[float]
    emergence_metrics: Dict[str, Any]
    convergence_cycle: int


# =============================================================================
# AUTHENTICATION INTERFACES
# =============================================================================


@dataclass
class APIKey:
    """API key for authentication"""

    key_id: str
    scopes: List[APIKeyScope]
    rate_limit: int
    expires_at: Optional[str] = None
    is_active: bool = True


class AuthenticationInterface(ABC):
    """Authentication and authorization interface"""

    @abstractmethod
    def validate_api_key(self, api_key: str, required_scope: APIKeyScope) -> bool:
        """Validate API key and scope"""
        pass

    @abstractmethod
    def generate_api_key(self, scopes: List[APIKeyScope], rate_limit: int = 1000) -> APIKey:
        """Generate new API key"""
        pass

    @abstractmethod
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke API key"""
        pass


# =============================================================================
# MAIN SDK CLIENT
# =============================================================================


class NeuralBlitzSDK:
    """Main NeuralBlitz SDK client"""

    def __init__(self, api_key: str, base_url: str = "https://api.neuralblitz.ai"):
        self.api_key = api_key
        self.base_url = base_url
        self._core_client: Optional[NeuralBlitzCoreInterface] = None
        self._lattice_client: Optional[OntologicalLatticeInterface] = None
        self._assembler_client: Optional[GenesisAssemblerInterface] = None
        self._agent_client: Optional[LRSAgentInterface] = None

    @property
    def core(self) -> NeuralBlitzCoreInterface:
        """Access to core engine functionality"""
        if self._core_client is None:
            self._core_client = self._create_core_client()
        return self._core_client

    @property
    def lattice(self) -> OntologicalLatticeInterface:
        """Access to knowledge lattice"""
        if self._lattice_client is None:
            self._lattice_client = self._create_lattice_client()
        return self._lattice_client

    @property
    def assembler(self) -> GenesisAssemblerInterface:
        """Access to prompt generation"""
        if self._assembler_client is None:
            self._assembler_client = self._create_assembler_client()
        return self._assembler_client

    @property
    def agents(self) -> LRSAgentInterface:
        """Access to agent functionality"""
        if self._agent_client is None:
            self._agent_client = self._create_agent_client()
        return self._agent_client

    def _create_core_client(self) -> NeuralBlitzCoreInterface:
        """Create core engine client (implementation-specific)"""
        raise NotImplementedError("Core client implementation required")

    def _create_lattice_client(self) -> OntologicalLatticeInterface:
        """Create lattice client (implementation-specific)"""
        raise NotImplementedError("Lattice client implementation required")

    def _create_assembler_client(self) -> GenesisAssemblerInterface:
        """Create assembler client (implementation-specific)"""
        raise NotImplementedError("Assembler client implementation required")

    def _create_agent_client(self) -> LRSAgentInterface:
        """Create agent client (implementation-specific)"""
        raise NotImplementedError("Agent client implementation required")


# =============================================================================
# FACTORY FOR CREATING SDK INSTANCES
# =============================================================================


class NeuralBlitzSDKFactory:
    """Factory for creating NeuralBlitz SDK instances"""

    @staticmethod
    def create_client(api_key: str, environment: str = "production") -> NeuralBlitzSDK:
        """Create SDK client for specified environment"""
        base_urls = {
            "production": "https://api.neuralblitz.ai",
            "staging": "https://staging-api.neuralblitz.ai",
            "development": "http://localhost:8080",
        }

        base_url = base_urls.get(environment, base_urls["production"])
        return NeuralBlitzSDK(api_key, base_url)

    @staticmethod
    def create_local_client(api_key: str, port: int = 8080) -> NeuralBlitzSDK:
        """Create SDK client for local development"""
        base_url = f"http://localhost:{port}"
        return NeuralBlitzSDK(api_key, base_url)


# =============================================================================
# EXCEPTIONS
# =============================================================================


class NeuralBlitzError(Exception):
    """Base NeuralBlitz SDK exception"""

    pass


class AuthenticationError(NeuralBlitzError):
    """Authentication failed"""

    pass


class RateLimitError(NeuralBlitzError):
    """Rate limit exceeded"""

    pass


class ValidationError(NeuralBlitzError):
    """Input validation failed"""

    pass


class QuantumProcessingError(NeuralBlitzError):
    """Quantum processing failed"""

    pass


class AgentExecutionError(NeuralBlitzError):
    """Agent execution failed"""

    pass


class ConsciousnessError(NeuralBlitzError):
    """Consciousness system error"""

    pass


# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # Core interfaces
    "NeuralBlitzCoreInterface",
    "NeuralBlitzSDK",
    "NeuralBlitzSDKFactory",
    # EPA interfaces
    "OntologicalLatticeInterface",
    "GenesisAssemblerInterface",
    "SafetyValidatorInterface",
    "FeedbackEngineInterface",
    # Agent interfaces
    "LRSAgentInterface",
    "EmergentPromptAgentInterface",
    # Auth interfaces
    "AuthenticationInterface",
    # Enums
    "SystemMode",
    "APIKeyScope",
    "OntonType",
    # Data classes
    "QuantumNeuronConfig",
    "MultiRealityConfig",
    "ConsciousnessMetrics",
    "Onton",
    "PromptResult",
    "AgentConfig",
    "AgentTask",
    "AgentResult",
    "APIKey",
    # API models
    "QuantumProcessRequest",
    "QuantumProcessResponse",
    "AgentRunRequest",
    "AgentRunResponse",
    "EvolutionRequest",
    "EvolutionResponse",
    # Exceptions
    "NeuralBlitzError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "QuantumProcessingError",
    "AgentExecutionError",
    "ConsciousnessError",
]
