"""
NeuralBlitz Core SDK
⚠️ INTERFACE DEFINITIONS ONLY - No implementation
⚠️ Actual processing done via SaaS API
"""

from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class QuantumNeuronConfig:
    """Configuration for quantum spiking neuron."""

    quantum_tunneling: float = 0.1
    coherence_time: float = 100.0
    membrane_time_constant: float = 10.0


class QuantumSpikingNeuron(ABC):
    """
    Quantum Spiking Neuron Interface.

    Represents a biologically-inspired neuron with quantum properties.
    Processing is performed via NeuralBlitz SaaS API.

    Example:
        >>> neuron = QuantumSpikingNeuron(config={
        ...     'quantum_tunneling': 0.1,
        ...     'coherence_time': 100.0
        ... })
        >>> result = neuron.process([0.1, 0.2, 0.3])
        >>> print(result['spikes'])
    """

    @abstractmethod
    def __init__(self, config: QuantumNeuronConfig):
        """
        Initialize quantum spiking neuron.

        Args:
            config: Neuron configuration parameters
        """
        pass

    @abstractmethod
    def process(self, input_data: np.ndarray) -> Dict:
        """
        Process input data through quantum neuron.

        Args:
            input_data: Input signal as numpy array

        Returns:
            Dict containing:
                - output: Processed output signal
                - spikes: Generated spike train
                - coherence: Quantum coherence value

        Note:
            This interface stub demonstrates usage.
            Actual processing requires API authentication.
        """
        pass


@dataclass
class MultiRealityConfig:
    """Configuration for multi-reality network."""

    num_realities: int = 4
    nodes_per_reality: int = 50
    reality_types: List[str] = field(default_factory=lambda: ["quantum", "classical", "hybrid"])


class MultiRealityNetwork(ABC):
    """
    Multi-Reality Neural Network Interface.

    Operates across multiple parallel computational realities.
    Processing is performed via NeuralBlitz SaaS API.

    Example:
        >>> network = MultiRealityNetwork(
        ...     num_realities=4,
        ...     nodes_per_reality=50
        ... })
        >>> metrics = network.evolve(cycles=50)
    """

    @abstractmethod
    def __init__(self, config: MultiRealityConfig):
        """Initialize multi-reality network."""
        pass

    @abstractmethod
    def evolve(self, cycles: int) -> Dict:
        """
        Evolve network across realities.

        Args:
            cycles: Number of evolution cycles

        Returns:
            Dict containing:
                - global_consciousness: System consciousness level
                - coherence: Cross-reality coherence
                - metrics: Evolution metrics
        """
        pass

    @abstractmethod
    def get_consciousness_level(self) -> float:
        """Get current consciousness integration level."""
        pass


class NeuralBlitzCore(ABC):
    """
    NeuralBlitz Core Engine Interface.

    Main entry point for NeuralBlitz capabilities.
    Processing is performed via NeuralBlitz SaaS API.

    Available Capabilities:
        - Quantum Spiking Neurons (10,705 ops/sec)
        - Multi-Reality Networks (2,710 cycles/sec)
        - Consciousness Integration (8 levels)
        - Cross-Reality Entanglement
        - Dimensional Computing (11 dimensions)

    Example:
        >>> from neuralblitz_core import NeuralBlitzCore
        >>> nb = NeuralBlitzCore(api_key="your-api-key")
        >>> capabilities = nb.get_capabilities()
    """

    @abstractmethod
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NeuralBlitz core.

        Args:
            api_key: NeuralBlitz API key for authentication
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict:
        """
        Get all available NeuralBlitz capabilities.

        Returns:
            Dict with capabilities information
        """
        pass

    @abstractmethod
    def process_quantum(
        self, input_data: List[float], current: float = 20.0, duration: float = 200.0
    ) -> Dict:
        """
        Process through quantum spiking neuron.

        Args:
            input_data: Input signal
            current: Input current (default: 20.0)
            duration: Simulation duration (default: 200.0)

        Returns:
            Dict with processing results
        """
        pass

    @abstractmethod
    def evolve_multi_reality(
        self, num_realities: int = 4, nodes_per_reality: int = 50, cycles: int = 50
    ) -> Dict:
        """
        Evolve multi-reality neural network.

        Args:
            num_realities: Number of parallel realities
            nodes_per_reality: Nodes in each reality
            cycles: Evolution cycles

        Returns:
            Dict with evolution results
        """
        pass


# Type definitions for common use cases
InputVector = List[float]
OutputVector = List[float]
QuantumState = Dict[str, float]
ConsciousnessLevel = float
RealityID = int
APICredentials = str


@dataclass
class QuantumProcessingResult:
    """Result from quantum processing."""
    spike_rate: float
    coherence: float
    qubits_active: int
    operations_per_second: float
    output: List[float]


@dataclass
class EvolutionResult:
    """Result from multi-reality evolution."""
    global_consciousness: float
    cross_reality_coherence: float
    dimensions: int
    reality_count: int
    metrics: Dict


@dataclass
class ConsciousnessMetrics:
    """Consciousness level and metrics."""
    level: int
    integration: float
    dimensions: int
    cosmic_bridge_status: str
    cosmic_bridge_strength: float


class QuantumEngineInterface(ABC):
    """Interface for quantum processing engine."""

    @abstractmethod
    def simulate(self, qubits: int, operations: int, coherence: float) -> Dict:
        pass

    @abstractmethod
    def entangle(self, qubits: int, strength: float) -> Dict:
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict:
        pass


class ConsciousnessInterface(ABC):
    """Interface for consciousness operations."""

    @abstractmethod
    def get_level(self) -> ConsciousnessMetrics:
        pass

    @abstractmethod
    def evolve(self, integration_level: float, dimensions: int) -> ConsciousnessMetrics:
        pass

    @abstractmethod
    def get_metrics(self) -> Dict:
        pass


class CrossRealityInterface(ABC):
    """Interface for cross-reality operations."""

    @abstractmethod
    def entangle_realities(self, realities: List[int], transfer_mode: str) -> Dict:
        pass

    @abstractmethod
    def transfer_reality(self, source_reality: int, target_reality: int) -> Dict:
        pass

    @abstractmethod
    def list_realities(self) -> List[Dict]:
        pass


__all__ = [
    "NeuralBlitzCore",
    "QuantumSpikingNeuron",
    "MultiRealityNetwork",
    "QuantumEngineInterface",
    "ConsciousnessInterface",
    "CrossRealityInterface",
    "QuantumNeuronConfig",
    "MultiRealityConfig",
    "QuantumProcessingResult",
    "EvolutionResult",
    "ConsciousnessMetrics",
]
