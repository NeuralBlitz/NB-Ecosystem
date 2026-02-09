"""
NeuralBlitz Agents SDK
⚠️ INTERFACE DEFINITIONS ONLY - No implementation
"""

from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for LRS agent."""

    agent_type: str = "general"
    memory_size: int = 1000
    learning_rate: float = 0.01
    autonomous_evolution: bool = True


@dataclass
class AgentTask:
    """Agent task definition."""

    task_id: str
    description: str
    priority: int = 0
    context: Optional[Dict] = None
    created_at: Optional[str] = None
    status: str = "pending"


@dataclass
class AgentResult:
    """Result from agent execution."""

    task_id: str
    success: bool
    output: Dict
    execution_time_ms: float
    consciousness_level: float


class LRSAgent(ABC):
    """
    Learning Record Store Agent Interface.

    Autonomous agents with learning and evolution capabilities.
    Execution is performed via NeuralBlitz SaaS API.

    Example:
        >>> agent = LRSAgent(config={
        ...     'agent_type': 'recognition',
        ...     'memory_size': 1000
        ... })
        >>> result = agent.run("Analyze this pattern")
    """

    @abstractmethod
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize LRS agent."""
        pass

    @abstractmethod
    def run(self, task: str, context: Optional[Dict] = None) -> Dict:
        """
        Execute agent on a task.

        Args:
            task: Task description
            context: Additional context

        Returns:
            Dict with execution results
        """
        pass

    @abstractmethod
    def learn(self, feedback: Dict) -> None:
        """
        Learn from feedback.

        Args:
            feedback: Learning feedback
        """
        pass


class EmergentPromptAgent(ABC):
    """
    Emergent Prompt Architecture Agent Interface.

    Uses prompt evolution for autonomous reasoning.
    Execution is performed via NeuralBlitz SaaS API.
    """

    @abstractmethod
    def __init__(self, prompts: List[str]):
        """Initialize EPA agent."""
        pass

    @abstractmethod
    def generate_response(self, query: str) -> str:
        """
        Generate response using evolved prompts.

        Args:
            query: User query

        Returns:
            Generated response
        """
        pass

    @abstractmethod
    def evolve_prompts(self) -> List[str]:
        """
        Evolve prompt set based on feedback.

        Returns:
            Evolved prompt set
        """
        pass
