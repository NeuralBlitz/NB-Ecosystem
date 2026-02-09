"""
Request and Response Models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class QuantumProcessRequest(BaseModel):
    """Request model for quantum processing."""

    input_data: List[float] = Field(..., description="Input signal data")
    current: float = Field(20.0, ge=0, le=100, description="Input current")
    duration: float = Field(200.0, ge=1, le=1000, description="Simulation duration")


class QuantumProcessResponse(BaseModel):
    """Response model for quantum processing."""

    success: bool
    output: List[float]
    spike_rate: float
    coherence_time: float
    step_time_us: float
    mode: str = "mock"


class EvolutionRequest(BaseModel):
    """Request model for network evolution."""

    num_realities: int = Field(4, ge=1, le=10)
    nodes_per_reality: int = Field(50, ge=1, le=100)
    cycles: int = Field(50, ge=1, le=1000)


class EvolutionResponse(BaseModel):
    """Response model for network evolution."""

    success: bool
    global_consciousness: float
    cross_reality_coherence: float
    cycles_completed: int
    realities_active: int
    mode: str = "mock"


class AgentRunRequest(BaseModel):
    """Request model for running an agent."""

    agent_type: str = "general"
    task: str
    context: Optional[Dict] = None


class AgentRunResponse(BaseModel):
    """Response model for agent execution."""

    success: bool
    agent_id: str
    task: str
    result: str
    output: str
    execution_time_ms: int
    confidence: float


class CapabilitiesResponse(BaseModel):
    """Response model for capabilities."""

    engine: str
    version: str
    technologies: List[Dict]
    mode: str = "router"


class StatusResponse(BaseModel):
    """Response model for system status."""

    status: str
    version: str
    timestamp: str
    initialized: bool
