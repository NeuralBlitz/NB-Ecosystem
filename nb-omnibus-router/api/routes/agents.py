"""
Agent Routes
LRS Agent endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime

from api.auth import verify_api_key
from engines.agents import LRSAgents, AgentConfig
from api.models import AgentRunRequest, AgentRunResponse

router = APIRouter()

# Cache for agent instances
_agent_cache = {}


async def get_agents(api_key: dict = Depends(verify_api_key)) -> LRSAgents:
    """Get or create agents instance."""
    partner_id = api_key["partner_id"]

    if partner_id not in _agent_cache:
        config = AgentConfig(
            agent_type="general", memory_size=api_key.get("memory_size", 1000)
        )
        _agent_cache[partner_id] = LRSAgents(config=config)

    return _agent_cache[partner_id]


@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    agents: LRSAgents = Depends(get_agents),
    api_key: dict = Depends(verify_api_key),
):
    """Run an LRS agent on a task."""
    result = await agents.run_agent(
        agent_id=request.agent_type, task=request.task, context=request.context
    )

    return AgentRunResponse(
        success=True,
        agent_id=result["agent_id"],
        task=result["task"],
        result=result["result"],
        output=result["output"],
        execution_time_ms=result["execution_time_ms"],
        confidence=result["confidence"],
    )


@router.get("/list")
async def list_agents(
    agents: LRSAgents = Depends(get_agents), api_key: dict = Depends(verify_api_key)
):
    """List all available agents."""
    agents_list = await agents.get_agents()
    return {"agents": agents_list, "count": len(agents_list)}


@router.get("/capabilities")
async def get_agent_capabilities(
    agents: LRSAgents = Depends(get_agents), api_key: dict = Depends(verify_api_key)
):
    """Get agent capabilities."""
    capabilities = await agents.get_capabilities()
    return capabilities


@router.post("/create")
async def create_agent(
    name: str,
    agent_type: str = "general",
    agents: LRSAgents = Depends(get_agents),
    api_key: dict = Depends(verify_api_key),
):
    """Create a new agent."""
    result = await agents.create_agent(name=name, agent_type=agent_type)
    return result
