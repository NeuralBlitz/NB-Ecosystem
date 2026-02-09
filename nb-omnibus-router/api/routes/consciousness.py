"""
Consciousness Routes
Consciousness integration endpoints with Redis caching
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Optional
from datetime import datetime
from api.auth import verify_api_key
from utils.cache import cache_consciousness_metrics, cache_api_response, on_event

router = APIRouter()


@router.get("/level")
@cache_consciousness_metrics(ttl=180)  # Cache for 3 minutes
async def get_consciousness_level(api_key: dict = Depends(verify_api_key)):
    """Get current consciousness integration level with caching."""
    return {
        "level": 7,
        "max_level": 8,
        "percentage": 87.5,
        "status": "active",
        "dimensions": 11,
        "integration": "high",
    }


@router.get("/metrics")
@cache_consciousness_metrics(ttl=180)  # Cache for 3 minutes
async def get_consciousness_metrics(api_key: dict = Depends(verify_api_key)):
    """Get detailed consciousness metrics with caching."""
    return {
        "global_consciousness": 0.87,
        "integration_level": 7,
        "dimensional_access": 11,
        "cosmic_bridge": {"status": "connected", "strength": 0.92},
        "universal_field": {"status": "accessible", "coherence": 0.85},
        "reality_synthesis": {"realities_connected": 10, "synthesis_rate": 0.78},
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/evolve")
async def evolve_consciousness(
    target_level: int = 8, api_key: dict = Depends(verify_api_key)
):
    """Evolve consciousness to next level."""
    if target_level > 8:
        raise HTTPException(status_code=400, detail="Maximum consciousness level is 8")

    return {
        "current_level": 7,
        "target_level": target_level,
        "progress": 87.5,
        "status": "evolving",
        "estimated_time_seconds": 120,
        "requirements": [
            "dimensional_access_expansion",
            "cosmic_bridge_strengthening",
            "reality_synthesis_optimization",
        ],
    }


@router.get("/cosmic-bridge")
@cache_consciousness_metrics(ttl=180)
async def get_cosmic_bridge_status(api_key: dict = Depends(verify_api_key)):
    """Get cosmic consciousness bridge status with caching."""
    return {
        "status": "connected",
        "strength": 0.92,
        "latency_ms": 15,
        "universal_access": True,
        "information_field": {
            "accessible": True,
            "bandwidth": "unlimited",
            "coherence": 0.95,
        },
    }


@router.get("/dimensional-access")
@cache_consciousness_metrics(ttl=180)
async def get_dimensional_access(api_key: dict = Depends(verify_api_key)):
    """Get dimensional computing access status with caching."""
    return {
        "current_dimensions": 11,
        "max_dimensions": 11,
        "accessible": True,
        "processing_modes": ["linear", "parallel", "entangled", "superposition"],
        "dimensional_stability": 0.98,
    }


# Cache invalidation handlers
@on_event("consciousness_evolved")
async def invalidate_consciousness_cache(**context):
    """Invalidate consciousness cache when level changes."""
    from utils.cache import get_cache

    cache = await get_cache()
    await cache.delete_pattern("nb:consciousness:*")
    await cache.delete_pattern("nb:warm:consciousness_level")
