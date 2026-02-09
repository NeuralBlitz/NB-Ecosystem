"""
Quantum Routes
Quantum simulation endpoints with Redis caching
"""

from fastapi import APIRouter, Depends
from api.auth import verify_api_key
from engines.quantum import QuantumEngine, QuantumConfig
from utils.cache import cache_quantum_state, cache_api_response

router = APIRouter()

# Cache for quantum instances
_quantum_cache = {}


async def get_quantum(api_key: dict = Depends(verify_api_key)) -> QuantumEngine:
    """Get or create quantum instance."""
    partner_id = api_key["partner_id"]

    if partner_id not in _quantum_cache:
        config = QuantumConfig(qubits=api_key.get("qubits", 4))
        _quantum_cache[partner_id] = QuantumEngine(config=config)

    return _quantum_cache[partner_id]


@router.post("/simulate")
@cache_quantum_state(ttl=300)  # Cache for 5 minutes
async def simulate_quantum(
    qubits: int = 4,
    circuit_depth: int = 3,
    quantum: QuantumEngine = Depends(get_quantum),
    api_key: dict = Depends(verify_api_key),
):
    """Run quantum simulation with caching."""
    result = await quantum.simulate(qubits=qubits, circuit_depth=circuit_depth)
    return {"success": True, **result}


@router.post("/entangle")
@cache_quantum_state(ttl=300)  # Cache for 5 minutes
async def create_entanglement(
    num_pairs: int = 2,
    quantum: QuantumEngine = Depends(get_quantum),
    api_key: dict = Depends(verify_api_key),
):
    """Create entangled qubit pairs with caching."""
    result = await quantum.entangle(num_pairs=num_pairs)
    return {"success": True, **result}


@router.get("/capabilities")
@cache_api_response(ttl=3600)  # Cache for 1 hour
async def get_quantum_capabilities(
    quantum: QuantumEngine = Depends(get_quantum),
    api_key: dict = Depends(verify_api_key),
):
    """Get quantum capabilities with caching."""
    capabilities = await quantum.get_capabilities()
    return capabilities
