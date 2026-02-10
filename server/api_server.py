"""
NeuralBlitz API Integration Layer
Production-ready REST API for DRS and HALIC components
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import asyncio
import uvicorn

# Import NeuralBlitz components
from drs_engine_enhanced import EnhancedDRSEngine
from halic_engine_enhanced import EnhancedHALICEngine


# Pydantic models for API
class ConceptRequest(BaseModel):
    id: str
    data: Dict[str, Any]
    connections: Optional[List[Dict[str, Any]]] = []


class InteractionRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = {}


class ConnectionRequest(BaseModel):
    start_concept: str
    end_concept: str


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)


class AuditSearchRequest(BaseModel):
    user_id: Optional[str] = None
    risk_level: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=200)


# API Application
app = FastAPI(
    title="NeuralBlitz API",
    description="Production API for NeuralBlitz DRS and HALIC components",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
drs_engine: EnhancedDRSEngine = None
halic_engine: EnhancedHALICEngine = None


@app.on_event("startup")
async def startup_event():
    """Initialize NeuralBlitz components."""
    global drs_engine, halic_engine

    # Database configuration
    db_config = {
        "host": "localhost",
        "port": 5432,
        "database": "neuralblitz",
        "user": "postgres",
        "password": "password",
        "pool_size": 10,
    }

    # Initialize engines
    drs_engine = EnhancedDRSEngine(db_config)
    halic_engine = EnhancedHALICEngine(db_config)

    await drs_engine.initialize()
    await halic_engine.initialize()

    print("NeuralBlitz API: All components initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources."""
    if drs_engine:
        await drs_engine.close()
    if halic_engine:
        await halic_engine.close()


# DRS Endpoints
@app.post("/api/v1/concepts", status_code=201)
async def create_concept(request: ConceptRequest):
    """Store a new concept in the DRS."""
    try:
        await drs_engine.store(request.id, request.data, request.connections)
        return {"message": f"Concept '{request.id}' stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/concepts/{concept_id}")
async def get_concept(concept_id: str):
    """Retrieve a concept by ID."""
    try:
        concept = await drs_engine.query(concept_id)
        if not concept:
            raise HTTPException(status_code=404, detail="Concept not found")
        return concept
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/concepts/connections")
async def find_connections(request: ConnectionRequest):
    """Find connections between two concepts."""
    try:
        path = await drs_engine.find_connections(
            request.start_concept, request.end_concept
        )
        return {"path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/concepts/search")
async def search_concepts(request: SearchRequest):
    """Search concepts by content."""
    try:
        results = await drs_engine.search(request.query, request.limit)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/concepts/{concept_id}/related")
async def get_related_concepts(concept_id: str, depth: int = 1):
    """Get concepts related to a given concept."""
    try:
        related = await drs_engine.get_related_concepts(concept_id, depth)
        return {"related_concepts": related}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# HALIC Endpoints
@app.post("/api/v1/interactions", status_code=201)
async def process_interaction(
    request: InteractionRequest, background_tasks: BackgroundTasks
):
    """Process an interaction with full audit trail."""
    try:
        result = await halic_engine.process_interaction(
            request.prompt, request.user_id, request.session_id, request.context
        )

        # Add background task for risk monitoring if needed
        if result["audit"]["risk_level"] in ["HIGH", "CRITICAL"]:
            background_tasks.add_task(
                monitor_high_risk_interaction, result["audit"]["trace_id"]
            )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/audit/{trace_id}")
async def get_audit_trail(trace_id: str):
    """Retrieve audit trail by trace ID."""
    try:
        trail = await halic_engine.get_audit_trail(trace_id)
        if not trail:
            raise HTTPException(status_code=404, detail="Audit trail not found")
        return trail
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/audit/search")
async def search_audit_trails(request: AuditSearchRequest):
    """Search audit trails with filters."""
    try:
        results = await halic_engine.search_audit_trails(
            request.user_id,
            request.risk_level,
            request.start_date,
            request.end_date,
            request.limit,
        )
        return {"audit_trails": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/audit/{trace_id}/verify")
async def verify_audit_trail(trace_id: str, golden_dag: str):
    """Verify audit trail integrity."""
    try:
        is_valid = await halic_engine.verify_audit_trail(trace_id, golden_dag)
        return {"trace_id": trace_id, "verified": is_valid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health and System Endpoints
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.get("/api/v1/stats")
async def get_system_stats():
    """Get system statistics."""
    try:
        # Get basic stats (implement based on your needs)
        stats = {
            "total_concepts": 0,  # Implement count query
            "total_interactions": 0,  # Implement count query
            "high_risk_incidents": 0,  # Implement count query
            "system_uptime": "0 days, 0 hours",
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background task for risk monitoring
async def monitor_high_risk_interaction(trace_id: str):
    """Background task to monitor high-risk interactions."""
    # Implement monitoring logic (alerts, notifications, etc.)
    print(f"Monitoring high-risk interaction: {trace_id}")


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
