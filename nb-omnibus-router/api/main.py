"""
Omnibus Router - NeuralBlitz SaaS API
FastAPI entry point
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import logging
import yaml
from pathlib import Path

from api.routes import (
    core,
    agents,
    quantum,
    consciousness,
    entanglement,
    agents_full,
    ui,
    monitoring,
    websocket,
)
from api.auth import verify_api_key
from engines.neuralblitz import NeuralBlitzCore


# Load configuration
def load_settings():
    """Load application settings."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

    defaults = {
        "ENVIRONMENT": "development",
        "DEBUG": True,
        "HOST": "0.0.0.0",
        "PORT": 8000,
        "ALLOWED_ORIGINS": ["*"],
        "LOG_LEVEL": "INFO",
    }

    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
            defaults.update(user_config)

    return defaults


settings = load_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting NeuralBlitz Omnibus Router...")
    logger.info(f"Environment: {settings.get('ENVIRONMENT', 'unknown')}")
    yield
    logger.info("Shutting down NeuralBlitz Omnibus Router...")


# Create FastAPI application
app = FastAPI(
    title="NeuralBlitz Omnibus Router",
    description="NeuralBlitz AI Platform - SaaS API Gateway",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("ALLOWED_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(core.router, prefix="/api/v1/core", tags=["Core"])
app.include_router(agents.router, prefix="/api/v1/agent", tags=["Agents"])
app.include_router(
    agents_full.router, prefix="/api/v1/agents", tags=["Advanced Agents"]
)
app.include_router(quantum.router, prefix="/api/v1/quantum", tags=["Quantum"])
app.include_router(
    consciousness.router, prefix="/api/v1/consciousness", tags=["Consciousness"]
)
app.include_router(
    entanglement.router, prefix="/api/v1/entanglement", tags=["Cross-Reality"]
)
app.include_router(ui.router, prefix="/api/v1/ui", tags=["UI"])
app.include_router(monitoring.router, tags=["Monitoring"])
app.include_router(websocket.router, prefix="/api/v1/ws", tags=["WebSocket"])


# Health check endpoints
@app.get("/health", tags=["System"])
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/", tags=["System"])
async def root():
    """Root redirect to docs."""
    return {
        "message": "NeuralBlitz Omnibus Router",
        "docs": "/docs",
        "version": "1.0.0",
    }


@app.get("/api/v1/capabilities", tags=["System"])
async def get_capabilities(api_key: dict = Depends(verify_api_key)):
    """Get all available capabilities."""
    nb = NeuralBlitzCore()
    capabilities = await nb.get_capabilities()

    return {
        "capabilities": capabilities,
        "partner_tier": api_key.get("tier", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.get("HOST", "0.0.0.0"),
        port=settings.get("PORT", 8000),
        reload=settings.get("DEBUG", True),
    )
