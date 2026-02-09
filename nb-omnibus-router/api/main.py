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
from utils.cache import (
    RedisCache,
    CacheConfig,
    CacheWarmer,
    get_cache,
    register_default_warming_tasks,
    cache_api_response,
)


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
        "REDIS": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "warmup_enabled": True,
        },
    }

    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
            defaults.update(user_config)

    return defaults


def load_cache_config(settings: dict) -> CacheConfig:
    """Load cache configuration from settings."""
    redis_config = settings.get("REDIS", {})

    return CacheConfig(
        host=redis_config.get("host", "localhost"),
        port=redis_config.get("port", 6379),
        db=redis_config.get("db", 0),
        password=redis_config.get("password"),
        max_connections=redis_config.get("max_connections", 50),
        socket_timeout=redis_config.get("socket_timeout", 5.0),
        socket_connect_timeout=redis_config.get("socket_connect_timeout", 5.0),
        retry_on_timeout=redis_config.get("retry_on_timeout", True),
        health_check_interval=redis_config.get("health_check_interval", 30),
        default_ttl=redis_config.get("default_ttl", 3600),
        quantum_state_ttl=redis_config.get("quantum_state_ttl", 300),
        multi_reality_ttl=redis_config.get("multi_reality_ttl", 600),
        consciousness_ttl=redis_config.get("consciousness_ttl", 180),
        api_response_ttl=redis_config.get("api_response_ttl", 900),
        warmup_enabled=redis_config.get("warmup_enabled", True),
        warmup_batch_size=redis_config.get("warmup_batch_size", 100),
        warmup_concurrency=redis_config.get("warmup_concurrency", 5),
        circuit_breaker_enabled=redis_config.get("circuit_breaker_enabled", True),
        circuit_breaker_threshold=redis_config.get("circuit_breaker_threshold", 5),
        circuit_breaker_timeout=redis_config.get("circuit_breaker_timeout", 60),
    )


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

    # Initialize cache
    cache_config = load_cache_config(settings)
    cache = RedisCache(cache_config)
    connected = await cache.connect()

    if connected:
        logger.info("Redis cache connected successfully")

        # Warm up cache if enabled
        if cache_config.warmup_enabled:
            logger.info("Starting cache warming...")
            warmer = CacheWarmer(cache)
            register_default_warming_tasks(warmer)
            warmup_results = await warmer.warm()
            logger.info(f"Cache warming completed: {warmup_results}")
    else:
        logger.warning("Redis cache unavailable, using local fallback")

    yield

    # Shutdown
    logger.info("Shutting down NeuralBlitz Omnibus Router...")
    await cache.disconnect()


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
@cache_api_response(ttl=3600)
async def get_capabilities(api_key: dict = Depends(verify_api_key)):
    """Get all available capabilities."""
    nb = NeuralBlitzCore()
    capabilities = await nb.get_capabilities()

    return {
        "capabilities": capabilities,
        "partner_tier": api_key.get("tier", "unknown"),
    }


@app.get("/cache/stats", tags=["System"])
async def get_cache_stats(api_key: dict = Depends(verify_api_key)):
    """Get cache statistics."""
    cache = await get_cache()
    return await cache.get_stats()


@app.post("/cache/clear", tags=["System"])
async def clear_cache(api_key: dict = Depends(verify_api_key)):
    """Clear all cache entries."""
    cache = await get_cache()
    success = await cache.clear()
    return {
        "status": "success" if success else "error",
        "message": "Cache cleared" if success else "Failed to clear cache",
    }


@app.get("/cache/health", tags=["System"])
async def get_cache_health():
    """Get cache health status."""
    cache = await get_cache()
    return await cache.health_check()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.get("HOST", "0.0.0.0"),
        port=settings.get("PORT", 8000),
        reload=settings.get("DEBUG", True),
    )
