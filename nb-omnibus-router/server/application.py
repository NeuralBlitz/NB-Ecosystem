"""
Server application factory
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI

from server.config import ServerConfig
from server.lifecycle import ServerLifecycle
from server.health import HealthManager, create_default_health_manager
from server.middleware import setup_middleware

logger = logging.getLogger(__name__)


class OmnibusServer:
    """
    Dedicated server class for NB Omnibus Router

    Encapsulates all server functionality including:
    - Application lifecycle management
    - Health checks
    - Middleware
    - Configuration
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.lifecycle = ServerLifecycle(self.config)
        self.health = create_default_health_manager()
        self.app: Optional[FastAPI] = None
        self._routers: List = []

        # Setup default lifecycle tasks
        self.lifecycle.setup_default_tasks()

        logger.info("OmnibusServer initialized")

    def add_router(self, router, prefix: str = "", tags: Optional[List[str]] = None):
        """Add a router to the server"""
        self._routers.append({"router": router, "prefix": prefix, "tags": tags or []})

    def _create_lifespan(self):
        """Create the lifespan context manager"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.lifecycle.on_startup()

            # Set cache reference in health manager
            cache = self.lifecycle.get_cache()
            if cache:
                self.health.set_cache(cache)

            yield

            # Shutdown
            await self.lifecycle.on_shutdown()

        return lifespan

    def create_application(self) -> FastAPI:
        """
        Create and configure the FastAPI application

        Returns:
            Configured FastAPI application
        """
        # Create FastAPI app
        self.app = FastAPI(
            title="NeuralBlitz Omnibus Server",
            description="Dedicated server for NeuralBlitz AI Platform",
            version="1.0.0",
            docs_url=self.config.docs_url,
            redoc_url=self.config.redoc_url,
            openapi_url=self.config.openapi_url,
            lifespan=self._create_lifespan(),
        )

        # Setup middleware
        setup_middleware(self.app, self.config)

        # Include routers
        for router_config in self._routers:
            self.app.include_router(
                router_config["router"], prefix=router_config["prefix"], tags=router_config["tags"]
            )

        # Add system routes
        self._add_system_routes()

        logger.info("✓ FastAPI application created")
        return self.app

    def _add_system_routes(self):
        """Add system-level routes (health, metrics, etc.)"""
        from fastapi import Depends
        from datetime import datetime

        # Health check
        @self.app.get(self.config.health_check_path, tags=["System"])
        async def health_check():
            """Basic health check endpoint"""
            report = await self.health.run_all_checks()
            return {
                "status": report.status.value,
                "version": report.version,
                "timestamp": report.timestamp.isoformat(),
                "uptime_seconds": report.uptime_seconds,
            }

        # Readiness probe
        @self.app.get(self.config.readiness_path, tags=["System"])
        async def readiness_check():
            """Readiness probe for orchestrators"""
            is_ready = await self.health.check_readiness()
            status_code = 200 if is_ready else 503
            return {
                "ready": is_ready,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Detailed health check
        @self.app.get(f"{self.config.health_check_path}/detailed", tags=["System"])
        async def detailed_health_check():
            """Detailed health check with all components"""
            report = await self.health.run_all_checks()

            return {
                "status": report.status.value,
                "version": report.version,
                "timestamp": report.timestamp.isoformat(),
                "uptime_seconds": report.uptime_seconds,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "message": check.message,
                        "response_time_ms": check.response_time_ms,
                        "metadata": check.metadata,
                    }
                    for check in report.checks
                ],
            }

        # Root endpoint
        @self.app.get("/", tags=["System"])
        async def root():
            """Root endpoint with API information"""
            return {
                "name": "NeuralBlitz Omnibus Server",
                "version": "1.0.0",
                "description": "Dedicated server for NeuralBlitz AI Platform",
                "docs": self.config.docs_url,
                "health": self.config.health_check_path,
            }

        logger.info("✓ System routes added")


def create_application(
    config: Optional[ServerConfig] = None, routers: Optional[List] = None
) -> FastAPI:
    """
    Factory function to create a fully configured FastAPI application

    Args:
        config: Server configuration (uses defaults if not provided)
        routers: List of FastAPI routers to include

    Returns:
        Configured FastAPI application
    """
    server = OmnibusServer(config)

    # Add routers if provided
    if routers:
        for router_config in routers:
            if isinstance(router_config, dict):
                server.add_router(
                    router_config["router"],
                    prefix=router_config.get("prefix", ""),
                    tags=router_config.get("tags"),
                )
            else:
                server.add_router(router_config)

    return server.create_application()
