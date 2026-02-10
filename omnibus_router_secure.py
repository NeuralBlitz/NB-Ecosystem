"""
NeuralBlitz Omnibus Router - Enhanced Security Version
=======================================================

This is the production-ready API gateway that provides secure access
to NeuralBlitz capabilities while protecting sensitive internal systems.

Security Features:
- API key authentication with scope-based access
- Rate limiting and circuit breaker patterns
- Request validation and sanitization
- Comprehensive audit logging
- Isolated internal systems

Last Updated: 2026-02-09
Security Version: 2.0
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import time
import hashlib
import hmac

from api.routes import (
    core,
    agents,
    quantum,
    consciousness,
    entanglement,
    ui,
    monitoring,
    websocket,
)
from api.auth_enhanced import EnhancedAuth, RateLimiter, AuditLogger
from utils.cache import SecureRedisCache, CacheConfig, get_cache
from monitoring.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry


# Security configuration
class SecurityConfig:
    """Security configuration settings"""

    # API key settings
    MIN_KEY_LENGTH = 32
    MAX_KEY_LENGTH = 128
    KEY_ROTATION_DAYS = 90

    # Rate limiting
    DEFAULT_RATE_LIMIT = 1000  # requests per hour
    BURST_LIMIT = 100  # requests per minute

    # Request validation
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_HOSTS = ["localhost", "127.0.0.1", "*.neuralblitz.ai"]

    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
    }


class OmnibusRouter:
    """Enhanced NeuralBlitz Omnibus Router with security features"""

    def __init__(self):
        self.settings = self._load_settings()
        self.auth = EnhancedAuth(self.settings)
        self.rate_limiter = RateLimiter(self.settings)
        self.audit_logger = AuditLogger(self.settings)
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.logger = self._setup_logging()

    def _load_settings(self) -> dict:
        """Load and validate application settings"""
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

        defaults = {
            "ENVIRONMENT": "production",
            "DEBUG": False,
            "HOST": "0.0.0.0",
            "PORT": 8443,  # HTTPS port
            "ALLOWED_ORIGINS": ["https://app.neuralblitz.ai"],
            "LOG_LEVEL": "INFO",
            "SECURITY": {
                "require_https": True,
                "api_key_rotation_days": SecurityConfig.KEY_ROTATION_DAYS,
                "max_login_attempts": 5,
                "lockout_duration_minutes": 15,
            },
            "REDIS": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None,  # Loaded from env in production
                "ssl": True,
            },
            "MONITORING": {
                "enable_tracing": True,
                "enable_metrics": True,
                "sample_rate": 0.1,
            },
        }

        if config_path.exists():
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f) or {}
                defaults.update(user_config)

        # Validate critical settings
        self._validate_settings(defaults)

        return defaults

    def _validate_settings(self, settings: dict) -> None:
        """Validate critical security settings"""
        if settings.get("ENVIRONMENT") == "production":
            required_settings = [
                "REDIS.password",
                "SECURITY.api_key_rotation_days",
                "ALLOWED_ORIGINS",
            ]

            for setting in required_settings:
                if not self._get_nested_setting(settings, setting):
                    raise ValueError(f"Required setting '{setting}' missing for production")

    def _get_nested_setting(self, settings: dict, path: str) -> Optional[str]:
        """Get nested setting by dot path"""
        keys = path.split(".")
        value = settings

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _setup_logging(self) -> logging.Logger:
        """Setup secure logging configuration"""
        logger = logging.getLogger("neuralblitz_omnibus")

        # Prevent log injection
        class SecureFormatter(logging.Formatter):
            def format(self, record):
                # Sanitize log messages to prevent injection
                message = super().format(record)
                return message.replace("\n", "\\n").replace("\r", "\\r")

        handler = logging.StreamHandler()
        handler.setFormatter(
            SecureFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.settings.get("LOG_LEVEL", "INFO")))

        return logger

    def create_app(self) -> FastAPI:
        """Create FastAPI application with security middleware"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan with security initialization"""
            self.logger.info("Starting NeuralBlitz Omnibus Router v2.0...")
            self.logger.info(f"Environment: {self.settings.get('ENVIRONMENT')}")

            # Initialize secure cache
            cache_config = self._load_cache_config()
            cache = SecureRedisCache(cache_config)
            connected = await cache.connect()

            if not connected:
                self.logger.critical("Failed to connect to secure cache - aborting startup")
                raise RuntimeError("Secure cache connection failed")

            self.logger.info("Secure cache connected successfully")

            # Initialize circuit breakers
            self._initialize_circuit_breakers()

            # Start audit logging
            await self.audit_logger.start()

            yield

            # Secure shutdown
            self.logger.info("Shutting down NeuralBlitz Omnibus Router...")
            await self.audit_logger.stop()
            await cache.disconnect()

        # Create FastAPI app
        app = FastAPI(
            title="NeuralBlitz Omnibus Router",
            description="NeuralBlitz AI Platform - Secure SaaS API Gateway v2.0",
            version="2.0.0",
            lifespan=lifespan,
            docs_url="/docs" if self.settings.get("DEBUG") else None,
            redoc_url="/redoc" if self.settings.get("DEBUG") else None,
        )

        # Add security middleware
        self._add_security_middleware(app)

        # Add rate limiting middleware
        self._add_rate_limiting_middleware(app)

        # Add audit logging middleware
        self._add_audit_middleware(app)

        # Include secure routers
        self._include_routers(app)

        # Add secure endpoints
        self._add_secure_endpoints(app)

        # Add exception handlers
        self._add_exception_handlers(app)

        return app

    def _add_security_middleware(self, app: FastAPI) -> None:
        """Add security middleware"""
        # Trusted host middleware
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=SecurityConfig.ALLOWED_HOSTS)

        # CORS middleware (restrictive in production)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.settings.get("ALLOWED_ORIGINS", []),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

        # Security headers middleware
        @app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)

            for header, value in SecurityConfig.SECURITY_HEADERS.items():
                response.headers[header] = value

            return response

    def _add_rate_limiting_middleware(self, app: FastAPI) -> None:
        """Add rate limiting middleware"""

        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Extract API key from header
            api_key = request.headers.get("X-API-Key") or request.headers.get(
                "Authorization", ""
            ).replace("Bearer ", "")

            if api_key:
                # Check rate limits
                if not await self.rate_limiter.check_limit(api_key, request.url.path):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={"Retry-After": "3600"},
                    )

            response = await call_next(request)
            return response

    def _add_audit_middleware(self, app: FastAPI) -> None:
        """Add audit logging middleware"""

        @app.middleware("http")
        async def audit_middleware(request: Request, call_next):
            start_time = time.time()

            # Log request
            await self.audit_logger.log_request(
                method=request.method,
                path=str(request.url.path),
                headers=dict(request.headers),
                client_ip=request.client.host if request.client else None,
            )

            try:
                response = await call_next(request)

                # Log response
                await self.audit_logger.log_response(
                    status_code=response.status_code, duration=time.time() - start_time
                )

                return response

            except Exception as e:
                # Log error
                await self.audit_logger.log_error(error=str(e), duration=time.time() - start_time)
                raise

    def _include_routers(self, app: FastAPI) -> None:
        """Include secure API routers"""
        routers = [
            (core.router, "/api/v2/core", "Core"),
            (agents.router, "/api/v2/agents", "Agents"),
            (quantum.router, "/api/v2/quantum", "Quantum"),
            (consciousness.router, "/api/v2/consciousness", "Consciousness"),
            (entanglement.router, "/api/v2/entanglement", "Cross-Reality"),
            (ui.router, "/api/v2/ui", "UI"),
            (monitoring.router, "/api/v2/monitoring", "Monitoring"),
            (websocket.router, "/api/v2/ws", "WebSocket"),
        ]

        for router, prefix, tag in routers:
            app.include_router(
                router, prefix=prefix, tags=[tag], dependencies=[Depends(self.auth.verify_api_key)]
            )

    def _add_secure_endpoints(self, app: FastAPI) -> None:
        """Add secure system endpoints"""

        @app.get("/health", tags=["System"])
        async def health_check():
            """System health check (public)"""
            return {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "security": "enabled",
            }

        @app.get("/api/v2/capabilities", tags=["System"])
        async def get_capabilities(user_data: dict = Depends(self.auth.verify_api_key)):
            """Get available capabilities (authenticated)"""
            # Filter capabilities based on user permissions
            filtered_capabilities = await self._filter_capabilities(user_data)

            return {
                "capabilities": filtered_capabilities,
                "partner_tier": user_data.get("tier", "unknown"),
                "rate_limit": user_data.get("rate_limit", SecurityConfig.DEFAULT_RATE_LIMIT),
            }

        @app.post("/api/v2/auth/keys", tags=["Authentication"])
        async def create_api_key(
            request_data: dict, user_data: dict = Depends(self.auth.verify_admin_key)
        ):
            """Create new API key (admin only)"""
            return await self.auth.create_api_key(
                user_id=user_data["user_id"],
                scopes=request_data.get("scopes", []),
                tier=request_data.get("tier", "basic"),
            )

        @app.delete("/api/v2/auth/keys/{key_id}", tags=["Authentication"])
        async def revoke_api_key(
            key_id: str, user_data: dict = Depends(self.auth.verify_admin_key)
        ):
            """Revoke API key (admin only)"""
            success = await self.auth.revoke_api_key(key_id, user_data["user_id"])
            return {"status": "success" if success else "error"}

    def _add_exception_handlers(self, app: FastAPI) -> None:
        """Add secure exception handlers"""

        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            # Log security exceptions
            if exc.status_code in [401, 403, 429]:
                await self.audit_logger.log_security_event(
                    event_type="http_exception",
                    details={
                        "status_code": exc.status_code,
                        "path": str(request.url.path),
                        "client_ip": request.client.host if request.client else None,
                    },
                )

            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            # Log unexpected errors
            self.logger.error(f"Unexpected error: {exc}", exc_info=True)
            await self.audit_logger.log_error(error=str(exc), path=str(request.url.path))

            # Don't expose internal errors in production
            if self.settings.get("ENVIRONMENT") == "production":
                return JSONResponse(status_code=500, content={"error": "Internal server error"})
            else:
                return JSONResponse(status_code=500, content={"error": str(exc)})

    def _load_cache_config(self) -> CacheConfig:
        """Load secure cache configuration"""
        redis_config = self.settings.get("REDIS", {})

        return CacheConfig(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password"),
            ssl=redis_config.get("ssl", True),
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
            warmup_enabled=redis_config.get("warmup_enabled", False),  # Disabled for security
            circuit_breaker_enabled=redis_config.get("circuit_breaker_enabled", True),
            circuit_breaker_threshold=redis_config.get("circuit_breaker_threshold", 5),
            circuit_breaker_timeout=redis_config.get("circuit_breaker_timeout", 60),
        )

    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for critical services"""
        critical_services = [
            "core.quantum_processing",
            "core.consciousness_integration",
            "agents.execution",
            "cache.redis",
        ]

        for service in critical_services:
            self.circuit_breaker_registry.register(
                name=service, failure_threshold=5, timeout=60.0, expected_exception=Exception
            )

    async def _filter_capabilities(self, user_data: dict) -> dict:
        """Filter capabilities based on user permissions"""
        # This would integrate with the actual capability system
        # For now, return basic capability structure

        tier = user_data.get("tier", "basic")

        capabilities = {
            "basic": {
                "core": ["get_capabilities", "health_check"],
                "quantum": ["process_basic"],
                "agents": ["run_basic_agent"],
            },
            "professional": {
                "core": ["get_capabilities", "health_check", "system_status"],
                "quantum": ["process_basic", "process_advanced"],
                "agents": ["run_basic_agent", "run_advanced_agent"],
                "consciousness": ["get_metrics"],
            },
            "enterprise": {
                "core": ["get_capabilities", "health_check", "system_status", "admin"],
                "quantum": ["process_basic", "process_advanced", "process_quantum"],
                "agents": ["run_basic_agent", "run_advanced_agent", "train_agent"],
                "consciousness": ["get_metrics", "set_mode"],
                "entanglement": ["evolve_reality"],
            },
        }

        return capabilities.get(tier, capabilities["basic"])


# Factory function
def create_omnibus_router() -> FastAPI:
    """Create enhanced Omnibus Router instance"""
    router = OmnibusRouter()
    return router.create_app()


# Main application
if __name__ == "__main__":
    import uvicorn

    router = OmnibusRouter()
    app = router.create_app()

    uvicorn.run(
        app,
        host=router.settings.get("HOST", "0.0.0.0"),
        port=router.settings.get("PORT", 8443),
        ssl_keyfile="server.key",  # Required for HTTPS
        ssl_certfile="server.crt",  # Required for HTTPS
        log_level=router.settings.get("LOG_LEVEL", "info").lower(),
    )
