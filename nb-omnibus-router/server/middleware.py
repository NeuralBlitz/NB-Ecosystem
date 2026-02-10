"""
Server middleware setup
"""

import time
import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from server.config import ServerConfig

logger = logging.getLogger(__name__)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request timing"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        # Log slow requests
        if process_time > 1.0:  # More than 1 second
            logger.warning(
                f"Slow request: {request.method} {request.url.path} took {process_time:.3f}s"
            )

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests"""

    def __init__(self, app: ASGIApp, log_level: str = "INFO"):
        super().__init__(app)
        self.log_level = log_level

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(f"→ {request.method} {request.url.path}")

        try:
            response = await call_next(request)

            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"← {request.method} {request.url.path} "
                f"{response.status_code} ({process_time:.3f}s)"
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"✗ {request.method} {request.url.path} ERROR: {e} ({process_time:.3f}s)")
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Basic rate limiting middleware"""

    def __init__(self, app: ASGIApp, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = request.headers.get(
            "X-API-Key", request.client.host if request.client else "unknown"
        )

        # Simple rate limiting logic
        current_time = time.time()

        if client_id in self.requests:
            request_times = self.requests[client_id]
            # Remove old requests (> 1 minute)
            self.requests[client_id] = [t for t in request_times if current_time - t < 60]

            if len(self.requests[client_id]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for client: {client_id}")
                from fastapi.responses import JSONResponse

                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

            self.requests[client_id].append(current_time)
        else:
            self.requests[client_id] = [current_time]

        return await call_next(request)


def setup_middleware(app: FastAPI, config: ServerConfig):
    """
    Setup all middleware for the application

    Order matters - middleware is executed in reverse order of addition
    (last added = first executed)
    """

    # 1. Security headers (outermost - applied last)
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # 3. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=config.allow_credentials,
        allow_methods=config.allowed_methods,
        allow_headers=config.allowed_headers,
    )

    # 4. Request timing
    app.add_middleware(RequestTimingMiddleware)

    # 5. Rate limiting (if enabled)
    if config.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

    # 6. Request logging (innermost - first to execute)
    app.add_middleware(RequestLoggingMiddleware, log_level=config.log_level)

    logger.info("✓ Middleware setup completed")
