"""
NB Omnibus Server - Main Entry Point

Dedicated server for NeuralBlitz AI Platform.
This module provides the main entry point for running the server.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import create_application, load_server_config, ServerConfig


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)

    # Reduce noise from external libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="NB Omnibus Server - NeuralBlitz AI Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m server.main
  
  # Run with custom config
  python -m server.main --config config/production.yaml
  
  # Run in production mode
  python -m server.main --host 0.0.0.0 --port 8000 --env production
  
  # Run with specific log level
  python -m server.main --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("NB_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("NB_PORT", "8000")),
        help="Port to bind to (default: 8000)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=os.getenv("NB_ENVIRONMENT", "development"),
        choices=["development", "staging", "production"],
        help="Environment (default: development)",
    )

    parser.add_argument(
        "--config", type=str, default=os.getenv("NB_CONFIG_FILE"), help="Path to configuration file"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("NB_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default: INFO)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("NB_RELOAD", "false").lower() == "true",
        help="Enable auto-reload (development only)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("NB_WORKERS", "1")),
        help="Number of worker processes (production only)",
    )

    parser.add_argument("--no-cache", action="store_true", help="Disable Redis cache")

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ServerConfig:
    """Create server configuration from command line arguments"""

    # Load base configuration
    if args.config:
        config = load_server_config(args.config)
    else:
        config = load_server_config()

    # Override with CLI arguments
    config.host = args.host
    config.port = args.port
    config.environment = args.env
    config.log_level = args.log_level
    config.debug = args.env == "development"

    if args.no_cache:
        config.cache_enabled = False

    return config


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()

    # Setup logging early
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create configuration
    config = create_config_from_args(args)

    logger.info("=" * 60)
    logger.info("NB Omnibus Server")
    logger.info("=" * 60)
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Host: {config.host}:{config.port}")
    logger.info(f"Debug: {config.debug}")
    logger.info(f"Cache: {'enabled' if config.cache_enabled else 'disabled'}")
    logger.info("=" * 60)

    # Import and register API routes
    try:
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

        routers = [
            {"router": core.router, "prefix": "/api/v1/core", "tags": ["Core"]},
            {"router": agents.router, "prefix": "/api/v1/agent", "tags": ["Agents"]},
            {"router": agents_full.router, "prefix": "/api/v1/agents", "tags": ["Advanced Agents"]},
            {"router": quantum.router, "prefix": "/api/v1/quantum", "tags": ["Quantum"]},
            {
                "router": consciousness.router,
                "prefix": "/api/v1/consciousness",
                "tags": ["Consciousness"],
            },
            {
                "router": entanglement.router,
                "prefix": "/api/v1/entanglement",
                "tags": ["Cross-Reality"],
            },
            {"router": ui.router, "prefix": "/api/v1/ui", "tags": ["UI"]},
            {"router": monitoring.router, "tags": ["Monitoring"]},
            {"router": websocket.router, "prefix": "/api/v1/ws", "tags": ["WebSocket"]},
        ]

        logger.info(f"Registered {len(routers)} API route modules")

    except ImportError as e:
        logger.warning(f"Could not import API routes: {e}")
        logger.warning("Starting server without API routes")
        routers = []

    # Create application
    app = create_application(config=config, routers=routers)

    # Run server
    import uvicorn

    uvicorn_config = {
        "host": config.host,
        "port": config.port,
        "log_level": config.log_level.lower(),
        "reload": args.reload if config.debug else False,
    }

    # Only use workers in production
    if config.environment == "production" and args.workers > 1:
        uvicorn_config["workers"] = args.workers
        logger.info(f"Running with {args.workers} workers")

    logger.info("Starting server...")
    logger.info(f"API documentation: http://{config.host}:{config.port}{config.docs_url}")
    logger.info(f"Health check: http://{config.host}:{config.port}{config.health_check_path}")

    try:
        uvicorn.run(app, **uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
