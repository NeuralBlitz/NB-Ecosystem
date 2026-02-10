"""
NB Omnibus Server - Dedicated server module for NeuralBlitz Omnibus Router

This module provides a clean separation between the server infrastructure
and the API business logic.
"""

from server.application import create_application, OmnibusServer
from server.config import ServerConfig, load_server_config
from server.lifecycle import ServerLifecycle
from server.middleware import setup_middleware
from server.health import HealthManager, HealthStatus

__version__ = "1.0.0"
__all__ = [
    "create_application",
    "OmnibusServer",
    "ServerConfig",
    "load_server_config",
    "ServerLifecycle",
    "setup_middleware",
    "HealthManager",
    "HealthStatus",
]
