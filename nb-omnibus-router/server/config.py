"""
Server configuration management
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ServerConfig:
    """Server configuration container"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"

    # CORS settings
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allowed_methods: List[str] = field(default_factory=lambda: ["*"])
    allowed_headers: List[str] = field(default_factory=lambda: ["*"])

    # API settings
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # Security settings
    rate_limit_enabled: bool = True
    auth_enabled: bool = True
    api_key_header: str = "X-API-Key"

    # Performance settings
    max_workers: int = 4
    request_timeout: int = 30
    keep_alive_timeout: int = 5

    # Cache settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    cache_enabled: bool = True
    cache_ttl: int = 3600

    # Monitoring settings
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    health_check_path: str = "/health"
    readiness_path: str = "/ready"

    # Partner configuration
    partners_file: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ServerConfig":
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ServerConfig":
        """Load config from YAML file"""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        return cls.from_dict(config_dict)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load config from environment variables"""
        config = cls()

        # Server settings
        if os.getenv("NB_HOST"):
            config.host = os.getenv("NB_HOST")
        if os.getenv("NB_PORT"):
            config.port = int(os.getenv("NB_PORT"))
        if os.getenv("NB_DEBUG"):
            config.debug = os.getenv("NB_DEBUG").lower() == "true"
        if os.getenv("NB_ENVIRONMENT"):
            config.environment = os.getenv("NB_ENVIRONMENT")
        if os.getenv("NB_LOG_LEVEL"):
            config.log_level = os.getenv("NB_LOG_LEVEL")

        # Redis settings
        if os.getenv("REDIS_HOST"):
            config.redis_host = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            config.redis_port = int(os.getenv("REDIS_PORT"))
        if os.getenv("REDIS_DB"):
            config.redis_db = int(os.getenv("REDIS_DB"))
        if os.getenv("REDIS_PASSWORD"):
            config.redis_password = os.getenv("REDIS_PASSWORD")

        # Partners file
        if os.getenv("NB_PARTNERS_FILE"):
            config.partners_file = os.getenv("NB_PARTNERS_FILE")

        return config


def load_server_config(config_path: Optional[str] = None) -> ServerConfig:
    """
    Load server configuration from multiple sources in order of precedence:
    1. Environment variables (highest priority)
    2. YAML config file
    3. Default values (lowest priority)
    """
    # Start with defaults
    config = ServerConfig()

    # Load from YAML if provided or found at default location
    if config_path:
        config = ServerConfig.from_yaml(config_path)
    else:
        default_paths = [
            Path("config/settings.yaml"),
            Path("/config/settings.yaml"),
            Path.home() / ".neuralblitz" / "settings.yaml",
        ]
        for path in default_paths:
            if path.exists():
                config = ServerConfig.from_yaml(str(path))
                break

    # Override with environment variables
    env_config = ServerConfig.from_env()

    # Merge environment config (only set values)
    for field_name in ServerConfig.__dataclass_fields__:
        env_value = getattr(env_config, field_name)
        default_value = getattr(ServerConfig(), field_name)
        if env_value != default_value:
            setattr(config, field_name, env_value)

    return config
