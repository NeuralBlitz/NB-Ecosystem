"""
NB Omnibus Server CLI Commands
"""

import click
import logging
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from server.main import main as server_main
from server import load_server_config, OmnibusServer

logger = logging.getLogger(__name__)


@click.group(name="server")
def server_cli():
    """NB Omnibus Server management commands"""
    pass


@server_cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", type=int, default=8000, help="Port to bind to")
@click.option(
    "--env",
    "-e",
    type=click.Choice(["development", "staging", "production"]),
    default="development",
    help="Environment",
)
@click.option("--config", "-c", help="Path to configuration file")
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Log level",
)
@click.option("--reload/--no-reload", default=None, help="Enable auto-reload")
@click.option("--workers", "-w", type=int, default=1, help="Number of workers")
@click.option("--no-cache", is_flag=True, help="Disable cache")
def start(host, port, env, config, log_level, reload, workers, no_cache):
    """Start the NB Omnibus Server"""
    import os

    # Set environment variables for the server
    os.environ["NB_HOST"] = host
    os.environ["NB_PORT"] = str(port)
    os.environ["NB_ENVIRONMENT"] = env
    os.environ["NB_LOG_LEVEL"] = log_level

    if config:
        os.environ["NB_CONFIG_FILE"] = config
    if reload is not None:
        os.environ["NB_RELOAD"] = str(reload).lower()
    if workers:
        os.environ["NB_WORKERS"] = str(workers)
    if no_cache:
        os.environ["NB_CACHE_ENABLED"] = "false"

    # Run the server
    server_main()


@server_cli.command()
@click.option("--config", "-c", help="Path to configuration file")
def config(config):
    """Display current server configuration"""
    cfg = load_server_config(config)

    click.echo("\n" + "=" * 60)
    click.echo("NB Omnibus Server Configuration")
    click.echo("=" * 60)

    click.echo(f"\nServer Settings:")
    click.echo(f"  Host: {cfg.host}")
    click.echo(f"  Port: {cfg.port}")
    click.echo(f"  Environment: {cfg.environment}")
    click.echo(f"  Debug: {cfg.debug}")
    click.echo(f"  Log Level: {cfg.log_level}")

    click.echo(f"\nAPI Settings:")
    click.echo(f"  API Prefix: {cfg.api_prefix}")
    click.echo(f"  Docs URL: {cfg.docs_url}")
    click.echo(f"  Health Check: {cfg.health_check_path}")

    click.echo(f"\nCache Settings:")
    click.echo(f"  Enabled: {cfg.cache_enabled}")
    click.echo(f"  Redis Host: {cfg.redis_host}:{cfg.redis_port}")
    click.echo(f"  Redis DB: {cfg.redis_db}")
    click.echo(f"  TTL: {cfg.cache_ttl}s")

    click.echo(f"\nSecurity Settings:")
    click.echo(f"  Rate Limit: {cfg.rate_limit_enabled}")
    click.echo(f"  Auth: {cfg.auth_enabled}")
    click.echo(f"  API Key Header: {cfg.api_key_header}")

    click.echo(f"\nCORS Settings:")
    click.echo(f"  Origins: {cfg.allowed_origins}")
    click.echo(f"  Credentials: {cfg.allow_credentials}")

    click.echo("\n" + "=" * 60)


@server_cli.command()
@click.option("--output", "-o", help="Output file path")
def generate_config(output):
    """Generate a sample configuration file"""
    from server.config import ServerConfig
    import yaml

    # Create sample config
    sample = {
        "host": "0.0.0.0",
        "port": 8000,
        "environment": "production",
        "debug": False,
        "log_level": "INFO",
        "allowed_origins": ["https://app.neuralblitz.ai"],
        "allow_credentials": True,
        "api_prefix": "/api/v1",
        "docs_url": "/docs",
        "rate_limit_enabled": True,
        "auth_enabled": True,
        "api_key_header": "X-API-Key",
        "max_workers": 4,
        "request_timeout": 30,
        "redis_host": "redis",
        "redis_port": 6379,
        "redis_db": 0,
        "cache_enabled": True,
        "cache_ttl": 3600,
        "metrics_enabled": True,
        "metrics_path": "/metrics",
        "health_check_path": "/health",
        "readiness_path": "/ready",
        "partners_file": "/config/partners.yaml",
    }

    yaml_content = yaml.dump(sample, default_flow_style=False, sort_keys=False)

    if output:
        with open(output, "w") as f:
            f.write("# NB Omnibus Server Configuration\n")
            f.write("# Generated automatically\n\n")
            f.write(yaml_content)
        click.echo(f"✓ Configuration written to: {output}")
    else:
        click.echo("# NB Omnibus Server Configuration")
        click.echo("# Generated automatically\n")
        click.echo(yaml_content)


@server_cli.command()
def health():
    """Check server health (requires running server)"""
    import requests

    cfg = load_server_config()
    url = f"http://{cfg.host}:{cfg.port}{cfg.health_check_path}"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code == 200:
            click.echo(f"✓ Server is healthy")
            click.echo(f"  Status: {data.get('status')}")
            click.echo(f"  Version: {data.get('version')}")
            click.echo(f"  Uptime: {data.get('uptime_seconds', 0):.1f}s")
        else:
            click.echo(f"✗ Server health check failed: {data}")
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        click.echo(f"✗ Cannot connect to server at {url}")
        click.echo(f"  Is the server running?")
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Health check error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    server_cli()
