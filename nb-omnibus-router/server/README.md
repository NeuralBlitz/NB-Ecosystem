# NB Omnibus Server

Dedicated server module for the NeuralBlitz Omnibus Router.

## Overview

The NB Omnibus Server provides a clean separation between server infrastructure and API business logic. It offers:

- **Lifecycle Management**: Proper startup/shutdown sequence with dependency resolution
- **Health Checks**: Comprehensive health monitoring with multiple probe types
- **Middleware Stack**: Configurable middleware including CORS, compression, security headers
- **Configuration**: Multi-source configuration (files, environment, CLI arguments)
- **Monitoring**: Built-in Prometheus metrics and structured logging

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NB Omnibus Server                     │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐│
│  │              ServerLifecycle                         ││
│  │  - Startup tasks (cache, connections, etc.)         ││
│  │  - Shutdown tasks (cleanup, disconnections)         ││
│  └─────────────────────────────────────────────────────┘│
│                           │                              │
│  ┌────────────────────────┼─────────────────────────────┐│
│  │                        ▼                              ││
│  │              OmnibusServer                           ││
│  │  - Application factory                               ││
│  │  - Router management                                 ││
│  │  - Middleware setup                                  ││
│  └─────────────────────────────────────────────────────┘│
│                           │                              │
│  ┌────────────────────────┼─────────────────────────────┐│
│  │                        ▼                              ││
│  │              FastAPI Application                     ││
│  │  - API routes                                        ││
│  │  - System endpoints (/health, /ready)                ││
│  │  - Documentation                                     ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Run the Server

```bash
# Development mode with auto-reload
python -m server.main --env development --reload

# Production mode with multiple workers
python -m server.main --env production --workers 4

# With custom configuration
python -m server.main --config config/production.yaml
```

### Using CLI

```bash
# Start the server
python -m cli.main server start --env production --port 8000

# Check configuration
python -m cli.main server config

# Check health
python -m cli.main server health
```

### Docker Deployment

```bash
# Using dedicated server compose file
docker-compose -f docker-compose.server.yml up -d

# Scale workers
docker-compose -f docker-compose.server.yml up -d --scale nb-server=3
```

## Configuration

Configuration is loaded from multiple sources in order of precedence:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (e.g., `NB_HOST`, `REDIS_HOST`)
3. **Configuration files** (YAML)
4. **Default values** (lowest priority)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NB_HOST` | Server bind host | `0.0.0.0` |
| `NB_PORT` | Server port | `8000` |
| `NB_ENVIRONMENT` | Environment name | `development` |
| `NB_LOG_LEVEL` | Logging level | `INFO` |
| `NB_WORKERS` | Number of workers | `1` |
| `REDIS_HOST` | Redis server host | `localhost` |
| `REDIS_PORT` | Redis server port | `6379` |

### Configuration File

Generate a sample configuration:

```bash
python -m server.main generate-config --output config/server.yaml
```

## Module Structure

```
server/
├── __init__.py           # Public API exports
├── application.py        # FastAPI application factory
├── config.py            # Configuration management
├── health.py            # Health checks and probes
├── lifecycle.py         # Startup/shutdown management
├── middleware.py        # Middleware setup
└── main.py              # Entry point
```

### Key Components

#### 1. Application Factory (`application.py`)

```python
from server import create_application, ServerConfig

config = ServerConfig(host="0.0.0.0", port=8000)
app = create_application(config=config)
```

#### 2. Configuration (`config.py`)

```python
from server import load_server_config

# Load from file and environment
config = load_server_config("config/server.yaml")

# Access settings
print(config.host, config.port)
```

#### 3. Lifecycle Management (`lifecycle.py`)

```python
from server import ServerLifecycle, ServerConfig

config = ServerConfig()
lifecycle = ServerLifecycle(config)

# Register custom startup task
async def init_custom_service():
    # Your initialization code
    pass

lifecycle.register_startup_task(
    name="custom_init",
    callable=init_custom_service,
    priority=20  # Lower = earlier execution
)
```

#### 4. Health Checks (`health.py`)

```python
from server import HealthManager, HealthStatus

health = HealthManager()

# Register custom health check
async def check_database():
    # Your check logic
    return HealthStatus.HEALTHY, "Database connected", {}

health.register_check("database", check_database, critical=True)

# Run checks
report = await health.run_all_checks()
```

## Health Endpoints

The server provides built-in health endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Basic health status |
| `GET /health/detailed` | Detailed component health |
| `GET /ready` | Readiness probe for Kubernetes |

### Response Format

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-02-09T12:00:00Z",
  "uptime_seconds": 3600
}
```

## Middleware

The server includes pre-configured middleware:

1. **Security Headers**: Adds security-related HTTP headers
2. **GZip Compression**: Compresses responses > 1KB
3. **CORS**: Cross-origin resource sharing
4. **Request Timing**: Tracks and logs request duration
5. **Request Logging**: Logs all requests with timing
6. **Rate Limiting**: Basic rate limiting (if enabled)

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Type Checking

```bash
mypy server/
```

### Code Formatting

```bash
black server/
isort server/
```

## Production Deployment

### Using Docker Compose

```yaml
services:
  nb-server:
    build:
      dockerfile: Dockerfile.server
    environment:
      - NB_ENVIRONMENT=production
      - NB_WORKERS=4
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nb-omnibus-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: server
        image: neuralblitz/nb-omnibus-server:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
```

## Monitoring

The server exposes Prometheus metrics at `/metrics`:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency
- `http_request_size_bytes` - Request size
- `http_response_size_bytes` - Response size

## License

Proprietary - NeuralBlitz AI Platform
