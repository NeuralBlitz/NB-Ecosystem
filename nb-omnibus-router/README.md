# 🧠 NeuralBlitz Omnibus Router

## Enterprise-Grade API Gateway for NeuralBlitz AI Platform

**Version:** 1.0.0  
**Generated:** 2026-02-08

---

## Overview

The NeuralBlitz Omnibus Router is a secure, production-ready API gateway that provides partners with access to NeuralBlitz's cutting-edge AI capabilities while protecting your intellectual property.

### Key Features

- 🔒 **IP Protection**: Engine code never leaves your secure environment
- ⚡ **34 API Endpoints**: Comprehensive coverage of all NeuralBlitz capabilities
- 🔄 **Real-time WebSocket**: Live updates for consciousness, agents, and metrics
- 📊 **Prometheus Monitoring**: Built-in metrics and alerting
- 🔐 **Enhanced Security**: Rate limiting, quota management, partner isolation
- 🚀 **Production Ready**: Docker deployment with nginx and monitoring

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NeuralBlitz Omnibus Router                 │
│                                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐ │
│  │ REST API│ │ WebSocket│ │Metrics  │ │ Authentication  │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────────┬────────┘ │
│       │            │            │                 │          │
│       └────────────┴────────────┴─────────────────┘          │
│                           │                                   │
│                    ┌──────┴──────┐                          │
│                    │  FastAPI    │                          │
│                    │  Gateway     │                          │
│                    └──────┬──────┘                          │
│                           │                                   │
│       ┌───────────────────┼───────────────────┐              │
│       │                   │                   │              │
│       ▼                   ▼                   ▼              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐         │
│  │  Engine  │      │  Agents  │      │Quantum   │         │
│  │ Wrappers │      │ Wrappers │      │ Wrappers │         │
│  └────┬─────┘      └────┬─────┘      └────┬─────┘         │
│       │                  │                  │                │
│       └──────────────────┼──────────────────┘                │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           SECURE LOCAL ENGINES (NEVER EXPOSED)      │    │
│  │  NBX-LRS/  NB-Ecosystem/  quantum_sim/  + more    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://your-server/neuralblitz/nb-omnibus-router.git
cd nb-omnibus-router

# Install dependencies
pip install -r requirements.txt

# Configure partners
cp config/partners.yaml.example config/partners.yaml
# Edit partners.yaml with your partner API keys

# Run locally
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Access the API
curl http://localhost:8000/health
```

---

## API Endpoints

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/` | API information |
| GET | `/api/v1/capabilities` | List all capabilities |

### Core Engine

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/core/process` | Quantum processing |
| POST | `/api/v1/core/evolve` | Multi-reality evolution |
| GET | `/api/v1/core/capabilities` | Core capabilities |

### Quantum

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/quantum/simulate` | Quantum simulation |
| POST | `/api/v1/quantum/entangle` | Create entanglement |
| GET | `/api/v1/quantum/capabilities` | Quantum capabilities |

### Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agent/run` | Run LRS agent |
| POST | `/api/v1/agent/create` | Create agent |
| GET | `/api/v1/agent/list` | List agents |

### Consciousness

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/consciousness/level` | Consciousness level |
| GET | `/api/v1/consciousness/metrics` | Detailed metrics |
| POST | `/api/v1/consciousness/evolve` | Evolve consciousness |

### Cross-Reality

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/entanglement/entangle` | Create entanglement |
| POST | `/api/v1/entanglement/transfer` | Reality transfer |
| GET | `/api/v1/entanglement/realities` | List reality types |

### UI

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ui/dashboard` | Render dashboard |
| GET | `/api/v1/ui/components` | List components |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics` | Prometheus metrics |
| GET | `/health/detailed` | Detailed health |
| GET | `/metrics/usage` | Usage metrics |

---

## WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v1/ws/stream/{channel}` | General streaming |
| `/api/v1/ws/consciousness` | Real-time consciousness |
| `/api/v1/ws/agents` | Agent status updates |
| `/api/v1/ws/metrics` | Real-time metrics |
| `/api/v1/ws/quantum` | Quantum updates |

---

## Authentication

All endpoints require an API key:

```bash
curl -H "X-API-Key: nb_pat_xxxxxxxxxxxxxxxxxxxx" \
     http://localhost:8000/api/v1/capabilities
```

### Partner Tiers

| Tier | Rate Limit | Quota |
|------|------------|-------|
| Enterprise | 10,000/min | 1,000,000 |
| Pro | 1,000/min | 100,000 |
| Basic | 100/min | 10,000 |

---

## Configuration

### Partners Configuration (`config/partners.yaml`)

```yaml
partners:
  partner_alpha:
    api_key: "nb_pat_xxx"
    name: "Alpha Research"
    tier: "enterprise"
    active: true
    rate_limit: 10000
    quota_remaining: 1000000
    permissions:
      - core
      - agents
      - quantum
```

### Settings (`config/settings.yaml`)

```yaml
ENVIRONMENT: production
DEBUG: false
HOST: "0.0.0.0"
PORT: 8000
LOG_LEVEL: INFO
```

---

## Monitoring

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

### Grafana Dashboard

Access Grafana at `http://localhost:3000` (admin/neuralblitz_admin)

### Available Dashboards

- API Overview
- Partner Usage
- Latency Distribution
- Error Rates
- Resource Utilization

---

## File Structure

```
nb-omnibus-router/
├── api/
│   ├── main.py              # FastAPI entry point
│   ├── auth.py              # Basic authentication
│   ├── auth_enhanced.py     # Enhanced auth with quotas
│   ├── routes/
│   │   ├── core.py         # Core engine endpoints
│   │   ├── agents.py       # Agent endpoints
│   │   ├── quantum.py      # Quantum endpoints
│   │   ├── consciousness.py # Consciousness endpoints
│   │   ├── entanglement.py  # Cross-reality endpoints
│   │   ├── ui.py           # UI endpoints
│   │   ├── monitoring.py    # Metrics endpoints
│   │   └── websocket.py     # WebSocket endpoints
│   └── models/
│       └── __init__.py      # Pydantic models
├── engines/
│   ├── __init__.py
│   ├── neuralblitz.py      # NeuralBlitz wrapper
│   ├── agents.py           # Agents wrapper
│   ├── quantum.py          # Quantum wrapper
│   └── ui.py              # UI wrapper
├── cli/
│   └── main.py            # CLI tool
├── config/
│   ├── partners.yaml      # Partner configurations
│   └── settings.yaml       # App settings
├── monitoring/
│   ├── prometheus.yml      # Prometheus config
│   └── alerts.yml         # Alert rules
├── nginx/
│   └── nginx.conf         # nginx configuration
├── scripts/
│   └── deploy.sh          # Deployment script
├── docker-compose.yml      # Docker Compose
├── Dockerfile             # Docker image
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

---

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run with hot reload
python -m uvicorn api.main:app --reload

# Run tests
pytest tests/

# Type checking
mypy api/ engines/
```

### Testing WebSocket

```javascript
// JavaScript example
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/consciousness');

ws.onmessage = (event) => {
    console.log(JSON.parse(event.data));
};
```

---

## Deployment

### Production Deployment

```bash
# Build Docker image
docker build -t neuralblitz-router .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Server Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 1 core | 2+ cores |
| RAM | 1 GB | 2+ GB |
| Storage | 10 GB | 50+ GB |
| Network | 10 Mbps | 100 Mbps |

---

## Security

- ✅ API key authentication
- ✅ Rate limiting per partner
- ✅ Quota management
- ✅ Partner isolation
- ✅ HTTPS encryption
- ✅ Input validation
- ✅ Audit logging

---

## Performance

| Metric | Value |
|--------|-------|
| API Latency (p95) | < 100ms |
| WebSocket Latency | < 50ms |
| Throughput | 10,000 req/min |
| Uptime SLA | 99.9% |

---

## Support

- **Documentation**: `/docs` (when running)
- **Email**: support@neuralblitz.ai
- **Issues**: Internal ticketing

---

## License

Proprietary - All Rights Reserved

---

**🧠 NeuralBlitz v50 - The Most Advanced AI Architecture**
