# NeuralBlitz API Reference

## Overview

The NeuralBlitz Omnibus Router provides a comprehensive API for accessing NeuralBlitz AI capabilities. All endpoints require authentication via API key.

## Base URL

```
https://your-server.com/api/v1
```

## Authentication

All endpoints require an API key in the header:

```
X-API-Key: nb_pat_xxxxxxxxxxxxxxxxxxxx
```

## Endpoints

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/` | API information |
| GET | `/api/v1/capabilities` | List all capabilities |

### Core Engine

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/core/process` | Process data through quantum neuron |
| POST | `/api/v1/core/evolve` | Evolve multi-reality network |
| GET | `/api/v1/core/capabilities` | List core capabilities |
| GET | `/api/v1/core/status` | Core engine status |

### Quantum

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/quantum/simulate` | Run quantum simulation |
| POST | `/api/v1/quantum/entangle` | Create entangled pairs |
| GET | `/api/v1/quantum/capabilities` | List quantum capabilities |

### Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agent/run` | Run LRS agent |
| POST | `/api/v1/agent/create` | Create new agent |
| GET | `/api/v1/agent/list` | List available agents |
| GET | `/api/v1/agent/capabilities` | List agent capabilities |

### Advanced Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agents/create` | Create new agent |
| POST | `/api/v1/agents/evolve` | Evolve agent |
| POST | `/api/v1/agents/learn` | Train with feedback |
| GET | `/api/v1/agents/types` | List agent types |
| GET | `/api/v1/agents/{id}/status` | Agent status |
| GET | `/api/v1/agents/{id}/capabilities` | Agent capabilities |

### Consciousness

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/consciousness/level` | Get consciousness level |
| GET | `/api/v1/consciousness/metrics` | Get detailed metrics |
| POST | `/api/v1/consciousness/evolve` | Evolve consciousness |
| GET | `/api/v1/consciousness/cosmic-bridge` | Cosmic bridge status |
| GET | `/api/v1/consciousness/dimensional-access` | Dimensional access |

### Cross-Reality

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/entanglement/entangle` | Create entanglement |
| GET | `/api/v1/entanglement/entanglements` | List entanglements |
| POST | `/api/v1/entanglement/transfer` | Reality transfer |
| GET | `/api/v1/entanglement/realities` | List reality types |
| POST | `/api/v1/entanglement/synchronize` | Synchronize realities |
| GET | `/api/v1/entanglement/coherence` | Coherence metrics |

### UI

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/ui/dashboard` | Render dashboard |
| GET | `/api/v1/ui/components` | List components |
| GET | `/api/v1/ui/capabilities` | UI capabilities |

---

## Request Examples

### Quantum Processing

```bash
curl -X POST https://your-server.com/api/v1/core/process \
  -H "X-API-Key: nb_pat_xxx" \
  -H "Content-Type: application/json" \
  -d '{"input_data": [0.1, 0.2, 0.3], "current": 20.0, "duration": 200.0}'
```

Response:
```json
{
  "success": true,
  "output": [0.01, 0.02, 0.03],
  "spike_rate": 35.0,
  "coherence_time": 100.0,
  "step_time_us": 93.41,
  "mode": "mock"
}
```

### List Capabilities

```bash
curl -X GET https://your-server.com/api/v1/capabilities \
  -H "X-API-Key: nb_pat_xxx"
```

Response:
```json
{
  "capabilities": {
    "engine": "NeuralBlitz v50",
    "version": "50.0.0",
    "technologies": [
      {"name": "Quantum Spiking Neurons", "ops_per_sec": 10705},
      {"name": "Multi-Reality Networks", "cycles_per_sec": 2710},
      {"name": "Consciousness Integration", "levels": 8}
    ]
  },
  "partner_tier": "pro"
}
```

---

## Rate Limits

| Tier | Requests/Day |
|------|-------------|
| Enterprise | 1,000,000 |
| Pro | 100,000 |
| Basic | 10,000 |

---

## Error Codes

| Code | Description |
|------|-------------|
| 401 | Invalid or missing API key |
| 403 | API key deactivated |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## SDK Integration

### Python

```python
import requests

class NeuralBlitzClient:
    def __init__(self, api_key, base_url="https://your-server.com"):
        self.api_key = api_key
        self.base_url = base_url
    
    def _request(self, method, endpoint, data=None):
        response = requests.request(
            method,
            f"{self.base_url}{endpoint}",
            headers={"X-API-Key": self.api_key},
            json=data
        )
        return response.json()
    
    def process_quantum(self, input_data):
        return self._request("POST", "/api/v1/core/process", {
            "input_data": input_data
        })
    
    def list_capabilities(self):
        return self._request("GET", "/api/v1/capabilities")
```

---

*Generated: 2026-02-08*
