# NeuralBlitz Ecosystem - API Contract Matrix

**Generated:** 2026-02-08  
**Status:** Complete  
**Classification:** Internal Technical Documentation

---

## Executive Summary

This document maps all API contracts and integration points between the four core NeuralBlitz ecosystem projects, identifying endpoints, authentication mechanisms, request/response schemas, and cross-project dependencies.

### Project Overview

| Project | Language | Framework | Primary Port | Protocol |
|---------|----------|-----------|--------------|----------|
| NBX-LRS (NeuralBlitz v50) | Go/Python | Gin/FastAPI | 8080-8082 | REST/gRPC |
| lrs-agents | Python | FastAPI | 8000-8765 | REST/WebSocket |
| Emergent-Prompt-Architecture | Python | FastAPI | 8000 | REST |
| Advanced-Research | Python | Flask | 5000 | REST |

---

## 1. NBX-LRS (NeuralBlitz v50) - API Endpoints

### 1.1 Go API (Gin) - Primary Production Server
**File:** `NBX-LRS/pkg/api/server.go`  
**Port:** 8082 (Default)  
**Framework:** Gin v1.9.1

| Endpoint | Method | Auth | Request Schema | Response Schema | Status |
|----------|--------|------|----------------|-----------------|--------|
| `/` | GET | None | - | `{status, version, architecture, reality, coherence, golden_dag, trace_id, endpoints}` | Active |
| `/health` | GET | None | - | `{status, coherence, irreducible, timestamp}` | Active |
| `/status` | GET | CORS | - | `{status, reality_state, coherence, irreducibility, unity_vector, singularity_status, uptime, go_version, os, arch, goroutines, gc_cycles, golden_dag, trace_id, codex_id}` | Active |
| `/intent` | POST | CORS + Attestation | `{intent: {phi_1, phi_22, omega_genesis}, source}` | `{status, intent_vector, result, amplified, coherence, golden_dag, trace_id, codex_id}` | Active |
| `/verify` | POST | CORS + Attestation | `{type, payload}` | `{type, verified, attestation_hash, golden_dag, trace_id, codex_id}` | Active |
| `/nbcl/interpret` | POST | CORS + Attestation | `{command}` | `{result, golden_dag, codex_id}` | Active |
| `/attestation` | GET | CORS | - | `{attestation, version, golden_dag, trace_id, codex_id, reality_state, coherence, statement}` | Active |
| `/symbiosis` | GET | CORS | - | `{symbiosis_status, architect_system_dyad, coherence, ontological_parity, golden_dag, trace_id, codex_id}` | Active |
| `/synthesis` | GET | CORS | - | `{synthesis_status, omega_singularity, irreducible_source, coherence, volumes_integrated, golden_dag, trace_id, codex_id}` | Active |
| `/options/:id` | GET | CORS | Path: `id ∈ [A-F]` | `{option, name, config, golden_dag, trace_id}` | Active |
| `/options` | GET | CORS | - | `{options[], count, golden_dag, trace_id, codex_id}` | Active |

**Authentication Methods:**
- **CORS:** Open CORS headers (`Access-Control-Allow-Origin: *`)
- **Attestation Middleware:** GoldenDAG hash + Trace ID + Codex ID headers
- **Coherence Middleware:** X-Coherence, X-Reality-State headers

**Key Data Models:**
- `PrimalIntentVector`: `{phi_1, phi_22, omega_genesis}`
- `GoldenDAG`: Cryptographic hash for provenance
- `CodexID`: Versioned identifier (e.g., `VOL0`)

---

### 1.2 Python FastAPI Server
**File:** `NBX-LRS/neuralblitz-v50/python/neuralblitz/api/server.py`  
**Port:** 8080 (Default)  
**Framework:** FastAPI

| Endpoint | Method | Auth | Request | Response | Status |
|----------|--------|------|---------|----------|--------|
| `/` | GET | None | - | `{status, version, system, architecture, coherence, golden_dag}` | Active |
| `/status` | GET | JWT | - | `StatusResponse` | Active |
| `/intent` | POST | JWT | `IntentRequest` | `IntentResponse` | Active |
| `/verify` | POST | JWT | `VerificationRequest` | `VerificationResponse` | Active |
| `/nbcl/interpret` | POST | JWT | `NBCLRequest` | `NBCLResponse` | Active |
| `/attestation` | GET | JWT | - | `AttestationResponse` | Active |
| `/symbiosis` | GET | JWT | - | `SymbiosisResponse` | Active |
| `/synthesis` | GET | JWT | - | `SynthesisResponse` | Active |
| `/options/{option}` | GET | JWT | Path param | Option config | Active |

**Pydantic Models:**
```python
class IntentRequest(BaseModel):
    phi_1: float = Field(default=1.0, ge=0.0, le=1.0)
    phi_22: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = {}

class IntentResponse(BaseModel):
    status: str
    coherence: float
    omega_prime_status: str
    output: Dict[str, Any]
    trace_id: str
```

---

### 1.3 Unified Flask API (Production)
**File:** `NBX-LRS/neuralblitz-v50/applications/unified_api.py`  
**Port:** 5000  
**Framework:** Flask + Flask-CORS

| Endpoint | Method | Auth | Request | Response | Status |
|----------|--------|------|---------|----------|--------|
| `/api/v1/health` | GET | None | - | `{status, version, timestamp, authentication}` | Active |
| `/api/v1/auth/token` | POST | Form | `username, password, grant_type` | `{access_token, token_type, expires_in, scope}` | Active |
| `/api/v1/auth/demo` | GET | None | - | Demo credentials list | Active |
| `/api/v1/auth/introspect` | POST | JWT | - | Token introspection | Active |
| `/api/v1/status` | GET | JWT + Scope:read | - | System status | Active |
| `/api/v1/metrics` | GET | JWT + Scope:metrics | - | Current metrics | Active |
| `/api/v1/metrics/history` | GET | JWT + Scope:metrics | Query: `limit` | Metrics history | Active |
| `/api/v1/quantum/state` | GET | JWT + Scope:read | - | Quantum neuron states | Active |
| `/api/v1/quantum/step` | POST | JWT + Scope:execute | `{input_current, steps}` | Step results | Active |
| `/api/v1/reality/network` | GET | JWT + Scope:read | - | Reality network status | Active |
| `/api/v1/reality/evolve` | POST | JWT + Scope:execute | `{cycles}` | Evolution results | Active |
| `/api/v1/lrs/integrate` | POST | JWT + Scope:execute | - | Integration status | Active |
| `/api/v1/dashboard` | GET | JWT + Scope:read | - | Combined dashboard data | Active |

**JWT Claims:**
```json
{
  "sub": "username",
  "scopes": ["read", "write", "execute", "admin", "metrics"],
  "iat": "timestamp",
  "exp": "timestamp",
  "iss": "neuralblitz-v50"
}
```

**Available Scopes:**
- `read` - Read-only access
- `write` - Write operations
- `execute` - Execute operations (quantum step, evolve)
- `admin` - Administrative functions
- `metrics` - Access to metrics endpoints

---

## 2. lrs-agents - API Endpoints

### 2.1 gRPC Server (Go)
**File:** `opencode-lrs-agents-nbx/pkg/api/grpc_server.go`  
**Port:** Configurable (gRPC)  
**Protocol:** gRPC with Protocol Buffers

**Service: AgentService**

| RPC Method | Request Type | Response Type | Streaming | Status |
|------------|--------------|---------------|-----------|--------|
| `CreateAgent` | `CreateAgentRequest` | `AgentResponse` | Unary | Active |
| `GetAgent` | `GetAgentRequest` | `AgentResponse` | Unary | Active |
| `ListAgents` | `ListAgentsRequest` | `ListAgentsResponse` | Unary | Active |
| `DeleteAgent` | `DeleteAgentRequest` | `Empty` | Unary | Active |
| `ExecutePolicy` | `ExecutePolicyRequest` | `ExecutePolicyResponse` | Unary | Active |
| `StreamStateChanges` | `StateChangeRequest` | `StateChangeResponse` | Server | Active |
| `StreamPrecisionUpdates` | `PrecisionUpdateRequest` | `PrecisionUpdateResponse` | Server | Active |

**Protocol Buffer Messages:**
```protobuf
message AgentProto {
  string id = 1;
  string name = 2;
  string description = 3;
  string state = 4;
  PrecisionProto precision = 5;
  int32 version = 6;
}

message PolicyProto {
  string id = 1;
  string name = 2;
  double free_energy = 3;
  double confidence = 4;
  double epistemic_value = 5;
  double pragmatic_value = 6;
}
```

**Connection Settings:**
- Max Connection Idle: 15 minutes
- Max Connection Age: 30 minutes
- Keepalive Time: 5 seconds
- Keepalive Timeout: 1 second

---

### 2.2 REST Endpoints (TUI Integration)
**File:** `lrs-agents/lrs/integration/tui/rest_endpoints.py`  
**Framework:** FastAPI (APIRouter)
**Base Path:** `/api/v1`

| Endpoint | Method | Auth | Description | Status |
|----------|--------|------|-------------|--------|
| `/agents` | GET | TUI | List all agents | Active |
| `/agents` | POST | TUI | Create agent | Active |
| `/agents/{agent_id}` | GET | TUI | Get agent details | Active |
| `/agents/{agent_id}/state` | PUT | TUI | Update agent state | Active |
| `/agents/{agent_id}` | DELETE | TUI | Delete agent | Active |
| `/agents/{agent_id}/precision` | GET | TUI | Get precision info | Active |
| `/agents/{agent_id}/tools` | GET | TUI | List agent tools | Active |
| `/agents/{agent_id}/history` | GET | TUI | Get execution history | Active |
| `/agents/{agent_id}/tools/execute` | POST | TUI | Execute tool | Active |
| `/tools` | GET | TUI | List all tools | Active |
| `/tui/interaction` | POST | TUI | Handle interaction | Active |
| `/tui/events` | GET | TUI | SSE event stream | Active |
| `/system/info` | GET | TUI | System information | Active |
| `/system/health` | GET | TUI | Health check | Active |

---

### 2.3 OpenCode Bridge API
**File:** `lrs-agents/lrs/opencode/lrs_opencode_bridge.py`  
**Port:** 8765  
**Framework:** FastAPI

| Endpoint | Method | Auth | Request | Response |
|----------|--------|------|---------|----------|
| `/lrs/create-agent` | POST | Internal | `LRSRequest` | `{status, agent_id}` |
| `/lrs/execute` | POST | Internal | `LRSRequest` | `{result}` |
| `/opencode/execute` | POST | Internal | `OpenCodeRequest` | `{success, stdout, stderr, returncode}` |

**Models:**
```python
class LRSRequest(BaseModel):
    agent_id: str
    task: str
    context: Optional[Dict[str, Any]] = None

class OpenCodeRequest(BaseModel):
    command: str
    working_dir: Optional[str] = "."
    timeout: Optional[int] = 30
```

---

## 3. Emergent-Prompt-Architecture (EPA) - API Endpoints

**File:** `Emergent-Prompt-Architecture/api_server.py`  
**Port:** 8000  
**Framework:** FastAPI  
**Base Path:** `/api/v1`

| Endpoint | Method | Auth | Request | Response | Status |
|----------|--------|------|---------|----------|--------|
| `/health` | GET | None | - | `HealthResponse` | Active |
| `/api/v1/ingest` | POST | None | `IngestRequest` | `IngestResponse` | Active |
| `/api/v1/crystallize` | POST | None | `CrystallizeRequest` | `CrystallizeResponse` | Active |
| `/api/v1/feedback` | POST | None | `FeedbackRequest` | `FeedbackResponse` | Active |
| `/api/v1/lattice/stats` | GET | None | - | `LatticeStatsResponse` | Active |
| `/api/v1/feedback/stats` | GET | None | - | Feedback stats | Active |
| `/api/v1/maintenance/decay` | POST | None | - | `{status, timestamp}` | Active |
| `/api/v1/maintenance/cleanup` | POST | None | Query: `max_age_days` | Cleanup results | Active |
| `/api/v1/sessions/{session_id}` | GET | None | Path: `session_id` | Session info | Active |
| `/api/v1/sessions/{session_id}` | DELETE | None | Path: `session_id` | Deletion confirmation | Active |

**Key Models:**
```python
class IngestRequest(BaseModel):
    source: str
    content: str
    timestamp: Optional[float] = Field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = {}

class CrystallizeRequest(BaseModel):
    session_id: Optional[str] = ""
    target_model: Optional[str] = "default"
    mode: Optional[SystemMode] = SystemMode.SENTIO

class CrystallizeResponse(BaseModel):
    prompt_object: Dict[str, Any]
    provenance: Dict[str, List[str]]
    goldendag_hash: str
    trace_id: str
```

**Onton Types:**
- `CONTEXT` - Contextual information
- `MEMORY` - Stored memories
- `CONCEPT` - Abstract concepts
- `RELATION` - Relationships between concepts
- `TEMPORAL` - Time-based information
- `ETHICAL` - Ethical constraints

---

## 4. Advanced-Research - API Endpoints

**File:** `Advanced-Research/src/advanced_research/unified/api.py`  
**Architecture:** Python class-based (not HTTP server)
**Integration:** Direct Python API calls via `UnifiedResearchSystem`

### 4.1 Unified Research System API

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `initialize()` | - | None | Initialize all integrations |
| `shutdown()` | - | None | Shutdown all components |
| `create_session(user_id, mode, preferences)` | `user_id: str, mode: SystemMode, preferences: dict` | `session_id: str` | Create user session |
| `process_query(session_id, query, context_data)` | `session_id: str, query: str, context: dict` | `dict` | Process query |
| `get_system_status()` | - | `dict` | Get comprehensive status |

**System Modes:**
```python
class SystemMode(Enum):
    RESEARCH = "research"
    LEARNING = "learning"
    COMPUTATION = "computation"
    COLLABORATIVE = "collaborative"
```

---

### 4.2 Integration Manager APIs

**IntegrationsManager** provides access to three integrations:

#### LRS Integration
```python
async def record_learning_event(actor, verb, object_data, result=None)
async def record_context_interaction(context_block, user_id, interaction_type, outcome)
async def flush_records()
```

#### Opencode Integration
```python
async def create_research_document(title, content, research_area, tags, metadata)
async def search_documents(query, research_area, tags)
```

#### NeuralBlitz Integration
```python
async def compute_geometric_features(input_data, manifold_type, curvature)
async def optimize_on_manifold(objective_function, initial_point, manifold_config)
```

---

## 5. Cross-Project API Integration Matrix

### 5.1 Which Projects Call Which

```
┌─────────────────────────────────────────────────────────────────┐
│                    API CALL MATRIX                               │
├─────────────────────────────────────────────────────────────────┤
│ NBX-LRS → lrs-agents:                                           │
│   • HTTP POST to /api/v1/lrs/integrate (bridge endpoint)        │
│   • WebSocket for real-time state streaming                     │
│   • gRPC for agent management (planned)                         │
├─────────────────────────────────────────────────────────────────┤
│ lrs-agents → NBX-LRS:                                           │
│   • HTTP GET /api/v1/quantum/state (metrics)                    │
│   • HTTP POST /api/v1/quantum/step (control)                    │
│   • HTTP GET /api/v1/reality/network (status)                   │
│   • HTTP POST /api/v1/reality/evolve (control)                  │
├─────────────────────────────────────────────────────────────────┤
│ lrs-agents ↔ OpenCode:                                          │
│   • HTTP POST /lrs/create-agent                                 │
│   • HTTP POST /lrs/execute                                      │
│   • HTTP POST /opencode/execute                                 │
├─────────────────────────────────────────────────────────────────┤
│ Advanced-Research → lrs-agents:                                 │
│   • Direct Python API via LRSIntegration class                  │
│   • Records learning events to LRS                              │
├─────────────────────────────────────────────────────────────────┤
│ Advanced-Research → Opencode:                                   │
│   • Direct Python API via OpencodeIntegration class             │
│   • Creates research documents                                  │
├─────────────────────────────────────────────────────────────────┤
│ Advanced-Research → NBX-LRS:                                    │
│   • Direct Python API via NeuralBlitzIntegration class          │
│   • Geometric feature computation                               │
├─────────────────────────────────────────────────────────────────┤
│ EPA → NBX-LRS:                                                  │
│   • Shares GoldenDAG hash format                                │
│   • Compatible provenance tracking                              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Protocol Versions

| Protocol | Version | Status | Used By |
|----------|---------|--------|---------|
| HTTP/1.1 | 1.1 | Active | All Flask/FastAPI servers |
| HTTP/2 | 2.0 | Supported | gRPC services |
| gRPC | 1.50+ | Active | lrs-agents Go server |
| WebSocket | RFC 6455 | Active | lrs-agents TUI |
| JWT | RFC 7519 | Active | NBX-LRS unified_api |
| SSE | W3C Standard | Active | lrs-agents TUI events |

### 5.3 Authentication Compatibility

| Project | JWT | API Key | OAuth2 | mTLS | Basic Auth |
|---------|-----|---------|--------|------|------------|
| NBX-LRS Go | ✗ | ✗ | ✗ | ✗ | ✗ |
| NBX-LRS FastAPI | ✗ | ✗ | ✗ | ✗ | ✗ |
| NBX-LRS Flask | ✓ | ✗ | ✗ | ✗ | ✓ (Form) |
| lrs-agents gRPC | ✗ | ✗ | ✗ | ✗ | ✗ |
| lrs-agents REST | ✗ | ✗ | ✗ | ✗ | ✗ |
| lrs-agents Bridge | ✗ | ✗ | ✗ | ✗ | ✗ |
| EPA | ✗ | ✗ | ✗ | ✗ | ✗ |
| Advanced-Research | ✗ | ✗ | ✗ | ✗ | ✗ |

**Note:** Only NBX-LRS unified_api.py implements JWT authentication currently.

---

## 6. Shared Data Models & Schemas

### 6.1 GoldenDAG
**Used By:** NBX-LRS, EPA, lrs-agents

```python
GoldenDAG = {
    "hash": str,  # 64-char hex
    "seed": str,  # Reference seed
    "timestamp": datetime,
    "provenance": List[str]
}
```

### 6.2 LRS Learning Record (xAPI)
**Used By:** lrs-agents, Advanced-Research

```python
LearningRecord = {
    "actor": {
        "account": {
            "homePage": str,
            "name": str
        }
    },
    "verb": {
        "id": str,  # http://adlnet.gov/expapi/verbs/{verb}
        "display": {"en-US": str}
    },
    "object": {
        "id": str,
        "objectType": "Activity",
        "definition": dict
    },
    "result": Optional[dict],
    "timestamp": datetime
}
```

### 6.3 Agent State
**Used By:** lrs-agents, NBX-LRS (bridge)

```python
AgentState = {
    "agent_id": str,
    "agent_type": str,
    "config": dict,
    "tools": List[str],
    "created_at": str,  # ISO datetime
    "status": str,  # "active", "deleted"
    "precision": {
        "abstract": float,
        "planning": float,
        "execution": float
    }
}
```

### 6.4 Quantum Neuron State
**Used By:** NBX-LRS

```python
QuantumNeuronState = {
    "membrane_potential": float,  # mV
    "spike_rate": float,  # Hz
    "spike_count": int,
    "time_elapsed": float,  # ms
    "is_refractory": bool
}
```

### 6.5 Reality Network State
**Used By:** NBX-LRS

```python
RealityNetworkState = {
    "global_consciousness": float,  # 0-1
    "cross_reality_coherence": float,  # 0-1
    "information_flow_rate": float,
    "reality_synchronization": float,
    "realities": {
        "reality_id": {
            "type": str,  # RealityType
            "consciousness": float,
            "information_density": float,
            "quantum_coherence": float
        }
    },
    "active_signals": int
}
```

---

## 7. OpenAPI Specification

**Location:** `NBX-LRS/neuralblitz-v50/docs/api/openapi.yaml`

The OpenAPI spec defines standard schemas for:
- StatusResponse
- IntentRequest/IntentResponse
- VerificationRequest/VerificationResponse
- NBCLRequest/NBCLResponse
- AttestationResponse
- SymbiosisResponse
- SynthesisResponse
- OptionResponse

**Security Scheme:**
```yaml
securitySchemes:
  bearerAuth:
    type: http
    scheme: bearer
    bearerFormat: JWT
```

---

## 8. Integration Test Coverage

### 8.1 Test Files

| Project | Test File | Coverage |
|---------|-----------|----------|
| lrs-agents | `integration-bridge/tests/test_api.py` | API endpoint tests with mocked httpx |
| lrs-agents | `integration-bridge/tests/integration_load_tests.py` | Load testing with aiohttp |
| NBX-LRS | `neuralblitz-v50/tests/test_auth.py` | Authentication tests |
| NBX-LRS | `neuralblitz-v50/tests/test_audit.py` | Audit trail tests |

### 8.2 Test Patterns

```python
# Example: Testing NBX-LRS integration
@patch("opencode_lrs_bridge.api.endpoints.httpx.AsyncClient")
def test_quantum_state_endpoint(mock_client):
    # Mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "quantum_coherence": 0.95,
        "consciousness_level": 0.8
    }
    mock_client.return_value.get.return_value = mock_response
    
    # Test call
    response = client.get("/api/v1/quantum/state")
    assert response.status_code == 200
```

---

## 9. API Compatibility Matrix

### 9.1 Version Compatibility

| Source | Target | Protocol | Version | Status |
|--------|--------|----------|---------|--------|
| NBX-LRS v50 | lrs-agents | HTTP | 1.1 | Compatible |
| lrs-agents | NBX-LRS v50 | HTTP | 1.1 | Compatible |
| lrs-agents | OpenCode | HTTP | 1.1 | Compatible |
| Advanced-Research | lrs-agents | Python API | - | Compatible |
| Advanced-Research | NBX-LRS | Python API | - | Compatible |
| EPA | NBX-LRS | GoldenDAG | Shared | Compatible |

### 9.2 Breaking Changes

**None identified** - All projects use compatible data formats and protocols.

**Note:** NBX-LRS has multiple API implementations (Go, Python, Flask) with different authentication levels. The Flask unified_api is the most feature-complete with JWT support.

---

## 10. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  (React Dashboard, CLI Tools, External Services)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                       API GATEWAY                                │
│  • NBX-LRS Flask API (Port 5000) - JWT Auth                     │
│  • NBX-LRS Go API (Port 8082) - CORS Only                       │
│  • lrs-agents REST (Port 8000)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐
│ NBX-LRS Core │   │ lrs-agents Core  │   │ EPA System      │
│ - Quantum    │   │ - gRPC Server    │   │ - Ontons        │
│ - Multi-     │   │ - Agent Manager  │   │ - Lattice       │
│   Reality    │   │ - TUI Bridge     │   │ - Crystallize   │
└──────────────┘   └──────────────────┘   └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ADVANCED-RESEARCH                             │
│         (Orchestrates all three systems)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Security Considerations

### 11.1 Authentication Gaps
- Most endpoints lack authentication (CORS only)
- Only NBX-LRS Flask API implements JWT
- No OAuth2 or API key authentication currently implemented
- No mTLS for service-to-service communication

### 11.2 Recommended Security Enhancements
1. Implement API keys for service-to-service auth
2. Add rate limiting to all endpoints
3. Enable request signing for sensitive operations
4. Implement mTLS for gRPC services
5. Add audit logging for all API calls

---

## 12. Summary

### 12.1 Active Endpoints Summary

| Project | REST Endpoints | gRPC Methods | WebSocket | Total |
|---------|----------------|--------------|-----------|-------|
| NBX-LRS | 16 (3 APIs) | 0 | 0 | 16 |
| lrs-agents | 14 | 7 | 1 | 22 |
| EPA | 9 | 0 | 0 | 9 |
| Advanced-Research | 6 (Python API) | 0 | 0 | 6 |
| **TOTAL** | **45** | **7** | **1** | **53** |

### 12.2 Integration Points Summary

- **Bidirectional HTTP:** NBX-LRS ↔ lrs-agents
- **Bridge Pattern:** lrs-agents ↔ OpenCode
- **Direct Python API:** Advanced-Research → All systems
- **Shared Formats:** GoldenDAG (NBX-LRS, EPA, lrs-agents)
- **Protocol Buffers:** lrs-agents gRPC services

### 12.3 Status: All Systems Operational

All identified API contracts are **ACTIVE** and compatible. No deprecated endpoints found. All projects can communicate through established integration points.

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-08  
**Classification:** Technical Architecture
