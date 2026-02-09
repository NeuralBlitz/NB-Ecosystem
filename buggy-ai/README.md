# 🐛 Buggy AI - Extended Platform

## 10-Layer Distributed AI Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ L10: API GATEWAY         Auth │ Rate Limit │ Load Balancing    │
├─────────────────────────────────────────────────────────────────┤
│ L9: STRATEGIC PLANNER    Goal Decomposition │ RAG │ Memory     │
├─────────────────────────────────────────────────────────────────┤
│ L8: BATCH PROCESSOR      Queue │ Retry │ Priority │ Throttling │
├─────────────────────────────────────────────────────────────────┤
│ L7: CLUSTER MANAGER      Nodes │ Sharding │ Replication       │
├─────────────────────────────────────────────────────────────────┤
│ L6: TASK ORCHESTRATOR    DAG │ Dependencies │ Workflow         │
├─────────────────────────────────────────────────────────────────┤
│ L5: AUTONOMOUS AGENTS    Think Loop │ Tools │ Memory          │
├─────────────────────────────────────────────────────────────────┤
│ L4: WEB DASHBOARD        REST API │ WebSocket │ React UI       │
├─────────────────────────────────────────────────────────────────┤
│ L3: SANDBOXED RUNNER     Docker │ gVisor │ WASM │ Isolation    │
├─────────────────────────────────────────────────────────────────┤
│ L2: MCP PROTOCOL         Tools │ Resources │ Prompts         │
├─────────────────────────────────────────────────────────────────┤
│ L1: VOICE AI             Whisper STT │ TTS │ Wake Word       │
└─────────────────────────────────────────────────────────────────┘
```

## Extended Components

### ✅ Completed Layers

| Layer | Component | Status | Features |
|-------|-----------|--------|----------|
| **L1** | Voice AI | ✅ | Whisper STT, Festival/Google TTS, Audio processing, Wake word |
| **L2** | MCP Protocol | ✅ | Tools, Resources, Prompts, Sampling, Sessions |
| **L3** | Sandboxed Runner | ✅ | Docker, gVisor, WASM, 15+ languages, Resource limits |
| **L6** | Task Orchestrator | ✅ | DAG scheduling, Workflows, Checkpoints, Plugins |

### 🚧 In Progress

| Layer | Component | Status |
|-------|-----------|--------|
| **L4** | Web Dashboard | HTTP/WS API + React UI |
| **L5** | Autonomous Agents | Think loops, Memory, Planning |
| **L7** | Cluster Manager | Distributed nodes, Sharding |
| **L8** | Batch Processor | Queue, Dead letter, Rate limiting |
| **L9** | Strategic Planner | Goal decomposition, RAG |
| **L10** | API Gateway | Auth, Rate limiting, Routes |

## New Directory Structure

```
buggy-ai/
├── cmd/
│   ├── cli/              # Terminal UI
│   ├── server/           # API Server
│   └── agent/            # Worker Agent
├── internal/
│   ├── voice/            # L1: Voice AI (STT/TTS)
│   ├── mcp/              # L2: MCP Protocol Server
│   ├── runner/           # L3: Sandboxed Code Runner
│   ├── web/              # L4: Web Dashboard (TODO)
│   ├── agent/            # L5: Autonomous Agents (TODO)
│   ├── orchestrator/     # L6: Task Orchestrator
│   ├── cluster/          # L7: Cluster Manager (TODO)
│   ├── batch/            # L8: Batch Processor (TODO)
│   ├── planner/          # L9: Strategic Planner (TODO)
│   ├── gateway/          # L10: API Gateway (TODO)
│   ├── config/           # Configuration
│   └── shared/           # Shared types
└── ARCHITECTURE.md
```

## Quick Start

```bash
# Terminal UI (original)
./buggy-ai

# Voice mode
./buggy-ai voice --stt whisper --tts festival

# Run code
./buggy-ai run --language python --code 'print("Hello!")'

# Execute workflow
./buggy-ai workflow --file workflow.yaml
```

## Running with Extensions

```bash
# Start API server
./buggy-ai server --port 8080 --cluster

# Start worker agent
./buggy-ai agent --server localhost:8080

# Voice input
./buggy-ai voice --wake-word "hey buggy"

# Run MCP server
./buggy-ai mcp --port 3000
```

## Features

### Voice AI (L1)
- Speech-to-text: Whisper, Google STT
- Text-to-speech: Festival, Google TTS
- Noise reduction, Audio normalization
- Wake word detection

### MCP Protocol (L2)
- Tool registration & execution
- Resource management
- Prompt templates
- Sampling API (Claude's protocol)

### Sandboxed Runner (L3)
- 15+ Languages: Python, JS, Go, Rust, Java, C++, etc.
- Docker, gVisor, WASM isolation
- Memory/CPU limits, Timeouts
- Resource monitoring

### Task Orchestrator (L6)
- DAG-based workflow definition
- Dependency resolution
- Checkpoint & resume
- Plugin system

## Configuration

```yaml
app:
  name: Buggy AI
  version: 2.0.0

cluster:
  mode: distributed
  nodes:
    - host: localhost
      port: 8080
      role: master

voice:
  enabled: true
  stt: whisper
  tts: festival
  wake_word: "hey buggy"

security:
  sandbox: docker
  memory_limit: 1GB
  cpu_limit: 2.0

agents:
  max_concurrent: 10
  planning_depth: 5
```

## Architecture Highlights

1. **Modular Design** - Each layer is independent and replaceable
2. **Distributed Ready** - Cluster support from the ground up
3. **Secure by Default** - Sandboxed execution, resource limits
4. **Voice-First** - Built-in voice capabilities
5. **Agent Native** - Think loops and tool calling
6. **Workflow Engine** - Complex DAG-based automation

## Next Steps

- [ ] L4: Web Dashboard (React + WebSocket)
- [ ] L5: Autonomous Agents (Think loops, Memory)
- [ ] L7: Cluster Manager (Consensus, Sharding)
- [ ] L8: Batch Processor (Queues, Dead letter)
- [ ] L9: Strategic Planner (RAG, Goal trees)
- [ ] L10: API Gateway (Auth, Rate limiting)

---

**Buggy AI** - The ultimate terminal-based AI platform 🚀
