# 🐛 Buggy AI - Ultimate Architecture

## 10-Layer Distributed AI Platform

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 10: API GATEWAY                        │
│         Auth │ Rate Limit │ Routes │ Load Balancing             │
├─────────────────────────────────────────────────────────────────┤
│                     LAYER 9: STRATEGIC PLANNER                   │
│       Goal Decomposition │ RAG │ Memory │ Decision Trees         │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 8: BATCH PROCESSOR                      │
│         Queue │ Retry │ Priority │ Throttling │ Dead Letter      │
├─────────────────────────────────────────────────────────────────┤
│                    LAYER 7: CLUSTER MANAGER                      │
│        Nodes │ Sharding │ Replication │ Consensus │ Discovery   │
├─────────────────────────────────────────────────────────────────┤
│                   LAYER 6: TASK ORCHESTRATOR                     │
│           DAG │ Dependencies │ Workflow │ Checkpoints │ State    │
├─────────────────────────────────────────────────────────────────┤
│                   LAYER 5: AUTONOMOUS AGENTS                     │
│        Think Loop │ Tools │ Memory │ Planning │ Self-Improve    │
├─────────────────────────────────────────────────────────────────┤
│                     LAYER 4: WEB DASHBOARD                       │
│           REST API │ WebSocket │ React UI │ Real-time Stats     │
├─────────────────────────────────────────────────────────────────┤
│                  LAYER 3: SANDBOXED RUNNER                       │
│         Docker │ gVisor │ WASM │ Resource Limits │ Isolation    │
├─────────────────────────────────────────────────────────────────┤
│                   LAYER 2: MCP PROTOCOL                          │
│       Tools │ Resources │ Prompts │ Sampling │ Sessions          │
├─────────────────────────────────────────────────────────────────┤
│                     LAYER 1: VOICE AI                            │
│         Whisper STT │ TTS │ Noise Reduction │ Wake Word          │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### CLI (`cmd/cli/`)
- Interactive terminal interface
- Voice commands
- Agent control

### Server (`cmd/server/`)
- HTTP/WebSocket API
- Cluster coordination
- Task distribution

### Agent (`cmd/agent/`)
- Autonomous worker nodes
- Task execution
- Health reporting

## Quick Start

```bash
# Run terminal UI
./buggy-ai

# Start API server
./buggy-ai server --port 8080 --cluster

# Start agent node
./buggy-ai agent --server localhost:8080

# Run voice mode
./buggy-ai voice --stt whisper --tts festival

# Execute task plan
./buggy-ai plan --file plan.yaml
```

## Configuration

```yaml
cluster:
  mode: distributed
  nodes:
    - host: localhost
      port: 8080
      role: master
    - host: node1
      port: 8080
      role: worker

voice:
  stt: whisper
  tts: festival
  wake_word: "hey buggy"

security:
  sandbox: gvisor
  memory_limit: 1GB
  cpu_limit: 2.0

agents:
  max_concurrent: 10
  timeout: 300s
  retry: 3
```
