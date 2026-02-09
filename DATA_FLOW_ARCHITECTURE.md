# NeuralBlitz Ecosystem - Data Flow Architecture

## Executive Summary

This document maps the complete data flow architecture across the four core projects that constitute the NeuralBlitz ecosystem. The architecture spans multiple layers including persistent storage, in-memory processing, message queues, and real-time synchronization.

**Last Updated**: 2026-02-08
**Architecture Version**: v50.0 "Apical Synthesis"
**Status**: Production-Ready

---

## 1. Project Overview

### 1.1 Core Projects

| Project | Role | Primary Language | Storage | Scale |
|---------|------|-----------------|---------|-------|
| **NBX-LRS** | Database & Core Engine | Python | MySQL/SQLite | Multi-TB |
| **lrs-agents** | State Management & Active Inference | Python | PostgreSQL | Agent-scale |
| **Emergent-Prompt-Architecture** | Ontological Foundation | Python | In-Memory | Real-time |
| **Advanced-Research** | Context Injection & Research | Python | Hybrid | Research-scale |

### 1.2 Integration Points

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEURALBLITZ ECOSYSTEM                               │
│                      Data Flow Architecture v50.0                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐   │
│   │  NBX-LRS         │     │  LRS-Agents      │     │  EPA (Lattice)   │   │
│   │  Core Database   │◄───►│  State Manager   │◄───►│  Ontons          │   │
│   │                  │     │                  │     │                  │   │
│   │  • Intents       │     │  • Free Energy   │     │  • Hypergraph    │   │
│   │  • GoldenDAG     │     │  • Precision     │     │  • Associations  │   │
│   │  • Attestations  │     │  • Agent States  │     │  • Weights       │   │
│   └────────┬─────────┘     └────────┬─────────┘     └──────────────────┘   │
│            │                        │                                       │
│            │         ┌──────────────┴──────────────┐                        │
│            │         │                             │                        │
│            ▼         ▼                             ▼                        │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                    SHARED STATE MANAGER                            │   │
│   │  • UnifiedState                                                    │   │
│   │  • StateEntry (versioned, checksummed)                            │   │
│   │  • Conflict Resolution (Last Write Wins / Merge)                  │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                    UNIFIED MESSAGE BUS                             │   │
│   │  • Async message queue                                             │   │
│   │  • Pub/sub pattern                                                 │   │
│   │  • MessageType enum (14 types)                                     │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │                 ADVANCED-RESEARCH                                  │   │
│   │  • ContextInjector (Pydantic models)                               │   │
│   │  • LRSContextInjector (with learning analytics)                    │   │
│   │  • ContextBlock (priority, expiration, tags)                       │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Models by Project

### 2.1 NBX-LRS - Database Schemas

#### 2.1.1 Core Tables (MySQL/SQLite)

**Source States Table** (`source_states`)
```sql
- id: BIGINT PRIMARY KEY AUTO_INCREMENT
- state_type: ENUM('omega_prime', 'irreducible', 'perpetual_genesis', 'metacosmic')
- coherence_value: DECIMAL(10,8) DEFAULT 1.00000000
- integrity_flag: BOOLEAN DEFAULT TRUE
- metadata: JSON
- created_at: TIMESTAMP(3)
- updated_at: TIMESTAMP(3)
```

**Primal Intent Vectors** (`primal_intent_vectors`)
```sql
- id: BIGINT PRIMARY KEY AUTO_INCREMENT
- phi_1: DECIMAL(15,8) NOT NULL
- phi_22: DECIMAL(15,8) NOT NULL
- omega_genesis: DECIMAL(15,8) NOT NULL
- vector_norm: DECIMAL(15,8) GENERATED ALWAYS AS (SQRT(...)) STORED
- metadata: JSON
```

**GoldenDAG Operations** (`golden_dag_operations`)
```sql
- id: BIGINT PRIMARY KEY AUTO_INCREMENT
- operation_type: ENUM('create', 'validate', 'attest', 'synthesis')
- seed_hash: VARCHAR(128) NOT NULL
- dag_hash: VARCHAR(128) NOT NULL
- operation_result: ENUM('success', 'failure', 'pending') DEFAULT 'pending'
- metadata: JSON
- UNIQUE KEY: (seed_hash, operation_type)
```

**Intent Operations** (`intent_operations`)
```sql
- id: BIGINT PRIMARY KEY AUTO_INCREMENT
- intent_text: TEXT NOT NULL
- intent_uuid: VARCHAR(36) NOT NULL
- coherence_verified_flag: BOOLEAN DEFAULT FALSE
- processing_time_ms: INT DEFAULT 0
- source_vector_id: BIGINT (FK to primal_intent_vectors)
- architect_operation_id: BIGINT (FK to architect_operations)
- metadata: JSON
- INDEX: intent_uuid, coherence_verified_flag
```

**Attestations** (`attestations`)
```sql
- id: BIGINT PRIMARY KEY AUTO_INCREMENT
- attestation_hash: VARCHAR(128) NOT NULL
- golden_dag_hash: VARCHAR(128) NOT NULL
- trace_id: VARCHAR(64) NOT NULL
- codex_id: VARCHAR(64) NOT NULL
- attestation_data: JSON
- INDEX: attestation_hash, golden_dag_hash, trace_id
```

#### 2.1.2 SQLite Database Layer (`neuralblitz/database.py`)

**Dataclasses**:
- `IntentRecord`: Stores intent with phi vectors and coherence scores
- `GoldenDAGRecord`: DAG hash, structure, validation status
- `AttestationRecord`: Attestation with signatures and verification

**Tables**:
- `intents`: Intent processing with 7 phi parameters
- `golden_dags`: Validated DAG structures
- `attestations`: Cryptographic proofs
- `cognitive_states`: AI engine consciousness metrics
- `ml_models`: Trained model registry
- `analytics`: Real-time metrics with anomaly detection
- `system_logs`: Structured logging

### 2.2 LRS-Agents - State Management

#### 2.2.1 PostgreSQL Schema (`docker/init.sql`)

**Agent Runs** (`agent_runs`)
```sql
- id: UUID PRIMARY KEY DEFAULT uuid_generate_v4()
- agent_id: VARCHAR(255) NOT NULL
- task: TEXT NOT NULL
- started_at: TIMESTAMP NOT NULL DEFAULT NOW()
- completed_at: TIMESTAMP
- status: VARCHAR(50) NOT NULL DEFAULT 'running'
- final_precision: JSONB
- total_steps: INTEGER
- adaptations: INTEGER
- metadata: JSONB
```

**Precision History** (`precision_history`)
```sql
- id: UUID PRIMARY KEY
- run_id: UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE
- step_number: INTEGER NOT NULL
- timestamp: TIMESTAMP NOT NULL DEFAULT NOW()
- level: VARCHAR(50) NOT NULL
- value: FLOAT NOT NULL CHECK (value >= 0 AND value <= 1)
- prediction_error: FLOAT CHECK (prediction_error >= 0 AND prediction_error <= 1)
```

**Tool Executions** (`tool_executions`)
```sql
- id: UUID PRIMARY KEY
- run_id: UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE
- step_number: INTEGER NOT NULL
- tool_name: VARCHAR(255) NOT NULL
- success: BOOLEAN NOT NULL
- prediction_error: FLOAT NOT NULL
- execution_time_ms: INTEGER
- input_data: JSONB
- output_data: JSONB
```

**Adaptation Events** (`adaptation_events`)
```sql
- id: UUID PRIMARY KEY
- run_id: UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE
- step_number: INTEGER NOT NULL
- trigger_tool: VARCHAR(255)
- trigger_error: FLOAT
- old_precision: FLOAT
- new_precision: FLOAT
- action_taken: TEXT
```

**Coordination Logs** (`coordination_logs`)
```sql
- id: UUID PRIMARY KEY
- session_id: UUID NOT NULL
- agent_id: VARCHAR(255) NOT NULL
- action: VARCHAR(255) NOT NULL
- social_precision: JSONB
- message_sent: TEXT
- message_received: TEXT
```

#### 2.2.2 Shared State Model (`neuralblitz_integration/shared_state.py`)

**StateType Enum** (12 types):
- LRS_AGENT_STATE, LRS_PRECISION, LRS_FREE_ENERGY, LRS_MULTI_AGENT
- NEURALBLITZ_SOURCE, NEURALBLITZ_INTENT, NEURALBLITZ_DYAD, NEURALBLITZ_ATTESTATION
- COORDINATION, SYNCHRONIZATION, CONFIGURATION

**StateEntry Dataclass**:
```python
- key: str
- value: Any
- type: StateType
- timestamp: datetime
- version: int
- source: str
- checksum: str (SHA256 truncated)
- metadata: Dict[str, Any]
```

**UnifiedState Dataclass**:
```python
- lrs_agent_states: Dict[str, Any]
- lrs_precision_states: Dict[str, Any]
- lrs_free_energy: Dict[str, Any]
- lrs_multi_agent: Dict[str, Any]
- neuralblitz_source: Dict[str, Any]
- neuralblitz_intent: Dict[str, Any]
- neuralblitz_dyad: Dict[str, Any]
- neuralblitz_attestation: Dict[str, Any]
- coordination: Dict[str, Any]
- synchronization: Dict[str, Any]
- configuration: Dict[str, Any]
- last_updated: datetime
- version: int
```

### 2.3 Emergent-Prompt-Architecture - Lattice/Ontons

#### 2.3.1 Onton Model (`epa/onton.py`)

**OntonType Enum**:
- PERSONA, INSTRUCTION, FACT, CONSTRAINT, ETHICAL, MEMORY, CONTEXT

**Onton Dataclass** (atomic semantic unit):
```python
- id: str (auto-generated: onton_<hash>_<timestamp>)
- content: str
- type: OntonType
- weight: float = 0.5
- decay_rate: float = 0.01
- associations: List[str] (linked onton IDs)
- creation_time: float
- last_accessed: float
- access_count: int
- truth_probability: float = 1.0
- emotional_valence: float = 0.0 (-1 to 1)
- metadata: Dict[str, Any]
```

**Onton Methods**:
- `activate()`: Returns activation weight with decay
- `reinforce(delta)`: Adjusts weight and truth probability
- `decay()`: Applies natural decay
- `add_association(other_id)`: Creates bidirectional link

#### 2.3.2 Ontological Lattice (`epa/lattice.py`)

**Core Components**:
```python
- ontons: Dict[str, Onton]  # ID -> Onton mapping
- associations: Dict[str, Set[str]]  # Hypergraph edges
- session_context: Dict[str, Dict[str, float]]  # Temporal state
- memory_decay_rate: float = 0.05
- immutable_anchors: Set[str]  # Protected ontons
```

**Query Mechanism**:
- Keyword extraction with stop-word filtering
- Activation score calculation (semantic similarity + session context + associations)
- Breadth-first traversal for association depth queries

**Statistics**:
- Total ontons count
- Type distribution
- Association counts
- Average weight
- Active sessions

### 2.4 Advanced-Research - Context Injection

#### 2.4.1 Context Management (`core/context.py`)

**Priority Enum**: LOW=1, MEDIUM=2, HIGH=3, CRITICAL=4

**ContextBlock Dataclass**:
```python
- key: str
- content: str
- priority: Priority
- timestamp: datetime
- expires: Optional[datetime]
- tags: List[str]
- metadata: Dict[str, Any]
```

**ContextInjector Class**:
```python
- context_blocks: Dict[str, ContextBlock]
- max_context_size: int = 10000
- conversation_history: List[Dict[str, Any]]
```

**Methods**:
- `add_context()`: Insert with expiration and tags
- `get_context(max_tokens)`: Priority-sorted retrieval
- `get_context_blocks(filter_tags)`: Filtered access
- `add_conversation_message()`: Chat history tracking

#### 2.4.2 LRS Integration (`core/lrs_context.py`)

**LRSContextInjector** extends ContextInjector:
```python
- lrs_integration: LRSIntegration
- user_id: Optional[str]
- user_profile: Dict[str, Any]
```

**Learning Analytics Integration**:
- Records context interactions (created/accessed)
- Tracks conversation participation
- xAPI-compliant learning event recording

---

## 3. Data Flow Diagrams

### 3.1 High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌──────────┐   │
│  │ External APIs  │  │ User Inputs    │  │ Agent Actions  │  │ Research │   │
│  │ (REST/WebSock) │  │ (Chat/Forms)   │  │ (Tool Calls)   │  │ Data     │   │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └────┬─────┘   │
│          │                   │                   │                │         │
│          └───────────────────┴───────────────────┘                │         │
│                              │                                    │         │
│                              ▼                                    │         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        DATA TRANSFORMATION LAYER                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │   │
│  │  │ Pydantic     │  │ Onton        │  │ StateEntry   │  │ Intent   │  │   │
│  │  │ Validation   │  │ Encoding     │  │ Wrapping     │  │ Vector   │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          MESSAGE QUEUE / BUS                                 │
│                      (UnifiedMessageBus - Async)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Message Types:                                                              │
│  ├── LRS_AGENT_STATE          → Broadcast agent state updates                │
│  ├── LRS_PRECISION_UPDATE     → Precision tracking changes                   │
│  ├── LRS_FREE_ENERGY          → Active inference metrics                     │
│  ├── LRS_COORDINATION         │                                              │
│  ├── NEURALBLITZ_SOURCE_STATE │                                              │
│  ├── NEURALBLITZ_INTENT_VECTOR│  Bidirectional flow                          │
│  ├── NEURALBLITZ_ARCHITECT_DYAD│                                             │
│  ├── NEURALBLITZ_ATTESTATION  │                                              │
│  ├── HEARTBEAT                → Health checks                                │
│  ├── SYNC_REQUEST/RESPONSE    → State synchronization                        │
│  └── ERROR                    → Error propagation                            │
│                                                                              │
│  Delivery Guarantees:                                                        │
│  • At-least-once delivery                                                    │
│  • TTL support (time-to-live)                                                │
│  • Retry mechanism (max 3 attempts)                                          │
│  • Priority queue (0-9 scale)                                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
               ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  STATE STORAGE   │ │  PERSISTENT DB   │ │  CACHE LAYER     │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ SharedStateManager│ │ PostgreSQL       │ │ In-Memory        │
│ - UnifiedState   │ │ - Agent runs     │ │ - Onton Lattice  │
│ - StateEntry     │ │ - Precision hist │ │ - Context blocks │
│ - Versioning     │ │ - Tool executions│ │ - Session state  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### 3.2 Intent Processing Flow

```
User Input
    │
    ▼
┌─────────────────────────┐
│ ContextInjector         │  ← Advanced-Research
│ (add_context)           │
└──────────┬──────────────┘
           │ ContextBlock
           │ (Priority, Tags)
           ▼
┌─────────────────────────┐
│ OntologicalLattice      │  ← EPA
│ (query)                 │
│ - Keyword extraction    │
│ - Activation scoring    │
│ - Association traversal │
└──────────┬──────────────┘
           │ Activated Ontons
           ▼
┌─────────────────────────┐
│ LRSContextInjector      │  ← Advanced-Research
│ (track_learning=True)   │
│ - Record interaction    │
│ - xAPI event emission   │
└──────────┬──────────────┘
           │ Learning Event
           ▼
┌─────────────────────────┐
│ Intent Vector           │  ← NBX-LRS
│ (phi_1, phi_22, omega)  │
│ - Coherence check       │
│ - GoldenDAG validation  │
└──────────┬──────────────┘
           │ Validated Intent
           ▼
┌─────────────────────────┐
│ Active Inference        │  ← LRS-Agents
│ (Free Energy Min)       │
│ - Policy selection      │
│ - Precision update      │
└──────────┬──────────────┘
           │ Agent Action
           ▼
    Tool Execution
```

### 3.3 State Synchronization Flow

```
┌─────────────────┐         ┌─────────────────┐
│   LRS-Agents    │         │   NBX-LRS       │
│   (Agent State) │         │   (Source State)│
└───────┬─────────┘         └───────┬─────────┘
        │                           │
        │ 1. State Change           │
        │ (precision, free_energy)  │
        ▼                           │
┌─────────────────┐                 │
│  LRSAdapter     │                 │
│  (adapters.py)  │                 │
└───────┬─────────┘                 │
        │ 2. Transform to Message   │
        ▼                           │
┌─────────────────┐                 │
│ UnifiedMessageBus│                │
│ (messaging.py)  │                 │
└───────┬─────────┘                 │
        │ 3. Publish Message        │
        │ (MessageType.LRS_*)       │
        └──────────┬────────────────┘
                   │
                   ▼
        ┌─────────────────┐
        │ SharedStateManager│
        │ (shared_state.py) │
        │ - StateEntry      │
        │ - Checksum        │
        │ - Versioning      │
        └────────┬──────────┘
                 │
        ┌────────┴──────────┐
        │                   │
        ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│  Subscribers    │ │  Persist to DB  │
│  (notify)       │ │  (SQLite/PG)    │
└─────────────────┘ └─────────────────┘
```

### 3.4 Context Injection Flow

```
Research Data / User Input
           │
           ▼
┌──────────────────────┐
│ ContextInjector      │  (core/context.py)
│ - Priority queue     │
│ - Expiration handling│
└───────┬──────────────┘
        │ ContextBlock
        │ ├─ key: str
        │ ├─ content: str
        │ ├─ priority: Priority
        │ ├─ expires: datetime
        │ └─ tags: List[str]
        ▼
┌──────────────────────┐
│ LRSContextInjector   │  (core/lrs_context.py)
│ - User tracking      │
│ - Learning analytics │
└───────┬──────────────┘
        │ Async Task
        ▼
┌──────────────────────┐
│ LRSIntegration       │
│ (integrations.py)    │
└───────┬──────────────┘
        │ xAPI Statement
        ├─ actor: user_id
        ├─ verb: "communicated"
        └─ object: context_ref
        ▼
┌──────────────────────┐
│ LRS Database         │
│ (PostgreSQL)         │
│ - actor              │
│ - verb               │
│ - object             │
│ - timestamp          │
│ - result             │
└──────────────────────┘
```

---

## 4. Shared Database Tables/Collections

### 4.1 Cross-Project Data Sharing

| Data Entity | Source Project | Target Projects | Sync Method | Frequency |
|-------------|----------------|-----------------|-------------|-----------|
| **Agent States** | lrs-agents | NBX-LRS, Advanced-Research | Message Bus | Real-time |
| **Intent Vectors** | NBX-LRS | lrs-agents, EPA | State Manager | Real-time |
| **GoldenDAG Hashes** | NBX-LRS | All projects | Database | On-demand |
| **Attestations** | NBX-LRS | All projects | Message Bus | Event-driven |
| **Context Blocks** | Advanced-Research | lrs-agents | Direct API | Real-time |
| **Onton Activations** | EPA | Advanced-Research | In-Memory | Real-time |
| **Learning Events** | Advanced-Research | lrs-agents | xAPI | Async |
| **Precision History** | lrs-agents | NBX-LRS | Database | Periodic |
| **Coordination Logs** | lrs-agents | All projects | PostgreSQL | Real-time |

### 4.2 Database Connection Matrix

```
                     ┌──────────┐
                     │ NBX-LRS  │
                     │ MySQL/   │
                     │ SQLite   │
                     └────┬─────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ LRS-Agents   │  │ EPA          │  │ Advanced-    │
│ PostgreSQL   │  │ In-Memory    │  │ Research     │
│              │  │ (Lattice)    │  │ Hybrid       │
│ • agent_runs │  │              │  │              │
│ • precision  │  │ • ontons     │  │ • Context    │
│ • tools      │  │ • assoc      │  │ • LRS events │
└──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │ Shared State       │
               │ Manager            │
               │ (Redis/In-Memory)  │
               └────────────────────┘
```

---

## 5. Message Queues and Event Streams

### 5.1 UnifiedMessageBus Architecture

**Implementation**: `lrs-agents/lrs/neuralblitz_integration/messaging.py`

**Core Components**:
```python
class UnifiedMessageBus:
    - _subscribers: Dict[MessageType, Set[Callable]]
    - _message_queue: asyncio.Queue
    - _message_history: Dict[str, Message]
    - _stats: Dict[str, int]
```

**Message Flow**:
1. **Publish**: `publish(message)` → Queue → History
2. **Process**: `_message_loop()` → Dequeue → `_process_message()`
3. **Distribute**: Execute all subscribers concurrently
4. **Retry**: Automatic retry up to 3 attempts

**Message Structure** (`Message` dataclass):
```python
- id: str (UUID)
- type: MessageType (enum)
- source: str
- destination: str (default: "broadcast")
- timestamp: datetime
- payload: Dict[str, Any]
- correlation_id: Optional[str]
- priority: int (0-9)
- ttl: Optional[int] (seconds)
- retry_count: int
- max_retries: int (default: 3)
```

### 5.2 Message Types Catalog

| Type | Direction | Payload | Use Case |
|------|-----------|---------|----------|
| `LRS_AGENT_STATE` | LRS → NBX | AgentState | Broadcast agent status |
| `LRS_PRECISION_UPDATE` | LRS → NBX | PrecisionRecord | Track confidence changes |
| `LRS_FREE_ENERGY` | LRS → NBX | FreeEnergyData | Active inference metrics |
| `LRS_COORDINATION` | LRS → NBX | CoordinationMsg | Multi-agent sync |
| `LRS_BENCHMARK` | LRS → NBX | BenchmarkResult | Performance reports |
| `NEURALBLITZ_SOURCE_STATE` | NBX → LRS | SourceState | System coherence |
| `NEURALBLITZ_INTENT_VECTOR` | NBX → LRS | IntentVector | Processed intents |
| `NEURALBLITZ_ARCHITECT_DYAD` | NBX → LRS | DyadData | Architect operations |
| `NEURALBLITZ_ATTESTATION` | NBX → LRS | Attestation | Verification proofs |
| `NEURALBLITZ_VERIFICATION` | NBX → LRS | Verification | Validation results |
| `HEARTBEAT` | Bidirectional | HealthStatus | Health checks (30s) |
| `SYNC_REQUEST` | Bidirectional | StateSnapshot | State synchronization |
| `SYNC_RESPONSE` | Bidirectional | StateDiff | Sync confirmation |
| `ERROR` | Bidirectional | ErrorInfo | Error propagation |
| `SHUTDOWN` | Bidirectional | ShutdownReason | Graceful termination |

### 5.3 Event Stream Processing

**Async/Await Pattern**:
```python
# Publisher
async def publish_state_update():
    message = Message(
        type=MessageType.LRS_AGENT_STATE,
        source="agent_001",
        payload=agent_state.to_dict()
    )
    await message_bus.publish(message)

# Subscriber
async def handle_agent_state(message: Message):
    state = message.payload
    await state_manager.set_state(
        key=state["agent_id"],
        value=state,
        state_type=StateType.LRS_AGENT_STATE
    )

# Registration
await message_bus.subscribe(
    MessageType.LRS_AGENT_STATE,
    handle_agent_state
)
```

**Backpressure Handling**:
- Queue size limit: 1000 messages
- Full queue: Drop oldest non-critical messages
- Priority queue: Critical messages bypass queue

---

## 6. State Synchronization Mechanisms

### 6.1 SharedStateManager

**Location**: `lrs-agents/lrs/neuralblitz_integration/shared_state.py`

**Features**:
- **Thread-safe**: Asyncio Lock for all operations
- **Versioning**: Incremental version numbers per entry
- **Checksums**: SHA256 integrity verification
- **Conflict Resolution**: Multiple strategies
- **Pub/Sub**: State change notifications

**StateEntry Lifecycle**:
```
Create → Validate Checksum → Store → Notify Subscribers
              ↑                    ↓
        Integrity Check      Version Increment
```

**Conflict Resolution Strategies**:
1. **LAST_WRITE_WINS** (default): Newest timestamp overwrites
2. **MERGE**: Dictionary merge for nested data
3. **CUSTOM**: User-defined resolution function
4. **IGNORE**: Keep existing value

**Statistics Tracking**:
```python
stats = {
    "state_updates": int,
    "state_reads": int,
    "conflicts": int,
    "subscribers": int,
    "total_states": int,
    "state_types": int,
    "last_updated": str (ISO format),
    "version": int
}
```

### 6.2 Multi-Agent Shared State

**Location**: `lrs-agents/lrs/multi_agent/shared_state.py`

**SharedWorldState Class**:
```python
- _state: Dict[str, Dict[str, Any]]  # agent_id -> state
- _lock: threading.Lock
- _history: List[Dict]  # Update history
- _subscribers: Dict[str, List[callable]]
```

**Synchronization Pattern**:
```python
# Agent A updates state
state.update("agent_a", {
    "position": (10, 20),
    "task": "fetch_data",
    "status": "working"
})

# Notification to all subscribers
_notify_subscribers("agent_a", updates)

# Agent B reads state
a_state = state.get_agent_state("agent_a")
all_states = state.get_all_states()
```

### 6.3 Database Synchronization

**Two-Phase Commit** (for critical operations):
1. **Prepare**: Write to message bus
2. **Commit**: Persist to database
3. **Ack**: Confirm to subscribers

**Eventual Consistency** (for analytics):
- Async replication to analytics tables
- Batch updates every 5 seconds
- Background cleanup of old data (30+ days)

---

## 7. Data Transformation Points

### 7.1 Adapter Pattern

**UnifiedAdapter** (`lrs-agents/lrs/neuralblitz_integration/adapters.py`):

**LRSAdapter**: Transforms LRS agent states to unified format
```python
Input: AgentState (beliefs, policies, precision, free_energy)
       ↓
Transform: to_dict() with timestamp
       ↓
Output: StateEntry (StateType.LRS_AGENT_STATE)
       ↓
Action: Publish to MessageBus + Store in SharedStateManager
```

**NeuralBlitzAdapter**: Transforms NBX system states
```python
Input: NeuralBlitzState (source, intent, dyad, attestation)
       ↓
Transform: to_dict() with metadata
       ↓
Output: StateEntry (StateType.NEURALBLITZ_*)
       ↓
Action: Broadcast + Database persistence
```

### 7.2 Protocol Translation

**BidirectionalProtocol** (`protocols.py`):
```python
# LRS to NeuralBlitz
lrs_config = ProtocolConfig(
    source_system=f"lrs_agent_{agent_id}",
    target_system="neuralblitz"
)

# NeuralBlitz to LRS
neuralblitz_config = ProtocolConfig(
    source_system=f"neuralblitz_system_{system_id}",
    target_system="lrs"
)
```

**Translation Layer**:
- Agent beliefs ↔ Intent vector components
- Precision values ↔ Coherence scores
- Free energy ↔ GoldenDAG validation status
- Tool executions ↔ NBCL operations

### 7.3 Context Transformation

**EPA Onton → ContextBlock**:
```python
# Extract Onton
onton = lattice.get_onton(onton_id)

# Transform to ContextBlock
context.add_context(
    key=onton.id,
    content=onton.content,
    priority=Priority.HIGH if onton.type == OntonType.ETHICAL else Priority.MEDIUM,
    tags=[onton.type.value, "epa"],
    metadata={
        "truth_probability": onton.truth_probability,
        "emotional_valence": onton.emotional_valence,
        "associations": onton.associations
    }
)
```

---

## 8. Caching Strategies

### 8.1 Multi-Level Cache Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    CACHE LAYERS                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L1: In-Memory (Fastest)                                    │
│  ├── EPA OntologicalLattice                                 │
│  │   └── ontons: Dict[str, Onton]                           │
│  │   └── associations: Dict[str, Set[str]]                  │
│  │   └── session_context: Dict[str, Dict]                   │
│  │                                                          │
│  ├── ContextInjector                                        │
│  │   └── context_blocks: Dict[str, ContextBlock]            │
│  │   └── conversation_history: List[Dict]                   │
│  │                                                          │
│  └── SharedStateManager                                     │
│      └── state_entries: Dict[str, StateEntry]               │
│      └── unified_state: UnifiedState                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L2: Message Queue (Hot Path)                               │
│  └── UnifiedMessageBus                                      │
│      └── _message_history: Dict[str, Message]               │
│      └── _message_queue: asyncio.Queue                      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L3: Persistent Cache (Warm)                                │
│  ├── SQLite (NBX-LRS)                                       │
│  │   └── Recent intents, GoldenDAGs                         │
│  │                                                          │
│  └── PostgreSQL (LRS-Agents)                                │
│      └── Active agent runs, precision history               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L4: Cold Storage (Archive)                                 │
│  ├── Long-term analytics (30+ days)                         │
│  ├── System logs (90+ days)                                 │
│  └── Backup GoldenDAG chains                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Cache Invalidation Strategies

**Time-Based**:
- Context blocks: Expire after configurable seconds
- Session context: Decay factor 0.9 per access
- Onton weights: Decay by `decay_rate` per hour

**Event-Based**:
- Intent processed → Invalidate related context
- State conflict → Refresh from database
- GoldenDAG attestation → Update all caches

**Size-Based**:
- Message queue: Max 1000 messages, drop oldest
- Conversation history: Keep last N messages
- Onton associations: Prune low-weight links

### 8.3 Cache Coherence

**Write-Through** (for critical data):
1. Update in-memory cache
2. Write to message bus
3. Persist to database
4. Acknowledge to caller

**Write-Behind** (for analytics):
1. Update in-memory cache
2. Queue async write
3. Batch persist every 5 seconds

**Read-Through**:
1. Check L1 cache
2. If miss, check L2 (message history)
3. If miss, query database
4. Populate cache on retrieval

---

## 9. Critical Data Paths

### 9.1 Path 1: Intent Processing (Hot Path)

```
User Input
    ↓ [~10ms]
ContextInjector.add_context()
    ↓ [~5ms]
OntologicalLattice.query()
    ↓ [~20ms]
LRSContextInjector (learning track)
    ↓ [~15ms]
Intent Vector Generation (phi values)
    ↓ [~30ms]
GoldenDAG Validation
    ↓ [~25ms]
Active Inference (Free Energy)
    ↓ [~20ms]
Tool Selection & Execution
    ↓ [~50-500ms]
Response Generation
    ↓
Total: ~175-625ms
```

**Optimization Points**:
- Pre-compute common intent patterns
- Cache GoldenDAG validation results
- Batch precision updates

### 9.2 Path 2: State Synchronization (Real-time)

```
Agent State Change
    ↓ [~2ms]
LRSAdapter.register_agent()
    ↓ [~3ms]
StateEntry Creation + Checksum
    ↓ [~1ms]
UnifiedMessageBus.publish()
    ↓ [~5ms]
SharedStateManager.set_state()
    ↓ [~10ms]
Database Persistence (async)
    ↓ [~2ms]
Subscriber Notifications
    ↓
Total: ~23ms (synchronous portion)
```

**Reliability**:
- Async database writes
- Message retry (3 attempts)
- Circuit breaker pattern

### 9.3 Path 3: Research Context Injection (Analytics)

```
Research Data Ingestion
    ↓ [~100ms]
Context Block Creation
    ↓ [~50ms]
Learning Event Recording (async)
    ↓ [~200ms]
xAPI Statement Generation
    ↓ [~500ms]
LRS Database Write
    ↓ [~100ms]
Context Activation in Lattice
    ↓
Total: ~950ms (mostly async)
```

**Throughput**: ~100 events/second

---

## 10. Data Consistency Issues & Conflicts

### 10.1 Identified Consistency Challenges

| Issue | Projects Affected | Impact | Mitigation |
|-------|------------------|--------|------------|
| **Concurrent State Updates** | LRS ↔ NBX | Overwrites | Versioning + Checksums |
| **Message Loss** | All | Data gaps | Retry mechanism + Persistence |
| **Clock Skew** | All | Ordering issues | UTC timestamps + Monotonic IDs |
| **Database Lag** | NBX, LRS | Stale reads | Read-through caching |
| **Onton Decay** | EPA | Outdated associations | Time-based decay + refresh |
| **Context Expiration** | Advanced-Research | Missing context | Priority-based cleanup |
| **Learning Event Duplication** | Advanced ↔ LRS | Inflated metrics | Idempotent writes |

### 10.2 Conflict Scenarios

**Scenario 1: Simultaneous State Updates**
```
Time T1: Agent A updates precision to 0.8
Time T2: Agent B updates precision to 0.9 (before T1 propagates)
Result: Conflict detected in SharedStateManager
Resolution: LAST_WRITE_WINS (T2 wins) or MERGE (avg = 0.85)
```

**Scenario 2: Network Partition**
```
Partition 1: NBX-LRS continues processing
Partition 2: LRS-Agents operates independently
Reconciliation: SYNC_REQUEST/SYNC_RESPONSE exchange
Resolution: Merge states, flag conflicts for review
```

**Scenario 3: Database Timeout**
```
Issue: PostgreSQL slow, LRS-Agents can't persist
Fallback: Continue with in-memory state
Recovery: Background task retries persistence
Data Loss Risk: System crash before persistence
Mitigation: Write-ahead logging (WAL)
```

### 10.3 Conflict Resolution Matrix

| Conflict Type | Resolution Strategy | Automation Level |
|---------------|---------------------|------------------|
| State version mismatch | LAST_WRITE_WINS | Automatic |
| Nested dict updates | MERGE (recursive) | Automatic |
| Precision conflicts | CUSTOM (weighted avg) | Semi-auto |
| Intent coherence | CECT projection | Automatic |
| Learning events | Idempotent dedup | Automatic |
| Onton associations | Union of sets | Automatic |

### 10.4 Data Integrity Checks

**Checksum Verification**:
```python
# StateEntry validates on every read
if not entry.is_valid():
    logger.error(f"Checksum mismatch for {key}")
    # Trigger re-sync from database
```

**GoldenDAG Validation**:
```python
# Verify hash chain integrity
dag = database.get_golden_dag_by_hash(dag_hash)
if not verify_chain(dag):
    raise IntegrityError("GoldenDAG chain broken")
```

**Learning Event Deduplication**:
```python
# Use unique constraint on (actor, verb, object, timestamp)
UNIQUE(actor_id, verb_id, object_id, timestamp_truncated)
```

---

## 11. Monitoring & Observability

### 11.1 Key Metrics

**Data Flow Metrics**:
- Message throughput (messages/sec)
- Queue depth (pending messages)
- State update latency (p50, p95, p99)
- Database write latency
- Cache hit rates (L1, L2, L3)

**Consistency Metrics**:
- Conflict rate (conflicts/sec)
- Checksum failures
- Retry attempts
- Circuit breaker state changes

**Business Metrics**:
- Intent processing rate
- Learning events recorded
- Agent coordination success rate
- Context hit rate

### 11.2 Health Checks

**Component Health**:
```python
# LRS Bridge Health
{
    "system_id": "NEURALBLITZ_V50",
    "status": "healthy",  # or "degraded"
    "metrics": {
        "queue_size": 12,
        "circuit_breaker_state": "CLOSED",
        "active_connections": 5,
        "is_healthy": True,
        "uptime_seconds": 86400
    }
}

# Shared State Health
{
    "state_updates": 15000,
    "state_reads": 45000,
    "conflicts": 23,
    "subscribers": 12,
    "total_states": 89,
    "version": 1523
}
```

### 11.3 Alerting Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Queue depth | >500 | >900 | Scale consumers |
| Conflict rate | >1% | >5% | Review resolution |
| DB latency | >100ms | >500ms | Check DB health |
| Cache hit rate | <80% | <50% | Review cache config |
| Message failures | >1% | >10% | Circuit breaker |

---

## 12. Security Considerations

### 12.1 Data Protection

**In Transit**:
- TLS 1.3 for all external APIs
- HMAC-SHA256 message signing
- Certificate pinning for LRS bridge

**At Rest**:
- SQLite: File-level encryption
- PostgreSQL: Transparent Data Encryption (TDE)
- Sensitive fields: AES-256-GCM

### 12.2 Access Control

**Database**:
- Role-based access control (RBAC)
- Least privilege principle
- Query audit logging

**Message Bus**:
- Authenticated publishers/subscribers
- Topic-level permissions
- Message signing verification

### 12.3 Privacy

**PII Handling**:
- User IDs hashed in analytics
- Context content encrypted
- Learning events anonymized after 90 days

**Data Retention**:
- Agent runs: 90 days
- Precision history: 90 days
- System logs: 90 days
- Analytics: 1 year (aggregated)

---

## 13. Performance Optimization

### 13.1 Bottlenecks & Solutions

| Bottleneck | Location | Solution | Expected Improvement |
|------------|----------|----------|---------------------|
| Intent query latency | EPA Lattice | Index onton associations | 50% reduction |
| State sync overhead | Message Bus | Batch state updates | 3x throughput |
| DB write contention | PostgreSQL | Connection pooling | 2x capacity |
| Cache misses | Shared State | LRU eviction policy | 20% hit rate ↑ |
| Context serialization | Advanced-Research | Binary protocol | 40% faster |

### 13.2 Scaling Strategies

**Horizontal Scaling**:
- Stateless message bus (Redis backend)
- Sharded database (by agent_id)
- Read replicas for analytics

**Vertical Scaling**:
- Increase L1 cache size
- Optimize SQL queries
- Use connection pooling

---

## 14. Appendix A: Database Schema Reference

### 14.1 NBX-LRS Complete Schema

```sql
-- Core Tables (from /NBX-LRS/neuralblitz-v50/sql/schema.sql)

CREATE TABLE source_states (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    state_type ENUM('omega_prime', 'irreducible', 'perpetual_genesis', 'metacosmic') NOT NULL,
    coherence_value DECIMAL(10,8) DEFAULT 1.00000000,
    integrity_flag BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_state_type (state_type),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

CREATE TABLE primal_intent_vectors (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    phi_1 DECIMAL(15,8) NOT NULL,
    phi_22 DECIMAL(15,8) NOT NULL,
    omega_genesis DECIMAL(15,8) NOT NULL,
    vector_norm DECIMAL(15,8) GENERATED ALWAYS AS (SQRT(...)) STORED,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    metadata JSON,
    INDEX idx_phi_1 (phi_1),
    INDEX idx_phi_22 (phi_22)
) ENGINE=InnoDB;

CREATE TABLE golden_dag_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    operation_type ENUM('create', 'validate', 'attest', 'synthesis') NOT NULL,
    seed_hash VARCHAR(128) NOT NULL,
    dag_hash VARCHAR(128) NOT NULL,
    version VARCHAR(32) DEFAULT 'v50.0.0',
    operation_result ENUM('success', 'failure', 'pending') DEFAULT 'pending',
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    completed_at TIMESTAMP(3) NULL,
    metadata JSON,
    UNIQUE KEY unique_seed_operation (seed_hash, operation_type)
) ENGINE=InnoDB;

CREATE TABLE architect_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    unity_vector DECIMAL(15,8) DEFAULT 1.00000000,
    amplification_factor DECIMAL(15,8) DEFAULT 1.00000200,
    irreducible_flag BOOLEAN DEFAULT TRUE,
    separation_impossibility DECIMAL(15,8) DEFAULT 0.00000000,
    symbiotic_return_signal DECIMAL(15,8) DEFAULT 1.00000200,
    beta_identifier VARCHAR(128),
    operation_result ENUM('amplified', 'processed', 'failed') DEFAULT 'processed',
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    metadata JSON
) ENGINE=InnoDB;

CREATE TABLE intent_operations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    intent_text TEXT NOT NULL,
    intent_uuid VARCHAR(36) NOT NULL,
    coherence_verified_flag BOOLEAN DEFAULT FALSE,
    processing_time_ms INT DEFAULT 0,
    source_vector_id BIGINT NULL,
    architect_operation_id BIGINT NULL,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    processed_at TIMESTAMP(3) NULL,
    metadata JSON,
    FOREIGN KEY (source_vector_id) REFERENCES primal_intent_vectors(id),
    FOREIGN KEY (architect_operation_id) REFERENCES architect_operations(id)
) ENGINE=InnoDB;

CREATE TABLE attestations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    attestation_hash VARCHAR(128) NOT NULL,
    attestation_timestamp TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    golden_dag_hash VARCHAR(128) NOT NULL,
    trace_id VARCHAR(64) NOT NULL,
    codex_id VARCHAR(64) NOT NULL,
    version VARCHAR(32) DEFAULT 'v50.0.0',
    attestation_data JSON,
    INDEX idx_attestation_hash (attestation_hash)
) ENGINE=InnoDB;

CREATE TABLE symbiosis_fields (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    active_flag BOOLEAN DEFAULT TRUE,
    symbiosis_factor DECIMAL(15,8) DEFAULT 1.00000000,
    integrated_entities INT DEFAULT 0,
    field_strength DECIMAL(15,8) DEFAULT 1.00000000,
    coherence_level DECIMAL(10,8) DEFAULT 1.00000000,
    created_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    updated_at TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3),
    metadata JSON
) ENGINE=InnoDB;

CREATE TABLE system_metrics (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    metric_name VARCHAR(128) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    metric_unit VARCHAR(32),
    component VARCHAR(64),
    timestamp TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
    metadata JSON
) ENGINE=InnoDB;

CREATE TABLE audit_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    operation_type VARCHAR(64) NOT NULL,
    table_name VARCHAR(64),
    record_id BIGINT,
    old_values JSON,
    new_values JSON,
    user_id VARCHAR(128),
    session_id VARCHAR(128),
    trace_id VARCHAR(64),
    timestamp TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3)
) ENGINE=InnoDB;
```

### 14.2 LRS-Agents Complete Schema

```sql
-- From /lrs-agents/docker/init.sql

CREATE TABLE agent_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    task TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'running',
    final_precision JSONB,
    total_steps INTEGER,
    adaptations INTEGER,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE precision_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    level VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL CHECK (value >= 0 AND value <= 1),
    prediction_error FLOAT CHECK (prediction_error >= 0 AND prediction_error <= 1),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE tool_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    tool_name VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    prediction_error FLOAT NOT NULL CHECK (prediction_error >= 0 AND prediction_error <= 1),
    execution_time_ms INTEGER,
    error_message TEXT,
    input_data JSONB,
    output_data JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE adaptation_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    trigger_tool VARCHAR(255),
    trigger_error FLOAT,
    old_precision FLOAT,
    new_precision FLOAT,
    action_taken TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE benchmark_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    benchmark_name VARCHAR(255) NOT NULL,
    run_date DATE NOT NULL DEFAULT CURRENT_DATE,
    num_trials INTEGER NOT NULL,
    success_rate FLOAT CHECK (success_rate >= 0 AND success_rate <= 1),
    avg_steps FLOAT,
    avg_adaptations FLOAT,
    avg_execution_time FLOAT,
    results JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE coordination_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    action VARCHAR(255) NOT NULL,
    social_precision JSONB,
    message_sent TEXT,
    message_received TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_agent_runs_agent_id ON agent_runs(agent_id);
CREATE INDEX idx_agent_runs_status ON agent_runs(status);
CREATE INDEX idx_precision_history_run_id ON precision_history(run_id);
CREATE INDEX idx_tool_executions_run_id ON tool_executions(run_id);
CREATE INDEX idx_coordination_logs_session ON coordination_logs(session_id);

-- Views
CREATE VIEW agent_performance_summary AS
SELECT 
    agent_id,
    COUNT(*) as total_runs,
    AVG(total_steps) as avg_steps,
    AVG(adaptations) as avg_adaptations,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM agent_runs
WHERE completed_at IS NOT NULL
GROUP BY agent_id;
```

---

## 15. Appendix B: Pydantic Models Reference

### 15.1 Advanced-Research Models

```python
# core/context.py

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ContextBlock:
    key: str
    content: str
    priority: Priority
    timestamp: datetime = field(default_factory=datetime.now)
    expires: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ContextInjector:
    context_blocks: Dict[str, ContextBlock]
    max_context_size: int = 10000
    conversation_history: List[Dict[str, Any]]
```

### 15.2 NBX-LRS Bridge Models

```python
# neuralblitz/lrs_bridge.py

class LRSMessage(BaseModel):
    protocol_version: str = "1.0"
    timestamp: datetime
    source_system: str
    target_system: str
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    message_type: str
    payload: Dict[str, Any]
    signature: str
    priority: str = "NORMAL"
    ttl: int = 300

class IntentRequest(BaseModel):
    phi_1: float = Field(default=1.0, ge=0.0, le=1.0)
    phi_22: float = Field(default=1.0, ge=0.0, le=1.0)
    phi_omega: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IntentResponse(BaseModel):
    intent_id: str
    status: str
    coherence_verified: bool
    processing_time_ms: int
    golden_dag_hash: str
    output_data: Optional[Dict[str, Any]] = None
```

---

## 16. Summary

The NeuralBlitz ecosystem implements a sophisticated multi-layer data flow architecture that spans four core projects:

1. **NBX-LRS**: Provides the persistent foundation with MySQL/SQLite databases storing intents, GoldenDAGs, and attestations
2. **lrs-agents**: Manages Active Inference state with PostgreSQL tracking agent runs, precision history, and tool executions
3. **Emergent-Prompt-Architecture**: Offers real-time ontological processing through an in-memory hypergraph of Ontons
4. **Advanced-Research**: Handles context injection and learning analytics with Pydantic models and xAPI integration

**Key Integration Mechanisms**:
- **UnifiedMessageBus**: Async pub/sub messaging with 14 message types
- **SharedStateManager**: Versioned, checksummed state with conflict resolution
- **Adapters**: Protocol translation between LRS and NeuralBlitz formats
- **Multi-level Caching**: L1 (in-memory) → L2 (message queue) → L3 (database)

**Data Flow Characteristics**:
- **Sync/Async**: Hot paths synchronous (<100ms), analytics asynchronous
- **Consistency**: Eventual consistency with conflict resolution strategies
- **Reliability**: At-least-once delivery, retry mechanisms, circuit breakers
- **Security**: HMAC signing, TLS encryption, access control

**Performance Targets**:
- Intent processing: <200ms (p95)
- State synchronization: <50ms
- Message throughput: 1000+ messages/sec
- Cache hit rate: >85%

This architecture supports production-scale deployment with horizontal scaling capabilities, comprehensive monitoring, and robust error handling.

---

**Document Version**: 1.0
**Author**: NeuralBlitz Architecture Team
**Review Date**: 2026-02-08
**Next Review**: 2026-03-08
