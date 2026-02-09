package shared

import (
	"encoding/json"
	"time"
)

type TaskID string
type AgentID string
type NodeID string

type Task struct {
	ID           TaskID        `json:"id"`
	Type         string        `json:"type"`
	Payload      interface{}   `json:"payload"`
	Priority     int           `json:"priority"`
	State        TaskState     `json:"state"`
	AgentID      AgentID       `json:"agent_id,omitempty"`
	CreatedAt    time.Time     `json:"created_at"`
	StartedAt    time.Time     `json:"started_at,omitempty"`
	CompletedAt  time.Time     `json:"completed_at,omitempty"`
	Retries      int           `json:"retries"`
	MaxRetries   int           `json:"max_retries"`
	Timeout      time.Duration `json:"timeout"`
	Dependencies []TaskID      `json:"dependencies,omitempty"`
	Result       interface{}   `json:"result,omitempty"`
	Error        string        `json:"error,omitempty"`
	Metadata     Metadata      `json:"metadata,omitempty"`
}

type TaskState string

const (
	TaskPending   TaskState = "pending"
	TaskScheduled TaskState = "scheduled"
	TaskRunning   TaskState = "running"
	TaskCompleted TaskState = "completed"
	TaskFailed    TaskState = "failed"
	TaskCancelled TaskState = "cancelled"
	TaskRetrying  TaskState = "retrying"
)

type Metadata map[string]interface{}

type TaskResult struct {
	TaskID   TaskID        `json:"task_id"`
	Success  bool          `json:"success"`
	Output   interface{}   `json:"output,omitempty"`
	Error    string        `json:"error,omitempty"`
	Duration time.Duration `json:"duration"`
}

type Agent struct {
	ID           AgentID      `json:"id"`
	Name         string       `json:"name"`
	Type         string       `json:"type"`
	State        AgentState   `json:"state"`
	Capabilities []string     `json:"capabilities"`
	CurrentTask  *Task        `json:"current_task,omitempty"`
	Memory       AgentMemory  `json:"memory"`
	Metrics      AgentMetrics `json:"metrics"`
	CreatedAt    time.Time    `json:"created_at"`
	LastActive   time.Time    `json:"last_active"`
}

type AgentState string

const (
	AgentIdle     AgentState = "idle"
	AgentBusy     AgentState = "busy"
	AgentTraining AgentState = "training"
	AgentError    AgentState = "error"
	AgentOffline  AgentState = "offline"
)

type AgentMemory struct {
	ShortTerm []Message `json:"short_term"`
	LongTerm  []Memory  `json:"long_term"`
	Working   []string  `json:"working"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Tokens  int    `json:"tokens"`
}

type Memory struct {
	Type       string    `json:"type"` // episodic, semantic, procedural
	Content    string    `json:"content"`
	Embeddings []float32 `json:"embeddings,omitempty"`
	Importance float32   `json:"importance"`
	CreatedAt  time.Time `json:"created_at"`
}

type AgentMetrics struct {
	TasksCompleted int     `json:"tasks_completed"`
	TasksFailed    int     `json:"tasks_failed"`
	AvgLatency     float64 `json:"avg_latency_ms"`
	SuccessRate    float64 `json:"success_rate"`
}

type Plan struct {
	ID          string     `json:"id"`
	Goal        string     `json:"goal"`
	Steps       []PlanStep `json:"steps"`
	State       PlanState  `json:"state"`
	Progress    float64    `json:"progress"`
	CreatedAt   time.Time  `json:"created_at"`
	CompletedAt time.Time  `json:"completed_at,omitempty"`
	ParentPlan  string     `json:"parent_plan,omitempty"`
}

type PlanStep struct {
	ID          string      `json:"id"`
	Description string      `json:"description"`
	Tasks       []TaskID    `json:"tasks"`
	DependsOn   []string    `json:"depends_on"`
	State       StepState   `json:"state"`
	Result      interface{} `json:"result,omitempty"`
	Children    []Plan      `json:"children,omitempty"`
}

type PlanState string

const (
	PlanPending   PlanState = "pending"
	PlanRunning   PlanState = "running"
	PlanCompleted PlanState = "completed"
	PlanFailed    PlanState = "failed"
	PlanPaused    PlanState = "paused"
)

type StepState string

const (
	StepPending   StepState = "pending"
	StepReady     StepState = "ready"
	StepRunning   StepState = "running"
	StepCompleted StepState = "completed"
	StepFailed    StepState = "failed"
	StepSkipped   StepState = "skipped"
)

type Node struct {
	ID            NodeID       `json:"id"`
	Host          string       `json:"host"`
	Port          int          `json:"port"`
	Role          NodeRole     `json:"role"`
	State         NodeState    `json:"state"`
	Capacity      NodeCapacity `json:"capacity"`
	Metrics       NodeMetrics  `json:"metrics"`
	LastHeartbeat time.Time    `json:"last_heartbeat"`
}

type NodeRole string

const (
	NodeMaster NodeRole = "master"
	NodeWorker NodeRole = "worker"
	NodeEdge   NodeRole = "edge"
)

type NodeState string

const (
	NodeOnline  NodeState = "online"
	NodeOffline NodeState = "offline"
	NodeBusy    NodeState = "busy"
	NodeError   NodeState = "error"
)

type NodeCapacity struct {
	CPUs      int `json:"cpus"`
	MemoryMB  int `json:"memory_mb"`
	StorageMB int `json:"storage_mb"`
	TasksMax  int `json:"tasks_max"`
	AgentsMax int `json:"agents_max"`
}

type NodeMetrics struct {
	CPUUsage     float64 `json:"cpu_usage"`
	MemoryUsage  float64 `json:"memory_usage"`
	TasksActive  int     `json:"tasks_active"`
	AgentsActive int     `json:"agents_active"`
	Uptime       int     `json:"uptime_seconds"`
}

type Event struct {
	Type      string      `json:"type"`
	Source    string      `json:"source"`
	Timestamp time.Time   `json:"timestamp"`
	Payload   interface{} `json:"payload"`
}

type ContextKey string

const (
	TaskContextKey  ContextKey = "task"
	AgentContextKey ContextKey = "agent"
	NodeContextKey  ContextKey = "node"
	PlanContextKey  ContextKey = "plan"
)

func (t *Task) MarshalJSON() ([]byte, error) {
	type Alias Task
	return json.Marshal(&struct {
		*Alias
	}{
		Alias: (*Alias)(t),
	})
}

func (t *Task) SetState(state TaskState) {
	t.State = state
}

func (t *Task) ShouldRetry() bool {
	return t.Retries < t.MaxRetries
}

func (a *Agent) IsAvailable() bool {
	return a.State == AgentIdle
}

func (n *Node) IsHealthy() bool {
	return n.State == NodeOnline || n.State == NodeBusy
}
