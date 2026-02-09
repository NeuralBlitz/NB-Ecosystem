package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"buggy-ai/internal/shared"
)

type Agent struct {
	ID          shared.AgentID
	Name        string
	Type        string
	State       shared.AgentState
	Memory      AgentMemory
	Tools       []Tool
	Planner     *Planner
	CurrentTask *shared.Task
	mu          sync.RWMutex
	metrics     shared.AgentMetrics
	createdAt   time.Time
	lastActive  time.Time
}

type AgentMemory struct {
	ShortTerm []Message
	LongTerm  []Memory
	Working   []string
	Episodic  []Episode
}

type Message struct {
	Role    string
	Content string
	Tokens  int
}

type Memory struct {
	Type       string
	Content    string
	Importance float32
	CreatedAt  time.Time
}

type Episode struct {
	State     string
	Action    string
	Result    string
	Timestamp time.Time
}

type Tool interface {
	Name() string
	Execute(ctx context.Context, args map[string]interface{}) (interface{}, error)
}

type Planner struct {
	Goal        string
	Steps       []PlanStep
	CurrentStep int
	Depth       int
}

type PlanStep struct {
	Description string
	SubGoal     string
	Completed   bool
	Result      interface{}
}

func NewAgent(name, agentType string) *Agent {
	return &Agent{
		ID:        shared.AgentID(fmt.Sprintf("agent-%d", time.Now().UnixNano())),
		Name:      name,
		Type:      agentType,
		State:     shared.AgentIdle,
		Memory:    AgentMemory{},
		Tools:     []Tool{},
		Planner:   &Planner{},
		metrics:   shared.AgentMetrics{},
		createdAt: time.Now(),
	}
}

func (a *Agent) Think(ctx context.Context, goal string) ([]PlanStep, error) {
	a.mu.Lock()
	a.Planner.Goal = goal
	a.Planner.Steps = []PlanStep{
		{Description: "Analyze the goal", SubGoal: "analysis", Completed: false},
		{Description: "Break down into tasks", SubGoal: "decomposition", Completed: false},
		{Description: "Execute tasks sequentially", SubGoal: "execution", Completed: false},
		{Description: "Verify results", SubGoal: "verification", Completed: false},
		{Description: "Return final result", SubGoal: "completion", Completed: false},
	}
	a.Planner.CurrentStep = 0
	a.mu.Unlock()

	return a.Planner.Steps, nil
}

func (a *Agent) Execute(ctx context.Context, task *shared.Task) (*shared.TaskResult, error) {
	a.mu.Lock()
	a.State = shared.AgentBusy
	a.CurrentTask = task
	a.lastActive = time.Now()
	a.mu.Unlock()

	result := &shared.TaskResult{
		TaskID:   task.ID,
		Success:  true,
		Duration: time.Second,
	}

	a.mu.Lock()
	a.State = shared.AgentIdle
	a.CurrentTask = nil
	a.metrics.TasksCompleted++
	a.mu.Unlock()

	return result, nil
}

func (a *Agent) Remember(memory Memory) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.Memory.LongTerm = append(a.Memory.LongTerm, memory)
}

func (a *Agent) Recall(query string) []Memory {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.Memory.LongTerm
}

func (a *Agent) AddTool(tool Tool) {
	a.Tools = append(a.Tools, tool)
}

func (a *Agent) GetTool(name string) Tool {
	for _, tool := range a.Tools {
		if tool.Name() == name {
			return tool
		}
	}
	return nil
}

func (a *Agent) Learn(experience Episode) {
	a.mu.Lock()
	a.Memory.Episodic = append(a.Memory.Episodic, experience)
	if len(a.Memory.Episodic) > 100 {
		a.Memory.Episodic = a.Memory.Episodic[len(a.Memory.Episodic)-100:]
	}
	a.mu.Unlock()
}

func (a *Agent) GetMetrics() shared.AgentMetrics {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.metrics
}

func (a *Agent) IsAvailable() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.State == shared.AgentIdle
}

type AgentSystem struct {
	agents     map[shared.AgentID]*Agent
	dispatcher *Dispatcher
	mu         sync.RWMutex
}

type Dispatcher struct {
	queue   chan *shared.Task
	workers int
	mu      sync.Mutex
}

func NewAgentSystem(workers int) *AgentSystem {
	return &AgentSystem{
		agents: make(map[shared.AgentID]*Agent),
		dispatcher: &Dispatcher{
			queue:   make(chan *shared.Task, 100),
			workers: workers,
		},
	}
}

func (s *AgentSystem) RegisterAgent(agent *Agent) {
	s.mu.Lock()
	s.agents[agent.ID] = agent
	s.mu.Unlock()
}

func (s *AgentSystem) Dispatch(task *shared.Task) error {
	s.dispatcher.queue <- task
	return nil
}

func (s *AgentSystem) GetAgent(id shared.AgentID) (*Agent, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	agent, ok := s.agents[id]
	return agent, ok
}

func (s *AgentSystem) ListAgents() []*Agent {
	s.mu.RLock()
	defer s.mu.RUnlock()
	agents := make([]*Agent, 0, len(s.agents))
	for _, agent := range s.agents {
		agents = append(agents, agent)
	}
	return agents
}

func (s *AgentSystem) GetStats() AgentSystemStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var totalCompleted, totalFailed int
	available := 0

	for _, agent := range s.agents {
		m := agent.GetMetrics()
		totalCompleted += m.TasksCompleted
		totalFailed += m.TasksFailed
		if agent.IsAvailable() {
			available++
		}
	}

	return AgentSystemStats{
		TotalAgents:    len(s.agents),
		Available:      available,
		Busy:           len(s.agents) - available,
		TotalCompleted: totalCompleted,
		TotalFailed:    totalFailed,
	}
}

type AgentSystemStats struct {
	TotalAgents    int
	Available      int
	Busy           int
	TotalCompleted int
	TotalFailed    int
}

type BuiltInTool struct {
	ToolName string
}

func (t *BuiltInTool) Name() string { return t.ToolName }

func (t *BuiltInTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	return "Tool executed", nil
}

type SearchTool struct{ BuiltInTool }

func NewSearchTool() *SearchTool {
	return &SearchTool{BuiltInTool: BuiltInTool{ToolName: "search"}}
}

func (t *SearchTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	query := args["query"].(string)
	return fmt.Sprintf("Search results for: %s", query), nil
}

type ExecuteTool struct{ BuiltInTool }

func NewExecuteTool() *ExecuteTool {
	return &ExecuteTool{BuiltInTool: BuiltInTool{ToolName: "execute"}}
}

func (t *ExecuteTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
	code := args["code"].(string)
	language := args["language"].(string)
	return fmt.Sprintf("Executed %s code: %s", language, code), nil
}
