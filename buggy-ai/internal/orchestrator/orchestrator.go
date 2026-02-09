package orchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"

	"buggy-ai/internal/shared"
)

type Orchestrator struct {
	config    OrchestratorConfig
	tasks     *TaskStore
	workflows *WorkflowStore
	scheduler Scheduler
	plugins   []OrchestratorPlugin
	mu        sync.RWMutex
}

type OrchestratorConfig struct {
	MaxConcurrentTasks int
	DefaultTimeout     time.Duration
	MaxRetries         int
	RetryBackoff       time.Duration
	CheckpointInterval time.Duration
}

type TaskStore struct {
	mu      sync.RWMutex
	tasks   map[shared.TaskID]*shared.Task
	byState map[shared.TaskState][]shared.TaskID
}

type WorkflowStore struct {
	mu        sync.RWMutex
	workflows map[string]*Workflow
}

type Workflow struct {
	ID          string
	Name        string
	DAG         DAG
	State       WorkflowState
	Progress    float64
	StartedAt   time.Time
	CompletedAt time.Time
}

type DAG struct {
	ID        string
	Nodes     map[string]*DAGNode
	Edges     []DAGEdge
	RootNodes []string
}

type DAGNode struct {
	ID        string
	Task      *shared.Task
	State     StepState
	DependsOn []string
	Result    interface{}
}

type DAGEdge struct {
	From string
	To   string
}

type StepState string

const (
	StepPending   StepState = "pending"
	StepReady     StepState = "ready"
	StepRunning   StepState = "running"
	StepCompleted StepState = "completed"
	StepFailed    StepState = "failed"
	StepSkipped   StepState = "skipped"
)

type WorkflowState string

const (
	WFCreated   WorkflowState = "created"
	WFRunning   WorkflowState = "running"
	WFCompleted WorkflowState = "completed"
	WFFailed    WorkflowState = "failed"
	WFCancelled WorkflowState = "cancelled"
)

type Scheduler interface {
	Schedule(task *shared.Task) error
	GetNextTask() *shared.Task
	TaskComplete(taskID shared.TaskID, result *shared.TaskResult)
	TaskFailed(taskID shared.TaskID, err error)
}

type FIFOScheduler struct {
	mu       sync.Mutex
	queue    []*shared.Task
	priority map[int][]*shared.Task
}

func NewFIFOScheduler() *FIFOScheduler {
	return &FIFOScheduler{
		priority: make(map[int][]*shared.Task),
	}
}

func (s *FIFOScheduler) Schedule(task *shared.Task) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	queue := s.priority[task.Priority]
	s.priority[task.Priority] = append(queue, task)
	return nil
}

func (s *FIFOScheduler) GetNextTask() *shared.Task {
	s.mu.Lock()
	defer s.mu.Unlock()

	for priority := 100; priority >= 0; priority-- {
		if queue, ok := s.priority[priority]; ok && len(queue) > 0 {
			task := queue[0]
			s.priority[priority] = queue[1:]
			return task
		}
	}
	return nil
}

func (s *FIFOScheduler) TaskComplete(taskID shared.TaskID, result *shared.TaskResult) {}
func (s *FIFOScheduler) TaskFailed(taskID shared.TaskID, err error)                   {}

type OrchestratorPlugin interface {
	OnTaskScheduled(task *shared.Task)
	OnTaskStarted(task *shared.Task)
	OnTaskCompleted(task *shared.Task, result *shared.TaskResult)
	OnTaskFailed(task *shared.Task, err error)
	OnWorkflowCreated(workflow *Workflow)
	OnWorkflowCompleted(workflow *Workflow)
}

func NewOrchestrator(cfg OrchestratorConfig) *Orchestrator {
	return &Orchestrator{
		config:    cfg,
		tasks:     NewTaskStore(),
		workflows: NewWorkflowStore(),
		scheduler: NewFIFOScheduler(),
		plugins:   []OrchestratorPlugin{},
	}
}

func NewTaskStore() *TaskStore {
	return &TaskStore{
		tasks:   make(map[shared.TaskID]*shared.Task),
		byState: make(map[shared.TaskState][]shared.TaskID),
	}
}

func NewWorkflowStore() *WorkflowStore {
	return &WorkflowStore{
		workflows: make(map[string]*Workflow),
	}
}

func (o *Orchestrator) CreateWorkflow(name string, dag DAG) *Workflow {
	workflow := &Workflow{
		ID:        generateID(),
		Name:      name,
		DAG:       dag,
		State:     WFCreated,
		StartedAt: time.Now(),
	}

	o.workflows.mu.Lock()
	o.workflows.workflows[workflow.ID] = workflow
	o.workflows.mu.Unlock()

	for _, plugin := range o.plugins {
		plugin.OnWorkflowCreated(workflow)
	}

	return workflow
}

func (o *Orchestrator) ExecuteWorkflow(ctx context.Context, workflowID string) error {
	workflow, ok := o.getWorkflow(workflowID)
	if !ok {
		return fmt.Errorf("workflow not found: %s", workflowID)
	}

	workflow.State = WFRunning

	err := o.executeDAG(ctx, workflow.DAG)

	workflow.CompletedAt = time.Now()
	if err != nil {
		workflow.State = WFFailed
	} else {
		workflow.State = WFCompleted
		workflow.Progress = 100.0
	}

	for _, plugin := range o.plugins {
		plugin.OnWorkflowCompleted(workflow)
	}

	return err
}

func (o *Orchestrator) executeDAG(ctx context.Context, dag DAG) error {
	completed := make(map[string]bool)
	var mu sync.Mutex

	for len(completed) < len(dag.Nodes) {
		progress := float64(len(completed)) / float64(len(dag.Nodes)) * 100
		o.updateWorkflowProgress(dag.ID, progress)

		for _, node := range dag.Nodes {
			if completed[node.ID] {
				continue
			}

			if !o.canExecute(node, completed) {
				continue
			}

			task := node.Task
			task.State = shared.TaskRunning
			task.StartedAt = time.Now()

			for _, plugin := range o.plugins {
				plugin.OnTaskStarted(task)
			}

			result, err := o.executeTask(ctx, task)

			mu.Lock()
			if err != nil {
				return err
			}
			completed[node.ID] = true
			mu.Unlock()

			task.CompletedAt = time.Now()
			if err != nil {
				task.State = shared.TaskFailed
				for _, plugin := range o.plugins {
					plugin.OnTaskFailed(task, err)
				}
				return err
			}

			task.State = shared.TaskCompleted
			task.Result = result.Output
			for _, plugin := range o.plugins {
				plugin.OnTaskCompleted(task, result)
			}
		}
	}

	return nil
}

func (o *Orchestrator) canExecute(node *DAGNode, completed map[string]bool) bool {
	for _, dep := range node.DependsOn {
		if !completed[dep] {
			return false
		}
	}
	return true
}

func (o *Orchestrator) executeTask(ctx context.Context, task *shared.Task) (*shared.TaskResult, error) {
	return &shared.TaskResult{
		TaskID:   task.ID,
		Success:  true,
		Duration: time.Second,
	}, nil
}

func (o *Orchestrator) updateWorkflowProgress(dagID string, progress float64) {
	for _, workflow := range o.workflows.workflows {
		if workflow.DAG.ID == dagID {
			workflow.Progress = progress
		}
	}
}

func (o *Orchestrator) ScheduleTask(task *shared.Task) error {
	o.tasks.mu.Lock()
	o.tasks.tasks[task.ID] = task
	o.tasks.byState[task.State] = append(o.tasks.byState[task.State], task.ID)
	o.tasks.mu.Unlock()

	for _, plugin := range o.plugins {
		plugin.OnTaskScheduled(task)
	}

	return o.scheduler.Schedule(task)
}

func (o *Orchestrator) GetTask(taskID shared.TaskID) (*shared.Task, bool) {
	return o.getTask(taskID)
}

func (o *Orchestrator) getTask(taskID shared.TaskID) (*shared.Task, bool) {
	o.tasks.mu.RLock()
	defer o.tasks.mu.RUnlock()
	task, ok := o.tasks.tasks[taskID]
	return task, ok
}

func (o *Orchestrator) getWorkflow(workflowID string) (*Workflow, bool) {
	o.workflows.mu.RLock()
	defer o.workflows.mu.RUnlock()
	wf, ok := o.workflows.workflows[workflowID]
	return wf, ok
}

func (o *Orchestrator) RegisterPlugin(plugin OrchestratorPlugin) {
	o.plugins = append(o.plugins, plugin)
}

func (o *Orchestrator) GetStats() OrchestratorStats {
	o.tasks.mu.RLock()
	o.workflows.mu.RLock()
	defer o.tasks.mu.RUnlock()
	defer o.workflows.mu.RUnlock()

	return OrchestratorStats{
		TotalTasks:     len(o.tasks.tasks),
		PendingTasks:   len(o.tasks.byState[shared.TaskPending]),
		RunningTasks:   len(o.tasks.byState[shared.TaskRunning]),
		CompletedTasks: len(o.tasks.byState[shared.TaskCompleted]),
		TotalWorkflows: len(o.workflows.workflows),
	}
}

type OrchestratorStats struct {
	TotalTasks     int
	PendingTasks   int
	RunningTasks   int
	CompletedTasks int
	FailedTasks    int
	TotalWorkflows int
}

type Checkpoint struct {
	Timestamp      time.Time
	TaskStates     map[shared.TaskID]shared.TaskState
	WorkflowStates map[string]WorkflowState
}

func (o *Orchestrator) CreateCheckpoint() *Checkpoint {
	o.tasks.mu.RLock()
	o.workflows.mu.RLock()
	defer o.tasks.mu.RUnlock()
	defer o.workflows.mu.RUnlock()

	return &Checkpoint{
		Timestamp:      time.Now(),
		TaskStates:     make(map[shared.TaskID]shared.TaskState),
		WorkflowStates: make(map[string]WorkflowState),
	}
}

func (o *Orchestrator) RestoreCheckpoint(cp *Checkpoint) error {
	for taskID, state := range cp.TaskStates {
		if task, ok := o.tasks.tasks[taskID]; ok {
			task.State = state
		}
	}
	for wfID, state := range cp.WorkflowStates {
		if wf, ok := o.workflows.workflows[wfID]; ok {
			wf.State = state
		}
	}
	return nil
}

func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

func NewDAG() *DAG {
	return &DAG{
		Nodes:     make(map[string]*DAGNode),
		Edges:     []DAGEdge{},
		RootNodes: []string{},
	}
}

func (d *DAG) AddNode(id string, task *shared.Task) {
	d.Nodes[id] = &DAGNode{
		ID:   id,
		Task: task,
	}
}

func (d *DAG) AddEdge(from, to string) {
	d.Edges = append(d.Edges, DAGEdge{From: from, To: to})
	d.Nodes[to].DependsOn = append(d.Nodes[to].DependsOn, from)
}

func (d *DAG) DetectCycles() ([]string, bool) {
	visited := make(map[string]bool)
	recStack := make(map[string]bool)
	var cycle []string

	var dfs func(node string) bool
	dfs = func(node string) bool {
		if recStack[node] {
			cycle = append(cycle, node)
			return true
		}
		if visited[node] {
			return false
		}

		visited[node] = true
		recStack[node] = true

		for _, edge := range d.Edges {
			if edge.From == node {
				if dfs(edge.To) {
					cycle = append(cycle, edge.From)
					return true
				}
			}
		}

		recStack[node] = false
		return false
	}

	for node := range d.Nodes {
		if !visited[node] {
			if dfs(node) {
				return cycle, true
			}
		}
	}

	return cycle, false
}
