package pipeline

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type Pipeline struct {
	ID          string
	Name        string
	Description string
	Stages      []*Stage
	Triggers    []*Trigger
	Status      PipelineStatus
	Config      PipelineConfig
	Executions  []*Execution
	mu          sync.RWMutex
}

type PipelineStatus string

const (
	PipelineIdle    PipelineStatus = "idle"
	PipelineRunning PipelineStatus = "running"
	PipelinePaused  PipelineStatus = "paused"
	PipelineSuccess PipelineStatus = "success"
	PipelineFailed  PipelineStatus = "failed"
)

type PipelineConfig struct {
	MaxConcurrent  int
	Timeout        time.Duration
	RetryPolicy    RetryPolicy
	RollbackPolicy RollbackPolicy
	Notifications  []NotificationConfig
}

type RetryPolicy struct {
	MaxAttempts int
	Delay       time.Duration
	Multiplier  float64
	MaxDelay    time.Duration
}

type RollbackPolicy struct {
	Enabled   bool
	Timeout   time.Duration
	OnFailure bool
}

type Stage struct {
	ID        string
	Name      string
	Type      StageType
	Actions   []*Action
	DependsOn []string
	Status    StageStatus
	Config    StageConfig
	StartTime time.Time
	EndTime   time.Time
}

type StageType string

const (
	StageBuild     StageType = "build"
	StageTest      StageType = "test"
	StageDeploy    StageType = "deploy"
	StageIntegrate StageType = "integrate"
	StageNotify    StageType = "notify"
	StageCustom    StageType = "custom"
)

type StageStatus string

const (
	StagePending StageStatus = "pending"
	StageRunning StageStatus = "running"
	StageSuccess StageStatus = "success"
	StageFailed  StageStatus = "failed"
	StageSkipped StageStatus = "skipped"
)

type StageConfig struct {
	Timeout     time.Duration
	Conditions  []Condition
	Environment map[string]string
	Artifacts   []Artifact
}

type Action struct {
	ID        string
	Name      string
	Type      ActionType
	Provider  string
	Config    ActionConfig
	Status    ActionStatus
	Output    map[string]interface{}
	Error     string
	StartTime time.Time
	EndTime   time.Time
}

type ActionType string

const (
	ActionRun         ActionType = "run"
	ActionScript      ActionType = "script"
	ActionDockerBuild ActionType = "docker_build"
	ActionK8sDeploy   ActionType = "k8s_deploy"
	ActionTerraform   ActionType = "terraform"
	ActionSlackNotify ActionType = "slack_notify"
	ActionEmailNotify ActionType = "email_notify"
	ActionApproval    ActionType = "approval"
	ActionHTTP        ActionType = "http"
	ActionGraphQL     ActionType = "graphql"
)

type ActionStatus string

const (
	ActionPending ActionStatus = "pending"
	ActionRunning ActionStatus = "running"
	ActionSuccess ActionStatus = "success"
	ActionFailed  ActionStatus = "failed"
)

type ActionConfig struct {
	Command string
	Script  string
	Image   string
	Tag     string
	Path    string
	URL     string
	Method  string
	Headers map[string]string
	Body    string
	Timeout time.Duration
}

type Condition struct {
	Type       string
	Expression string
	Value      interface{}
}

type Artifact struct {
	Name string
	Path string
	Type string
}

type Trigger struct {
	ID          string
	Type        TriggerType
	Config      TriggerConfig
	Enabled     bool
	LastTrigger time.Time
}

type TriggerType string

const (
	TriggerManual   TriggerType = "manual"
	TriggerGitPush  TriggerType = "git_push"
	TriggerSchedule TriggerType = "schedule"
	TriggerWebhook  TriggerType = "webhook"
	TriggerAPI      TriggerType = "api"
	TriggerEvent    TriggerType = "event"
)

type TriggerConfig struct {
	Branch     string
	Tag        string
	Schedule   string
	Path       string
	EventTypes []string
}

type Execution struct {
	ID           string
	PipelineID   string
	Trigger      *Trigger
	Status       PipelineStatus
	StartTime    time.Time
	EndTime      time.Time
	StageResults []*StageResult
	Artifacts    []Artifact
	Metadata     map[string]interface{}
}

type StageResult struct {
	StageID       string
	Status        StageStatus
	StartTime     time.Time
	EndTime       time.Time
	ActionResults []*ActionResult
	Error         string
}

type ActionResult struct {
	ActionID  string
	Status    ActionStatus
	Output    map[string]interface{}
	Error     string
	StartTime time.Time
	EndTime   time.Time
}

type NotificationConfig struct {
	Type     string
	Endpoint string
	Events   []string
	Template string
}

type Engine struct {
	pipelines map[string]*Pipeline
	executors map[string]Executor
	mu        sync.RWMutex
}

type Executor interface {
	ID() string
	Type() string
	Execute(ctx context.Context, action *Action) error
	Cancel(ctx context.Context, executionID string) error
	Health(ctx context.Context) ExecutorHealth
}

type ExecutorHealth struct {
	Status      string
	JobsRunning int
	JobsQueued  int
}

func NewEngine() *Engine {
	return &Engine{
		pipelines: make(map[string]*Pipeline),
		executors: make(map[string]Executor),
	}
}

func (e *Engine) CreatePipeline(id, name string, config *PipelineConfig) *Pipeline {
	pipeline := &Pipeline{
		ID:     id,
		Name:   name,
		Config: *config,
		Status: PipelineIdle,
	}

	e.mu.Lock()
	e.pipelines[id] = pipeline
	e.mu.Unlock()

	return pipeline
}

func (e *Engine) AddStage(pipelineID string, stage *Stage) error {
	e.mu.RLock()
	pipeline, ok := e.pipelines[pipelineID]
	e.mu.RUnlock()

	if !ok {
		return ErrPipelineNotFound(pipelineID)
	}

	e.mu.Lock()
	pipeline.Stages = append(pipeline.Stages, stage)
	e.mu.Unlock()

	return nil
}

func (e *Engine) AddTrigger(pipelineID string, trigger *Trigger) error {
	e.mu.RLock()
	pipeline, ok := e.pipelines[pipelineID]
	e.mu.RUnlock()

	if !ok {
		return ErrPipelineNotFound(pipelineID)
	}

	e.mu.Lock()
	pipeline.Triggers = append(pipeline.Triggers, trigger)
	e.mu.Unlock()

	return nil
}

func (e *Engine) Execute(ctx context.Context, pipelineID string, trigger *Trigger) (*Execution, error) {
	e.mu.RLock()
	pipeline, ok := e.pipelines[pipelineID]
	e.mu.RUnlock()

	if !ok {
		return nil, ErrPipelineNotFound(pipelineID)
	}

	execution := &Execution{
		ID:         fmt.Sprintf("exec-%d", time.Now().UnixNano()),
		PipelineID: pipelineID,
		Trigger:    trigger,
		Status:     PipelineRunning,
		StartTime:  time.Now(),
	}

	e.mu.Lock()
	pipeline.Executions = append(pipeline.Executions, execution)
	e.mu.Unlock()

	go e.runPipeline(ctx, pipeline, execution)

	return execution, nil
}

func (e *Engine) runPipeline(ctx context.Context, pipeline *Pipeline, execution *Execution) {
	defer func() {
		execution.EndTime = time.Now()
		if len(execution.StageResults) > 0 {
			lastResult := execution.StageResults[len(execution.StageResults)-1]
			if lastResult.Status == StageFailed {
				execution.Status = PipelineFailed
			} else {
				execution.Status = PipelineSuccess
			}
		}
	}()

	for _, stage := range pipeline.Stages {
		if e.shouldSkipStage(stage, execution) {
			execution.StageResults = append(execution.StageResults, &StageResult{
				StageID:   stage.ID,
				Status:    StageSkipped,
				StartTime: time.Now(),
				EndTime:   time.Now(),
			})
			continue
		}

		stageResult := e.runStage(ctx, stage)
		execution.StageResults = append(execution.StageResults, stageResult)

		if stageResult.Status == StageFailed {
			return
		}
	}
}

func (e *Engine) shouldSkipStage(stage *Stage, execution *Execution) bool {
	for _, dep := range stage.DependsOn {
		for _, result := range execution.StageResults {
			if result.StageID == dep && result.Status != StageSuccess {
				return true
			}
		}
	}
	return false
}

func (e *Engine) runStage(ctx context.Context, stage *Stage) *StageResult {
	result := &StageResult{
		StageID:   stage.ID,
		Status:    StageRunning,
		StartTime: time.Now(),
	}

	for _, action := range stage.Actions {
		actionResult := e.runAction(ctx, action)
		result.ActionResults = append(result.ActionResults, actionResult)

		if actionResult.Status == ActionFailed {
			result.Status = StageFailed
			result.Error = actionResult.Error
			return result
		}
	}

	result.Status = StageSuccess
	result.EndTime = time.Now()
	return result
}

func (e *Engine) runAction(ctx context.Context, action *Action) *ActionResult {
	result := &ActionResult{
		ActionID:  action.ID,
		Status:    ActionRunning,
		StartTime: time.Now(),
	}

	action.Status = ActionRunning

	e.mu.RLock()
	executor := e.executors[string(action.Type)]
	e.mu.RUnlock()

	if executor != nil {
		if err := executor.Execute(ctx, action); err != nil {
			result.Status = ActionFailed
			result.Error = err.Error()
		} else {
			result.Status = ActionSuccess
			result.Output = action.Output
		}
	} else {
		result.Status = ActionSuccess
		result.Output = map[string]interface{}{
			"message": "Action executed (no executor configured)",
		}
	}

	action.Status = result.Status
	result.EndTime = time.Now()
	return result
}

func (e *Engine) RegisterExecutor(executorType string, executor Executor) {
	e.mu.Lock()
	e.executors[executorType] = executor
	e.mu.Unlock()
}

func (e *Engine) GetPipeline(id string) (*Pipeline, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	p, ok := e.pipelines[id]
	return p, ok
}

func (e *Engine) ListPipelines() []*Pipeline {
	e.mu.RLock()
	defer e.mu.RUnlock()

	pipelines := make([]*Pipeline, 0, len(e.pipelines))
	for _, p := range e.pipelines {
		pipelines = append(pipelines, p)
	}
	return pipelines
}

func (e *Engine) GetExecution(pipelineID, executionID string) (*Execution, error) {
	e.mu.RLock()
	pipeline, ok := e.pipelines[pipelineID]
	e.mu.RUnlock()

	if !ok {
		return nil, ErrPipelineNotFound(pipelineID)
	}

	for _, exec := range pipeline.Executions {
		if exec.ID == executionID {
			return exec, nil
		}
	}

	return nil, ErrExecutionNotFound(executionID)
}

var (
	ErrPipelineNotFound = func(id string) error {
		return &PipelineError{Code: "NOT_FOUND", Message: fmt.Sprintf("Pipeline not found: %s", id)}
	}
	ErrExecutionNotFound = func(id string) error {
		return &PipelineError{Code: "NOT_FOUND", Message: fmt.Sprintf("Execution not found: %s", id)}
	}
)

type PipelineError struct {
	Code    string
	Message string
}

func (e *PipelineError) Error() string {
	return e.Message
}

type LocalExecutor struct {
	id      string
	mu      sync.RWMutex
	running int
}

func NewLocalExecutor(id string) Executor {
	return &LocalExecutor{id: id}
}

func (l *LocalExecutor) ID() string {
	return l.id
}

func (l *LocalExecutor) Type() string {
	return "local"
}

func (l *LocalExecutor) Execute(ctx context.Context, action *Action) error {
	l.mu.Lock()
	l.running++
	l.mu.Unlock()

	defer func() {
		l.mu.Lock()
		l.running--
		l.mu.Unlock()
	}()

	action.Output = map[string]interface{}{
		"executed": true,
		"action":   action.Name,
		"time":     time.Now().Format(time.RFC3339),
	}

	return nil
}

func (l *LocalExecutor) Cancel(ctx context.Context, executionID string) error {
	return nil
}

func (l *LocalExecutor) Health(ctx context.Context) ExecutorHealth {
	l.mu.RLock()
	defer l.mu.RUnlock()

	status := "healthy"
	if l.running > 10 {
		status = "degraded"
	}

	return ExecutorHealth{
		Status:      status,
		JobsRunning: l.running,
		JobsQueued:  0,
	}
}

func init() {
	engine := NewEngine()
	engine.RegisterExecutor("run", NewLocalExecutor("local-run"))
	engine.RegisterExecutor("script", NewLocalExecutor("local-script"))
}
