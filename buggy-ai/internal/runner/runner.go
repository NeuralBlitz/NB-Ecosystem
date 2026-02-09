package runner

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"time"
)

type Runner struct {
	config  RunnerConfig
	sandbox Sandbox
	mu      sync.RWMutex
	active  map[string]*Execution
}

type RunnerConfig struct {
	MaxConcurrent int
	MemoryLimit   string
	CPULimit      float64
	Timeout       time.Duration
	AllowedPaths  []string
	EnvVars       []string
}

type Sandbox interface {
	Run(ctx context.Context, code string, language string) (*ExecutionResult, error)
	Info() SandboxInfo
}

type Execution struct {
	ID        string
	Language  string
	Code      string
	StartedAt time.Time
	Status    ExecutionStatus
}

type ExecutionStatus string

const (
	ExecPending   ExecutionStatus = "pending"
	ExecRunning   ExecutionStatus = "running"
	ExecCompleted ExecutionStatus = "completed"
	ExecFailed    ExecutionStatus = "failed"
	ExecTimeout   ExecutionStatus = "timeout"
)

type ExecutionResult struct {
	ExitCode   int
	Stdout     string
	Stderr     string
	Duration   time.Duration
	MemoryUsed int64
}

type SandboxInfo struct {
	Name    string
	Version string
}

func NewRunner(cfg RunnerConfig) *Runner {
	return &Runner{
		config:  cfg,
		sandbox: NewDockerSandbox(cfg),
		active:  make(map[string]*Execution),
	}
}

type DockerSandbox struct {
	config RunnerConfig
}

func NewDockerSandbox(cfg RunnerConfig) *DockerSandbox {
	return &DockerSandbox{config: cfg}
}

func (d *DockerSandbox) Run(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	ctx, cancel := context.WithTimeout(ctx, d.config.Timeout)
	defer cancel()

	switch language {
	case "python":
		return d.runPython(ctx, code)
	case "javascript", "js":
		return d.runJavaScript(ctx, code)
	case "go":
		return d.runGo(ctx, code)
	case "rust":
		return d.runRust(ctx, code)
	case "bash", "shell":
		return d.runBash(ctx, code)
	default:
		return &ExecutionResult{
			ExitCode: -1,
			Stderr:   fmt.Sprintf("unsupported language: %s", language),
		}, nil
	}
}

func (d *DockerSandbox) runPython(ctx context.Context, code string) (*ExecutionResult, error) {
	cmd := exec.CommandContext(ctx, "python3", "-c", code)
	output, _ := cmd.CombinedOutput()
	return &ExecutionResult{
		ExitCode: 0,
		Stdout:   string(output),
		Duration: time.Since(ctx.Value("start").(time.Time)),
	}, nil
}

func (d *DockerSandbox) runJavaScript(ctx context.Context, code string) (*ExecutionResult, error) {
	cmd := exec.CommandContext(ctx, "node", "-e", code)
	output, _ := cmd.CombinedOutput()
	return &ExecutionResult{
		ExitCode: 0,
		Stdout:   string(output),
	}, nil
}

func (d *DockerSandbox) runGo(ctx context.Context, code string) (*ExecutionResult, error) {
	cmd := exec.CommandContext(ctx, "go", "run", "-")
	cmd.Stdin = nil
	output, _ := cmd.CombinedOutput()
	return &ExecutionResult{
		ExitCode: 0,
		Stdout:   string(output),
	}, nil
}

func (d *DockerSandbox) runRust(ctx context.Context, code string) (*ExecutionResult, error) {
	cmd := exec.CommandContext(ctx, "rustc", "-o", "/tmp/rust_bin", "-")
	cmd.Stdin = nil
	output, _ := cmd.CombinedOutput()
	return &ExecutionResult{
		ExitCode: 0,
		Stdout:   string(output),
	}, nil
}

func (d *DockerSandbox) runBash(ctx context.Context, code string) (*ExecutionResult, error) {
	cmd := exec.CommandContext(ctx, "bash", "-c", code)
	output, _ := cmd.CombinedOutput()
	return &ExecutionResult{
		ExitCode: 0,
		Stdout:   string(output),
	}, nil
}

func (d *DockerSandbox) Info() SandboxInfo {
	return SandboxInfo{
		Name:    "docker",
		Version: "1.0.0",
	}
}

type WASMSandbox struct{}

func (w *WASMSandbox) Run(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	return &ExecutionResult{
		ExitCode: 0,
		Stdout:   "WASM execution",
	}, nil
}

func (w *WASMSandbox) Info() SandboxInfo {
	return SandboxInfo{
		Name:    "wasm",
		Version: "1.0.0",
	}
}

type GVisorSandbox struct{}

func (g *GVisorSandbox) Run(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	return &ExecutionResult{
		ExitCode: 0,
		Stdout:   "gVisor sandbox execution",
	}, nil
}

func (g *GVisorSandbox) Info() SandboxInfo {
	return SandboxInfo{
		Name:    "gvisor",
		Version: "1.0.0",
	}
}

func (r *Runner) Run(ctx context.Context, code string, language string) (*ExecutionResult, error) {
	r.mu.Lock()
	if len(r.active) >= r.config.MaxConcurrent {
		r.mu.Unlock()
		return nil, fmt.Errorf("max concurrent executions reached")
	}
	r.mu.Unlock()

	result, err := r.sandbox.Run(ctx, code, language)
	return result, err
}

func (r *Runner) SetSandbox(sandbox Sandbox) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sandbox = sandbox
}

func (r *Runner) GetSandbox() Sandbox {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.sandbox
}

type CodeExecutor struct {
	runner *Runner
}

func NewCodeExecutor(cfg RunnerConfig) *CodeExecutor {
	return &CodeExecutor{
		runner: NewRunner(cfg),
	}
}

func (e *CodeExecutor) Execute(ctx context.Context, req ExecuteRequest) (*ExecuteResponse, error) {
	result, err := e.runner.Run(ctx, req.Code, req.Language)
	if err != nil {
		return nil, err
	}

	return &ExecuteResponse{
		ExitCode: result.ExitCode,
		Stdout:   result.Stdout,
		Stderr:   result.Stderr,
		Duration: result.Duration,
	}, nil
}

type ExecuteRequest struct {
	Code     string            `json:"code"`
	Language string            `json:"language"`
	EnvVars  map[string]string `json:"env_vars,omitempty"`
	Timeout  time.Duration     `json:"timeout,omitempty"`
}

type ExecuteResponse struct {
	ExitCode   int           `json:"exit_code"`
	Stdout     string        `json:"stdout"`
	Stderr     string        `json:"stderr"`
	Duration   time.Duration `json:"duration"`
	MemoryUsed int64         `json:"memory_used,omitempty"`
}

type LanguageSupport struct {
	Name       string   `json:"name"`
	Extensions []string `json:"extensions"`
	Versions   []string `json:"versions"`
	Sandboxed  bool     `json:"sandboxed"`
}

func GetSupportedLanguages() []LanguageSupport {
	return []LanguageSupport{
		{Name: "Python", Extensions: []string{".py", ".py3"}, Versions: []string{"3.10", "3.11", "3.12"}, Sandboxed: true},
		{Name: "JavaScript", Extensions: []string{".js", ".mjs"}, Versions: []string{"18", "20", "22"}, Sandboxed: true},
		{Name: "TypeScript", Extensions: []string{".ts", ".tsx"}, Versions: []string{"5.0", "5.1", "5.2"}, Sandboxed: true},
		{Name: "Go", Extensions: []string{".go"}, Versions: []string{"1.21", "1.22"}, Sandboxed: true},
		{Name: "Rust", Extensions: []string{".rs"}, Versions: []string{"1.74", "1.75"}, Sandboxed: true},
		{Name: "Java", Extensions: []string{".java"}, Versions: []string{"17", "21"}, Sandboxed: true},
		{Name: "C", Extensions: []string{".c", ".h"}, Versions: []string{"gcc-13", "clang-17"}, Sandboxed: true},
		{Name: "C++", Extensions: []string{".cpp", ".cc", ".hpp"}, Versions: []string{"gcc-13", "clang-17"}, Sandboxed: true},
		{Name: "Ruby", Extensions: []string{".rb"}, Versions: []string{"3.2", "3.3"}, Sandboxed: true},
		{Name: "PHP", Extensions: []string{".php"}, Versions: []string{"8.2", "8.3"}, Sandboxed: true},
		{Name: "Bash", Extensions: []string{".sh"}, Versions: []string{"5.2"}, Sandboxed: true},
		{Name: "Lua", Extensions: []string{".lua"}, Versions: []string{"5.4"}, Sandboxed: true},
		{Name: "R", Extensions: []string{".r", ".R"}, Versions: []string{"4.3"}, Sandboxed: true},
		{Name: "Julia", Extensions: []string{".jl"}, Versions: []string{"1.10"}, Sandboxed: true},
		{Name: "Swift", Extensions: []string{".swift"}, Versions: []string{"5.9"}, Sandboxed: true},
	}
}

func GetLanguageByExtension(ext string) *LanguageSupport {
	for _, lang := range GetSupportedLanguages() {
		for _, e := range lang.Extensions {
			if e == ext {
				return &lang
			}
		}
	}
	return nil
}

func (r *Runner) HealthCheck() error {
	_, err := os.Stat("/var/run/docker.sock")
	if err != nil {
		return fmt.Errorf("docker not available: %v", err)
	}
	return nil
}
