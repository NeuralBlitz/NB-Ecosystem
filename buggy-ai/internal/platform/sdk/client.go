package sdk

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"
)

type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	mu         sync.RWMutex
	config     *ClientConfig
}

type ClientConfig struct {
	BaseURL       string
	APIKey        string
	Timeout       time.Duration
	RetryAttempts int
	RetryDelay    time.Duration
}

type SDKOption func(*ClientConfig)

func WithBaseURL(url string) SDKOption {
	return func(c *ClientConfig) {
		c.BaseURL = url
	}
}

func WithTimeout(timeout time.Duration) SDKOption {
	return func(c *ClientConfig) {
		c.Timeout = timeout
	}
}

func WithRetry(attempts int, delay time.Duration) SDKOption {
	return func(c *ClientConfig) {
		c.RetryAttempts = attempts
		c.RetryDelay = delay
	}
}

func NewClient(apiKey string, opts ...SDKOption) *Client {
	config := &ClientConfig{
		BaseURL:       "http://localhost:8080",
		Timeout:       30 * time.Second,
		RetryAttempts: 3,
		RetryDelay:    time.Second,
	}

	for _, opt := range opts {
		opt(config)
	}

	return &Client{
		baseURL: config.BaseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
		config: config,
	}
}

func (c *Client) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	var resp ChatResponse

	err := c.doRequest(ctx, "POST", "/api/v1/chat", req, &resp)
	if err != nil {
		return nil, err
	}

	return &resp, nil
}

func (c *Client) ExecuteTool(ctx context.Context, req *ToolRequest) (*ToolResponse, error) {
	var resp ToolResponse

	err := c.doRequest(ctx, "POST", "/api/v1/tools/execute", req, &resp)
	if err != nil {
		return nil, err
	}

	return &resp, nil
}

func (c *Client) CreatePipeline(ctx context.Context, req *CreatePipelineRequest) (*PipelineResponse, error) {
	var resp PipelineResponse

	err := c.doRequest(ctx, "POST", "/api/v1/pipelines", req, &resp)
	if err != nil {
		return nil, err
	}

	return &resp, nil
}

func (c *Client) ExecutePipeline(ctx context.Context, pipelineID string, req *ExecutePipelineRequest) (*ExecutionResponse, error) {
	var resp ExecutionResponse

	err := c.doRequest(ctx, "POST", fmt.Sprintf("/api/v1/pipelines/%s/execute", pipelineID), req, &resp)
	if err != nil {
		return nil, err
	}

	return &resp, nil
}

func (c *Client) ConnectIntegration(ctx context.Context, integrationType string, req *ConnectRequest) (*ConnectResponse, error) {
	var resp ConnectResponse

	err := c.doRequest(ctx, "POST", fmt.Sprintf("/api/v1/integrations/%s", integrationType), req, &resp)
	if err != nil {
		return nil, err
	}

	return &resp, nil
}

func (c *Client) RegisterComponent(ctx context.Context, req *ComponentRequest) (*ComponentResponse, error) {
	var resp ComponentResponse

	err := c.doRequest(ctx, "POST", "/api/v1/components", req, &resp)
	if err != nil {
		return nil, err
	}

	return &resp, nil
}

func (c *Client) GetClusterHealth(ctx context.Context) (*ClusterHealthResponse, error) {
	var resp ClusterHealthResponse

	err := c.doRequest(ctx, "GET", "/api/v1/cluster/health", nil, &resp)
	if err != nil {
		return nil, err
	}

	return &resp, nil
}

func (c *Client) doRequest(ctx context.Context, method, path string, body, result interface{}) error {
	u, err := url.Parse(c.baseURL)
	if err != nil {
		return fmt.Errorf("invalid base URL: %w", err)
	}

	u.Path = path
	req, err := http.NewRequestWithContext(ctx, method, u.String(), nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("X-SDK-Version", "1.0.0")

	client := &http.Client{
		Timeout: c.config.Timeout,
	}

	var lastErr error
	for attempt := 0; attempt <= c.config.RetryAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(c.config.RetryDelay * time.Duration(attempt))
		}

		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			lastErr = fmt.Errorf("request failed with status: %d", resp.StatusCode)
			continue
		}

		if result != nil {
			if err := decodeJSON(resp.Body, result); err != nil {
				lastErr = err
				continue
			}
		}

		return nil
	}

	return lastErr
}

type ChatRequest struct {
	Messages    []Message `json:"messages"`
	Model       string    `json:"model,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Model     string    `json:"model"`
	Usage     Usage     `json:"usage"`
	CreatedAt time.Time `json:"created_at"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ToolRequest struct {
	Tool    string                 `json:"tool"`
	Params  map[string]interface{} `json:"params"`
	Context map[string]interface{} `json:"context,omitempty"`
}

type ToolResponse struct {
	Success   bool          `json:"success"`
	Result    interface{}   `json:"result,omitempty"`
	Error     string        `json:"error,omitempty"`
	Execution time.Duration `json:"execution_time_ms"`
}

type CreatePipelineRequest struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Stages      []StageConfig   `json:"stages"`
	Triggers    []TriggerConfig `json:"triggers,omitempty"`
	Config      PipelineConfig  `json:"config,omitempty"`
}

type StageConfig struct {
	ID        string         `json:"id"`
	Name      string         `json:"name"`
	Type      string         `json:"type"`
	Actions   []ActionConfig `json:"actions"`
	DependsOn []string       `json:"depends_on,omitempty"`
}

type ActionConfig struct {
	ID     string                 `json:"id"`
	Name   string                 `json:"name"`
	Type   string                 `json:"type"`
	Config map[string]interface{} `json:"config,omitempty"`
}

type TriggerConfig struct {
	Type   string                 `json:"type"`
	Config map[string]interface{} `json:"config,omitempty"`
}

type PipelineConfig struct {
	Timeout       time.Duration `json:"timeout,omitempty"`
	MaxConcurrent int           `json:"max_concurrent,omitempty"`
	RetryPolicy   RetryPolicy   `json:"retry_policy,omitempty"`
}

type RetryPolicy struct {
	MaxAttempts int           `json:"max_attempts"`
	Delay       time.Duration `json:"delay"`
	Multiplier  float64       `json:"multiplier"`
}

type PipelineResponse struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description,omitempty"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
}

type ExecutePipelineRequest struct {
	Trigger    map[string]interface{} `json:"trigger,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type ExecutionResponse struct {
	ID           string        `json:"id"`
	PipelineID   string        `json:"pipeline_id"`
	Status       string        `json:"status"`
	StartTime    time.Time     `json:"start_time"`
	EndTime      time.Time     `json:"end_time,omitempty"`
	StageResults []StageResult `json:"stage_results,omitempty"`
}

type StageResult struct {
	StageID   string         `json:"stage_id"`
	Status    string         `json:"status"`
	StartTime time.Time      `json:"start_time"`
	EndTime   time.Time      `json:"end_time"`
	Actions   []ActionResult `json:"actions,omitempty"`
	Error     string         `json:"error,omitempty"`
}

type ActionResult struct {
	ActionID string                 `json:"action_id"`
	Status   string                 `json:"status"`
	Output   map[string]interface{} `json:"output,omitempty"`
	Error    string                 `json:"error,omitempty"`
}

type ConnectRequest struct {
	Config map[string]interface{} `json:"config"`
}

type ConnectResponse struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Status      string    `json:"status"`
	ConnectedAt time.Time `json:"connected_at"`
}

type ComponentRequest struct {
	ID     string                 `json:"id"`
	Type   string                 `json:"type"`
	Name   string                 `json:"name"`
	Config map[string]interface{} `json:"config,omitempty"`
}

type ComponentResponse struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	StartedAt time.Time `json:"started_at"`
}

type ClusterHealthResponse struct {
	TotalNodes int                   `json:"total_nodes"`
	AliveNodes int                   `json:"alive_nodes"`
	Quorum     bool                  `json:"quorum"`
	NodeHealth map[string]NodeHealth `json:"node_health"`
}

type NodeHealth struct {
	Status   string    `json:"status"`
	Role     string    `json:"role"`
	Leader   bool      `json:"leader"`
	LastSeen time.Time `json:"last_seen"`
}

func decodeJSON(r interface{}, v interface{}) error {
	return fmt.Errorf("JSON decode placeholder")
}

type EventHandler func(event Event)

type EventSubscriber struct {
	client *Client
	events chan Event
}

func (c *Client) Subscribe(ctx context.Context, eventTypes []string) *EventSubscriber {
	events := make(chan Event, 100)
	return &EventSubscriber{
		client: c,
		events: events,
	}
}

func (s *EventSubscriber) Events() <-chan Event {
	return s.events
}

func (s *EventSubscriber) Close() {
	close(s.events)
}

type Event struct {
	Type      string                 `json:"type"`
	Source    string                 `json:"source"`
	Timestamp time.Time              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
}

type Middleware func(Request) Response

type Request struct {
	Method  string
	Path    string
	Body    interface{}
	Context map[string]interface{}
}

type Response struct {
	StatusCode int
	Body       interface{}
	Headers    map[string]string
}

func Chain(middlewares ...Middleware) Middleware {
	return func(r Request) Response {
		var resp Response
		for _, m := range middlewares {
			resp = m(r)
		}
		return resp
	}
}
