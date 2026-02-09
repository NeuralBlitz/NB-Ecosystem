package integration

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type Hub struct {
	connectors map[string]Connector
	factories  map[string]ConnectorFactory
	mu         sync.RWMutex
}

var (
	DefaultHub *Hub
	once       sync.Once
	hubInit    []func(*Hub)
)

func RegisterConnectorFactory(connType string, factory ConnectorFactory) {
	hubInit = append(hubInit, func(h *Hub) {
		h.RegisterFactory(connType, factory)
	})
}

func GetDefaultHub() *Hub {
	once.Do(func() {
		DefaultHub = NewHub()
		for _, init := range hubInit {
			init(DefaultHub)
		}
	})
	return DefaultHub
}

type Connector interface {
	GetID() string
	GetType() string
	Connect(ctx context.Context, config map[string]interface{}) error
	Disconnect() error
	Execute(ctx context.Context, action string, params map[string]interface{}) (interface{}, error)
	Health(ctx context.Context) ConnectorHealth
	Events() <-chan ConnectorEvent
}

type ConnectorHealth struct {
	Status    string
	Latency   time.Duration
	UpdatedAt time.Time
}

type ConnectorEvent struct {
	Type   string
	Source string
	Data   map[string]interface{}
	Time   time.Time
}

type ConnectorFactory func(id string) Connector

type ConnectorConfig struct {
	ID         string
	Type       string
	Enabled    bool
	Timeout    time.Duration
	Retries    int
	RetryDelay time.Duration
	Auth       AuthConfig
	WebhookURL string
	Events     []string
}

type AuthConfig struct {
	Type     string
	APIKey   string
	Secret   string
	Token    string
	Endpoint string
	Scopes   []string
}

func NewHub() *Hub {
	return &Hub{
		connectors: make(map[string]Connector),
		factories:  make(map[string]ConnectorFactory),
	}
}

func (h *Hub) RegisterFactory(connectorType string, factory ConnectorFactory) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.factories[connectorType] = factory
}

func (h *Hub) CreateConnector(connType, id string, config map[string]interface{}) (Connector, error) {
	h.mu.RLock()
	factory, ok := h.factories[connType]
	h.mu.RUnlock()

	if !ok {
		return nil, ErrUnknownConnectorType(connType)
	}

	connector := factory(id)
	if err := connector.Connect(context.Background(), config); err != nil {
		return nil, err
	}

	h.mu.Lock()
	h.connectors[id] = connector
	h.mu.Unlock()

	return connector, nil
}

func (h *Hub) GetConnector(id string) (Connector, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	c, ok := h.connectors[id]
	return c, ok
}

func (h *Hub) RemoveConnector(id string) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if connector, ok := h.connectors[id]; ok {
		connector.Disconnect()
		delete(h.connectors, id)
	}
	return nil
}

func (h *Hub) ListConnectors() []Connector {
	h.mu.RLock()
	defer h.mu.RUnlock()

	connectors := make([]Connector, 0, len(h.connectors))
	for _, c := range h.connectors {
		connectors = append(connectors, c)
	}
	return connectors
}

func (h *Hub) Execute(id, action string, params map[string]interface{}) (interface{}, error) {
	h.mu.RLock()
	connector, ok := h.connectors[id]
	h.mu.RUnlock()

	if !ok {
		return nil, ErrConnectorNotFound(id)
	}

	return connector.Execute(context.Background(), action, params)
}

func (h *Hub) HealthCheckAll() map[string]ConnectorHealth {
	h.mu.RLock()
	defer h.mu.RUnlock()

	results := make(map[string]ConnectorHealth)
	for id, connector := range h.connectors {
		results[id] = connector.Health(context.Background())
	}
	return results
}

var (
	ErrUnknownConnectorType = func(connType string) error {
		return &IntegrationError{Code: "UNKNOWN_TYPE", Message: "Unknown connector type: " + connType}
	}
	ErrConnectorNotFound = func(id string) error {
		return &IntegrationError{Code: "NOT_FOUND", Message: "Connector not found: " + id}
	}
)

type IntegrationError struct {
	Code    string
	Message string
	Err     error
}

func (e *IntegrationError) Error() string {
	if e.Err != nil {
		return e.Message + ": " + e.Err.Error()
	}
	return e.Message
}

type BaseConnector struct {
	id            string
	connectorType string
	Config        ConnectorConfig
	events        chan ConnectorEvent
	mu            sync.RWMutex
	healthy       bool
	lastLatency   time.Duration
}

func (b *BaseConnector) GetID() string {
	return b.id
}

func (b *BaseConnector) GetType() string {
	return b.connectorType
}

func (b *BaseConnector) Events() <-chan ConnectorEvent {
	return b.events
}

func (b *BaseConnector) Health(ctx context.Context) ConnectorHealth {
	b.mu.RLock()
	defer b.mu.RUnlock()

	status := "healthy"
	if !b.healthy {
		status = "unhealthy"
	}

	return ConnectorHealth{
		Status:    status,
		Latency:   b.lastLatency,
		UpdatedAt: time.Now(),
	}
}

func (b *BaseConnector) setHealthy(healthy bool, latency time.Duration) {
	b.mu.Lock()
	b.healthy = healthy
	b.lastLatency = latency
	b.mu.Unlock()
}

func (b *BaseConnector) emit(eventType string, data map[string]interface{}) {
	select {
	case b.events <- ConnectorEvent{
		Type:   eventType,
		Source: b.id,
		Data:   data,
		Time:   time.Now(),
	}:
	default:
	}
}

type SlackConnector struct {
	BaseConnector
	token       string
	workspaceID string
}

func NewSlackConnector(id string) Connector {
	return &SlackConnector{
		BaseConnector: BaseConnector{
			id:            id,
			connectorType: "slack",
			Config:        ConnectorConfig{Retries: 3, Timeout: 30 * time.Second},
			events:        make(chan ConnectorEvent, 100),
		},
	}
}

func (s *SlackConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	if token, ok := config["token"].(string); ok {
		s.token = token
	}
	if ws, ok := config["workspace_id"].(string); ok {
		s.workspaceID = ws
	}
	s.setHealthy(true, 0)
	return nil
}

func (s *SlackConnector) Disconnect() error {
	s.setHealthy(false, 0)
	return nil
}

func (s *SlackConnector) Execute(ctx context.Context, action string, params map[string]interface{}) (interface{}, error) {
	start := time.Now()
	defer func() {
		s.setHealthy(true, time.Since(start))
	}()

	switch action {
	case "send_message":
		channel := params["channel"].(string)
		return map[string]interface{}{
			"ok":      true,
			"channel": channel,
			"ts":      time.Now().Unix(),
		}, nil
	case "list_channels":
		return map[string]interface{}{
			"channels": []string{"general", "random", "dev", "ops"},
		}, nil
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

type GitHubConnector struct {
	BaseConnector
	token string
	org   string
	repo  string
}

func NewGitHubConnector(id string) Connector {
	return &GitHubConnector{
		BaseConnector: BaseConnector{
			id:            id,
			connectorType: "github",
			Config:        ConnectorConfig{Retries: 3, Timeout: 30 * time.Second},
			events:        make(chan ConnectorEvent, 100),
		},
	}
}

func (g *GitHubConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	if token, ok := config["token"].(string); ok {
		g.token = token
	}
	if org, ok := config["org"].(string); ok {
		g.org = org
	}
	if repo, ok := config["repo"].(string); ok {
		g.repo = repo
	}
	g.setHealthy(true, 0)
	return nil
}

func (g *GitHubConnector) Disconnect() error {
	g.setHealthy(false, 0)
	return nil
}

func (g *GitHubConnector) Execute(ctx context.Context, action string, params map[string]interface{}) (interface{}, error) {
	start := time.Now()
	defer func() {
		g.setHealthy(true, time.Since(start))
	}()

	switch action {
	case "create_issue":
		title := params["title"].(string)
		body := params["body"].(string)
		return map[string]interface{}{
			"id":    123,
			"title": title,
			"body":  body,
			"url":   fmt.Sprintf("https://github.com/%s/%s/issues/123", g.org, g.repo),
		}, nil
	case "list_issues":
		return map[string]interface{}{
			"issues": []map[string]interface{}{
				{"id": 1, "title": "Bug fix", "state": "open"},
				{"id": 2, "title": "Feature request", "state": "open"},
			},
		}, nil
	case "create_pr":
		title := params["title"].(string)
		body := params["body"].(string)
		return map[string]interface{}{
			"id":     456,
			"title":  title,
			"body":   body,
			"state":  "open",
			"merged": false,
		}, nil
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

type JiraConnector struct {
	BaseConnector
	baseURL  string
	email    string
	apiToken string
	project  string
}

func NewJiraConnector(id string) Connector {
	return &JiraConnector{
		BaseConnector: BaseConnector{
			id:            id,
			connectorType: "jira",
			Config:        ConnectorConfig{Retries: 3, Timeout: 30 * time.Second},
			events:        make(chan ConnectorEvent, 100),
		},
	}
}

func (j *JiraConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	if url, ok := config["base_url"].(string); ok {
		j.baseURL = url
	}
	if email, ok := config["email"].(string); ok {
		j.email = email
	}
	if token, ok := config["api_token"].(string); ok {
		j.apiToken = token
	}
	if project, ok := config["project"].(string); ok {
		j.project = project
	}
	j.setHealthy(true, 0)
	return nil
}

func (j *JiraConnector) Disconnect() error {
	j.setHealthy(false, 0)
	return nil
}

func (j *JiraConnector) Execute(ctx context.Context, action string, params map[string]interface{}) (interface{}, error) {
	start := time.Now()
	defer func() {
		j.setHealthy(true, time.Since(start))
	}()

	switch action {
	case "create_issue":
		return map[string]interface{}{
			"id":      "PROJ-123",
			"key":     "PROJ-123",
			"summary": params["summary"].(string),
			"status":  "To Do",
		}, nil
	case "transition_issue":
		return map[string]interface{}{
			"id":     params["issue_key"].(string),
			"status": params["transition"].(string),
		}, nil
	case "search_issues":
		return map[string]interface{}{
			"issues": []map[string]interface{}{
				{"key": "PROJ-1", "summary": "First issue"},
				{"key": "PROJ-2", "summary": "Second issue"},
			},
		}, nil
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

type DatabaseConnector struct {
	BaseConnector
	driver string
	dsn    string
}

func NewDatabaseConnector(id string) Connector {
	return &DatabaseConnector{
		BaseConnector: BaseConnector{
			id:            id,
			connectorType: "database",
			Config:        ConnectorConfig{Retries: 3, Timeout: 30 * time.Second},
			events:        make(chan ConnectorEvent, 100),
		},
	}
}

func (d *DatabaseConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	if driver, ok := config["driver"].(string); ok {
		d.driver = driver
	}
	if dsn, ok := config["dsn"].(string); ok {
		d.dsn = dsn
	}
	d.setHealthy(true, 0)
	return nil
}

func (d *DatabaseConnector) Disconnect() error {
	d.setHealthy(false, 0)
	return nil
}

func (d *DatabaseConnector) Execute(ctx context.Context, action string, params map[string]interface{}) (interface{}, error) {
	start := time.Now()
	defer func() {
		d.setHealthy(true, time.Since(start))
	}()

	switch action {
	case "query":
		return map[string]interface{}{
			"rows": []map[string]interface{}{
				{"id": 1, "name": "test"},
				{"id": 2, "name": "test2"},
			},
			"count": 2,
		}, nil
	case "execute":
		return map[string]interface{}{
			"affected_rows": 1,
		}, nil
	default:
		return nil, fmt.Errorf("unknown action: %s", action)
	}
}

func init() {
	RegisterConnectorFactory("slack", NewSlackConnector)
	RegisterConnectorFactory("github", NewGitHubConnector)
	RegisterConnectorFactory("jira", NewJiraConnector)
	RegisterConnectorFactory("database", NewDatabaseConnector)
}
