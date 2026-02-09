package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

type MCP struct {
	tools     map[string]Tool
	resources map[string]Resource
	prompts   map[string]Prompt
	sessions  map[string]*Session
	config    MCPConfig
	mu        sync.RWMutex
}

type MCPConfig struct {
	MaxSessions     int
	Timeout         time.Duration
	SamplingEnabled bool
}

type Tool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema InputSchema `json:"inputSchema"`
	Handler     ToolHandler `json:"-"`
}

type InputSchema struct {
	Type       string              `json:"type"`
	Properties map[string]Property `json:"properties"`
	Required   []string            `json:"required"`
}

type Property struct {
	Type        string      `json:"type"`
	Description string      `json:"description"`
	Enum        []string    `json:"enum,omitempty"`
	Default     interface{} `json:"default,omitempty"`
}

type ToolHandler func(ctx context.Context, args map[string]interface{}) (*ToolResult, error)

type ToolResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MimeType    string `json:"mimeType"`
	Handler     ResourceHandler
}

type ResourceHandler func(ctx context.Context, uri string) (*ResourceResult, error)

type ResourceResult struct {
	Contents []ResourceContent `json:"contents"`
}

type ResourceContent struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType"`
	Text     string `json:"text"`
}

type Prompt struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Arguments   []Argument `json:"arguments"`
}

type Argument struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
}

type Session struct {
	ID         string
	Messages   []Message
	State      SessionState
	CreatedAt  time.Time
	LastActive time.Time
}

type SessionState string

const (
	SessionActive SessionState = "active"
	SessionIdle   SessionState = "idle"
	SessionClosed SessionState = "closed"
)

type Message struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type Request struct {
	Method    string          `json:"method"`
	Params    json.RawMessage `json:"params"`
	SessionID string          `json:"sessionId,omitempty"`
}

type Response struct {
	Result interface{} `json:"result,omitempty"`
	Error  *Error      `json:"error,omitempty"`
}

type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func NewMCP(cfg MCPConfig) *MCP {
	return &MCP{
		tools:     make(map[string]Tool),
		resources: make(map[string]Resource),
		prompts:   make(map[string]Prompt),
		sessions:  make(map[string]*Session),
		config:    cfg,
	}
}

func (m *MCP) RegisterTool(tool Tool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tools[tool.Name] = tool
}

func (m *MCP) RegisterResource(resource Resource) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.resources[resource.URI] = resource
}

func (m *MCP) RegisterPrompt(prompt Prompt) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.prompts[prompt.Name] = prompt
}

func (m *MCP) HandleRequest(ctx context.Context, req Request) (*Response, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	switch req.Method {
	case "initialize":
		return m.handleInitialize(req)
	case "tools/list":
		return m.handleListTools(req)
	case "tools/call":
		return m.handleCallTool(req)
	case "resources/list":
		return m.handleListResources(req)
	case "resources/read":
		return m.handleReadResource(req)
	case "prompts/list":
		return m.handleListPrompts(req)
	case "prompts/get":
		return m.handleGetPrompt(req)
	case "sampling/createMessage":
		return m.handleSampling(req)
	default:
		return nil, fmt.Errorf("unknown method: %s", req.Method)
	}
}

func (m *MCP) handleInitialize(req Request) (*Response, error) {
	return &Response{
		Result: map[string]interface{}{
			"protocolVersion": "2024-01-01",
			"capabilities": map[string]interface{}{
				"tools":     true,
				"resources": true,
				"prompts":   true,
				"sampling":  true,
			},
			"serverInfo": map[string]string{
				"name":    "Buggy AI MCP",
				"version": "1.0.0",
			},
		},
	}, nil
}

func (m *MCP) handleListTools(req Request) (*Response, error) {
	tools := make([]Tool, 0, len(m.tools))
	for _, tool := range m.tools {
		tools = append(tools, Tool{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		})
	}

	return &Response{
		Result: map[string]interface{}{
			"tools": tools,
		},
	}, nil
}

func (m *MCP) handleCallTool(req Request) (*Response, error) {
	var params struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}

	if err := json.Unmarshal(req.Params, &params); err != nil {
		return nil, err
	}

	tool, ok := m.tools[params.Name]
	if !ok {
		return nil, fmt.Errorf("tool not found: %s", params.Name)
	}

	result, err := tool.Handler(context.Background(), params.Arguments)
	if err != nil {
		return &Response{
			Error: &Error{
				Code:    -32603,
				Message: err.Error(),
			},
		}, nil
	}

	return &Response{
		Result: result,
	}, nil
}

func (m *MCP) handleListResources(req Request) (*Response, error) {
	resources := make([]Resource, 0, len(m.resources))
	for _, res := range m.resources {
		resources = append(resources, Resource{
			URI:         res.URI,
			Name:        res.Name,
			Description: res.Description,
			MimeType:    res.MimeType,
		})
	}

	return &Response{
		Result: map[string]interface{}{
			"resources": resources,
		},
	}, nil
}

func (m *MCP) handleReadResource(req Request) (*Response, error) {
	var params struct {
		URI string `json:"uri"`
	}

	if err := json.Unmarshal(req.Params, &params); err != nil {
		return nil, err
	}

	res, ok := m.resources[params.URI]
	if !ok {
		return nil, fmt.Errorf("resource not found: %s", params.URI)
	}

	result, err := res.Handler(context.Background(), params.URI)
	if err != nil {
		return nil, err
	}

	return &Response{
		Result: result,
	}, nil
}

func (m *MCP) handleListPrompts(req Request) (*Response, error) {
	prompts := make([]Prompt, 0, len(m.prompts))
	for _, p := range m.prompts {
		prompts = append(prompts, Prompt{
			Name:        p.Name,
			Description: p.Description,
			Arguments:   p.Arguments,
		})
	}

	return &Response{
		Result: map[string]interface{}{
			"prompts": prompts,
		},
	}, nil
}

func (m *MCP) handleGetPrompt(req Request) (*Response, error) {
	var params struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}

	if err := json.Unmarshal(req.Params, &params); err != nil {
		return nil, err
	}

	prompt, ok := m.prompts[params.Name]
	if !ok {
		return nil, fmt.Errorf("prompt not found: %s", params.Name)
	}

	return &Response{
		Result: map[string]interface{}{
			"description": prompt.Description,
			"messages": []map[string]interface{}{
				{
					"role": "user",
					"content": map[string]interface{}{
						"type": "text",
						"text": fmt.Sprintf("Prompt: %s", prompt.Description),
					},
				},
			},
		},
	}, nil
}

func (m *MCP) handleSampling(req Request) (*Response, error) {
	return &Response{
		Result: map[string]interface{}{
			"role": "assistant",
			"content": []map[string]interface{}{
				{
					"type": "text",
					"text": "Sampled response from AI",
				},
			},
		},
	}, nil
}

func RegisterBuiltInTools(mcp *MCP) {
	mcp.RegisterTool(Tool{
		Name:        "bash",
		Description: "Execute bash commands",
		InputSchema: InputSchema{
			Type: "object",
			Properties: map[string]Property{
				"command": {Type: "string", Description: "Command to execute"},
			},
			Required: []string{"command"},
		},
		Handler: func(ctx context.Context, args map[string]interface{}) (*ToolResult, error) {
			return &ToolResult{
				Content: []ContentBlock{
					{Type: "text", Text: "Command executed"},
				},
			}, nil
		},
	})

	mcp.RegisterTool(Tool{
		Name:        "read_file",
		Description: "Read a file from the filesystem",
		InputSchema: InputSchema{
			Type: "object",
			Properties: map[string]Property{
				"path": {Type: "string", Description: "File path to read"},
			},
			Required: []string{"path"},
		},
		Handler: func(ctx context.Context, args map[string]interface{}) (*ToolResult, error) {
			return &ToolResult{
				Content: []ContentBlock{
					{Type: "text", Text: "File content"},
				},
			}, nil
		},
	})

	mcp.RegisterTool(Tool{
		Name:        "write_file",
		Description: "Write content to a file",
		InputSchema: InputSchema{
			Type: "object",
			Properties: map[string]Property{
				"path":    {Type: "string", Description: "File path"},
				"content": {Type: "string", Description: "Content to write"},
			},
			Required: []string{"path", "content"},
		},
		Handler: func(ctx context.Context, args map[string]interface{}) (*ToolResult, error) {
			return &ToolResult{
				Content: []ContentBlock{
					{Type: "text", Text: "File written successfully"},
				},
			}, nil
		},
	})

	mcp.RegisterTool(Tool{
		Name:        "search",
		Description: "Search for information",
		InputSchema: InputSchema{
			Type: "object",
			Properties: map[string]Property{
				"query": {Type: "string", Description: "Search query"},
			},
			Required: []string{"query"},
		},
		Handler: func(ctx context.Context, args map[string]interface{}) (*ToolResult, error) {
			return &ToolResult{
				Content: []ContentBlock{
					{Type: "text", Text: "Search results"},
				},
			}, nil
		},
	})

	mcp.RegisterResource(Resource{
		URI:         "file:///etc/hostname",
		Name:        "Hostname",
		Description: "System hostname file",
		MimeType:    "text/plain",
		Handler: func(ctx context.Context, uri string) (*ResourceResult, error) {
			return &ResourceResult{
				Contents: []ResourceContent{
					{URI: uri, MimeType: "text/plain", Text: "buggy-ai"},
				},
			}, nil
		},
	})

	mcp.RegisterPrompt(Prompt{
		Name:        "code_review",
		Description: "Review code for best practices and potential issues",
		Arguments: []Argument{
			{Name: "language", Description: "Programming language", Required: false},
		},
	})

	mcp.RegisterPrompt(Prompt{
		Name:        "debug",
		Description: "Debug and fix code issues",
		Arguments: []Argument{
			{Name: "error", Description: "Error message", Required: true},
		},
	})
}
