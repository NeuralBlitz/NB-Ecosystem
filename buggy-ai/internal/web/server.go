package web

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Server struct {
	config ServerConfig
	mux    *http.ServeMux
	server *http.Server
	hub    *Hub
	auth   Authenticator
	mu     sync.RWMutex
}

type ServerConfig struct {
	Port         int
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	StaticDir    string
	APIKey       string
	CORSOrigins  []string
}

type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

type Client struct {
	hub     *Hub
	send    chan []byte
	agentID string
}

type Authenticator struct {
	APIKey string
}

type APIResponse struct {
	Success   bool        `json:"success"`
	Data      interface{} `json:"data,omitempty"`
	Error     string      `json:"error,omitempty"`
	RequestID string      `json:"request_id"`
}

type WSMessage struct {
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

type ChatRequest struct {
	Message   string                 `json:"message"`
	SessionID string                 `json:"session_id,omitempty"`
	Context   map[string]interface{} `json:"context,omitempty"`
}

type ChatResponse struct {
	Response   string  `json:"response"`
	SessionID  string  `json:"session_id"`
	Tokens     int     `json:"tokens"`
	DurationMs float64 `json:"duration_ms"`
}

func NewServer(cfg ServerConfig) *Server {
	mux := http.NewServeMux()

	s := &Server{
		config: cfg,
		mux:    mux,
		hub:    NewHub(),
		auth:   Authenticator{APIKey: cfg.APIKey},
	}

	s.setupRoutes()

	s.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      s.mux,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
	}

	go s.hub.Run()

	return s
}

func NewHub() *Hub {
	return &Hub{
		broadcast:  make(chan []byte),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		clients:    make(map[*Client]bool),
	}
}

func (h *Hub) Run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()
		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					close(client.send)
					delete(h.clients, client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

func (s *Server) setupRoutes() {
	s.mux.HandleFunc("/", s.handleIndex)
	s.mux.HandleFunc("/api/v1/chat", s.handleChat)
	s.mux.HandleFunc("/api/v1/chat/history", s.handleChatHistory)
	s.mux.HandleFunc("/api/v1/tasks", s.handleTasks)
	s.mux.HandleFunc("/api/v1/tasks/", s.handleTaskStatus)
	s.mux.HandleFunc("/api/v1/execute", s.handleExecute)
	s.mux.HandleFunc("/api/v1/tools", s.handleTools)
	s.mux.HandleFunc("/api/v1/agents", s.handleAgents)
	s.mux.HandleFunc("/api/v1/config", s.handleConfig)
	s.mux.HandleFunc("/api/v1/stats", s.handleStats)
	s.mux.HandleFunc("/api/v1/health", s.handleHealth)
	s.mux.HandleFunc("/api/v1/workflow", s.handleWorkflows)
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(`<!DOCTYPE html>
<html>
<head>
    <title>Buggy AI Dashboard</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: system-ui; margin: 0; background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 100%); color: #e0e0e0; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; padding: 20px; background: rgba(255,255,255,0.05); border-radius: 15px; }
        .logo { font-size: 28px; font-weight: bold; color: #00ffff; text-shadow: 0 0 20px rgba(0,255,255,0.5); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .stat-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 25px; border-radius: 15px; border: 1px solid rgba(0,255,255,0.2); transition: transform 0.3s; }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-value { font-size: 42px; font-weight: bold; color: #00ffff; }
        .stat-label { color: #888; margin-top: 8px; font-size: 14px; }
        .card { background: rgba(26,26,46,0.8); border-radius: 15px; padding: 25px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1); }
        .card h3 { margin-top: 0; color: #00ffff; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px; }
        .chat-container { height: 500px; display: flex; flex-direction: column; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 10px; margin-bottom: 15px; }
        .message { padding: 12px 18px; margin-bottom: 10px; border-radius: 10px; max-width: 80%; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .user { background: linear-gradient(135deg, #4a00e0 0%, #8e2de2 100%); margin-left: auto; border-bottom-right-radius: 2px; }
        .assistant { background: linear-gradient(135deg, #0f3460 0%, #16213e 100%); border-bottom-left-radius: 2px; }
        .input-container { display: flex; gap: 15px; }
        input { flex: 1; padding: 18px; border: none; border-radius: 10px; background: rgba(255,255,255,0.1); color: white; font-size: 16px; outline: none; }
        input:focus { background: rgba(255,255,255,0.15); }
        button { padding: 18px 35px; background: linear-gradient(135deg, #00ffff 0%, #00cccc 100%); color: #0f0f23; border: none; border-radius: 10px; cursor: pointer; font-weight: bold; font-size: 16px; transition: all 0.3s; }
        button:hover { transform: scale(1.05); box-shadow: 0 0 30px rgba(0,255,255,0.5); }
        .tabs { display: flex; gap: 10px; margin-bottom: 25px; flex-wrap: wrap; }
        .tab { padding: 12px 25px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; color: #888; cursor: pointer; transition: all 0.3s; }
        .tab:hover { background: rgba(255,255,255,0.1); }
        .tab.active { background: linear-gradient(135deg, #00ffff 0%, #00cccc 100%); color: #0f0f23; font-weight: bold; }
        .panel { display: none; }
        .panel.active { display: block; }
        .progress-bar { height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; margin: 10px 0; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #00ffff, #00cccc); transition: width 0.3s; }
        .badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .badge-success { background: rgba(0,255,0,0.2); color: #00ff00; }
        .badge-warning { background: rgba(255,200,0,0.2); color: #ffc800; }
        .badge-error { background: rgba(255,0,0,0.2); color: #ff0000; }
        .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .online { background: #00ff00; box-shadow: 0 0 10px #00ff00; }
        .offline { background: #ff0000; }
        .section-title { font-size: 18px; margin-bottom: 15px; color: #00ffff; }
        .tool-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
        .tool-item { padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; text-align: center; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🐛 Buggy AI Platform v2.0</div>
            <div><span class="status-dot online"></span>System Online</div>
        </div>
        
        <div class="grid">
            <div class="stat-card">
                <div class="stat-value" id="active-agents">0</div>
                <div class="stat-label">🤖 Active Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="pending-tasks">0</div>
                <div class="stat-label">📋 Pending Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="running-tasks">0</div>
                <div class="stat-label">⚡ Running Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="completed-tasks">0</div>
                <div class="stat-label">✅ Completed</div>
            </div>
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">💬 AI Chat</button>
            <button class="tab" onclick="showTab('agents')">🤖 Agents</button>
            <button class="tab" onclick="showTab('tasks')">📋 Task Queue</button>
            <button class="tab" onclick="showTab('workflows')">🔄 Workflows</button>
            <button class="tab" onclick="showTab('tools')">🛠️ Tools</button>
            <button class="tab" onclick="showTab('monitor')">📊 Monitor</button>
        </div>

        <div id="chat-panel" class="panel active">
            <div class="card">
                <div class="chat-container">
                    <div class="chat-messages" id="chat-messages">
                        <div class="message assistant">👋 Hello! I'm Buggy AI. How can I help you today?</div>
                    </div>
                    <div class="input-container">
                        <input type="text" id="chat-input" placeholder="Ask me anything..." onkeypress="if(event.key==='Enter')sendMessage()">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <div id="agents-panel" class="panel">
            <div class="card">
                <h3>🤖 Autonomous Agents</h3>
                <div id="agents-list">
                    <div class="tool-item">No active agents</div>
                </div>
            </div>
        </div>

        <div id="tasks-panel" class="panel">
            <div class="card">
                <h3>📋 Task Queue</h3>
                <div id="tasks-list">
                    <div class="tool-item">No pending tasks</div>
                </div>
            </div>
        </div>

        <div id="workflows-panel" class="panel">
            <div class="card">
                <h3>🔄 Active Workflows</h3>
                <div id="workflows-list">
                    <div class="tool-item">No active workflows</div>
                </div>
            </div>
        </div>

        <div id="tools-panel" class="panel">
            <div class="card">
                <h3>🛠️ Available Tools</h3>
                <div class="tool-grid">
                    <div class="tool-item">🔧 bash</div>
                    <div class="tool-item">🐍 python</div>
                    <div class="tool-item">📝 code_an</div>
                    <div class="tool-item">🔍 search</div>
                    <div class="tool-item">📥 fetch</div>
                    <div class="tool-item">🐳 docker</div>
                    <div class="tool-item">☸️ kubernetes</div>
                    <div class="tool-item">🔐 encrypt</div>
                    <div class="tool-item">🌐 dns</div>
                    <div class="tool-item">📊 stock</div>
                </div>
            </div>
        </div>

        <div id="monitor-panel" class="panel">
            <div class="card">
                <h3>📊 System Monitor</h3>
                <div class="section-title">CPU Usage</div>
                <div class="progress-bar"><div class="progress-fill" style="width: 45%"></div></div>
                <div class="section-title">Memory Usage</div>
                <div class="progress-bar"><div class="progress-fill" style="width: 62%"></div></div>
                <div class="section-title">Disk Usage</div>
                <div class="progress-bar"><div class="progress-fill" style="width: 78%"></div></div>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket('ws://' + location.host + '/ws');
        let sessionId = '';
        
        ws.onopen = () => { document.querySelector('.status-dot').className = 'status-dot online'; };
        ws.onclose = () => { document.querySelector('.status-dot').className = 'status-dot offline'; };
        
        ws.onmessage = function(event) {
            const msg = JSON.parse(event.data);
            if (msg.type === 'chat') {
                addMessage('assistant', msg.payload.response);
            } else if (msg.type === 'stats') {
                updateStats(msg.payload);
            } else if (msg.type === 'pong') {
                // Heartbeat response
            }
        };
        
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            
            addMessage('assistant', 'Processing...');
            
            setTimeout(() => {
                const lastMsg = document.querySelector('.message:last-child');
                lastMsg.textContent = 'Buggy AI: I received your message: "' + message + '"';
            }, 1000);
        }
        
        function addMessage(role, text) {
            const container = document.getElementById('chat-messages');
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.textContent = text;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        function updateStats(stats) {
            document.getElementById('active-agents').textContent = stats.agents || 0;
            document.getElementById('pending-tasks').textContent = stats.pending || 0;
            document.getElementById('running-tasks').textContent = stats.running || 0;
            document.getElementById('completed-tasks').textContent = stats.completed || 0;
        }
        
        function showTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tab + '-panel').classList.add('active');
        }
        
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'ping'}));
            }
        }, 30000);
        
        // Initial stats fetch
        fetch('/api/v1/stats').then(r => r.json()).then(updateStats);
    </script>
</body>
</html>`))
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req ChatRequest
	json.NewDecoder(r.Body).Decode(&req)

	response := ChatResponse{
		Response:   "Buggy AI: " + req.Message,
		SessionID:  req.SessionID,
		Tokens:     150,
		DurationMs: 0.5,
	}

	s.respondJSON(w, response)
}

func (s *Server) handleChatHistory(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, []interface{}{})
}

func (s *Server) handleTasks(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, []interface{}{})
}

func (s *Server) handleTaskStatus(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/v1/tasks/")
	w.Write([]byte(`{"task_id":"` + id + `","status":"pending"}`))
}

func (s *Server) handleExecute(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, map[string]string{"task_id": generateID(), "status": "submitted"})
}

func (s *Server) handleTools(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, []string{"bash", "python", "search", "fetch", "docker", "k8s", "encrypt"})
}

func (s *Server) handleAgents(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, []interface{}{})
}

func (s *Server) handleConfig(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, map[string]interface{}{
		"version":   "2.0.0",
		"layers":    10,
		"features":  []string{"voice", "mcp", "runner", "orchestrator", "web"},
		"dashboard": true,
	})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, map[string]interface{}{
		"agents":    0,
		"pending":   0,
		"running":   0,
		"completed": 0,
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("OK"))
}

func (s *Server) handleWorkflows(w http.ResponseWriter, r *http.Request) {
	s.respondJSON(w, []interface{}{})
}

func (s *Server) respondJSON(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(APIResponse{
		Success:   true,
		Data:      data,
		RequestID: generateID(),
	})
}

func (s *Server) Start() error {
	return s.server.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.server.Shutdown(ctx)
}

func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}
