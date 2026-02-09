package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"buggy-ai/internal/platform/integration"
	"buggy-ai/internal/platform/pipeline"
	"buggy-ai/internal/platform/scale"
)

type APIServer struct {
	hub    *integration.Hub
	engine *pipeline.Engine
	mu     sync.RWMutex
	config *APIConfig
}

type APIConfig struct {
	Port         int
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	APIKeys      []string
}

type Handler func(w http.ResponseWriter, r *http.Request)

func main() {
	config := &APIConfig{
		Port:         8080,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		APIKeys:      []string{},
	}

	server := &APIServer{
		hub:    integration.GetDefaultHub(),
		engine: pipeline.NewEngine(),
		config: config,
	}

	mux := http.NewServeMux()

	mux.HandleFunc("/api/v1/status", server.handleStatus)
	mux.HandleFunc("/api/v1/health", server.handleHealth)
	mux.HandleFunc("/api/v1/integrations", server.handleIntegrations)
	mux.HandleFunc("/api/v1/integrations/", server.handleIntegrationAction)
	mux.HandleFunc("/api/v1/pipelines", server.handlePipelines)
	mux.HandleFunc("/api/v1/pipelines/", server.handlePipelineAction)
	mux.HandleFunc("/api/v1/cluster/health", server.handleClusterHealth)

	mux.HandleFunc("/", server.handleIndex)

	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", config.Port),
		Handler:      mux,
		ReadTimeout:  config.ReadTimeout,
		WriteTimeout: config.WriteTimeout,
	}

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		srv.Shutdown(ctx)
	}()

	log.Printf("🚀 Buggy Platform API Server starting on :%d", config.Port)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func (s *APIServer) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	fmt.Fprintf(w, `🐛 Buggy Platform API
═══════════════════════════════════

Endpoints:
  GET  /api/v1/status           - Platform status
  GET  /api/v1/health            - Health check
  GET  /api/v1/integrations      - List integrations
  POST /api/v1/integrations/{type} - Connect integration
  DELETE /api/v1/integrations/{id} - Disconnect integration
  GET  /api/v1/pipelines         - List pipelines
  POST /api/v1/pipelines         - Create pipeline
  POST /api/v1/pipelines/{id}/execute - Execute pipeline
  GET  /api/v1/cluster/health    - Cluster health
`)
}

func (s *APIServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	response := map[string]interface{}{
		"status":       "healthy",
		"version":      "1.0.0",
		"uptime":       time.Since(time.Now()).String(),
		"integrations": len(s.hub.ListConnectors()),
		"pipelines":    len(s.engine.ListPipelines()),
	}

	json.NewEncoder(w).Encode(response)
}

func (s *APIServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	response := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"checks": map[string]string{
			"hub":     "healthy",
			"engine":  "healthy",
			"cluster": "not_configured",
		},
	}

	json.NewEncoder(w).Encode(response)
}

func (s *APIServer) handleIntegrations(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		integrations := s.hub.ListConnectors()
		list := make([]map[string]interface{}, 0)

		for _, conn := range integrations {
			list = append(list, map[string]interface{}{
				"id":     conn.GetID(),
				"type":   conn.GetType(),
				"status": "connected",
			})
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"integrations": list,
			"total":        len(list),
		})

	case "POST":
		var req struct {
			Type   string                 `json:"type"`
			ID     string                 `json:"id"`
			Config map[string]interface{} `json:"config"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		conn, err := s.hub.CreateConnector(req.Type, req.ID, req.Config)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to connect: %v", err), http.StatusBadRequest)
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "connected",
			"id":     conn.GetID(),
			"type":   conn.GetType(),
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *APIServer) handleIntegrationAction(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Path[len("/api/v1/integrations/"):]

	switch r.Method {
	case "GET":
		conn, ok := s.hub.GetConnector(id)
		if !ok {
			http.Error(w, "Integration not found", http.StatusNotFound)
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     conn.GetID(),
			"type":   conn.GetType(),
			"status": "connected",
		})

	case "DELETE":
		if err := s.hub.RemoveConnector(id); err != nil {
			http.Error(w, "Integration not found", http.StatusNotFound)
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "disconnected",
			"id":     id,
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *APIServer) handlePipelines(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch r.Method {
	case "GET":
		pipelines := s.engine.ListPipelines()
		list := make([]map[string]interface{}, 0)

		for _, p := range pipelines {
			list = append(list, map[string]interface{}{
				"id":     p.ID,
				"name":   p.Name,
				"stages": len(p.Stages),
				"status": string(p.Status),
			})
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"pipelines": list,
			"total":     len(list),
		})

	case "POST":
		var req struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		p := s.engine.CreatePipeline(req.ID, req.Name, &pipeline.PipelineConfig{
			MaxConcurrent: 5,
		})

		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     p.ID,
			"name":   p.Name,
			"status": string(p.Status),
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *APIServer) handlePipelineAction(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	pipelineID := ""
	action := ""

	if len(path) > len("/api/v1/pipelines/") {
		remaining := path[len("/api/v1/pipelines/"):]
		for i, c := range remaining {
			if c == '/' {
				pipelineID = remaining[:i]
				action = remaining[i+1:]
				break
			}
		}
		if pipelineID == "" {
			pipelineID = remaining
		}
	}

	switch {
	case r.Method == "POST" && action == "execute":
		exec, err := s.engine.Execute(context.Background(), pipelineID, &pipeline.Trigger{
			Type: pipeline.TriggerAPI,
		})
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to execute: %v", err), http.StatusBadRequest)
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"execution_id": exec.ID,
			"status":       string(exec.Status),
			"started":      true,
		})

	case r.Method == "GET":
		p, ok := s.engine.GetPipeline(pipelineID)
		if !ok {
			http.Error(w, "Pipeline not found", http.StatusNotFound)
			return
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     p.ID,
			"name":   p.Name,
			"stages": len(p.Stages),
			"status": string(p.Status),
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *APIServer) handleClusterHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	response := map[string]interface{}{
		"status":      "not_configured",
		"total_nodes": 0,
		"alive_nodes": 0,
		"quorum":      false,
		"message":     "Cluster mode not enabled",
	}

	json.NewEncoder(w).Encode(response)
}

var _ = scale.NewShardManager
