package gateway

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

type Gateway struct {
	config      GatewayConfig
	routes      map[string]*Route
	auth        Authenticator
	rateLimiter *RateLimiter
	metrics     Metrics
	mu          sync.RWMutex
}

type GatewayConfig struct {
	Port         int
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	MaxConns     int
	RateLimit    int
	RateWindow   time.Duration
	APIKeys      []string
	EnableAuth   bool
	EnableCORS   bool
	CORSOrigins  []string
}

type Route struct {
	Path       string
	Methods    []string
	Handler    http.Handler
	Middleware []Middleware
	RateLimit  int
	Auth       bool
}

type Middleware func(http.Handler) http.Handler

type Authenticator struct {
	APIKeys map[string]string
	JWTKey  string
}

type RateLimiter struct {
	limit    int
	window   time.Duration
	requests map[string][]time.Time
	mu       sync.Mutex
}

type RateLimitResult struct {
	Allowed   bool
	Remaining int
	ResetAt   time.Time
}

type Metrics struct {
	RequestsTotal   int64
	RequestsSuccess int64
	RequestsError   int64
	LatencyAvg      float64
	LatencyP95      float64
	RateLimitHits   int64
	AuthFailures    int64
	BytesIn         int64
	BytesOut        int64
	mu              sync.RWMutex
}

type Upstream struct {
	Name    string
	URL     string
	Health  UpstreamHealth
	Weight  int
	Timeout time.Duration
}

type UpstreamHealth struct {
	Status    string
	Checks    int
	Failures  int
	LastCheck time.Time
}

type LoadBalancer struct {
	upstreams []*Upstream
	current   int
	mu        sync.RWMutex
}

func NewGateway(cfg GatewayConfig) *Gateway {
	return &Gateway{
		config:      cfg,
		routes:      make(map[string]*Route),
		auth:        NewAuthenticator(cfg.APIKeys),
		rateLimiter: NewRateLimiter(cfg.RateLimit, cfg.RateWindow),
	}
}

func NewAuthenticator(apiKeys []string) Authenticator {
	auth := Authenticator{
		APIKeys: make(map[string]string),
	}
	for _, key := range apiKeys {
		auth.APIKeys[key] = "active"
	}
	return auth
}

func NewRateLimiter(limit int, window time.Duration) *RateLimiter {
	return &RateLimiter{
		limit:    limit,
		window:   window,
		requests: make(map[string][]time.Time),
	}
}

func (g *Gateway) RegisterRoute(route *Route) {
	g.mu.Lock()
	g.routes[route.Path] = route
	g.mu.Unlock()
}

func (g *Gateway) Handle(path string, handler http.Handler, methods []string) {
	g.RegisterRoute(&Route{
		Path:    path,
		Methods: methods,
		Handler: handler,
	})
}

func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	g.mu.RLock()
	route, ok := g.routes[r.URL.Path]
	g.mu.RUnlock()

	if !ok {
		http.NotFound(w, r)
		return
	}

	if !contains(route.Methods, r.Method) {
		http.Error(w, "Method not allowed", 405)
		return
	}

	if route.Auth && g.config.EnableAuth {
		if !g.authenticate(r) {
			http.Error(w, "Unauthorized", 401)
			return
		}
	}

	if route.RateLimit > 0 {
		clientIP := getClientIP(r)
		result := g.rateLimiter.Allow(clientIP)
		if !result.Allowed {
			w.Header().Set("Retry-After", fmt.Sprintf("%d", int(result.ResetAt.Sub(time.Now()).Seconds())))
			http.Error(w, "Rate limit exceeded", 429)
			return
		}
	}

	rec := &responseRecorder{ResponseWriter: w, status: 200, bytes: 0}
	for _, mw := range route.Middleware {
		route.Handler = mw(route.Handler)
	}
	route.Handler.ServeHTTP(rec, r)

	g.recordMetrics(r, rec.status, time.Since(start), rec.bytes)
}

func (g *Gateway) authenticate(r *http.Request) bool {
	apiKey := r.Header.Get("X-API-Key")
	if apiKey == "" {
		apiKey = r.URL.Query().Get("api_key")
	}
	_, valid := g.auth.APIKeys[apiKey]
	return valid
}

type responseRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (r *responseRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func (r *responseRecorder) Write(data []byte) (int, error) {
	n, err := r.ResponseWriter.Write(data)
	r.bytes += n
	return n, err
}

func (g *Gateway) recordMetrics(r *http.Request, status int, latency time.Duration, bytes int) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.metrics.RequestsTotal++
	if status < 400 {
		g.metrics.RequestsSuccess++
	} else {
		g.metrics.RequestsError++
	}
	g.metrics.LatencyAvg = (g.metrics.LatencyAvg*float64(g.metrics.RequestsTotal-1) + float64(latency.Microseconds())) / float64(g.metrics.RequestsTotal)
	g.metrics.BytesIn += int64(r.ContentLength)
	g.metrics.BytesOut += int64(bytes)
}

func (g *RateLimiter) Allow(key string) RateLimitResult {
	g.mu.Lock()
	defer g.mu.Unlock()

	now := time.Now()
	cutoff := now.Add(-g.window)

	var valid []time.Time
	for _, t := range g.requests[key] {
		if t.After(cutoff) {
			valid = append(valid, t)
		}
	}
	g.requests[key] = valid

	remaining := g.limit - len(valid)
	if remaining <= 0 {
		return RateLimitResult{
			Allowed:   false,
			Remaining: 0,
			ResetAt:   now.Add(g.window),
		}
	}

	g.requests[key] = append(g.requests[key], now)
	return RateLimitResult{
		Allowed:   true,
		Remaining: remaining - 1,
		ResetAt:   now.Add(g.window),
	}
}

func NewLoadBalancer(upstreams []*Upstream) *LoadBalancer {
	return &LoadBalancer{
		upstreams: upstreams,
		current:   0,
	}
}

func (lb *LoadBalancer) GetUpstream() *Upstream {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	lb.current = (lb.current + 1) % len(lb.upstreams)
	return lb.upstreams[lb.current]
}

func (lb *LoadBalancer) HealthCheck() {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	for _, upstream := range lb.upstreams {
		resp, err := http.Get(upstream.URL + "/health")
		if err != nil || resp.StatusCode != 200 {
			upstream.Health.Failures++
			upstream.Health.Status = "unhealthy"
		} else {
			upstream.Health.Status = "healthy"
			upstream.Health.Checks++
		}
		upstream.Health.LastCheck = time.Now()
	}
}

func (g *Gateway) GetMetrics() Metrics {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return Metrics{
		RequestsTotal:   g.metrics.RequestsTotal,
		RequestsSuccess: g.metrics.RequestsSuccess,
		RequestsError:   g.metrics.RequestsError,
		LatencyAvg:      g.metrics.LatencyAvg,
		LatencyP95:      g.metrics.LatencyP95,
		RateLimitHits:   g.metrics.RateLimitHits,
		AuthFailures:    g.metrics.AuthFailures,
		BytesIn:         g.metrics.BytesIn,
		BytesOut:        g.metrics.BytesOut,
	}
}

func (g *Gateway) StartHealthCheck(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Health check logic here
		}
	}
}

type HealthCheckHandler struct {
	upstream *Upstream
	mu       sync.RWMutex
}

func (h *HealthCheckHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(200)
	w.Write([]byte("OK"))
}

func (h *HealthCheckHandler) Check() {
	h.mu.Lock()
	defer h.mu.Unlock()

	resp, err := http.Get(h.upstream.URL + "/health")
	if err != nil {
		h.upstream.Health.Status = "unhealthy"
		return
	}
	defer resp.Body.Close()

	h.upstream.Health.Status = "healthy"
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func getClientIP(r *http.Request) string {
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		return strings.Split(xff, ",")[0]
	}
	return r.RemoteAddr
}

func (g *Gateway) AddMiddleware(path string, mw Middleware) {
	g.mu.RLock()
	route, ok := g.routes[path]
	g.mu.RUnlock()

	if ok {
		route.Middleware = append(route.Middleware, mw)
	}
}

func CORS(origins []string) Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Access-Control-Allow-Origin", strings.Join(origins, ", "))
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
			if r.Method == "OPTIONS" {
				w.WriteHeader(204)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func (g *Gateway) AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !g.authenticate(r) {
			http.Error(w, "Unauthorized", 401)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		fmt.Printf("%s %s %v\n", r.Method, r.URL.Path, time.Since(start))
	})
}

func TimeoutMiddleware(timeout time.Duration) Middleware {
	return func(next http.Handler) http.Handler {
		return http.TimeoutHandler(next, timeout, "Request timeout")
	}
}
