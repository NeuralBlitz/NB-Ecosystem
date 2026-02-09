package config

import (
	"os"
	"sync"
)

type Config struct {
	App      AppConfig
	Cluster  ClusterConfig
	Voice    VoiceConfig
	Security SecurityConfig
	Agents   AgentsConfig
	Database DatabaseConfig
	Cache    CacheConfig
	Queue    QueueConfig
}

type AppConfig struct {
	Name        string
	Version     string
	Environment string
	LogLevel    string
}

type ClusterConfig struct {
	Mode      string
	NodeID    string
	MasterURL string
	Nodes     []NodeConfig
}

type NodeConfig struct {
	ID     string
	Host   string
	Port   int
	Role   string
	Region string
}

type VoiceConfig struct {
	Enabled    bool
	STT        string
	TTS        string
	WakeWord   string
	Language   string
	SampleRate int
}

type SecurityConfig struct {
	Sandbox      string
	MemoryLimit  string
	CPULimit     float64
	Timeout      int
	AllowedPaths []string
}

type AgentsConfig struct {
	MaxConcurrent int
	Timeout       int
	RetryAttempts int
	PlanningDepth int
}

type DatabaseConfig struct {
	Type     string
	Host     string
	Port     int
	Database string
}

type CacheConfig struct {
	Type    string
	Host    string
	Port    int
	MaxSize int
	TTL     int
}

type QueueConfig struct {
	Type     string
	Host     string
	Port     int
	Durable  bool
	Prefetch int
}

var (
	cfg     *Config
	cfgLock sync.RWMutex
)

func Load(path string) (*Config, error) {
	cfgLock.Lock()
	defer cfgLock.Unlock()

	_, err := os.ReadFile(path)
	if err != nil {
		cfg = defaultConfig()
		return cfg, nil
	}

	cfg = &Config{}
	return cfg, nil
}

func Get() *Config {
	cfgLock.RLock()
	defer cfgLock.RUnlock()
	return cfg
}

func defaultConfig() *Config {
	return &Config{
		App: AppConfig{
			Name:        "Buggy AI",
			Version:     "1.0.0",
			Environment: "development",
			LogLevel:    "info",
		},
		Cluster: ClusterConfig{
			Mode:   "standalone",
			NodeID: "node-1",
		},
		Voice: VoiceConfig{
			Enabled:    false,
			STT:        "whisper",
			TTS:        "festival",
			Language:   "en",
			SampleRate: 16000,
		},
		Security: SecurityConfig{
			Sandbox:     "none",
			MemoryLimit: "1GB",
			CPULimit:    1.0,
			Timeout:     300,
		},
		Agents: AgentsConfig{
			MaxConcurrent: 10,
			Timeout:       300,
			RetryAttempts: 3,
			PlanningDepth: 5,
		},
	}
}

func Set(c *Config) {
	cfgLock.Lock()
	defer cfgLock.Unlock()
	cfg = c
}
