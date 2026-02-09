package memory

import (
	"sync"
	"time"
)

type Message struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

type Config struct {
	Enabled     bool
	MaxMessages int
	PersistPath string
}

type Memory struct {
	Messages []Message
	Config   *Config
	mu       sync.RWMutex
}

func New(config *Config) *Memory {
	return &Memory{
		Messages: []Message{},
		Config:   config,
	}
}

func (m *Memory) AddMessage(msg Message) {
	m.mu.Lock()
	defer m.mu.Unlock()
	msg.Timestamp = time.Now()
	m.Messages = append(m.Messages, msg)
}

func (m *Memory) GetMessages() ([]Message, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.Messages, nil
}

func (m *Memory) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Messages = []Message{}
}
