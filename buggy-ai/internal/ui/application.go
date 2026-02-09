package ui

import (
	"fmt"
	"time"
)

type Message struct {
	Role    string
	Content string
	Time    time.Time
}

type Engine interface {
	SendMessage(message string) (string, error)
	GetTools() []string
}

type Application struct {
	engine   Engine
	messages []Message
}

func NewApplication(engine Engine) *Application {
	return &Application{
		engine:   engine,
		messages: []Message{},
	}
}

func (a *Application) AddMessage(role, content string) {
	a.messages = append(a.messages, Message{
		Role:    role,
		Content: content,
		Time:    time.Now(),
	})
}

func (a *Application) Render() string {
	var output string
	output += "🐛 Buggy AI - Terminal Platform\n"
	output += "═══════════════════════════════\n\n"

	for _, msg := range a.messages {
		prefix := "🤖"
		if msg.Role == "user" {
			prefix = "👤"
		}
		output += fmt.Sprintf("%s [%s]\n%s\n\n", prefix, msg.Role, msg.Content)
	}

	return output
}

func (a *Application) Clear() {
	a.messages = []Message{}
}

func (a *Application) GetMessages() []Message {
	return a.messages
}
