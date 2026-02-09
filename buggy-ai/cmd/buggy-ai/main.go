package main

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"buggy-ai/internal/ai"
	"buggy-ai/internal/memory"
	"buggy-ai/internal/plugins"
	"buggy-ai/internal/tools"
	"buggy-ai/internal/ui"
)

type Engine struct {
	AIProvider ai.Provider
	Memory     *memory.Memory
	Toolbox    *tools.Toolbox
	PluginMgr  *plugins.PluginManager
}

func (e *Engine) SendMessage(message string) (string, error) {
	messages, _ := e.Memory.GetMessages()
	messages = append(messages, memory.Message{Role: "user", Content: message})

	aiMessages := make([]ai.Message, len(messages))
	for i, msg := range messages {
		aiMessages[i] = ai.Message{Role: msg.Role, Content: msg.Content}
	}

	response, err := e.AIProvider.Chat(aiMessages)
	if err != nil {
		return "", err
	}

	e.Memory.AddMessage(memory.Message{Role: "assistant", Content: response})
	return response, nil
}

func (e *Engine) GetTools() []string {
	return e.Toolbox.ListAll()
}

func showDemoCharts() string {
	var result strings.Builder

	chart := ui.NewChart("Server Performance")
	chart.AddPoint("CPU", 75.5)
	chart.AddPoint("Memory", 62.3)
	chart.AddPoint("Disk", 45.8)
	chart.AddPoint("Network", 88.2)
	result.WriteString(chart.Render())
	result.WriteString("\n")

	table := ui.NewTable()
	table.Headers = []string{"Metric", "Value", "Status"}
	table.AddRow([]string{"Uptime", "5d 3h 42m", "✓"})
	table.AddRow([]string{"Load Avg", "0.45", "✓"})
	table.AddRow([]string{"Processes", "142", "✓"})
	table.AddRow([]string{"Connections", "89", "!"})
	result.WriteString(table.Render())
	result.WriteString("\n")

	progress := ui.NewProgressBar(100)
	progress.Current = 73
	result.WriteString("Task Progress: " + progress.Render())
	result.WriteString("\n\n")

	result.WriteString("Activity: " + ui.Sparkline([]float64{10, 25, 40, 35, 50, 65, 55, 70, 85, 75, 90}))

	return result.String()
}

func main() {
	fmt.Print(`🐛 Buggy AI - Terminal AI Platform
═══════════════════════════════════════
A world-class AI experience in your terminal

┌─────────────────────────────────────────┐
│ Multi-Provider AI: OpenAI | Anthropic | Gemini | Ollama | Local
│ Developer Tools: Code, Git, Docker, K8s, Databases, APIs
│ Web Tools: Fetch, Search, Scrape, DNS, IPs, Ports
│ Cloud: AWS, GCP, Azure, Kubernetes, Terraform
│ Security: Encrypt, Hash, PGP, Certificates, Secrets
│ Creative: Image, Audio, Video, Documents
│ Research: News, Weather, Stocks, Crypto, Academic
│ Productivity: Calendar, Tasks, Notes, Email, Webhooks
└─────────────────────────────────────────┘

Commands: /help /tools /demo /clear /quit
`)

	engine := &Engine{
		AIProvider: ai.NewProvider("openai"),
		Memory:     memory.New(&memory.Config{Enabled: true, MaxMessages: 100}),
		Toolbox:    tools.New(),
		PluginMgr:  plugins.New(),
	}

	app := ui.NewApplication(engine)
	app.AddMessage("system", "Welcome to Buggy AI! Your terminal AI platform.")
	app.AddMessage("system", "Try /demo to see charts and visualizations!")

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		fmt.Println("\nGoodbye! 👋")
		os.Exit(0)
	}()

	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("You: ")
		input, _ := reader.ReadString('\n')
		if len(input) > 0 {
			input = input[:len(input)-1]
		}

		if input == "" {
			continue
		}

		switch input {
		case "/quit", "/exit":
			fmt.Println("Goodbye! 👋")
			return
		case "/help":
			app.AddMessage("user", "/help")
			app.AddMessage("assistant", `🐛 Buggy AI Commands
═══════════════════════

Core:
  /help       - Show this help
  /demo       - Show charts & data visualization demo
  /clear      - Clear conversation
  /quit       - Exit

AI Providers:
  /providers  - Switch AI provider
  /model      - Set model (gpt-4, claude-3, etc)

Tools:
  /tools      - List all 50+ tools
  /categories - Browse tools by category

Productivity:
  /calendar   - View calendar
  /tasks      - Manage tasks
  /notes      - Quick notes
  /email      - Send email
  /webhook    - Create webhook

Tips:
• "Analyze this code" - Code analysis
• "Search for X" - Web search
• "Deploy to k8s" - Cloud operations
• "Encrypt message" - Security tools`)
			fmt.Println(app.Render())
			continue
		case "/demo":
			app.AddMessage("user", "/demo")
			app.AddMessage("assistant", showDemoCharts())
			fmt.Println(app.Render())
			continue
		case "/tools":
			app.AddMessage("user", "/tools")
			toolsList := engine.GetTools()
			app.AddMessage("assistant", "🛠️ Available Tools (50+):\n\n"+strings.Join(toolsList, "\n"))
			fmt.Println(app.Render())
			continue
		case "/categories":
			app.AddMessage("assistant", `📁 Tool Categories
══════════════════

[Developer] 15 tools
  code_an, code_format, git, docker, k8s, shell, api_test, json_format...

[Web] 10 tools
  fetch, search, scrape, links, ip_geo, dns, port_scan, headers, ua_parse...

[Database] 5 tools
  mongo, postgres, redis, elastic, sql_gen

[Cloud] 4 tools
  aws, gcp, azure, kubernetes

[Security] 6 tools
  encrypt, decrypt, pgp, token, secrets_scan, password_hash

[Creative] 3 tools
  image_info, colors, tts

[Research] 5 tools
  academic, news, weather, stock, crypto

[System] 5 tools
  sysinfo, process, disk, netstat, uptime`)
			fmt.Println(app.Render())
			continue
		case "/clear":
			app.Clear()
			engine.Memory.Clear()
			fmt.Println("Conversation cleared.")
			continue
		case "/providers":
			app.AddMessage("assistant", `🌐 AI Providers
═════════════

• openai    - GPT-4, GPT-3.5-Turbo
• anthropic - Claude 3 Opus, Sonnet, Haiku
• gemini    - Gemini Pro, Ultra
• ollama    - Local models (llama2, mistral)
• local     - Custom local endpoints

Set provider: /provider openai`)
			fmt.Println(app.Render())
			continue
		case "/calendar", "/tasks", "/notes", "/email", "/webhook":
			app.AddMessage("assistant", "📋 Productivity Feature\n\nThis would integrate with:\n• Google Calendar / Outlook\n• Task management (Todoist, Notion)\n• Notes (Obsidian, Notion)\n• Email (SMTP, Gmail API)\n• Webhooks (Slack, Discord)\n\nFull integration coming soon!")
			fmt.Println(app.Render())
			continue
		}

		app.AddMessage("user", input)
		fmt.Println(app.Render())

		response, _ := engine.SendMessage(input)
		app.AddMessage("assistant", response)

		fmt.Println(app.Render())
	}
}
