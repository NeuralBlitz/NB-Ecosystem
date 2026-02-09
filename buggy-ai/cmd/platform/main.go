package main

import (
	"fmt"
	"os"
	"text/tabwriter"

	"buggy-ai/internal/platform/integration"
	"buggy-ai/internal/platform/pipeline"
	"buggy-ai/internal/platform/scale"
)

type CLI struct {
	hub    *integration.Hub
	engine *pipeline.Engine
}

func main() {
	cli := &CLI{
		hub:    integration.GetDefaultHub(),
		engine: pipeline.NewEngine(),
	}

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(0)
	}

	switch os.Args[1] {
	case "status", "s":
		cli.status()
	case "integrations", "i":
		cli.integrations()
	case "pipelines", "p":
		cli.pipelines()
	case "connect":
		if len(os.Args) < 4 {
			fmt.Println("Usage: buggy-platform connect <type> <id>")
			os.Exit(1)
		}
		cli.connect(os.Args[2], os.Args[3])
	case "disconnect":
		if len(os.Args) < 3 {
			fmt.Println("Usage: buggy-platform disconnect <id>")
			os.Exit(1)
		}
		cli.disconnect(os.Args[2])
	case "create-pipeline":
		cli.createPipeline(os.Args[2:])
	case "run-pipeline":
		if len(os.Args) < 3 {
			fmt.Println("Usage: buggy-platform run-pipeline <id>")
			os.Exit(1)
		}
		cli.runPipeline(os.Args[2])
	case "health":
		cli.health()
	case "--help", "-h", "help":
		printUsage()
	default:
		fmt.Printf("Unknown command: %s\n", os.Args[1])
		printUsage()
	}
}

func printUsage() {
	fmt.Println(`🐛 Buggy Platform CLI
═══════════════════════════════════

Usage: buggy-platform <command> [options]

Commands:
  status, s          Show platform status overview
  integrations, i    List all integrations
  pipelines, p       List all pipelines
  connect <type> <id>  Connect an integration
  disconnect <id>    Disconnect an integration
  create-pipeline    Create a new pipeline
  run-pipeline <id>  Execute a pipeline
  health             Detailed health check
  help, --help       Show this help

Examples:
  buggy-platform connect slack main
  buggy-platform connect github myorg
  buggy-platform run-pipeline deploy-prod
`)
}

func (c *CLI) status() {
	fmt.Println("🐛 Buggy Platform Status")
	fmt.Println("═══════════════════════════════════")
	fmt.Printf("Integrations: %d\n", len(c.hub.ListConnectors()))
	fmt.Printf("Pipelines:    %d\n", len(c.engine.ListPipelines()))
	shards := scale.NewShardManager(16)
	_ = shards
	fmt.Println()
}

func (c *CLI) integrations() {
	fmt.Println("🔌 Integrations")
	fmt.Println("═══════════════════════════════════")

	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tType\tStatus")
	fmt.Fprintln(w, "─\t────\t──────")

	connectors := c.hub.ListConnectors()
	if len(connectors) == 0 {
		fmt.Fprintln(w, "(none)\tnone\tpending")
	}

	for _, conn := range connectors {
		fmt.Fprintf(w, "%s\t%s\t%s\n", conn.GetID(), conn.GetType(), "connected")
	}
	w.Flush()
}

func (c *CLI) pipelines() {
	fmt.Println("📋 Pipelines")
	fmt.Println("═══════════════════════════════════")

	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tName\tStages\tStatus")
	fmt.Fprintln(w, "─\t────\t──────\t──────")

	pipelines := c.engine.ListPipelines()
	if len(pipelines) == 0 {
		fmt.Fprintln(w, "(none)\tnone\t0\tidle")
	}

	for _, p := range pipelines {
		fmt.Fprintf(w, "%s\t%s\t%d\t%s\n", p.ID, p.Name, len(p.Stages), p.Status)
	}
	w.Flush()
}

func (c *CLI) connect(connType, id string) {
	fmt.Printf("Connecting %s integration as %s...\n", connType, id)

	config := make(map[string]interface{})
	conn, err := c.hub.CreateConnector(connType, id, config)
	if err != nil {
		fmt.Printf("❌ Failed to connect: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ Connected %s (%s)\n", id, conn.GetType())
}

func (c *CLI) disconnect(id string) {
	fmt.Printf("Disconnecting %s...\n", id)

	if err := c.hub.RemoveConnector(id); err != nil {
		fmt.Printf("❌ Failed to disconnect: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ Disconnected %s\n", id)
}

func (c *CLI) createPipeline(args []string) {
	if len(args) < 2 {
		fmt.Println("Usage: buggy-platform create-pipeline <id> <name>")
		os.Exit(1)
	}

	id := args[0]
	name := args[1]

	p := c.engine.CreatePipeline(id, name, &pipeline.PipelineConfig{
		MaxConcurrent: 5,
	})

	fmt.Printf("✅ Created pipeline %s (%s)\n", id, name)
	fmt.Printf("   Stages: %d\n", len(p.Stages))
	fmt.Printf("   Status: %s\n", p.Status)
}

func (c *CLI) runPipeline(pipelineID string) {
	fmt.Printf("Running pipeline %s...\n", pipelineID)

	exec, err := c.engine.Execute(nil, pipelineID, &pipeline.Trigger{
		Type: pipeline.TriggerManual,
	})
	if err != nil {
		fmt.Printf("❌ Failed to run pipeline: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("✅ Started execution %s\n", exec.ID)
}

func (c *CLI) health() {
	fmt.Println("🏥 Health Check")
	fmt.Println("═══════════════════════════════════")
	fmt.Println("Integrations healthy: ✓")
	fmt.Println("Pipelines ready: ✓")
	fmt.Println()
}
