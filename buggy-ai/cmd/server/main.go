package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"buggy-ai/internal/web"
)

func main() {
	port := flag.Int("port", 8080, "Port to listen on")
	apiKey := flag.String("api-key", "", "API key for authentication")
	timeout := flag.Duration("timeout", 30*time.Second, "Request timeout")
	staticDir := flag.String("static", "", "Static file directory")

	flag.Parse()

	serverConfig := web.ServerConfig{
		Port:         *port,
		ReadTimeout:  *timeout,
		WriteTimeout: *timeout,
		StaticDir:    *staticDir,
		APIKey:       *apiKey,
	}

	server := web.NewServer(serverConfig)

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		fmt.Println("\nShutting down server...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		server.Shutdown(ctx)
		os.Exit(0)
	}()

	fmt.Printf("🐛 Buggy AI Web Dashboard v2.0\n")
	fmt.Printf("================================\n")
	fmt.Printf("🌐 Server running at http://localhost:%d\n", *port)
	fmt.Printf("📊 Dashboard: http://localhost:%d/\n", *port)
	fmt.Printf("🔌 WebSocket: ws://localhost:%d/ws\n", *port)
	fmt.Printf("\nPress Ctrl+C to stop\n\n")

	if err := server.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
