#!/bin/bash
# Multi-Shell Agent Launcher
# Launch multiple terminal agents that connect to the same server

AGENTS_DIR="/home/runner/workspace/codebase-monitor/agents"
mkdir -p "$AGENTS_DIR"

LAUNCH_MODE="${1:-tui}"

launch_agent() {
    local AGENT_NAME="$1"
    local AGENT_NUM="$2"
    
    echo -e "\x1b[32m[Launcher]\x1b[0m Starting agent: $AGENT_NAME ( #$AGENT_NUM )"
    
    if [ "$LAUNCH_MODE" = "tui" ]; then
        gnome-terminal -- bash -c "cd /home/runner/workspace/codebase-monitor && node multi-agent-tui.js '$AGENT_NAME' 2>&1; bash" 2>/dev/null || \
        xterm -bg black -fg white -e "cd /home/runner/workspace/codebase-monitor && node multi-agent-tui.js '$AGENT_NAME'" 2>/dev/null || \
        echo "Cannot open terminal for $AGENT_NAME - will run in background"
        node /home/runner/workspace/codebase-monitor/multi-agent-tui.js "$AGENT_NAME" &
    fi
}

if [ "$1" = "list" ]; then
    echo -e "\x1b[1;34m=== Active Agent Sessions ===\x1b[0m"
    ps aux | grep "multi-agent-tui" | grep -v grep
    exit 0
fi

if [ "$1" = "killall" ]; then
    echo -e "\x1b[31m[Launcher]\x1b[0m Killing all agent sessions..."
    pkill -f "multi-agent-tui" 2>/dev/null
    echo -e "\x1b[32mDone.\x1b[0m"
    exit 0
fi

if [ "$1" = "help" ]; then
    echo -e "\x1b[1;33mMulti-Agent Shell Launcher\x1b[0m"
    echo ""
    echo "Usage: $0 [mode] [count]"
    echo ""
    echo "Modes:"
    echo "  tui    - Launch terminal UI agents (default)"
    echo "  bg     - Launch background agents"
    echo "  list   - List running agents"
    echo "  killall - Kill all agents"
    echo "  help   - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                    # Launch 2 default agents"
    echo "  $0 tui 5              # Launch 5 terminal agents"
    echo "  $0 bg 10              # Launch 10 background agents"
    echo "  $0 list               # List running agents"
    echo "  $0 killall            # Kill all agents"
    exit 0
fi

COUNT="${2:-2}"

echo -e "\x1b[1;36m╔═══════════════════════════════════════════╗\x1b[0m"
echo -e "\x1b[1;36m║\x1b[0m \x1b[1;33mMulti-Agent Terminal Hub Launcher\x1b[0m           \x1b[1;36m║\x1b[0m"
echo -e "\x1b[1;36m║\x1b[0m \x1b[0mLaunching $COUNT agent sessions...              \x1b[1;36m║\x1b[0m"
echo -e "\x1b[1;36m╚═══════════════════════════════════════════╝\x1b[0m"

for i in $(seq 1 $COUNT); do
    AGENT_NAME="agent_$i"
    launch_agent "$AGENT_NAME" "$i"
    sleep 0.5
done

echo ""
echo -e "\x1b[32m[Launcher]\x1b[0m All agents started!"
echo -e "\x1b[90mEach agent connects to ws://localhost:3000\x1b[0m"
echo -e "\x1b[90mUse /help inside agents to see commands\x1b[0m"
