# Codebase Monitor

A dedicated server for monitoring and reporting codebase changes in real-time.

## Features

- **File Monitoring**: Watch directories for file additions, modifications, and deletions
- **Git Integration**: Track commits, branches, and repository status
- **Real-time Updates**: WebSocket-based live feed of changes
- **REST API**: Comprehensive API for fetching change history and statistics
- **Web Dashboard**: Visual interface for monitoring your codebase

## Installation

```bash
npm install
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
PORT=3000
WATCH_PATH=.
GIT_POLL_INTERVAL=30000
```

## Usage

```bash
npm start
```

The server will start on `http://localhost:3000`. Open the dashboard at `http://localhost:3000/`.

## API Endpoints

### GET /api/health
Check server health status.

### GET /api/status
Get comprehensive status including file watcher, git, and WebSocket statistics.

### GET /api/changes
Get change history.

Query parameters:
- `limit` (default: 100) - Maximum number of changes to return
- `type` - Filter by change type (added, modified, deleted)

### GET /api/git/commits
Get recent git commits.

Query parameters:
- `limit` (default: 50) - Maximum number of commits to return

### GET /api/git/status
Get current git working directory status.

### GET /api/git/stats
Get git repository statistics.

### GET /api/files/stats
Get file watcher statistics.

### GET /api/websocket/stats
Get WebSocket connection statistics.

## WebSocket Events

Connect to `ws://localhost:3000` to receive real-time updates:

- `file_change` - File added, modified, or deleted
- `git_commit` - New commit detected
- `status` - Periodic status update

## Architecture

```
codebase-monitor/
├── server.js              # Main Express server
├── services/
│   ├── FileWatcher.js     # Chokidar-based file monitoring
│   ├── GitMonitor.js      # Git repository tracking
│   └── WebSocketManager.js # WebSocket server
├── public/
│   ├── index.html         # Dashboard UI
│   ├── styles.css        # Dashboard styles
│   └── app.js            # Dashboard JavaScript
├── .env.example          # Configuration template
└── package.json
```

## Ignoring Files

The file watcher automatically ignores:
- `node_modules/`
- `.git/`
- `dist/`, `build/`
- `.cache/`, `coverage/`
- `.env`
- `package-lock.json`
- `.log` files

Add custom patterns in `server.js` if needed.

## License

MIT
