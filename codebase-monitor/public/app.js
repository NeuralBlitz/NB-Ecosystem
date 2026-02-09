class CodebaseMonitor {
    constructor() {
        this.ws = null;
        this.changes = [];
        this.isPaused = false;
        this.maxChanges = 100;

        this.initElements();
        this.initEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
    }

    initElements() {
        this.connectionStatus = document.getElementById('connectionStatus');
        this.feedContainer = document.getElementById('feedContainer');
        this.commitsContainer = document.getElementById('commitsContainer');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.clearBtn = document.getElementById('clearBtn');

        this.stats = {
            filesMonitored: document.getElementById('filesMonitored'),
            totalChanges: document.getElementById('totalChanges'),
            gitCommits: document.getElementById('gitCommits'),
            wsClients: document.getElementById('wsClients')
        };
    }

    initEventListeners() {
        this.pauseBtn.addEventListener('click', () => this.togglePause());
        this.clearBtn.addEventListener('click', () => this.clearFeed());
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.updateConnectionStatus(true);
            console.log('WebSocket connected');
        };

        this.ws.onclose = () => {
            this.updateConnectionStatus(false);
            console.log('WebSocket disconnected, reconnecting in 3s...');
            setTimeout(() => this.connectWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
    }

    updateConnectionStatus(connected) {
        const statusDot = this.connectionStatus.querySelector('.status-dot');
        const statusText = this.connectionStatus.querySelector('.status-text');

        if (connected) {
            statusDot.classList.remove('disconnected');
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusDot.classList.add('disconnected');
            statusText.textContent = 'Disconnected';
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'file_change':
                this.addChange(data.data);
                break;
            case 'git_commit':
                this.addCommit(data.data);
                break;
            case 'status':
                this.updateStats(data.data);
                break;
            case 'connected':
                console.log('Connected with client ID:', data.clientId);
                break;
        }
    }

    async loadInitialData() {
        try {
            const [status, changes, gitStats] = await Promise.all([
                this.fetchAPI('/api/status'),
                this.fetchAPI('/api/changes?limit=50'),
                this.fetchAPI('/api/git/stats').catch(() => null)
            ]);

            this.updateStats(status);

            if (changes.changes) {
                changes.changes.forEach(change => this.addChange(change, false));
            }

            if (gitStats && gitStats.totalCommits) {
                this.stats.gitCommits.textContent = gitStats.totalCommits;
                await this.loadCommits();
            } else {
                this.commitsContainer.innerHTML = '<div class="loading">Git monitoring not available</div>';
            }
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    async loadCommits() {
        try {
            const response = await this.fetchAPI('/api/git/commits?limit=20');
            this.commitsContainer.innerHTML = '';

            if (response.commits && response.commits.length > 0) {
                response.commits.forEach(commit => this.renderCommit(commit));
            } else {
                this.commitsContainer.innerHTML = '<div class="loading">No commits found</div>';
            }
        } catch (error) {
            console.error('Error loading commits:', error);
            this.commitsContainer.innerHTML = '<div class="loading">Error loading commits</div>';
        }
    }

    updateStats(data) {
        if (data.fileWatcher && data.fileWatcher.stats) {
            this.stats.filesMonitored.textContent = data.fileWatcher.stats.totalFiles || 0;
        }

        if (data.websocket && data.websocket.clients) {
            this.stats.wsClients.textContent = data.websocket.clients.totalClients || 0;
        }
    }

    addChange(change, animate = true) {
        this.changes.unshift(change);

        if (this.changes.length > this.maxChanges) {
            this.changes.pop();
        }

        if (!this.isPaused) {
            this.renderChange(change, animate);
        }

        this.stats.totalChanges.textContent = this.changes.length;
    }

    renderChange(change, animate = true) {
        if (this.isPaused) return;

        if (this.feedContainer.querySelector('.empty-state')) {
            this.feedContainer.innerHTML = '';
        }

        const item = document.createElement('div');
        item.className = `change-item ${change.type}`;
        item.innerHTML = `
            <span class="change-type ${change.type}">${change.type}</span>
            <span class="change-file">${change.file}</span>
            <span class="change-time">${this.formatTime(change.timestamp)}</span>
        `;

        this.feedContainer.insertBefore(item, this.feedContainer.firstChild);

        if (this.feedContainer.children.length > this.maxChanges) {
            this.feedContainer.removeChild(this.feedContainer.lastChild);
        }
    }

    renderCommit(commit) {
        const item = document.createElement('div');
        item.className = 'commit-item';
        item.innerHTML = `
            <span class="commit-hash">${commit.shortHash}</span>
            <div class="commit-info">
                <div class="commit-message">${this.escapeHtml(commit.message)}</div>
                <div class="commit-meta">${this.escapeHtml(commit.authorName)} • ${this.formatTime(commit.date)}</div>
            </div>
        `;
        this.commitsContainer.appendChild(item);
    }

    addCommit(commit) {
        this.renderCommit(commit);
        const currentCount = parseInt(this.stats.gitCommits.textContent) || 0;
        this.stats.gitCommits.textContent = currentCount + 1;
    }

    togglePause() {
        this.isPaused = !this.isPaused;
        this.pauseBtn.textContent = this.isPaused ? 'Resume' : 'Pause';
        this.pauseBtn.classList.toggle('paused', this.isPaused);
    }

    clearFeed() {
        this.changes = [];
        this.feedContainer.innerHTML = '<div class="empty-state">Waiting for changes...</div>';
        this.stats.totalChanges.textContent = '0';
    }

    async fetchAPI(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) {
            return 'Just now';
        } else if (diff < 3600000) {
            const minutes = Math.floor(diff / 60000);
            return `${minutes}m ago`;
        } else if (diff < 86400000) {
            const hours = Math.floor(diff / 3600000);
            return `${hours}h ago`;
        } else {
            return date.toLocaleDateString();
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new CodebaseMonitor();
});
