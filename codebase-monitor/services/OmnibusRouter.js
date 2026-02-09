const EventEmitter = require('events');
const WebSocket = require('ws');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const ALL_PROJECTS = [
  { id: 'nbx-lrs', name: 'NBX-LRS', type: 'lrs', basePath: '/home/runner/workspace/NBX-LRS', description: 'NeuralBlitz Learning Record Store' },
  { id: 'nb-ecosystem', name: 'NB-Ecosystem', type: 'frontend', basePath: '/home/runner/workspace/NB-Ecosystem', description: 'NeuralBlitz Frontend UI' },
  { id: 'nb-omnibus-router', name: 'NB-Omnibus-Router', type: 'router', basePath: '/home/runner/workspace/nb-omnibus-router', description: 'NB Omnibus Router Service' },
  { id: 'nbos', name: 'NBOS', type: 'os', basePath: '/home/runner/workspace/NBOS', description: 'NeuralBlitz Operating System' },
  { id: 'neuralblitz-core', name: 'NeuralBlitz-Core', type: 'core', basePath: '/home/runner/workspace/neuralblitz-core', description: 'NeuralBlitz Core Engine' },
  { id: 'neuralblitz_core', name: 'NeuralBlitz_Core', type: 'core', basePath: '/home/runner/workspace/neuralblitz_core', description: 'NeuralBlitz Core (Alt)' },
  { id: 'advanced-research', name: 'Advanced-Research', type: 'research', basePath: '/home/runner/workspace/Advanced-Research', description: 'Advanced Research Division' },
  { id: 'aetheria-project', name: 'Aetheria-Project', type: 'project', basePath: '/home/runner/workspace/aetheria-project', description: 'Aetheria Project' },
  { id: 'buggy-ai', name: 'Buggy-AI', type: 'sandbox', basePath: '/home/runner/workspace/buggy-ai', description: 'Buggy AI Sandbox' },
  { id: 'emergent-prompt-architecture', name: 'Emergent-Prompt-Architecture', type: 'research', basePath: '/home/runner/workspace/Emergent-Prompt-Architecture', description: 'Emergent Prompt Architecture' },
  { id: 'forge-ai', name: 'Forge-AI', type: 'sandbox', basePath: '/home/runner/workspace/Forge-ai', description: 'Forge AI Lab' },
  { id: 'lrs-agents', name: 'LRS-Agents', type: 'agents', basePath: '/home/runner/workspace/lrs-agents', description: 'LRS Agent Framework' },
  { id: 'ncx', name: 'NCX', type: 'system', basePath: '/home/runner/workspace/ncx', description: 'NCX System' },
  { id: 'neuralblitz-agents', name: 'NeuralBlitz-Agents', type: 'agents', basePath: '/home/runner/workspace/neuralblitz-agents', description: 'NeuralBlitz Agent Framework' },
  { id: 'neuralblitz-ui', name: 'NeuralBlitz-UI', type: 'ui', basePath: '/home/runner/workspace/neuralblitz-ui', description: 'NeuralBlitz UI Components' },
  { id: 'ontological-playground', name: 'Ontological-Playground', type: 'research', basePath: '/home/runner/workspace/ontological-playground-designer', description: 'Ontological Playground Designer' },
  { id: 'opencode', name: 'OpenCode', type: 'tool', basePath: '/home/runner/workspace/opencode', description: 'OpenCode CLI Tool' },
  { id: 'opencode-lrs-agents', name: 'OpenCode-LRS-Agents', type: 'agents', basePath: '/home/runner/workspace/opencode-lrs-agents-nbx', description: 'OpenCode LRS Agents Bridge' },
  { id: 'prompt-nexus', name: 'Prompt-Nexus', type: 'research', basePath: '/home/runner/workspace/prompt_nexus', description: 'Prompt Nexus' },
  { id: 'symai', name: 'SymAI', type: 'research', basePath: '/home/runner/workspace/SymAI', description: 'Symbiotic AI' },
  { id: 'computational-axioms', name: 'Computational-Axioms', type: 'research', basePath: '/home/runner/workspace/ComputationalAxioms', description: 'Computational Axioms' },
  { id: 'grant', name: 'Grant', type: 'system', basePath: '/home/runner/workspace/grant', description: 'Grant System' },
  { id: 'quantum-sim', name: 'Quantum-Sim', type: 'simulation', basePath: '/home/runner/workspace/quantum_sim', description: 'Quantum Simulation' },
  { id: 'codebase-monitor', name: 'Codebase-Monitor', type: 'monitoring', basePath: '/home/runner/workspace/codebase-monitor', description: 'Codebase Monitor & Observatory' }
];

class ProjectConnection extends EventEmitter {
  constructor(projectId, config) {
    super();
    this.projectId = projectId;
    this.name = config.name || projectId;
    this.type = config.type || 'generic';
    this.basePath = config.basePath;
    this.wsUrl = config.wsUrl;
    this.apiUrl = config.apiUrl;
    this.status = 'disconnected';
    this.clients = new Map();
    this.subscriptionChannels = new Set(['all', 'project:' + projectId]);
    this.metrics = {
      messagesReceived: 0,
      messagesSent: 0,
      bytesTransferred: 0,
      lastActivity: null
    };
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 5000;
  }

  async connect() {
    try {
      if (this.wsUrl) {
        await this.connectWebSocket();
      }
      this.status = 'connected';
      this.emit('connected', { projectId: this.projectId });
      return true;
    } catch (error) {
      this.status = 'error';
      this.emit('error', { projectId: this.projectId, error: error.message });
      return false;
    }
  }

  connectWebSocket() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      const timeout = setTimeout(() => {
        ws.close();
        reject(new Error('Connection timeout'));
      }, 10000);

      ws.on('open', () => {
        clearTimeout(timeout);
        ws.send(JSON.stringify({
          type: 'register',
          projectId: this.projectId,
          name: this.name,
          type: this.type
        }));
        resolve();
      });

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(message);
        } catch (e) {
          console.error(`[Omnibus] Error parsing message from ${this.projectId}:`, e.message);
        }
      });

      ws.on('close', () => {
        this.status = 'disconnected';
        this.attemptReconnect();
      });

      ws.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
      });
    });
  }

  handleMessage(message) {
    this.metrics.messagesReceived++;
    this.metrics.lastActivity = new Date().toISOString();
    this.emit('message', {
      projectId: this.projectId,
      projectName: this.name,
      message
    });
  }

  attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.status = 'failed';
      return;
    }
    this.reconnectAttempts++;
    setTimeout(() => {
      this.connect().catch(() => {});
    }, this.reconnectDelay * this.reconnectAttempts);
  }

  broadcast(message, channels = ['all', 'project:' + this.projectId]) {
    const envelope = {
      from: this.projectId,
      fromName: this.name,
      timestamp: new Date().toISOString(),
      ...message
    };

    this.metrics.messagesSent++;
    this.metrics.bytesTransferred += JSON.stringify(envelope).length;

    this.emit('broadcast', {
      projectId: this.projectId,
      message: envelope,
      channels
    });
  }

  getStatus() {
    return {
      projectId: this.projectId,
      name: this.name,
      type: this.type,
      basePath: this.basePath,
      status: this.status,
      metrics: this.metrics,
      subscriptions: Array.from(this.subscriptionChannels),
      clients: this.clients.size
    };
  }
}

class OmnibusRouter extends EventEmitter {
  constructor(options = {}) {
    super();
    this.port = options.port || 3001;
    this.projects = new Map();
    this.clients = new Map();
    this.messageHistory = [];
    this.maxHistory = 1000;
    this.wss = null;
    this.httpServer = null;
    this.server = null;
  }

  async initialize() {
    this.server = http.createServer();
    this.wss = new WebSocket.Server({ server: this.server });

    this.wss.on('connection', (ws, req) => {
      this.handleClientConnection(ws, req);
    });

    this.setupHTTPRoutes();

    await this.autoRegisterProjects();

    return new Promise((resolve) => {
      this.server.listen(this.port, () => {
        console.log(`[Omnibus Router] Started on port ${this.port}`);
        resolve();
      });
    });
  }

  async autoRegisterProjects() {
    const registered = [];
    for (const config of ALL_PROJECTS) {
      try {
        const stats = fs.statSync(config.basePath);
        if (stats.isDirectory()) {
          if (!this.projects.has(config.id)) {
            const project = new ProjectConnection(config.id, {
              name: config.name,
              type: config.type,
              basePath: config.basePath
            });
            this.projects.set(config.id, project);
            project.on('message', (msg) => this.routeMessage(msg));
            project.on('broadcast', (data) => this.broadcastToClients(data));
            console.log(`[Omnibus] Auto-registered: ${config.name} (${config.type})`);
            registered.push({ id: config.id, name: config.name, type: config.type });
          }
        }
      } catch (error) {
        console.warn(`[Omnibus] Cannot access ${config.basePath}: ${error.message}`);
      }
    }
    return registered;
  }

  setupHTTPRoutes() {
    const express = require('express');
    const app = express();
    app.use(express.json());

    app.get('/api/health', (req, res) => {
      res.json({ status: 'ok', port: this.port, projects: this.projects.size });
    });

    app.get('/api/projects', (req, res) => {
      const projectList = Array.from(this.projects.values()).map(p => p.getStatus());
      res.json({ projects: projectList });
    });

    app.get('/api/projects/:projectId', (req, res) => {
      const project = this.projects.get(req.params.projectId);
      if (project) {
        res.json(project.getStatus());
      } else {
        res.status(404).json({ error: 'Project not found' });
      }
    });

    app.post('/api/projects/:projectId/connect', async (req, res) => {
      const { wsUrl, apiUrl, type, name, basePath } = req.body;
      const projectId = req.params.projectId;

      if (this.projects.has(projectId)) {
        return res.status(400).json({ error: 'Project already connected' });
      }

      const project = new ProjectConnection(projectId, { wsUrl, apiUrl, type, name, basePath });
      this.projects.set(projectId, project);

      project.on('message', (msg) => this.routeMessage(msg));
      project.on('broadcast', (data) => this.broadcastToClients(data));

      const connected = await project.connect();
      res.json({ success: connected, project: project.getStatus() });
    });

    app.delete('/api/projects/:projectId', (req, res) => {
      const project = this.projects.get(req.params.projectId);
      if (project) {
        project.status = 'disconnected';
        this.projects.delete(req.params.projectId);
        res.json({ success: true });
      } else {
        res.status(404).json({ error: 'Project not found' });
      }
    });

    app.post('/api/broadcast', (req, res) => {
      const { from, message, channels = ['all'] } = req.body;
      this.broadcastToClients({
        from: 'router',
        message: { from, ...message },
        channels
      });
      res.json({ success: true, delivered: this.clients.size });
    });

    app.get('/api/messages', (req, res) => {
      const { limit = 100, project } = req.query;
      let messages = this.messageHistory;
      if (project) {
        messages = messages.filter(m => m.from === project || m.message?.from === project);
      }
      res.json({ messages: messages.slice(-parseInt(limit)) });
    });

    app.get('/api/routes', (req, res) => {
      const routes = [];
      this.projects.forEach((project, id) => {
        routes.push({
          projectId: id,
          name: project.name,
          type: project.type,
          status: project.status,
          subscriptions: Array.from(project.subscriptionChannels)
        });
      });
      res.json({ routes });
    });

    app.get('/api/all-projects', (req, res) => {
      res.json({ projects: ALL_PROJECTS });
    });

    app.post('/api/batch/register', (req, res) => {
      const { projectIds } = req.body;
      const registered = [];
      const failed = [];
      
      for (const pid of projectIds || []) {
        const config = ALL_PROJECTS.find(p => p.id === pid);
        if (config && !this.projects.has(pid)) {
          try {
            const stats = fs.statSync(config.basePath);
            if (stats.isDirectory()) {
              const project = new ProjectConnection(config.id, {
                name: config.name,
                type: config.type,
                basePath: config.basePath
              });
              this.projects.set(config.id, project);
              project.on('message', (msg) => this.routeMessage(msg));
              project.on('broadcast', (data) => this.broadcastToClients(data));
              registered.push({ id: config.id, name: config.name, type: config.type });
            }
          } catch (error) {
            failed.push({ id: pid, error: error.message });
          }
        } else {
          failed.push({ id: pid, error: 'Project not found or already registered' });
        }
      }
      res.json({ success: true, registered, failed, total: registered.length });
    });

    app.post('/api/batch/status', (req, res) => {
      const { projectIds } = req.body;
      const statuses = [];
      for (const pid of projectIds || []) {
        const project = this.projects.get(pid);
        statuses.push({
          projectId: pid,
          registered: !!project,
          status: project?.status || 'not_registered'
        });
      }
      res.json({ statuses });
    });

    app.post('/api/batch/broadcast', (req, res) => {
      const { projectIds, message, channels = ['all'] } = req.body;
      const delivered = [];
      for (const pid of projectIds || []) {
        const project = this.projects.get(pid);
        if (project) {
          project.broadcast({ type: 'batch_message', content: message }, channels);
          delivered.push(pid);
        }
      }
      res.json({ success: true, delivered, count: delivered.length });
    });

    app.post('/api/batch/scan', async (req, res) => {
      const { projectIds } = req.body;
      const results = [];
      for (const pid of projectIds || []) {
        const project = this.projects.get(pid);
        if (project && project.basePath) {
          try {
            const files = this.scanProjectFiles(project.basePath);
            results.push({ projectId: pid, files: files.length, fileTree: this.buildFileTree(files) });
          } catch (error) {
            results.push({ projectId: pid, error: error.message });
          }
        }
      }
      res.json({ success: true, results });
    });

    app.get('/api/tasks', (req, res) => {
      const tasks = Array.from(this.batchTasks?.values() || []);
      res.json({ tasks });
    });

    app.post('/api/tasks/create', (req, res) => {
      const { name, tasks, parallel = false } = req.body;
      const taskId = uuidv4();
      if (!this.batchTasks) this.batchTasks = new Map();
      
      this.batchTasks.set(taskId, {
        id: taskId,
        name: name || `Batch_${Date.now()}`,
        tasks: tasks || [],
        parallel,
        status: 'pending',
        createdAt: new Date().toISOString(),
        results: []
      });
      res.json({ success: true, taskId });
    });

    app.post('/api/tasks/:taskId/execute', async (req, res) => {
      const task = this.batchTasks?.get(req.params.taskId);
      if (!task) {
        return res.status(404).json({ error: 'Task not found' });
      }
      
      task.status = 'executing';
      task.startedAt = new Date().toISOString();
      
      const results = [];
      for (const t of task.tasks) {
        const result = await this.executeTask(t);
        results.push({ ...t, result, executedAt: new Date().toISOString() });
      }
      task.results = results;
      task.status = 'completed';
      task.completedAt = new Date().toISOString();
      
      res.json({ success: true, task });
    });

    app.get('/api/tasks/:taskId', (req, res) => {
      const task = this.batchTasks?.get(req.params.taskId);
      if (task) {
        res.json({ task });
      } else {
        res.status(404).json({ error: 'Task not found' });
      }
    });

    this.httpServer = app;
  }

  scanProjectFiles(basePath) {
    const files = [];
    const scanDir = (dir) => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory() && !entry.name.match(/^\.|node_modules|dist|build|\.git/)) {
            scanDir(fullPath);
          } else if (entry.isFile()) {
            files.push(path.relative(basePath, fullPath));
          }
        }
      } catch (e) {}
    };
    scanDir(basePath);
    return files;
  }

  buildFileTree(files) {
    const tree = {};
    for (const file of files) {
      const parts = file.split('/');
      let current = tree;
      for (let i = 0; i < parts.length - 1; i++) {
        if (!current[parts[i]]) current[parts[i]] = {};
        current = current[parts[i]];
      }
      current[parts[parts.length - 1]] = null;
    }
    return tree;
  }

  async executeTask(task) {
    return new Promise((resolve) => {
      const http = require('http');
      const req = http.request(`${task.apiUrl || 'http://localhost:3000'}${task.endpoint}`, {
        method: task.method || 'GET',
        timeout: task.timeout || 5000
      }, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          try {
            resolve({ status: res.statusCode, data: JSON.parse(data) });
          } catch {
            resolve({ status: res.statusCode, data });
          }
        });
      });
      req.on('error', (e) => resolve({ error: e.message }));
      if (task.body) req.write(JSON.stringify(task.body));
      req.end();
    });
  }

  handleClientConnection(ws, req) {
    const clientId = this.generateClientId();
    const clientInfo = {
      id: clientId,
      ws,
      subscriptions: new Set(['all']),
      connectedAt: new Date(),
      lastActivity: new Date()
    };

    this.clients.set(clientId, clientInfo);
    console.log(`[Omnibus] Client connected: ${clientId}`);

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        this.handleClientMessage(clientId, msg);
      } catch (e) {
        console.error(`[Omnibus] Client message error:`, e.message);
      }
    });

    ws.on('close', () => {
      this.clients.delete(clientId);
      console.log(`[Omnibus] Client disconnected: ${clientId}`);
    });

    ws.on('error', (error) => {
      console.error(`[Omnibus] Client error ${clientId}:`, error.message);
    });

    ws.send(JSON.stringify({
      type: 'connected',
      clientId,
      projects: Array.from(this.projects.keys()),
      timestamp: new Date().toISOString()
    }));
  }

  handleClientMessage(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    client.lastActivity = new Date();

    switch (message.type) {
      case 'subscribe':
        if (message.channels) {
          message.channels.forEach(ch => client.subscriptions.add(ch));
        }
        break;

      case 'unsubscribe':
        if (message.channels) {
          message.channels.forEach(ch => client.subscriptions.delete(ch));
        }
        break;

      case 'publish':
        this.routeMessage({
          from: clientId,
          message: message.data,
          channels: message.channels || ['all']
        });
        break;

      case 'request':
        this.handleClientRequest(clientId, message);
        break;
    }
  }

  async handleClientRequest(clientId, request) {
    const { action, payload } = request;

    let result;
    switch (action) {
      case 'listProjects':
        result = Array.from(this.projects.values()).map(p => p.getStatus());
        break;

      case 'getProjectStatus':
        const project = this.projects.get(payload.projectId);
        result = project ? project.getStatus() : { error: 'Project not found' };
        break;

      case 'sendToProject':
        const targetProject = this.projects.get(payload.projectId);
        if (targetProject) {
          targetProject.broadcast(payload.message, payload.channels);
          result = { success: true };
        } else {
          result = { error: 'Project not found' };
        }
        break;

      case 'getRoutes':
        result = [];
        this.projects.forEach((p, id) => {
          result.push({
            projectId: id,
            name: p.name,
            type: p.type,
            status: p.status
          });
        });
        break;

      case 'getHistory':
        result = this.messageHistory.slice(-(payload.limit || 100));
        break;

      default:
        result = { error: 'Unknown action' };
    }

    const client = this.clients.get(clientId);
    if (client) {
      client.ws.send(JSON.stringify({
        type: 'response',
        requestId: request.id,
        result
      }));
    }
  }

  routeMessage(data) {
    const envelope = {
      id: this.generateMessageId(),
      ...data,
      timestamp: new Date().toISOString()
    };

    this.messageHistory.push(envelope);
    if (this.messageHistory.length > this.maxHistory) {
      this.messageHistory.shift();
    }

    this.emit('message', envelope);
    this.broadcastToClients({
      from: 'router',
      message: envelope,
      channels: data.channels || ['all']
    });
  }

  broadcastToClients(data) {
    const messageStr = JSON.stringify(data);
    const channels = data.channels || ['all'];

    this.clients.forEach((client) => {
      const hasSubscription = channels.some(ch =>
        client.subscriptions.has(ch)
      );

      if (hasSubscription && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(messageStr);
      }
    });
  }

  registerProject(projectId, config) {
    if (this.projects.has(projectId)) {
      console.warn(`[Omnibus] Project ${projectId} already registered`);
      return;
    }

    const project = new ProjectConnection(projectId, config);
    this.projects.set(projectId, project);

    project.on('message', (msg) => this.routeMessage(msg));
    project.on('broadcast', (data) => this.broadcastToClients(data));

    console.log(`[Omnibus] Project registered: ${projectId} (${config.type})`);
    return project;
  }

  generateClientId() {
    return 'client_' + Math.random().toString(36).substring(2, 15);
  }

  generateMessageId() {
    return 'msg_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
  }

  getAllProjects() {
    return Array.from(this.projects.values()).map(p => p.getStatus());
  }

  getRoutes() {
    const routes = [];
    this.projects.forEach((project, id) => {
      routes.push({
        projectId: id,
        name: project.name,
        type: project.type,
        status: project.status,
        subscriptions: Array.from(project.subscriptionChannels)
      });
    });
    return routes;
  }

  getStatistics() {
    return {
      server: {
        port: this.port,
        uptime: process.uptime()
      },
      projects: {
        total: this.projects.size,
        connected: Array.from(this.projects.values()).filter(p => p.status === 'connected').length
      },
      clients: {
        total: this.clients.size
      },
      messages: {
        total: this.messageHistory.length,
        recent: this.messageHistory.slice(-100).length
      }
    };
  }

  shutdown() {
    this.projects.forEach(project => {
      project.status = 'disconnected';
    });
    this.clients.forEach((client) => {
      client.ws.close();
    });
    this.clients.clear();
    this.projects.clear();
    if (this.wss) {
      this.wss.close();
    }
    if (this.server) {
      this.server.close();
    }
    console.log('[Omnibus Router] Shutdown complete');
  }
}

module.exports = OmnibusRouter;
