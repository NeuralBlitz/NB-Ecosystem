require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const { exec } = require('child_process');
const util = require('util');

const FileWatcher = require('./services/FileWatcher');
const GitMonitor = require('./services/GitMonitor');
const WebSocketManager = require('./services/WebSocketManager');
const AICommandHandler = require('./services/AICommandHandler');
const OmnibusRouter = require('./services/OmnibusRouter');
const NeuralBlitzBridge = require('./services/NeuralBlitzBridge');

class CodebaseMonitorServer {
  constructor() {
    this.app = express();
    this.port = process.env.PORT || 3000;
    
    this.server = null;
    this.fileWatcher = null;
    this.gitMonitor = null;
    this.wsManager = null;
    this.omnibusRouter = null;
    this.neuralBlitzBridge = null;
    
    this.changeHistory = [];
    this.maxHistorySize = 1000;
    
    this.setupMiddleware();
    this.setupRoutes();
  }

  setupMiddleware() {
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.static(path.join(__dirname, 'public')));
  }

  setupAIHandlers() {
    this.aiHandler = new AICommandHandler();

    this.aiHandler.addTool('getStatus', 'Get server status and health', {
      type: 'object',
      properties: {}
    }, async () => {
      return {
        status: 'running',
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        port: this.port,
        timestamp: new Date().toISOString()
      };
    });

    this.aiHandler.addTool('getFiles', 'Get watched files and recent changes', {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Max files to return' }
      }
    }, async ({ limit = 50 }) => {
      return {
        totalFiles: this.fileWatcher?.getStatistics()?.totalFiles || 0,
        changes: this.changeHistory.slice(-limit)
      };
    });

    this.aiHandler.addTool('getGitCommits', 'Get recent git commits', {
      type: 'object',
      properties: {
        limit: { type: 'number', description: 'Max commits to return' }
      }
    }, async ({ limit = 10 }) => {
      return {
        commits: this.gitMonitor?.commitHistory?.slice(0, limit) || []
      };
    });

    this.aiHandler.addTool('executeCommand', 'Execute a bash command', {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'Command to execute' }
      },
      required: ['command']
    }, async ({ command }) => {
      const execPromise = util.promisify(exec);
      try {
        const { stdout, stderr } = await execPromise(command, { timeout: 30000 });
        return { stdout, stderr, success: true };
      } catch (error) {
        return { error: error.message, success: false };
      }
    });

    this.aiHandler.addTool('searchCode', 'Search code in watched files', {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
        type: { type: 'string', description: 'File type to search' }
      },
      required: ['query']
    }, async ({ query, type }) => {
      return {
        query,
        message: 'Code search functionality - implement grep integration'
      };
    });

    this.aiHandler.addTool('readFile', 'Read file contents', {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to read' }
      },
      required: ['path']
    }, async ({ path: filePath }) => {
      const fs = require('fs');
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        return { content, success: true };
      } catch (error) {
        return { error: error.message, success: false };
      }
    });

    this.aiHandler.addTool('help', 'Get available commands', {
      type: 'object',
      properties: {}
    }, async () => {
      return {
        commands: this.aiHandler.getRegisteredTools(),
        message: 'Available AI-powered commands. Also try natural language!'
      };
    });

    this.wsManager.registerHandler('aiCommand', async ({ clientId, payload }) => {
      const { command, context } = payload || {};
      if (!command) {
        return { success: false, error: 'No command provided' };
      }
      return this.aiHandler.processCommand(command, { clientId, server: this });
    });

    this.wsManager.registerHandler('nlp', async ({ clientId, payload }) => {
      const { query } = payload || {};
      if (!query) {
        return { success: false, error: 'No query provided' };
      }
      return this.aiHandler.processCommand(query, { clientId, server: this });
    });

    this.wsManager.registerHandler('agent', async ({ clientId, payload }) => {
      const { instruction, mode } = payload || {};
      if (!instruction) {
        return { success: false, error: 'No instruction provided' };
      }
      return this.processAgentInstruction(instruction, mode, { clientId, server: this });
    });

    this.wsManager.registerHandler('terminal', async ({ clientId, payload }) => {
      const { command, timeout } = payload || {};
      if (!command) {
        return { success: false, error: 'No command provided' };
      }
      return this.executeTerminalCommand(command, timeout || 30000, { clientId, server: this });
    });

    console.log('[Server] AI Command Handler initialized');
  }

  processAgentInstruction(instruction, mode = 'auto', context = {}) {
    const normalized = instruction.toLowerCase();
    
    if (mode === 'direct' || !this.openai) {
      return this.directAgentProcessing(instruction, context);
    }
    
    return this.aiHandler.processCommand(instruction, context);
  }

  executeTerminalCommand(command, timeout, context = {}) {
    return new Promise((resolve) => {
      const execPromise = util.promisify(exec);
      execPromise(command, { timeout, maxBuffer: 10 * 1024 * 1024 })
        .then(({ stdout, stderr }) => {
          resolve({
            success: true,
            stdout,
            stderr,
            exitCode: 0
          });
        })
        .catch((error) => {
          resolve({
            success: false,
            error: error.message,
            stdout: error.stdout || '',
            stderr: error.stderr || '',
            exitCode: error.code || 1
          });
        });
    });
  }

  directAgentProcessing(instruction, context = {}) {
    const normalized = instruction.toLowerCase();
    const original = instruction;
    
    if (normalized.startsWith('broadcast ')) {
      const message = normalized.substring(10).trim();
      this.wsManager.broadcast({
        type: 'agent_message',
        from: context.clientId || 'agent',
        content: message,
        timestamp: new Date().toISOString()
      }, ['commands', 'all']);
      return {
        success: true,
        response: `Broadcast sent: ${message}`,
        broadcast: true
      };
    }
    
    if (normalized.startsWith('relay ')) {
      const message = normalized.substring(6).trim();
      const agentMessage = {
        type: 'agent_message',
        from: context.clientId || 'agent',
        content: message,
        timestamp: new Date().toISOString(),
        relay: true
      };
      this.wsManager.broadcast(agentMessage, ['commands', 'all']);
      return {
        success: true,
        response: `Relayed to all agents: ${message}`,
        message: agentMessage
      };
    }

    if (normalized.startsWith('register ')) {
      const agentName = normalized.substring(9).trim();
      return {
        success: true,
        response: `Agent registered: ${agentName}`,
        agentConnected: true,
        agentId: agentName
      };
    }
    
    const parts = instruction.split(' ');
    const action = parts[0].toLowerCase();
    const args = parts.slice(1).join(' ');

    const agentCommands = {
      status: () => ({
        status: 'running',
        uptime: process.uptime(),
        port: this.port,
        timestamp: new Date().toISOString()
      }),
      files: () => ({
        totalFiles: this.fileWatcher?.getStatistics()?.totalFiles || 0,
        recentChanges: this.changeHistory.slice(-20)
      }),
      commits: (limit = 10) => ({
        commits: this.gitMonitor?.commitHistory?.slice(0, limit) || []
      }),
      clients: () => this.wsManager?.getStatistics() || { error: 'WS not available' },
      exec: (cmd) => this.executeTerminalCommand(cmd, 30000, context),
      read: (path) => {
        const fs = require('fs');
        try {
          return { content: fs.readFileSync(path, 'utf8'), success: true };
        } catch (e) {
          return { error: e.message, success: false };
        }
      },
      grep: (query) => ({
        query,
        message: 'Implement grep integration',
        results: []
      }),
      ping: () => ({ pong: true, timestamp: new Date().toISOString() }),
      echo: (msg) => ({ echo: msg }),
      help: () => ({
        commands: ['status', 'files', 'commits [n]', 'clients', 'exec <cmd>', 'read <path>', 'grep <query>', 'broadcast <msg>', 'relay <msg>', 'register <name>', 'ping', 'echo <msg>', 'help']
      })
    };

    if (agentCommands[action]) {
      try {
        const result = agentCommands[action](args);
        return {
          success: true,
          response: typeof result === 'object' ? JSON.stringify(result, null, 2) : result,
          mode: 'direct',
          agent: 'terminal'
        };
      } catch (e) {
        return { success: false, error: e.message };
      }
    }

    return {
      success: false,
      error: `Unknown command: ${action}. Try 'help'`,
      available: Object.keys(agentCommands)
    };
  }

  setupRoutes() {
    this.app.get('/api/health', (req, res) => {
      res.json({ 
        status: 'ok', 
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
      });
    });

    this.app.post('/api/ai/command', (req, res) => {
      const { command } = req.body;
      if (!command) {
        return res.status(400).json({ error: 'No command provided' });
      }
      this.aiHandler.processCommand(command, { server: this })
        .then(result => res.json(result))
        .catch(error => res.status(500).json({ error: error.message }));
    });

    this.app.post('/api/agent/execute', (req, res) => {
      const { instruction, mode } = req.body;
      if (!instruction) {
        return res.status(400).json({ error: 'No instruction provided' });
      }
      const result = this.processAgentInstruction(instruction, mode || 'direct', { server: this });
      res.json(result);
    });

    this.app.post('/api/terminal/execute', (req, res) => {
      const { command, timeout } = req.body;
      if (!command) {
        return res.status(400).json({ error: 'No command provided' });
      }
      this.executeTerminalCommand(command, timeout || 30000, { server: this })
        .then(result => res.json(result))
        .catch(error => res.status(500).json({ error: error.message }));
    });

    this.app.get('/api/status', (req, res) => {
      const status = {
        server: {
          port: this.port,
          uptime: process.uptime()
        },
        fileWatcher: this.fileWatcher ? {
          active: true,
          watching: this.fileWatcher.watchPath,
          stats: this.fileWatcher.getStatistics()
        } : null,
        git: this.gitMonitor ? {
          active: true,
          stats: this.gitMonitor.getStats()
        } : null,
        websocket: this.wsManager ? {
          active: true,
          clients: this.wsManager.getStatistics()
        } : null
      };
      res.json(status);
    });

    this.app.get('/api/changes', (req, res) => {
      const { limit = 100, type } = req.query;
      let changes = this.changeHistory;

      if (type) {
        changes = changes.filter(c => c.type === type);
      }

      res.json({
        total: changes.length,
        changes: changes.slice(-parseInt(limit))
      });
    });

    this.app.get('/api/git/commits', async (req, res) => {
      if (!this.gitMonitor) {
        return res.status(503).json({ error: 'Git monitoring not available' });
      }

      const { limit = 50 } = req.query;
      const commits = this.gitMonitor.commitHistory.slice(0, parseInt(limit));
      res.json({ commits });
    });

    this.app.get('/api/git/status', async (req, res) => {
      if (!this.gitMonitor) {
        return res.status(503).json({ error: 'Git monitoring not available' });
      }

      const status = await this.gitMonitor.getStatus();
      res.json(status);
    });

    this.app.get('/api/git/stats', async (req, res) => {
      if (!this.gitMonitor) {
        return res.status(503).json({ error: 'Git monitoring not available' });
      }

      const stats = await this.gitMonitor.getStats();
      res.json(stats);
    });

    this.app.get('/api/files/stats', (req, res) => {
      if (!this.fileWatcher) {
        return res.status(503).json({ error: 'File watcher not available' });
      }

      const stats = this.fileWatcher.getStatistics();
      res.json(stats);
    });

    this.app.get('/api/websocket/stats', (req, res) => {
      if (!this.wsManager) {
        return res.status(503).json({ error: 'WebSocket not available' });
      }

      const stats = this.wsManager.getStatistics();
      res.json(stats);
    });

    this.app.post('/api/websocket/send', (req, res) => {
      if (!this.wsManager) {
        return res.status(503).json({ error: 'WebSocket not available' });
      }

      const { clientId, message, type = 'custom_message' } = req.body;

      if (!clientId || !message) {
        return res.status(400).json({ error: 'clientId and message are required' });
      }

      const client = this.wsManager.clients.get(clientId);
      if (!client) {
        return res.status(404).json({ error: 'Client not found', clientId });
      }

      this.wsManager.sendToClient(clientId, {
        type,
        message,
        timestamp: new Date().toISOString()
      });

      res.json({ success: true, clientId, message: 'Message sent' });
    });

    this.app.post('/api/websocket/broadcast', (req, res) => {
      if (!this.wsManager) {
        return res.status(503).json({ error: 'WebSocket not available' });
      }

      const { message, type = 'broadcast', channels = ['all'] } = req.body;

      if (!message) {
        return res.status(400).json({ error: 'message is required' });
      }

      this.wsManager.broadcast({
        type,
        message,
        timestamp: new Date().toISOString()
      }, channels);

      const stats = this.wsManager.getStatistics();
      res.json({ 
        success: true, 
        clientsNotified: stats.totalClients,
        message: 'Broadcast sent'
      });
    });

    this.app.get('/api/omnibus/projects', (req, res) => {
      if (!this.omnibusRouter) {
        return res.status(503).json({ error: 'OmnibusRouter not available' });
      }
      const projects = this.omnibusRouter.getAllProjects();
      res.json({ projects });
    });

    this.app.get('/api/omnibus/routes', (req, res) => {
      if (!this.omnibusRouter) {
        return res.status(503).json({ error: 'OmnibusRouter not available' });
      }
      const routes = this.omnibusRouter.getRoutes();
      res.json({ routes });
    });

    this.app.post('/api/omnibus/broadcast', (req, res) => {
      if (!this.omnibusRouter) {
        return res.status(503).json({ error: 'OmnibusRouter not available' });
      }
      const { message, channels = ['all'] } = req.body;
      if (!message) {
        return res.status(400).json({ error: 'message is required' });
      }
      this.omnibusRouter.broadcast(message, channels);
      res.json({ success: true, message: 'Broadcast sent via Omnibus' });
    });

    this.app.get('/api/neuralblitz/projects', (req, res) => {
      if (!this.neuralBlitzBridge) {
        return res.status(503).json({ error: 'NeuralBlitzBridge not available' });
      }
      const projects = Array.from(this.neuralBlitzBridge.projects?.values() || []);
      res.json({ projects });
    });

    this.app.post('/api/neuralblitz/scan', (req, res) => {
      if (!this.neuralBlitzBridge) {
        return res.status(503).json({ error: 'NeuralBlitzBridge not available' });
      }
      const { projectId } = req.body;
      const results = this.neuralBlitzBridge.scanProject(projectId);
      res.json({ success: true, results });
    });

    this.app.get('/api/neuralblitz/file-tree/:projectId', (req, res) => {
      if (!this.neuralBlitzBridge) {
        return res.status(503).json({ error: 'NeuralBlitzBridge not available' });
      }
      const tree = this.neuralBlitzBridge.getFileTree(req.params.projectId);
      if (tree) {
        res.json({ tree });
      } else {
        res.status(404).json({ error: 'Project not found' });
      }
    });

    this.app.get('/api/neuralblitz/search', (req, res) => {
      if (!this.neuralBlitzBridge) {
        return res.status(503).json({ error: 'NeuralBlitzBridge not available' });
      }
      const { q } = req.query;
      if (!q) {
        return res.status(400).json({ error: 'Query parameter q is required' });
      }
      const results = this.neuralBlitzBridge.searchAllProjects(q);
      res.json({ results });
    });

    this.app.get('/api/projects/all', (req, res) => {
      const OmnibusRouter = require('./services/OmnibusRouter');
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
      res.json({ projects: ALL_PROJECTS, total: ALL_PROJECTS.length });
    });

    this.app.get('/api/projects/registered', (req, res) => {
      if (!this.omnibusRouter) {
        return res.status(503).json({ error: 'OmnibusRouter not available' });
      }
      const projects = this.omnibusRouter.getAllProjects();
      res.json({ projects, total: projects.length });
    });

    this.app.post('/api/batch/register', (req, res) => {
      if (!this.omnibusRouter) {
        return res.status(503).json({ error: 'OmnibusRouter not available' });
      }
      const { projectIds } = req.body;
      const registered = [];
      const failed = [];
      
      for (const pid of projectIds || []) {
        const project = this.omnibusRouter.projects?.get(pid);
        if (project) {
          registered.push({ id: pid, status: project.status });
        } else {
          failed.push({ id: pid, error: 'Project not found' });
        }
      }
      res.json({ success: true, registered, failed, total: registered.length });
    });

    this.app.post('/api/batch/status', (req, res) => {
      if (!this.omnibusRouter) {
        return res.status(503).json({ error: 'OmnibusRouter not available' });
      }
      const { projectIds } = req.body;
      const statuses = [];
      for (const pid of projectIds || []) {
        const project = this.omnibusRouter.projects?.get(pid);
        statuses.push({
          projectId: pid,
          registered: !!project,
          status: project?.status || 'not_registered'
        });
      }
      res.json({ statuses });
    });

    this.app.post('/api/batch/scan', (req, res) => {
      const { projectIds } = req.body;
      const fs = require('fs');
      const path = require('path');
      const results = [];
      
      const scanProject = (pid) => {
        const project = this.omnibusRouter.projects?.get(pid);
        if (project?.basePath) {
          try {
            const files = [];
            const scanDir = (dir) => {
              try {
                const entries = fs.readdirSync(dir, { withFileTypes: true });
                for (const entry of entries) {
                  const fullPath = path.join(dir, entry.name);
                  if (entry.isDirectory() && !entry.name.match(/^\.|node_modules|dist|build|\.git/)) {
                    scanDir(fullPath);
                  } else if (entry.isFile()) {
                    files.push(path.relative(project.basePath, fullPath));
                  }
                }
              } catch (e) {}
            };
            scanDir(project.basePath);
            results.push({ projectId: pid, files: files.length, path: project.basePath });
          } catch (error) {
            results.push({ projectId: pid, error: error.message });
          }
        } else {
          results.push({ projectId: pid, error: 'Base path not found' });
        }
      };
      
      (projectIds || []).forEach(scanProject);
      res.json({ success: true, results });
    });

    this.app.get('/api/tasks', (req, res) => {
      if (!this.omnibusRouter?.batchTasks) {
        return res.json({ tasks: [] });
      }
      const tasks = Array.from(this.omnibusRouter.batchTasks.values());
      res.json({ tasks });
    });

    this.app.post('/api/tasks/create', (req, res) => {
      if (!this.omnibusRouter) {
        return res.status(503).json({ error: 'OmnibusRouter not available' });
      }
      const { name, tasks, parallel = false } = req.body;
      const { v4: uuidv4 } = require('uuid');
      const taskId = uuidv4();
      if (!this.omnibusRouter.batchTasks) this.omnibusRouter.batchTasks = new Map();
      
      this.omnibusRouter.batchTasks.set(taskId, {
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

    this.app.post('/api/tasks/:taskId/execute', async (req, res) => {
      const task = this.omnibusRouter.batchTasks?.get(req.params.taskId);
      if (!task) {
        return res.status(404).json({ error: 'Task not found' });
      }

      const fs = require('fs');
      const path = require('path');
      const { exec } = require('child_process');
      const util = require('util');

      task.status = 'executing';
      task.startedAt = new Date().toISOString();
      task.results = [];

      const execPromise = util.promisify(exec);

      const executeTaskItem = async (item, index) => {
        const result = { action: item.action, target: item.target, index, executedAt: new Date().toISOString() };

        try {
          const project = this.omnibusRouter.projects?.get(item.target);
          const basePath = project?.basePath || item.basePath;

          switch (item.action) {
            case 'scan':
              if (basePath) {
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
                const byType = {};
                files.forEach(f => {
                  const ext = path.extname(f).substring(1) || 'no-ext';
                  byType[ext] = (byType[ext] || 0) + 1;
                });
                result.success = true;
                result.files = files.length;
                result.byType = byType;
              } else {
                result.success = false;
                result.error = 'Base path not found';
              }
              break;

            case 'status':
              if (basePath && fs.existsSync(path.join(basePath, '.git'))) {
                try {
                  const { stdout } = await execPromise('git status --short', { cwd: basePath });
                  const { stdout: log } = await execPromise('git log --oneline -5', { cwd: basePath });
                  result.success = true;
                  result.git = { changed: stdout.trim().split('\n').filter(Boolean), recentCommits: log.trim().split('\n') };
                } catch (e) {
                  result.success = false;
                  result.error = e.message;
                }
              } else {
                result.success = true;
                result.note = 'Not a git repository';
              }
              break;

            case 'package':
              if (basePath) {
                const pkgPath = path.join(basePath, 'package.json');
                const reqPath = path.join(basePath, 'requirements.txt');
                if (fs.existsSync(pkgPath)) {
                  const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
                  result.success = true;
                  result.dependencies = Object.keys(pkg.dependencies || {}).length;
                  result.scripts = Object.keys(pkg.scripts || {});
                } else if (fs.existsSync(reqPath)) {
                  const req = fs.readFileSync(reqPath, 'utf8').split('\n').filter(Boolean);
                  result.success = true;
                  result.pythonPackages = req.length;
                } else {
                  result.success = false;
                  result.error = 'No package.json or requirements.txt found';
                }
              } else {
                result.success = false;
                result.error = 'Base path not found';
              }
              break;

            case 'tree':
              if (basePath) {
                const buildTree = (dir, prefix = '') => {
                  const items = [];
                  try {
                    const entries = fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
                    entries.forEach((entry, i) => {
                      const isLast = i === entries.length - 1;
                      const name = entry.name;
                      if (entry.isDirectory()) {
                        items.push({ name, type: 'directory', children: buildTree(path.join(dir, name)) });
                      } else {
                        items.push({ name, type: 'file' });
                      }
                    });
                  } catch (e) {}
                  return items;
                };
                result.success = true;
                result.tree = buildTree(basePath).slice(0, 50);
              } else {
                result.success = false;
                result.error = 'Base path not found';
              }
              break;

            default:
              result.success = false;
              result.error = `Unknown action: ${item.action}`;
          }
        } catch (error) {
          result.success = false;
          result.error = error.message;
        }

        task.results.push(result);
        this.wsManager?.broadcast({ type: 'task_progress', taskId: task.id, result });
      };

      for (let i = 0; i < task.tasks.length; i++) {
        if (task.parallel) {
          await Promise.all(task.tasks.map((item, idx) => executeTaskItem(item, idx)));
        } else {
          await executeTaskItem(task.tasks[i], i);
        }
      }

      task.status = 'completed';
      task.completedAt = new Date().toISOString();

      res.json({ success: true, task });
    });

    this.app.get('/api/summary', (req, res) => {
      const omnibusProjects = this.omnibusRouter?.getAllProjects() || [];
      const neuralProjects = Array.from(this.neuralBlitzBridge?.projects?.values() || []);
      res.json({
        server: { port: this.port, uptime: process.uptime() },
        omnibus: { total: omnibusProjects.length, connected: omnibusProjects.filter(p => p.status === 'connected').length },
        neuralblitz: { total: neuralProjects.length },
        websocket: this.wsManager?.getStatistics()
      });
    });

    this.app.get('/api/projects/:projectId/info', (req, res) => {
      const project = this.omnibusRouter?.projects?.get(req.params.projectId);
      if (project) {
        res.json({
          id: project.projectId,
          name: project.name,
          type: project.type,
          basePath: project.basePath,
          status: project.status,
          metrics: project.metrics,
          subscriptions: Array.from(project.subscriptionChannels)
        });
      } else {
        res.status(404).json({ error: 'Project not found' });
      }
    });

    this.app.post('/api/projects/:projectId/connect', async (req, res) => {
      const project = this.omnibusRouter?.projects?.get(req.params.projectId);
      if (!project) {
        return res.status(404).json({ error: 'Project not found' });
      }
      const connected = await project.connect();
      res.json({ success: connected, status: project.status });
    });

    this.app.get('/api/search', (req, res) => {
      const { q, type } = req.query;
      if (!q) {
        return res.status(400).json({ error: 'Query parameter q is required' });
      }

      const fs = require('fs');
      const path = require('path');
      const results = [];
      const query = q.toLowerCase();

      const searchDir = (basePath, depth = 0) => {
        if (depth > 3) return;
        try {
          const entries = fs.readdirSync(basePath, { withFileTypes: true });
          for (const entry of entries) {
            const fullPath = path.join(basePath, entry.name);
            if (entry.isDirectory() && !entry.name.match(/^\.|node_modules|\.git/)) {
              if (entry.name.toLowerCase().includes(query)) {
                results.push({ path: fullPath.replace('/home/runner/workspace/', ''), type: 'directory', name: entry.name });
              }
              searchDir(fullPath, depth + 1);
            } else if (entry.isFile() && entry.name.toLowerCase().includes(query)) {
              results.push({ path: fullPath.replace('/home/runner/workspace/', ''), type: 'file', name: entry.name });
            }
          }
        } catch (e) {}
      };

      searchDir('/home/runner/workspace');
      res.json({ query: q, total: results.length, results: results.slice(0, 100) });
    });

    this.app.get('/', (req, res) => {
      res.sendFile(path.join(__dirname, 'public', 'index.html'));
    });

    this.app.get('/dashboard', (req, res) => {
      res.sendFile(path.join(__dirname, 'public', 'dashboard.html'));
    });
  }

  async initialize() {
    const watchPath = process.env.WATCH_PATH || '.';

    this.server = this.app.listen(this.port);

    this.wsManager = new WebSocketManager();
    this.wsManager.initialize(this.server);

    this.wsManager.registerHandler('getStatus', ({clientId, payload}) => {
      return { status: 'running', timestamp: new Date().toISOString() };
    });

    this.wsManager.registerHandler('getHandlers', ({clientId, payload}) => {
      return { handlers: this.wsManager.handlers ? Array.from(this.wsManager.handlers.keys()) : [] };
    });

    this.wsManager.registerHandler('getClients', ({clientId, payload}) => {
      return this.wsManager.getStatistics();
    });

    this.wsManager.registerHandler('pingClient', ({clientId, payload}) => {
      return { pong: true, timestamp: new Date().toISOString(), from: 'server' };
    });

    this.setupAIHandlers();

    this.omnibusRouter = new OmnibusRouter();
    this.omnibusRouter.initialize();

    this.neuralBlitzBridge = new NeuralBlitzBridge();
    this.neuralBlitzBridge.initialize();

    this.wsManager.connectToOmnibus('http://localhost:3001');

    console.log('[Server] OmnibusRouter and NeuralBlitzBridge initialized');

    this.gitMonitor = new GitMonitor(watchPath);
    const hasGit = await this.gitMonitor.initialize();

    if (hasGit) {
      this.gitMonitor.onNewCommit((commit) => {
        console.log(`[Server] New commit: ${commit.shortHash} - ${commit.message}`);
        this.wsManager.broadcastGitCommit(commit);
      });

      this.gitMonitor.startPolling(30000);
    }

    this.fileWatcher = new FileWatcher(watchPath, {
      ignorePatterns: [
        /node_modules/,
        /\.git/,
        /dist/,
        /build/,
        /\.cache/,
        /coverage/,
        /\.env/,
        /package-lock\.json/,
        /\.log$/,
        /\.DS_Store/
      ]
    });

    this.fileWatcher.on('change', (change) => {
      console.log(`[Server] File ${change.type}: ${change.file}`);
      
      this.addToHistory(change);
      this.wsManager.broadcastFileChange(change);
    });

    this.fileWatcher.on('error', (error) => {
      console.error('[Server] File watcher error:', error.message);
    });

    this.fileWatcher.start();

    return this;
  }

  addToHistory(change) {
    this.changeHistory.push(change);
    
    if (this.changeHistory.length > this.maxHistorySize) {
      this.changeHistory.shift();
    }
  }

  start() {
    console.log(`
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   Codebase Monitor Server                             ║
║   ───────────────────────                            ║
║   Status: Running                                     ║
║   Port: ${this.port}                                          ║
║   Dashboard: http://localhost:${this.port}/                 ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
    `);

    return this.server;
  }

  stop() {
    console.log('[Server] Shutting down...');

    if (this.fileWatcher) {
      this.fileWatcher.stop();
    }

    if (this.gitMonitor) {
      this.gitMonitor.stopPolling();
    }

    if (this.wsManager) {
      this.wsManager.shutdown();
    }

    if (this.server) {
      this.server.close();
    }

    console.log('[Server] Shutdown complete');
  }
}

const server = new CodebaseMonitorServer();

server.initialize().then(() => {
  server.start();
});

process.on('SIGINT', () => {
  server.stop();
  process.exit(0);
});

process.on('SIGTERM', () => {
  server.stop();
  process.exit(0);
});

module.exports = CodebaseMonitorServer;
