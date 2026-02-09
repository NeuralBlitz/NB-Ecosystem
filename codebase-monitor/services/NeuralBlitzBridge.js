const EventEmitter = require('events');
const fs = require('fs');
const path = require('path');

class NeuralBlitzBridge extends EventEmitter {
  constructor(config = {}) {
    super();
    this.basePath = config.basePath || '/home/runner/workspace';
    this.projects = new Map();
    this.filePatterns = {
      python: ['*.py', '**/*.py'],
      javascript: ['*.js', '**/*.js', '*.jsx', '**/*.jsx'],
      typescript: ['*.ts', '**/*.ts', '*.tsx', '**/*.tsx'],
      markdown: ['*.md', '**/*.md'],
      config: ['package.json', 'requirements.txt', 'pyproject.toml', 'docker-compose.yml', 'Dockerfile']
    };
    this.changeBuffer = [];
    this.maxBufferSize = 100;
    this.flushInterval = null;
  }

  initialize() {
    this.discoverProjects();
    this.startChangeBuffering();
    console.log('[NeuralBlitz Bridge] Initialized with projects:', Array.from(this.projects.keys()));
    return this;
  }

  discoverProjects() {
    const projectConfigs = [
      {
        id: 'nbx-lrs',
        name: 'NBX-LRS',
        path: 'NBX-LRS',
        type: 'lrs',
        description: 'NeuralBlitz Learning Record Store'
      },
      {
        id: 'nb-ecosystem',
        name: 'NB-Ecosystem',
        path: 'NB-Ecosystem',
        type: 'frontend',
        description: 'NB Frontend Ecosystem'
      },
      {
        id: 'codebase-monitor',
        name: 'Codebase Monitor',
        path: 'codebase-monitor',
        type: 'monitoring',
        description: 'Codebase Monitoring Server'
      }
    ];

    for (const config of projectConfigs) {
      const fullPath = path.join(this.basePath, config.path);
      try {
        if (fs.existsSync(fullPath)) {
          const project = this.createProject(fullPath, config);
          this.projects.set(config.id, project);
          console.log(`[NeuralBlitz Bridge] Discovered: ${config.name} at ${fullPath}`);
        }
      } catch (error) {
        console.warn(`[NeuralBlitz Bridge] Failed to access ${config.path}: ${error.message}`);
      }
    }
  }

  createProject(basePath, config) {
    const project = {
      id: config.id,
      name: config.name,
      type: config.type,
      path: basePath,
      description: config.description,
      status: 'active',
      files: new Map(),
      lastScan: null,
      metadata: {}
    };

    this.scanProject(project);
    return project;
  }

  scanProject(project) {
    const startTime = Date.now();
    const files = [];

    try {
      const scanDir = (dir, patterns) => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });

        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          const relativePath = path.relative(project.path, fullPath);

          if (entry.isDirectory()) {
            if (!this.shouldSkip(relativePath)) {
              scanDir(fullPath, patterns);
            }
          } else if (entry.isFile()) {
            if (this.matchesPatterns(entry.name, patterns)) {
              const stats = fs.statSync(fullPath);
              files.push({
                path: fullPath,
                relativePath,
                name: entry.name,
                size: stats.size,
                mtime: stats.mtime,
                type: this.getFileType(entry.name)
              });
            }
          }
        }
      };

      const patterns = [
        ...this.filePatterns.python,
        ...this.filePatterns.javascript,
        ...this.filePatterns.typescript,
        ...this.filePatterns.markdown,
        ...this.filePatterns.config
      ];

      scanDir(project.path, patterns);

      project.files = new Map(files.map(f => [f.relativePath, f]));
      project.lastScan = new Date().toISOString();
      project.fileCount = files.length;

      const duration = Date.now() - startTime;
      console.log(`[NeuralBlitz Bridge] Scanned ${project.name}: ${files.length} files in ${duration}ms`);

    } catch (error) {
      console.error(`[NeuralBlitz Bridge] Scan error for ${project.name}:`, error.message);
    }

    return project;
  }

  shouldSkip(relativePath) {
    const skipPatterns = [
      /node_modules/,
      /\.git/,
      /__pycache__/,
      /\.cache/,
      /dist\//,
      /build\//,
      /\.env/,
      /venv\//,
      /\.idea/,
      /\.vscode/
    ];

    return skipPatterns.some(pattern => pattern.test(relativePath));
  }

  matchesPatterns(filename, patterns) {
    return patterns.some(pattern => {
      if (pattern.startsWith('*')) {
        const regex = new RegExp(pattern.replace('*', '.*'));
        return regex.test(filename);
      }
      return filename === pattern;
    });
  }

  getFileType(filename) {
    if (filename.endsWith('.py')) return 'python';
    if (filename.endsWith('.js') || filename.endsWith('.jsx')) return 'javascript';
    if (filename.endsWith('.ts') || filename.endsWith('.tsx')) return 'typescript';
    if (filename.endsWith('.md')) return 'markdown';
    if (filename.endsWith('.json')) return 'json';
    if (filename === 'Dockerfile' || filename.includes('docker')) return 'docker';
    return 'other';
  }

  getProjectStats() {
    const stats = {
      totalProjects: this.projects.size,
      totalFiles: 0,
      byType: {},
      projects: []
    };

    this.projects.forEach((project, id) => {
      stats.totalFiles += project.fileCount || 0;

      const typeCount = stats.byType[project.type] || 0;
      stats.byType[project.type] = typeCount + (project.fileCount || 0);

      stats.projects.push({
        id: project.id,
        name: project.name,
        type: project.type,
        files: project.fileCount,
        lastScan: project.lastScan,
        status: project.status
      });
    });

    return stats;
  }

  getFileTree(projectId) {
    const project = this.projects.get(projectId);
    if (!project) return null;

    const tree = {
      name: project.name,
      type: 'directory',
      children: {}
    };

    project.files.forEach((file) => {
      const parts = file.relativePath.split(path.sep);
      let current = tree.children;

      for (let i = 0; i < parts.length; i++) {
        const part = parts[i];
        if (i === parts.length - 1) {
          current[part] = {
            name: part,
            type: 'file',
            size: file.size,
            path: file.relativePath,
            fileType: file.type
          };
        } else {
          if (!current[part]) {
            current[part] = {
              name: part,
              type: 'directory',
              children: {}
            };
          }
          current = current[part].children;
        }
      }
    });

    return tree;
  }

  searchFiles(projectId, query, options = {}) {
    const project = this.projects.get(projectId);
    if (!project) return [];

    const results = [];
    const { type, limit = 50 } = options;

    project.files.forEach((file) => {
      if (type && file.type !== type) return;
      if (query && !file.name.toLowerCase().includes(query.toLowerCase())) return;

      results.push({
        path: file.relativePath,
        name: file.name,
        size: file.size,
        type: file.type,
        mtime: file.mtime
      });
    });

    return results.slice(0, limit);
  }

  startChangeBuffering() {
    this.flushInterval = setInterval(() => {
      if (this.changeBuffer.length > 0) {
        this.flushChanges();
      }
    }, 1000);
  }

  flushChanges() {
    const changes = [...this.changeBuffer];
    this.changeBuffer = [];

    this.emit('changes', {
      count: changes.length,
      changes,
      timestamp: new Date().toISOString()
    });
  }

  recordChange(projectId, filePath, changeType) {
    const project = this.projects.get(projectId);
    if (!project) return;

    const change = {
      projectId,
      projectName: project.name,
      file: filePath,
      type: changeType,
      timestamp: new Date().toISOString()
    };

    this.changeBuffer.push(change);

    if (this.changeBuffer.length >= this.maxBufferSize) {
      this.flushChanges();
    }

    this.emit('change', change);
  }

  async getFileContent(projectId, filePath) {
    const project = this.projects.get(projectId);
    if (!project) throw new Error('Project not found');

    const fullPath = path.join(project.path, filePath);
    try {
      const content = fs.readFileSync(fullPath, 'utf8');
      return content;
    } catch (error) {
      throw new Error(`Cannot read file: ${error.message}`);
    }
  }

  getProjectFiles(projectId) {
    const project = this.projects.get(projectId);
    if (!project) return [];

    return Array.from(project.files.values()).map(f => ({
      path: f.relativePath,
      name: f.name,
      size: f.size,
      type: f.type,
      mtime: f.mtime
    }));
  }

  rescanProject(projectId) {
    const project = this.projects.get(projectId);
    if (!project) return null;
    return this.scanProject(project);
  }

  shutdown() {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    console.log('[NeuralBlitz Bridge] Shutdown complete');
  }
}

module.exports = NeuralBlitzBridge;
