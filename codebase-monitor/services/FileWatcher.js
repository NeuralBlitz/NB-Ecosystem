const chokidar = require('chokidar');
const EventEmitter = require('events');
const path = require('path');
const fs = require('fs');

class FileWatcher extends EventEmitter {
  constructor(watchPath, options = {}) {
    super();
    this.watchPath = watchPath;
    this.options = options;
    this.watcher = null;
    this.fileStates = new Map();
    this.ignorePatterns = [
      /node_modules/,
      /\.git/,
      /dist/,
      /build/,
      /\.cache/,
      /coverage/,
      /\.env/,
      /package-lock\.json/,
      /\.log$/
    ];
  }

  start() {
    const ignorePatterns = this.options.ignorePatterns || this.ignorePatterns;

    this.watcher = chokidar.watch(this.watchPath, {
      ignored: ignorePatterns,
      persistent: true,
      awaitWriteFinish: {
        stabilityThreshold: 300,
        pollInterval: 100
      },
      usePolling: false,
      interval: 100,
      binaryInterval: 300
    });

    this.watcher
      .on('add', (filePath) => this.handleFileAdd(filePath))
      .on('change', (filePath) => this.handleFileChange(filePath))
      .on('unlink', (filePath) => this.handleFileUnlink(filePath))
      .on('ready', () => this.handleReady())
      .on('error', (error) => this.handleError(error));

    console.log(`[FileWatcher] Started watching: ${this.watchPath}`);
  }

  handleFileAdd(filePath) {
    const stats = fs.statSync(filePath);
    const relativePath = path.relative(process.cwd(), filePath);

    this.fileStates.set(filePath, {
      type: 'added',
      size: stats.size,
      mtime: stats.mtime
    });

    this.emit('change', {
      type: 'added',
      file: relativePath,
      size: stats.size,
      timestamp: new Date().toISOString(),
      path: filePath
    });
  }

  handleFileChange(filePath) {
    if (!this.fileStates.has(filePath)) {
      this.handleFileAdd(filePath);
      return;
    }

    const stats = fs.statSync(filePath);
    const previousState = this.fileStates.get(filePath);
    const relativePath = path.relative(process.cwd(), filePath);

    this.fileStates.set(filePath, {
      type: 'modified',
      size: stats.size,
      mtime: stats.mtime,
      previousSize: previousState.size
    });

    this.emit('change', {
      type: 'modified',
      file: relativePath,
      size: stats.size,
      previousSize: previousState.previousSize,
      timestamp: new Date().toISOString(),
      path: filePath
    });
  }

  handleFileUnlink(filePath) {
    const previousState = this.fileStates.get(filePath);
    const relativePath = path.relative(process.cwd(), filePath);

    this.fileStates.delete(filePath);

    this.emit('change', {
      type: 'deleted',
      file: relativePath,
      timestamp: new Date().toISOString(),
      path: filePath,
      previousSize: previousState ? previousState.size : 0
    });
  }

  handleReady() {
    console.log('[FileWatcher] Initial scan complete');
    this.emit('ready');
  }

  handleError(error) {
    console.error('[FileWatcher] Error:', error.message);
    this.emit('error', error);
  }

  stop() {
    if (this.watcher) {
      this.watcher.close();
      console.log('[FileWatcher] Stopped watching');
    }
  }

  getFileStates() {
    return Array.from(this.fileStates.entries()).map(([path, state]) => ({
      path,
      ...state
    }));
  }

  getStatistics() {
    const stats = {
      totalFiles: this.fileStates.size,
      byType: {
        added: 0,
        modified: 0,
        deleted: 0
      }
    };

    this.fileStates.forEach((state) => {
      stats.byType[state.type]++;
    });

    return stats;
  }
}

module.exports = FileWatcher;
