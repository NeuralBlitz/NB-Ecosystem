const simpleGit = require('simple-git');
const path = require('path');

class GitMonitor {
  constructor(repoPath = '.') {
    this.git = simpleGit(repoPath);
    this.repoPath = repoPath;
    this.commitHistory = [];
    this.branches = [];
    this.pollInterval = null;
    this.onCommitCallbacks = [];
  }

  async initialize() {
    try {
      const isRepo = await this.git.checkIsRepo();
      if (!isRepo) {
        console.log('[GitMonitor] Not a git repository, will track file changes only');
        return false;
      }

      await this.fetchBranches();
      await this.fetchCommitHistory();
      console.log('[GitMonitor] Initialized for git repository');
      return true;
    } catch (error) {
      console.error('[GitMonitor] Initialization error:', error.message);
      return false;
    }
  }

  async fetchBranches() {
    try {
      const branchSummary = await this.git.branch();
      this.branches = branchSummary.all.map(branch => ({
        name: branch,
        current: branch === branchSummary.current
      }));
      return this.branches;
    } catch (error) {
      console.error('[GitMonitor] Error fetching branches:', error.message);
      return [];
    }
  }

  async fetchCommitHistory(limit = 50) {
    try {
      const log = await this.git.log({ maxCount: limit });
      this.commitHistory = log.all.map(commit => ({
        hash: commit.hash,
        shortHash: commit.hash.substring(0, 7),
        message: commit.message,
        authorName: commit.author_name,
        authorEmail: commit.author_email,
        date: commit.date,
        refs: commit.refs
      }));
      return this.commitHistory;
    } catch (error) {
      console.error('[GitMonitor] Error fetching commit history:', error.message);
      return [];
    }
  }

  async getDiff(oldCommit, newCommit) {
    try {
      const diff = await this.git.diff([oldCommit, newCommit]);
      return diff;
    } catch (error) {
      console.error('[GitMonitor] Error getting diff:', error.message);
      return '';
    }
  }

  async getFileDiff(commitHash, filePath) {
    try {
      const diff = await this.git.diff([`${commitHash}^`, commitHash, '--', filePath]);
      return diff;
    } catch (error) {
      console.error('[GitMonitor] Error getting file diff:', error.message);
      return '';
    }
  }

  async getStatus() {
    try {
      const status = await this.git.status();
      return {
        current: status.current,
        clean: status.clean,
        files: status.files.map(file => ({
          path: file.path,
          index: file.index,
          working_dir: file.working_dir
        }))
      };
    } catch (error) {
      console.error('[GitMonitor] Error getting status:', error.message);
      return null;
    }
  }

  async getLastCommit() {
    if (this.commitHistory.length === 0) {
      await this.fetchCommitHistory(1);
    }
    return this.commitHistory[0] || null;
  }

  async getStats() {
    const lastCommit = await this.getLastCommit();
    const status = await this.getStatus();

    return {
      repository: this.repoPath,
      currentBranch: status ? status.current : null,
      totalCommits: this.commitHistory.length,
      lastCommit: lastCommit,
      pendingChanges: status ? status.files.length : 0,
      branches: this.branches
    };
  }

  startPolling(interval = 30000) {
    this.pollInterval = setInterval(async () => {
      await this.checkForNewCommits();
    }, interval);
    console.log(`[GitMonitor] Started polling every ${interval}ms`);
  }

  stopPolling() {
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
      console.log('[GitMonitor] Stopped polling');
    }
  }

  async checkForNewCommits() {
    try {
      const log = await this.git.log({ maxCount: 1 });

      if (this.commitHistory.length === 0 || 
          this.commitHistory[0].hash !== log.latest.hash) {
        await this.fetchCommitHistory();
        
        const newCommits = this.commitHistory.filter(commit => 
          !this.onCommitCallbacks.some(cb => cb.id === commit.hash)
        );

        for (const commit of newCommits) {
          this.notifyNewCommit(commit);
        }
      }
    } catch (error) {
      console.error('[GitMonitor] Polling error:', error.message);
    }
  }

  onNewCommit(callback) {
    this.onCommitCallbacks.push(callback);
  }

  notifyNewCommit(commit) {
    this.onCommitCallbacks.forEach(callback => {
      try {
        callback(commit);
      } catch (error) {
        console.error('[GitMonitor] Callback error:', error.message);
      }
    });
  }
}

module.exports = GitMonitor;
