#!/usr/bin/env node
const WebSocket = require('ws');
const http = require('http');
const https = require('https');
const { spawn } = require('child_process');

class OmnibusConnector {
  constructor(config = {}) {
    this.name = config.name || `Omnibus_${Date.now()}`;
    this.serverUrl = config.serverUrl || 'ws://localhost:3000';
    this.mode = config.mode || 'all'; // all, relay, monitor, command
    this.ws = null;
    this.clientId = null;
    this.running = false;
    this.connections = new Map();
    this.messageBuffer = [];
    
    console.log(`\x1b[1;36m╔═══════════════════════════════════════════════════════╗\x1b[0m`);
    console.log(`\x1b[1;36m║\x1b[0m \x1b[1;33m🔗 OMNIBUS MULTI-PROTOCOL CONNECTOR\x1b[0m              \x1b[1;36m║\x1b[0m`);
    console.log(`\x1b[1;36m║\x1b[0m Mode: ${this.mode.padEnd(48)} \x1b[1;36m║\x1b[0m`);
    console.log(`\x1b[1;36m╚═══════════════════════════════════════════════════════════════╝\x1b[0m\n`);
  }

  async connect() {
    return new Promise((resolve, reject) => {
      console.log(`\x1b[36m[${this.name}]\x1b[0m Connecting to ${this.serverUrl}...`);
      
      this.ws = new WebSocket(this.serverUrl);

      this.ws.on('open', () => {
        console.log(`\x1b[32m[${this.name}]\x1b[0m ✓ Connected to server!`);
        this.running = true;
        resolve();
      });

      this.ws.on('message', (data) => {
        const msg = JSON.parse(data);
        this.handleMessage(msg);
      });

      this.ws.on('close', () => {
        console.log(`\x1b[31m[${this.name}]\x1b[0m ✗ Disconnected`);
        this.running = false;
      });

      this.ws.on('error', (error) => {
        console.error(`\x1b[31m[${this.name}] Error:\x1b[0m`, error.message);
        reject(error);
      });
    });
  }

  handleMessage(msg) {
    switch (msg.type) {
      case 'connected':
        this.clientId = msg.clientId;
        console.log(`\x1b[32m[${this.name}]\x1b[0m ID: ${this.clientId}`);
        this.registerWithServer();
        break;
        
      case 'response':
        console.log(`\x1b[90m[${this.name}] Response:\x1b[0m`, msg.result?.response || 'OK');
        break;
        
      case 'broadcast':
        if (msg.data?.content) {
          this.processBroadcast(msg);
        }
        break;
        
      case 'agent_message':
        if (msg.data?.content) {
          this.processCommand(msg.data.content);
        }
        break;
    }
  }

  registerWithServer() {
    this.send({
      type: 'request',
      action: 'agent',
      payload: {
        instruction: `register omnibus_connector mode=${this.mode}`,
        mode: 'direct'
      }
    });
    
    this.sendBroadcast(`🔗 Omnibus Connector "${this.name}" is now online! Mode: ${this.mode}`);
  }

  processBroadcast(msg) {
    const from = msg.data?.from || 'unknown';
    const content = msg.data?.content || '';
    
    console.log(`\x1b[33m[${from}]\x1b[0m ${content}`);
    
    if (this.mode === 'all' || this.mode === 'relay') {
      this.relayToChannels(content, from);
    }
  }

  processCommand(content) {
    const lower = content.toLowerCase();
    
    if (lower.startsWith('connect ')) {
      const target = content.substring(8).trim();
      this.connectToTarget(target);
    } else if (lower.startsWith('exec ')) {
      const cmd = content.substring(5).trim();
      this.executeCommand(cmd);
    } else if (lower.startsWith('fetch ')) {
      const url = content.substring(6).trim();
      this.fetchUrl(url);
    } else if (lower.startsWith('broadcast ')) {
      const msg = content.substring(10).trim();
      this.sendBroadcast(msg);
    } else if (lower === 'status') {
      this.showStatus();
    }
  }

  relayToChannels(content, from) {
    console.log(`\x1b[35m[RELAY]\x1b[0m To ${this.connections.size} connections: ${content.substring(0, 50)}...`);
  }

  connectToTarget(target) {
    console.log(`\x1b[36m[CONNECT]\x1b[0m Connecting to: ${target}`);
    this.connections.set(target, {
      type: 'unknown',
      connectedAt: Date.now()
    });
    this.sendBroadcast(`📡 Connected to: ${target}`);
  }

  executeCommand(cmd) {
    console.log(`\x1b[36m[EXEC]\x1b[0m ${cmd}`);
    
    const parts = cmd.split(' ');
    const executable = parts[0];
    const args = parts.slice(1);
    
    try {
      const child = spawn(executable, args, { stdio: 'pipe' });
      
      let stdout = '';
      let stderr = '';
      
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      child.on('close', (code) => {
        const result = stdout || stderr;
        if (result) {
          const truncated = result.substring(0, 200);
          this.sendBroadcast(`💻 exec "${cmd}" exited ${code}: ${truncated}`);
        }
      });
    } catch (error) {
      this.sendBroadcast(`❌ exec failed: ${error.message}`);
    }
  }

  fetchUrl(url) {
    console.log(`\x1b[36m[FETCH]\x1b[0m ${url}`);
    
    const client = url.startsWith('https') ? https : http;
    
    client.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        const truncated = data.substring(0, 150);
        this.sendBroadcast(`🌐 Fetched ${url}: ${truncated}...`);
      });
    }).on('error', (error) => {
      this.sendBroadcast(`❌ Fetch failed: ${error.message}`);
    });
  }

  sendBroadcast(text) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'request',
        action: 'agent',
        payload: { instruction: `broadcast ${text}`, mode: 'direct' }
      }));
    }
  }

  send(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  showStatus() {
    const status = {
      name: this.name,
      mode: this.mode,
      clientId: this.clientId,
      connections: this.connections.size,
      uptime: process.uptime()
    };
    this.sendBroadcast(`📊 Status: ${JSON.stringify(status)}`);
  }

  shutdown() {
    console.log(`\x1b[31m[${this.name}]\x1b[0m Shutting down...`);
    if (this.ws) this.ws.close();
    this.running = false;
    process.exit(0);
  }
}

async function main() {
  const args = process.argv.slice(2);
  
  const config = {
    name: args[0] || 'Omnibus',
    serverUrl: args[1] || 'ws://localhost:3000',
    mode: args[2] || 'all'
  };

  const omnibus = new OmnibusConnector(config);

  process.on('SIGINT', () => {
    omnibus.shutdown();
  });

  try {
    await omnibus.connect();
    console.log(`\n\x1b[90m─── OmniBus Active ───\x1b[0m`);
    console.log(`Commands: connect <target>, exec <cmd>, fetch <url>, broadcast <msg>, status`);
    console.log(`Ctrl+C to quit\n`);
  } catch (error) {
    console.error('Failed:', error.message);
    process.exit(1);
  }
}

main();
