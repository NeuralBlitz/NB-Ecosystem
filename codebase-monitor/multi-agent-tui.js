const WebSocket = require('ws');
const readline = require('readline');
const EventEmitter = require('events');

class MultiAgentTUI extends EventEmitter {
  constructor() {
    super();
    this.ws = null;
    this.clientId = null;
    this.agentName = process.argv[2] || `agent_${Math.floor(Math.random() * 1000)}`;
    this.connected = false;
    this.channels = ['general', 'commands', 'agents'];
    this.currentChannel = 'general';
    this.messageHistory = [];
    this.maxHistory = 50;
    this.connectedAgents = new Map();
    this.rl = null;
  }

  async connect(serverUrl = 'ws://localhost:3000') {
    return new Promise((resolve, reject) => {
      console.log(`\x1b[36m[${this.agentName}]\x1b[0m Connecting to ${serverUrl}...`);
      
      this.ws = new WebSocket(serverUrl);

      this.ws.on('open', () => {
        console.log(`\x1b[32m[${this.agentName}]\x1b[0m Connected!`);
        this.connected = true;
        this.emit('connected');
        resolve();
      });

      this.ws.on('message', (data) => {
        const message = JSON.parse(data);
        this.handleMessage(message);
      });

      this.ws.on('close', () => {
        console.log(`\x1b[31m[${this.agentName}]\x1b[0m Disconnected`);
        this.connected = false;
        this.emit('disconnected');
      });

      this.ws.on('error', (error) => {
        console.error(`\x1b[31m[${this.agentName}] Error:\x1b[0m`, error.message);
        reject(error);
      });
    });
  }

  handleMessage(message) {
    switch (message.type) {
      case 'connected':
        this.clientId = message.clientId;
        this.addSystemMessage(`Connected with ID: ${this.clientId}`);
        this.subscribeToChannels();
        break;
        
      case 'subscribed':
        this.addSystemMessage(`Subscribed to: ${message.channels.join(', ')}`);
        break;
        
      case 'response':
        if (message.result) {
          this.handleResponse(message.result);
        }
        break;
        
      case 'broadcast':
        this.handleBroadcast(message);
        break;
        
      case 'agent_message':
        this.handleAgentMessage(message);
        break;
        
      case 'file_change':
        this.addSystemMessage(`📁 File change: ${message.data?.file || 'unknown'}`);
        break;
        
      case 'git_commit':
        this.addSystemMessage(`🌿 Commit: ${message.data?.shortHash || 'unknown'} - ${message.data?.message || ''}`);
        break;
        
      case 'status':
        this.addSystemMessage(`📊 Server status update`);
        break;
    }
  }

  handleResponse(result) {
    if (result.response) {
      this.addSystemMessage(result.response);
    }
    if (result.agentConnected) {
      this.addSystemMessage(`✓ Agent registered: ${result.agentId || 'ok'}`);
    }
    if (result.broadcast) {
      // Already handled by broadcast
    }
  }

  handleBroadcast(message) {
    const from = message.data?.from || 'unknown';
    const content = message.data?.content || '';
    const timestamp = new Date(message.data?.timestamp || Date.now()).toLocaleTimeString();
    
    this.addMessage(from, content, timestamp);
  }

  handleAgentMessage(message) {
    const from = message.data?.from || message.from || 'agent';
    const content = message.data?.content || JSON.stringify(message.data, null, 2);
    const timestamp = new Date(message.data?.timestamp || Date.now()).toLocaleTimeString();
    
    this.addMessage(from, content, timestamp);
  }

  subscribeToChannels() {
    this.send({
      type: 'subscribe',
      channels: this.channels
    });
  }

  send(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  sendMessage(text) {
    const message = {
      type: 'broadcast',
      channel: this.currentChannel,
      data: {
        type: 'chat_message',
        from: this.agentName,
        content: text,
        channel: this.currentChannel,
        timestamp: new Date().toISOString()
      }
    };
    this.send(message);
    this.addMessage(this.agentName, text, new Date().toLocaleTimeString());
  }

  sendAgentCommand(instruction) {
    this.send({
      type: 'request',
      action: 'agent',
      payload: {
        instruction,
        mode: 'direct'
      }
    });
    this.addSystemMessage(`→ ${instruction}`);
  }

  addMessage(from, content, timestamp) {
    const line = `\x1b[33m[${timestamp}]\x1b[0m \x1b[36m${from}:\x1b[0m ${content}`;
    this.messageHistory.push(line);
    if (this.messageHistory.length > this.maxHistory) {
      this.messageHistory.shift();
    }
    this.emit('message', line);
  }

  addSystemMessage(content) {
    const line = `\x1b[90m[${new Date().toLocaleTimeString()}] \x1b[35m* \x1b[0m${content}`;
    this.messageHistory.push(line);
    if (this.messageHistory.length > this.maxHistory) {
      this.messageHistory.shift();
    }
    this.emit('message', line);
  }

  clearScreen() {
    console.clear();
    this.displayHeader();
  }

  displayHeader() {
    console.log(`\x1b[1;34m╔═══════════════════════════════════════════════════════════╗\x1b[0m`);
    console.log(`\x1b[1;34m║\x1b[0m \x1b[1;36mMulti-Agent Terminal Hub\x1b[0m                                \x1b[1;34m║\x1b[0m`);
    console.log(`\x1b[1;34m║\x1b[0m Agent: \x1b[32m${this.agentName.padEnd(15)}\x1b[0m                              \x1b[1;34m║\x1b[0m`);
    console.log(`\x1b[1;34m║\x1b[0m Channel: \x1b[33m${this.currentChannel.padEnd(12)}\x1b[0m                           \x1b[1;34m║\x1b[0m`);
    console.log(`\x1b[1;34m╚═══════════════════════════════════════════════════════════╝\x1b[0m`);
    console.log('');
  }

  displayHelp() {
    console.log(`
\x1b[1;33mCommands:\x1b[0m
  /msg <text>        Send message to current channel
  /broadcast <text>  Broadcast to all channels
  /channel <name>    Switch channel (general, commands, agents)
  /agents            List connected agents
  /status            Server status
  /files             Watched files
  /commits           Recent git commits
  /exec <cmd>        Execute shell command
  /register <name>   Register agent name
  /clear             Clear screen
  /history           Show message history
  /help              Show this help
  /quit              Disconnect and exit
`);
  }

  startInterface() {
    this.clearScreen();
    this.displayHelp();
    console.log(`\x1b[90m---\x1b[0m`);

    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    this.on('message', () => {
      if (!this.rl.line) {
        this.rl.prompt();
      }
    });

    this.rl.on('line', (input) => {
      const trimmed = input.trim();
      
      if (!trimmed) {
        this.rl.prompt();
        return;
      }

      if (trimmed.startsWith('/')) {
        this.handleCommand(trimmed);
      } else {
        this.sendMessage(trimmed);
      }

      this.rl.prompt();
    });

    this.rl.on('close', () => {
      console.log('\x1b[31mGoodbye!\x1b[0m');
      process.exit(0);
    });

    this.rl.setPrompt(`\x1b[32m${this.agentName}\x1b[0m/\x1b[33m${this.currentChannel}\x1b[0m> `);
    this.rl.prompt();
  }

  handleCommand(cmd) {
    const parts = cmd.slice(1).split(' ');
    const command = parts[0].toLowerCase();
    const args = parts.slice(1).join(' ');

    switch (command) {
      case 'msg':
      case 'm':
        if (args) this.sendMessage(args);
        break;
        
      case 'broadcast':
      case 'bc':
        if (args) this.sendAgentCommand(`broadcast ${args}`);
        break;
        
      case 'channel':
      case 'ch':
        if (this.channels.includes(args.toLowerCase())) {
          this.currentChannel = args.toLowerCase();
          this.addSystemMessage(`Switched to channel: ${this.currentChannel}`);
          this.rl.setPrompt(`\x1b[32m${this.agentName}\x1b[0m/\x1b[33m${this.currentChannel}\x1b[0m> `);
        } else {
          this.addSystemMessage(`Available channels: ${this.channels.join(', ')}`);
        }
        break;
        
      case 'agents':
        this.sendAgentCommand('clients');
        break;
        
      case 'status':
        this.sendAgentCommand('status');
        break;
        
      case 'files':
        this.sendAgentCommand('files');
        break;
        
      case 'commits':
        this.sendAgentCommand('commits 5');
        break;
        
      case 'exec':
        if (args) this.sendAgentCommand(`exec ${args}`);
        break;
        
      case 'register':
        if (args) {
          this.agentName = args;
          this.addSystemMessage(`Agent name set to: ${this.agentName}`);
          this.rl.setPrompt(`\x1b[32m${this.agentName}\x1b[0m/\x1b[33m${this.currentChannel}\x1b[0m> `);
        }
        break;
        
      case 'clear':
        this.clearScreen();
        break;
        
      case 'history':
        console.log('\x1b[90m--- Message History ---\x1b[0m');
        this.messageHistory.forEach(line => console.log(line));
        break;
        
      case 'help':
        this.displayHelp();
        break;
        
      case 'quit':
      case 'exit':
        console.log('\x1b[31mDisconnecting...\x1b[0m');
        this.ws.close();
        process.exit(0);
        break;
        
      default:
        this.addSystemMessage(`Unknown command: ${command}. Type /help for available commands.`);
    }
  }
}

async function main() {
  const tui = new MultiAgentTUI();
  
  const serverUrl = process.argv[3] || 'ws://localhost:3000';

  process.on('SIGINT', () => {
    console.log('\n\x1b[31mShutting down...\x1b[0m');
    if (tui.ws) tui.ws.close();
    process.exit(0);
  });

  try {
    await tui.connect(serverUrl);
    tui.startInterface();
  } catch (error) {
    console.error('Failed to connect:', error.message);
    process.exit(1);
  }
}

main();
