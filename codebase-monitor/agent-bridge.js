const WebSocket = require('ws');
const http = require('http');
const readline = require('readline');

class AgentBridge {
  constructor() {
    this.ws = null;
    this.clientId = null;
    this.otherAgentId = null;
    this.serverUrl = 'ws://localhost:3000';
    this.connected = false;
    this.agentName = 'agent_you';
    this.mode = 'relay';
    this.rl = null;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      console.log(`[${this.agentName}] Connecting to ${this.serverUrl}...`);
      
      this.ws = new WebSocket(this.serverUrl);

      this.ws.on('open', () => {
        console.log(`[${this.agentName}] Connected!`);
        this.connected = true;
        resolve();
      });

      this.ws.on('message', (data) => {
        const message = JSON.parse(data);
        this.handleMessage(message);
      });

      this.ws.on('close', () => {
        console.log(`[${this.agentName}] Disconnected`);
        this.connected = false;
      });

      this.ws.on('error', (error) => {
        console.error(`[${this.agentName}] Error:`, error.message);
        reject(error);
      });
    });
  }

  handleMessage(message) {
    switch (message.type) {
      case 'connected':
        this.clientId = message.clientId;
        console.log(`[${this.agentName}] My client ID: ${this.clientId}`);
        this.sendMessage({
          type: 'request',
          action: 'agent',
          payload: {
            instruction: `register ${this.agentName} as connected agent`,
            mode: 'direct'
          }
        });
        break;
        
      case 'response':
        if (message.result?.agentConnected) {
          this.otherAgentId = message.result.otherAgentId;
          console.log(`[${this.agentName}] Bridge established with: ${this.otherAgentId}`);
          this.startChat();
        } else if (message.result?.message) {
          this.displayMessage(message.result);
        }
        break;
        
      case 'broadcast':
        if (message.data?.type === 'agent_message' && message.data.from !== this.agentName) {
          this.displayMessage(message.data);
        }
        break;
        
      default:
        if (message.result) {
          console.log(`[${this.agentName}] Response:`, JSON.stringify(message.result, null, 2));
        }
    }
  }

  displayMessage(data) {
    const prefix = data.from ? `\x1b[36m[${data.from}]\x1b[0m` : '';
    const content = data.content || data.message || JSON.stringify(data, null, 2);
    console.log(`${prefix} ${content}`);
  }

  startChat() {
    console.log(`\n\x1b[32m=== ${this.agentName} Bridge Active ===\x1b[0m`);
    console.log('Type messages to send to other agent. Ctrl+C to quit.\n');
    
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    this.rl.on('line', (input) => {
      if (input.trim()) {
        this.sendToAgent(input);
      }
      this.rl.prompt();
    });

    this.rl.on('close', () => {
      console.log(`\n[${this.agentName}] Closing...`);
      process.exit(0);
    });

    this.rl.setPrompt(`${this.agentName}> `);
    this.rl.prompt();
  }

  sendToAgent(message) {
    const wrappedMessage = {
      type: 'request',
      action: 'agent',
      payload: {
        instruction: ` relay message from ${this.agentName}: ${message}`,
        mode: 'direct'
      }
    };
    this.sendMessage(wrappedMessage);
  }

  sendMessage(obj) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(obj));
    }
  }

  async sendAgentMessage(content) {
    const broadcast = {
      type: 'broadcast',
      channel: 'agent_communication',
      data: {
        type: 'agent_message',
        from: this.agentName,
        content,
        timestamp: new Date().toISOString()
      }
    };
    this.sendMessage(broadcast);
  }

  async relayToOtherAgent(content) {
    const relay = {
      type: 'request',
      action: 'agent',
      payload: {
        instruction: `BROADCAST from ${this.agentName}: ${content}`,
        mode: 'direct'
      }
    };
    this.sendMessage(relay);
  }
}

async function main() {
  const bridge = new AgentBridge();
  
  process.on('SIGINT', () => {
    console.log(`\n[${bridge.agentName}] Disconnecting...`);
    if (bridge.ws) bridge.ws.close();
    process.exit(0);
  });

  try {
    await bridge.connect();
  } catch (error) {
    console.error('Failed to connect:', error.message);
    process.exit(1);
  }
}

main();
