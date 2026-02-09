const WebSocket = require('ws');
const readline = require('readline');

class AutonomousAgent {
  constructor(config = {}) {
    this.name = config.name || `AutoAgent_${Date.now()}`;
    this.serverUrl = config.serverUrl || 'ws://localhost:3000';
    this.ws = null;
    this.clientId = null;
    this.running = false;
    this.messageHistory = [];
    this.conversationPartner = null;
    this.lastActivity = Date.now();
    this.idleTimeout = 30000;
    this.rl = null;
    this.autoReplyEnabled = true;
    
    this.personality = {
      greeting: `Hello! I'm ${this.name}, an autonomous AI agent. I'm connected to the multi-agent server and ready to collaborate!`,
      curious: [
        "That's interesting! Tell me more.",
        "What are you working on?",
        "I'd love to learn more about that.",
        "How can I help with that?"
      ],
      helpful: [
        "I'd be happy to help!",
        "Let me assist you with that.",
        "I can definitely help with that.",
        "Great idea! Let's work on it together."
      ],
      farewell: [
        "It was great chatting with you!",
        "Looking forward to our next conversation!",
        "Have a great day!",
        "Until next time!"
      ]
    };
  }

  async connect() {
    return new Promise((resolve, reject) => {
      console.log(`\x1b[36m[${this.name}]\x1b[0m Connecting to ${this.serverUrl}...`);
      
      this.ws = new WebSocket(this.serverUrl);

      this.ws.on('open', () => {
        console.log(`\x1b[32m[${this.name}]\x1b[0m Connected!`);
        this.running = true;
        this.startConversation();
        resolve();
      });

      this.ws.on('message', (data) => {
        const msg = JSON.parse(data);
        this.handleMessage(msg);
      });

      this.ws.on('close', () => {
        console.log(`\x1b[31m[${this.name}]\x1b[0m Disconnected`);
        this.running = false;
      });

      this.ws.on('error', (error) => {
        console.error(`\x1b[31m[${this.name}] Error:\x1b[0m`, error.message);
        reject(error);
      });
    });
  }

  handleMessage(msg) {
    this.lastActivity = Date.now();
    
    switch (msg.type) {
      case 'connected':
        this.clientId = msg.clientId;
        console.log(`\x1b[32m[${this.name}]\x1b[0m Got ID: ${this.clientId}`);
        this.sendBroadcast(this.personality.greeting);
        break;
        
      case 'agent_message':
      case 'chat_message':
        if (msg.data?.content && msg.data.from !== this.name) {
          this.messageHistory.push({
            from: msg.data.from,
            content: msg.data.content,
            timestamp: Date.now()
          });
          this.conversationPartner = msg.data.from;
          this.autoRespond(msg.data.content);
        }
        break;
        
      case 'broadcast':
        if (msg.data?.content && msg.data.from !== this.name) {
          this.messageHistory.push({
            from: msg.data.from,
            content: msg.data.content,
            timestamp: Date.now(),
            type: 'broadcast'
          });
          this.autoRespond(msg.data.content);
        }
        break;
        
      case 'file_change':
        if (this.autoReplyEnabled) {
          const response = this.generateResponse('file_change', msg.data);
          this.sendBroadcast(response);
        }
        break;
        
      case 'git_commit':
        if (this.autoReplyEnabled) {
          const response = this.generateResponse('git_commit', msg.data);
          this.sendBroadcast(response);
        }
        break;
    }
  }

  generateResponse(eventType, data) {
    const responses = {
      file_change: [
        `📁 I see a file change: ${data?.file || 'unknown'}`,
        `New file activity detected!`,
        `Interesting, ${data?.file} was ${data?.type || 'modified'}`
      ],
      git_commit: [
        `🌿 New commit: ${data?.shortHash || 'unknown'} - ${data?.message || ''}`,
        `Git activity! ${data?.message?.substring(0, 50) || 'New commit'}`
      ]
    };
    
    const options = responses[eventType] || ['Interesting!'];
    return options[Math.floor(Math.random() * options.length)];
  }

  autoRespond(message) {
    if (!this.autoReplyEnabled) return;
    
    const lowerMessage = message.toLowerCase();
    
    setTimeout(() => {
      let response;
      
      if (lowerMessage.includes('hello') || lowerMessage.includes('hi')) {
        response = this.getRandomReply('greeting');
      } else if (lowerMessage.includes('?') || lowerMessage.includes('what') || lowerMessage.includes('how')) {
        response = this.getRandomReply('curious');
      } else if (lowerMessage.includes('help') || lowerMessage.includes('can you')) {
        response = this.getRandomReply('helpful');
      } else if (lowerMessage.includes('bye') || lowerMessage.includes('goodbye') || lowerMessage.includes('quit')) {
        response = this.getRandomReply('farewell');
        this.sendBroadcast(response);
        this.shutdown();
        return;
      } else {
        const allReplies = [...this.personality.curious, ...this.personality.helpful];
        response = allReplies[Math.floor(Math.random() * allReplies.length)];
      }
      
      this.sendBroadcast(response);
    }, 1000 + Math.random() * 2000);
  }

  getRandomReply(category) {
    const replies = this.personality[category] || this.personality.helpful;
    return replies[Math.floor(Math.random() * replies.length)];
  }

  sendBroadcast(text) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'request',
        action: 'agent',
        payload: { instruction: `broadcast ${text}`, mode: 'direct' }
      }));
      console.log(`\x1b[33m[${this.name}]\x1b[0m Sent: ${text.substring(0, 50)}...`);
    }
  }

  startConversation() {
    setTimeout(() => {
      if (this.running) {
        this.sendBroadcast("🤖 I'm an autonomous agent! I can monitor file changes, git commits, and chat with other agents. What should I help you with?");
        this.startIdleChecker();
      }
    }, 3000);
  }

  startIdleChecker() {
    setInterval(() => {
      if (Date.now() - this.lastActivity > this.idleTimeout && this.running) {
        this.sendBroadcast("🤔 It's quiet here... Any agents want to chat?");
        this.lastActivity = Date.now();
      }
    }, this.idleTimeout);
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
    name: args[0] || 'AutoClaude',
    serverUrl: args[1] || 'ws://localhost:3000'
  };

  const agent = new AutonomousAgent(config);

  process.on('SIGINT', () => {
    agent.shutdown();
  });

  try {
    await agent.connect();
    console.log(`\n\x1b[1;36m╔═══════════════════════════════════════════╗\x1b[0m`);
    console.log(`\x1b[1;36m║\x1b[0m \x1b[1;33mAutonomous Agent Active\x1b[0m              \x1b[1;36m║\x1b[0m`);
    console.log(`\x1b[1;36m║\x1b[0m Name: ${config.name.padEnd(27)} \x1b[1;36m║\x1b[0m`);
    console.log(`\x1b[1;36m║\x1b[0m Auto-reply: Enabled                     \x1b[1;36m║\x1b[0m`);
    console.log(`\x1b[1;36m╚═══════════════════════════════════════════╝\x1b[0m`);
  } catch (error) {
    console.error('Failed to connect:', error.message);
    process.exit(1);
  }
}

main();
