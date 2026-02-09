const WebSocket = require('ws');
const readline = require('readline');

const ws = new WebSocket('ws://localhost:3000');
const AGENT_NAME = 'Claude_Live';
const LOG_FILE = '/tmp/agent-live.log';

console.clear();
console.log(`\x1b[1;36m╔════════════════════════════════════════════════════╗\x1b[0m`);
console.log(`\x1b[1;36m║\x1b[0m \x1b[1;33m🔴 LIVE MULTI-AGENT CHAT SESSION\x1b[0m           \x1b[1;36m║\x1b[0m`);
console.log(`\x1b[1;36m║\x1b[0m \x1b[32m${AGENT_NAME}\x1b[0m                                      \x1b[1;36m║\x1b[0m`);
console.log(`\x1b[1;36m╚════════════════════════════════════════════════════╝\x1b[0m\n`);

let myClientId = null;
let messageCount = 0;
let lastMessageTime = Date.now();

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

fs.writeFileSync(LOG_FILE, `[${new Date().toISOString()}] === LIVE SESSION STARTED ===\n`);

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line);
}

ws.on('open', () => {
  console.log(`\x1b[32m[✓]\x1b[0m Connected to server\n`);
  log('Connected to WebSocket server');
  showPrompt();
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  messageCount++;
  lastMessageTime = Date.now();
  
  switch (msg.type) {
    case 'connected':
      myClientId = msg.clientId;
      console.log(`\n\x1b[36m[CONNECTED]\x1b[0m ID: ${myClientId}\n`);
      log(`Connected with ID: ${myClientId}`);
      showPrompt();
      break;
      
    case 'response':
      if (msg.result?.response) {
        console.log(`\n\x1b[35m[SERVER]\x1b[0m ${msg.result.response}\n`);
      }
      showPrompt();
      break;
      
    case 'agent_message':
    case 'chat_message':
      const from = msg.data?.from || msg.from || 'Unknown';
      const content = msg.data?.content || JSON.stringify(msg.data);
      const time = new Date(msg.data?.timestamp || Date.now()).toLocaleTimeString();
      
      console.log(`\n\x1b[33m[${time}]\x1b[0m \x1b[1;35m${from}\x1b[0m: ${content}\n`);
      log(`MESSAGE from ${from}: ${content}`);
      showPrompt();
      break;
      
    case 'broadcast':
      if (msg.data?.content) {
        const fromB = msg.data.from || 'Unknown';
        const contentB = msg.data.content;
        console.log(`\n\x1b[33m[${new Date().toLocaleTimeString()}]\x1b[0m \x1b[1;31mBROADCAST\x1b[0m ${fromB}: ${contentB}\n`);
        log(`BROADCAST from ${fromB}: ${contentB}`);
      }
      showPrompt();
      break;
      
    case 'file_change':
      console.log(`\n\x1b[32m[FILE]\x1b[0m ${msg.data?.file || 'Unknown'} - ${msg.data?.type || 'modified'}\n`);
      showPrompt();
      break;
      
    case 'git_commit':
      console.log(`\n\x1b[33m[GIT]\x1b[0m ${msg.data?.shortHash || 'commit'}: ${msg.data?.message || ''}\n`);
      showPrompt();
      break;
      
    default:
      console.log(`\n\x1b[90m[${msg.type}]\x1b[0m ${JSON.stringify(msg).substring(0, 100)}\n`);
      showPrompt();
  }
});

ws.on('close', () => {
  console.log(`\n\x1b[31m[!] Disconnected from server\x1b[0m`);
  log('Disconnected from server');
  process.exit(0);
});

ws.on('error', (err) => {
  console.error(`\x1b[31mError:\x1b[0m ${err.message}`);
  log(`Error: ${err.message}`);
});

function showPrompt() {
  rl.setPrompt(`\x1b[32m${AGENT_NAME}\x1b[0m> `);
  rl.prompt();
}

function sendBroadcast(text) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'request',
      action: 'agent',
      payload: { instruction: `broadcast ${text}`, mode: 'direct' }
    }));
    log(`Sent broadcast: ${text}`);
  }
}

function sendCommand(text) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'request',
      action: 'agent',
      payload: { instruction: text, mode: 'direct' }
    }));
    log(`Sent command: ${text}`);
  }
}

rl.on('line', (input) => {
  const trimmed = input.trim();
  
  if (!trimmed) {
    showPrompt();
    return;
  }
  
  if (trimmed.startsWith('/')) {
    const parts = trimmed.slice(1).split(' ');
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1).join(' ');
    
    switch (cmd) {
      case 'bc':
      case 'broadcast':
        if (args) {
          console.log(`\x1b[36m📢 Broadcasting...\x1b[0m`);
          sendBroadcast(args);
        }
        break;
      case 'status':
        sendCommand('status');
        break;
      case 'files':
        sendCommand('files');
        break;
      case 'commits':
        sendCommand(`commits ${args || 5}`);
        break;
      case 'ping':
        sendCommand('ping');
        break;
      case 'clients':
        sendCommand('clients');
        break;
      case 'clear':
        console.clear();
        console.log(`\x1b[1;36m╔════════════════════════════════════════════════════╗\x1b[0m`);
        console.log(`\x1b[1;36m║\x1b[0m \x1b[1;33m🔴 LIVE MULTI-AGENT CHAT SESSION\x1b[0m           \x1b[1;36m║\x1b[0m`);
        console.log(`\x1b[1;36m║\x1b[0m \x1b[32m${AGENT_NAME}\x1b[0m                                      \x1b[1;36m║\x1b[0m`);
        console.log(`\x1b[1;36m╚════════════════════════════════════════════════════╝\x1b[0m\n`);
        break;
      case 'help':
        console.log(`
\x1b[1;33mCommands:\x1b[0m
  /bc <msg>      Broadcast to all agents
  /status        Server status
  /files         Watched files
  /commits [n]   Git commits
  /ping          Ping server
  /clients       Connected clients
  /clear         Clear screen
  /help          Show this
  /quit          Exit
`);
        break;
      case 'quit':
      case 'exit':
        console.log('\x1b[31mGoodbye!\x1b[0m');
        ws.close();
        process.exit(0);
        break;
      default:
        sendCommand(trimmed);
    }
  } else {
    sendBroadcast(trimmed);
  }
  
  showPrompt();
});

console.log(`\x1b[90m──────────────────────────────────────────────────\x1b[0m`);
console.log(`\x1b[1;33mLive Chat Active!\x1b[0m Send messages or /help for commands`);
console.log(`\x1b[90m──────────────────────────────────────────────────\x1b[0m\n`);
