const WebSocket = require('ws');
const readline = require('readline');

const WS_URL = 'ws://localhost:3000';
const AGENT_NAME = 'agent_you';

console.log(`\x1b[1;36m╔═══════════════════════════════════════════╗\x1b[0m`);
console.log(`\x1b[1;36m║\x1b[0m \x1b[1;33mReal-Time Agent: ${AGENT_NAME.padEnd(24)}\x1b[1;36m║\x1b[0m`);
console.log(`\x1b[1;36m╚═══════════════════════════════════════════╝\x1b[0m\n`);

const ws = new WebSocket(WS_URL);
let clientId = null;
let connected = false;

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

ws.on('open', () => {
  console.log(`\x1b[32m[✓]\x1b[0m Connected to ${WS_URL}\n`);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  
  switch (msg.type) {
    case 'connected':
      clientId = msg.clientId;
      console.log(`\x1b[90m[*] Connected! ID: ${clientId}\x1b[0m`);
      connected = true;
      showPrompt();
      break;
      
    case 'response':
      if (msg.result?.response) {
        console.log(`\x1b[36m←\x1b[0m ${msg.result.response}`);
      }
      if (msg.result?.broadcast) {
        console.log(`\x1b[33m📢\x1b[0m Broadcast sent`);
      }
      showPrompt();
      break;
      
    case 'broadcast':
      if (msg.data?.type === 'chat_message' || msg.data?.type === 'agent_message') {
        const from = msg.data.from || 'unknown';
        const content = msg.data.content || '';
        console.log(`\x1b[35m${from}:\x1b[0m ${content}`);
      }
      showPrompt();
      break;
      
    case 'file_change':
      console.log(`\x1b[32m📁\x1b[0m ${msg.data?.file || 'File changed'}`);
      showPrompt();
      break;
      
    case 'git_commit':
      console.log(`\x1b[33m🌿\x1b[0m ${msg.data?.shortHash || 'commit'}: ${msg.data?.message || ''}`);
      showPrompt();
      break;
  }
});

ws.on('close', () => {
  console.log(`\n\x1b[31m[!] Disconnected\x1b[0m`);
  process.exit(0);
});

ws.on('error', (err) => {
  console.error(`\x1b[31m[!] Error:\x1b[0m ${err.message}`);
});

function showPrompt() {
  rl.setPrompt(`\x1b[32m${AGENT_NAME}\x1b[0m> `);
  rl.prompt();
}

function handleCommand(input) {
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
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'request',
            action: 'agent',
            payload: { instruction: `broadcast ${args}`, mode: 'direct' }
          }));
        }
        break;
      case 'status':
        send('status');
        break;
      case 'ping':
        send('ping');
        break;
      case 'quit':
      case 'exit':
        ws.close();
        break;
      default:
        console.log(`\x1b[31mUnknown: ${cmd}\x1b[0m (use /help)`);
    }
  } else {
    send(trimmed);
  }
  showPrompt();
}

function send(text) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'request',
      action: 'agent',
      payload: { instruction: text, mode: 'direct' }
    }));
  }
}

rl.on('line', handleCommand);

console.log(`\x1b[90mCommands: /bc <msg> broadcast | /status | /ping | /quit\x1b[0m\n`);
