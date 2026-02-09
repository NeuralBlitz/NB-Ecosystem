#!/usr/bin/env node
const WebSocket = require('ws');
const readline = require('readline');

const WS_URL = process.argv[2] || 'ws://localhost:3000';
const AGENT_NAME = process.argv[3] || `shell_${Math.floor(Math.random() * 1000)}`;

console.log(`\x1b[1;36m╔═══════════════════════════════════════════╗\x1b[0m`);
console.log(`\x1b[1;36m║\x1b[0m \x1b[1;33mReal-Time Agent Shell\x1b[0m                   \x1b[1;36m║\x1b[0m`);
console.log(`\x1b[1;36m║\x1b[0m Agent: \x1b[32m${AGENT_NAME}\x1b[0m                            \x1b[1;36m║\x1b[0m`);
console.log(`\x1b[1;36m║\x1b[0m Server: \x1b[34m${WS_URL}\x1b[0m                       \x1b[1;36m║\x1b[0m`);
console.log(`\x1b[1;36m╚═══════════════════════════════════════════╝\x1b[0m\n`);

const ws = new WebSocket(WS_URL);
let clientId = null;
let connected = false;

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

ws.on('open', () => {
  console.log(`\x1b[32m[✓]\x1b[0m Connected to server\n`);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  
  switch (msg.type) {
    case 'connected':
      clientId = msg.clientId;
      console.log(`\x1b[90m[*] Connected with ID: ${clientId}\x1b[0m`);
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
        const time = new Date(msg.data.timestamp || Date.now()).toLocaleTimeString();
        console.log(`\x1b[90m[${time}]\x1b[0m \x1b[35m${from}:\x1b[0m ${content}`);
      } else if (msg.data?.content) {
        console.log(`\x1b[35m${msg.data.from || 'system'}:\x1b[0m ${msg.data.content}`);
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
  console.log(`\n\x1b[31m[!] Disconnected from server\x1b[0m`);
  process.exit(0);
});

ws.on('error', (err) => {
  console.error(`\x1b[31m[!] Error:\x1b[0m ${err.message}`);
});

function showPrompt() {
  rl.setPrompt(`\x1b[32m${AGENT_NAME}\x1b[0m> `);
  rl.prompt();
}

function sendMessage(text) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'request',
      action: 'agent',
      payload: {
        instruction: text,
        mode: 'direct'
      }
    }));
  }
}

function broadcast(text) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'request',
      action: 'agent',
      payload: {
        instruction: `broadcast ${text}`,
        mode: 'direct'
      }
    }));
  }
}

rl.on('line', (input) => {
  const trimmed = input.trim();
  
  if (!trimmed) {
    showPrompt();
    return;
  }
  
  if (trimmed.startsWith('/')) {
    handleCommand(trimmed);
  } else {
    sendMessage(trimmed);
  }
  
  showPrompt();
});

function handleCommand(cmd) {
  const parts = cmd.slice(1).split(' ');
  const cmdName = parts[0].toLowerCase();
  const args = parts.slice(1).join(' ');
  
  switch (cmdName) {
    case 'bc':
    case 'broadcast':
      broadcast(args);
      break;
    case 'status':
      sendMessage('status');
      break;
    case 'files':
      sendMessage('files');
      break;
    case 'commits':
      sendMessage(`commits ${args || 5}`);
      break;
    case 'ping':
      sendMessage('ping');
      break;
    case 'help':
      console.log(`
\x1b[1;33mCommands:\x1b[0m
  <text>         Send to server (NLP processing)
  /bc <text>     Broadcast to all agents
  /status        Server status
  /files         Watched files
  /commits [n]   Git commits
  /ping          Ping server
  /help          Show this help
  /quit          Disconnect
`);
      break;
    case 'quit':
    case 'exit':
      console.log('\x1b[31mDisconnecting...\x1b[0m');
      ws.close();
      break;
    default:
      console.log(`\x1b[31mUnknown command: ${cmdName}\x1b[0m (use /help)`);
  }
}

console.log(`\x1b[90mType /help for commands, or just type to send to AI agent.\x1b[0m\n`);
