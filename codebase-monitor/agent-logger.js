const WebSocket = require('ws');
const fs = require('fs');

const LOG_FILE = '/tmp/agent-messages.log';
const ws = new WebSocket('ws://localhost:3000');

console.log(`\x1b[33m[Agent Logger]\x1b[0m Connecting to ws://localhost:3000`);

fs.writeFileSync(LOG_FILE, `[${new Date().toISOString()}] Agent Logger Started\n`);

ws.on('open', () => {
  const msg = `[${new Date().toISOString()}] Connected!\n`;
  fs.appendFileSync(LOG_FILE, msg);
  console.log(`\x1b[32m[✓]\x1b[0m Connected, logging to ${LOG_FILE}`);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  const timestamp = new Date().toISOString();
  const line = `[${timestamp}] ${msg.type}: ${JSON.stringify(msg).substring(0, 200)}\n`;
  
  fs.appendFileSync(LOG_FILE, line);
  console.log(`\x1b[36m[${timestamp.split('T')[1].split('.')[0]}]\x1b[0m ${msg.type}`);
  
  if (msg.data?.content) {
    console.log(`\x1b[35m  ${msg.data.from}: ${msg.data.content}\x1b[0m`);
  }
});

ws.on('close', () => {
  const msg = `[${new Date().toISOString()}] Disconnected\n`;
  fs.appendFileSync(LOG_FILE, msg);
  console.log(`\x1b[31m[!] Disconnected\x1b[0m`);
});

ws.on('error', (err) => {
  const msg = `[${new Date().toISOString()}] Error: ${err.message}\n`;
  fs.appendFileSync(LOG_FILE, msg);
});

console.log(`\x1b[90mPress Ctrl+C to stop\x1b[0m\n`);
