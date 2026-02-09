const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3000');

let connected = false;
const commandQueue = [];

console.log('Connecting to server...\n');

ws.on('open', () => {
  console.log('✓ Connected!\n');
  connected = true;
  
  while (commandQueue.length > 0) {
    const cmd = commandQueue.shift();
    ws.send(cmd);
  }
  
  console.log('Available commands:');
  console.log('  subscribe <channel> - Subscribe to a channel');
  console.log('  broadcast <msg>    - Send broadcast to all clients');
  console.log('  request <action>   - Make a request');
  console.log('  clients            - List all connected clients');
  console.log('  help               - Show commands\n');
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  if (msg.type === 'response' || msg.type === 'subscribed' || msg.type === 'broadcast') {
    console.log('\n📨 Received:', JSON.stringify(msg, null, 2));
  }
});

ws.on('close', () => {
  console.log('\n❌ Disconnected');
});

ws.on('error', (err) => {
  console.error('Error:', err.message);
});

function send(data) {
  if (connected && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  } else {
    commandQueue.push(JSON.stringify(data));
  }
}

const readline = require('readline');
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

rl.on('line', (input) => {
  const parts = input.trim().split(' ');
  const cmd = parts[0].toLowerCase();
  
  switch (cmd) {
    case 'subscribe':
      const channel = parts[1] || 'all';
      send({ type: 'subscribe', channels: [channel] });
      console.log(`→ Subscribed to: ${channel}`);
      break;
      
    case 'request':
      const action = parts[1];
      const payloadStr = parts.slice(2).join(' ');
      let payload = {};
      try {
        payload = payloadStr ? JSON.parse(payloadStr) : {};
      } catch (e) {
        payload = { message: payloadStr };
      }
      send({ type: 'request', requestId: `req-${Date.now()}`, action, payload });
      console.log(`→ Request: ${action}`);
      break;
      
    case 'clients':
      send({ type: 'request', requestId: `req-${Date.now()}`, action: 'getClients', payload: {} });
      console.log('→ Requesting clients...');
      break;
      
    case 'broadcast':
      const broadcastMsg = parts.slice(1).join(' ');
      send({ type: 'broadcast', event: 'chat', data: { message: broadcastMsg, from: 'shell-client' } });
      console.log(`→ Broadcast sent: "${broadcastMsg}"`);
      break;
      
    case 'help':
      console.log('\nCommands:');
      console.log('  subscribe <channel>  - Subscribe to channel');
      console.log('  request <action> <json> - Make request');
      console.log('  broadcast <message>   - Send broadcast');
      console.log('  clients              - List clients');
      console.log('  quit                 - Exit\n');
      break;
      
    case 'quit':
    case 'exit':
      ws.close();
      rl.close();
      process.exit(0);
      break;
      
    default:
      if (input.trim()) {
        console.log(`→ Unknown command: ${cmd}. Type 'help' for options.`);
      }
  }
  
  rl.prompt();
});

console.log('Type "help" for commands, "quit" to exit.\n');
rl.setPrompt('> ');
rl.prompt();
