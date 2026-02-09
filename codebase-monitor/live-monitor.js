const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3000');

console.log('[Agent] Connecting to monitor server...\n');

ws.on('open', () => {
  console.log('[Agent] Connected!');
  ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['all', 'file_changes', 'git_commits', 'commands']
  }));
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  const time = new Date(msg.timestamp).toLocaleTimeString();
  
  console.log(`\n[${time}] === ${msg.type.toUpperCase()} ===`);
  console.log(JSON.stringify(msg, null, 2));
});

ws.on('close', () => {
  console.log('\n[Agent] Disconnected');
});

ws.on('error', (err) => {
  console.error('[Agent] Error:', err.message);
});

process.on('SIGINT', () => {
  console.log('\n[Agent] Closing...');
  ws.close();
  process.exit(0);
});
