const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3000');

console.log('Connecting...\n');

ws.on('open', () => {
  console.log('✓ Connected!\n');
  
  console.log('1. Getting clients...');
  ws.send(JSON.stringify({ type: 'request', requestId: '1', action: 'getClients', payload: {} }));
  
  setTimeout(() => {
    console.log('\n2. Subscribing to file_changes channel...');
    ws.send(JSON.stringify({ type: 'subscribe', channels: ['file_changes'] }));
  }, 300);
  
  setTimeout(() => {
    console.log('\n3. Getting server status...');
    ws.send(JSON.stringify({ type: 'request', requestId: '2', action: 'getStatus', payload: {} }));
  }, 600);
  
  setTimeout(() => {
    console.log('\n4. Broadcasting message to all clients...');
    ws.send(JSON.stringify({ type: 'broadcast', event: 'chat', data: { message: 'Hello from shell!', from: 'opencode' } }));
  }, 900);
  
  setTimeout(() => {
    console.log('\n5. Getting handlers...');
    ws.send(JSON.stringify({ type: 'request', requestId: '3', action: 'getHandlers', payload: {} }));
  }, 1200);
  
  setTimeout(() => {
    console.log('\n6. Getting clients again (should show this client)...');
    ws.send(JSON.stringify({ type: 'request', requestId: '4', action: 'getClients', payload: {} }));
  }, 1500);
  
  setTimeout(() => {
    console.log('\n✓ All commands sent! Closing...\n');
    ws.close();
    process.exit(0);
  }, 2000);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  console.log('\n📨 Response:', JSON.stringify(msg, null, 2));
});

ws.on('close', () => {
  console.log('\n❌ Disconnected');
});

ws.on('error', (err) => {
  console.error('Error:', err.message);
});
