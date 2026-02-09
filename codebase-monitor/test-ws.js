const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3000');

ws.on('open', () => {
  console.log('Connected!');
  
  ws.send(JSON.stringify({
    type: 'request',
    requestId: '1',
    action: 'getHandlers',
    payload: {}
  }));

  setTimeout(() => {
    ws.send(JSON.stringify({
      type: 'request',
      requestId: '2',
      action: 'getClients',
      payload: {}
    }));
  }, 100);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  console.log('\nReceived:', JSON.stringify(msg, null, 2));
});

ws.on('error', (err) => {
  console.error('Error:', err.message);
});

setTimeout(() => {
  console.log('\nClosing connection...');
  ws.close();
  process.exit(0);
}, 3000);
