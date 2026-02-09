const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3000');

ws.on('open', () => {
  console.log('[Client] Connected');
  
  setTimeout(() => {
    console.log('[Client] Sending ping request to server...');
    ws.send(JSON.stringify({
      type: 'request',
      requestId: 'req-1',
      action: 'pingClient',
      payload: { message: 'ping' }
    }));
  }, 1000);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  console.log('[Client] Received:', msg.type);
  
  if (msg.type === 'response' && msg.requestId === 'req-1') {
    console.log('[Client] Server responded:', msg.result);
    ws.close();
    process.exit(0);
  }
});

ws.on('close', () => {
  console.log('[Client] Disconnected');
});

ws.on('error', (err) => {
  console.error('[Client] Error:', err.message);
  process.exit(1);
});

setTimeout(() => {
  console.log('[Client] Timeout - closing');
  ws.close();
  process.exit(0);
}, 5000);
