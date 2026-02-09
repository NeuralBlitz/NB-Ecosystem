const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');

class BidirectionalProtocol {
  static MESSAGE_TYPES = {
    CONNECT: 'connect',
    CONNECTED: 'connected',
    DISCONNECT: 'disconnect',
    DISCONNECTED: 'disconnected',
    PING: 'ping',
    PONG: 'pong',
    SUBSCRIBE: 'subscribe',
    SUBSCRIBED: 'subscribed',
    UNSUBSCRIBE: 'unsubscribe',
    UNSUBSCRIBED: 'unsubscribed',
    BROADCAST: 'broadcast',
    FILE_CHANGE: 'file_change',
    GIT_COMMIT: 'git_commit',
    STATUS: 'status',
    REQUEST: 'request',
    RESPONSE: 'response',
    ERROR: 'error',
    HEARTBEAT: 'heartbeat',
    ACK: 'ack',
    EVENT: 'event'
  };

  static CHANNELS = {
    ALL: 'all',
    FILE_CHANGES: 'file_changes',
    GIT_COMMITS: 'git_commits',
    STATUS: 'status',
    COMMANDS: 'commands'
  };

  constructor() {
    this.wss = null;
    this.clients = new Map();
    this.pendingRequests = new Map();
    this.handlers = new Map();
    this.listeners = new Map();
    this.heartbeatInterval = null;
    this.heartbeatTimeout = 30000;
    this.requestTimeout = 10000;
  }

  initialize(server) {
    this.wss = new WebSocket.Server({ server });

    this.wss.on('connection', (ws, req) => {
      const clientId = this.generateId();
      const clientInfo = this.createClient(clientId, ws);

      this.clients.set(clientId, clientInfo);
      this.setupClientHandlers(ws, clientId);

      this.sendToClient(clientId, {
        type: BidirectionalProtocol.MESSAGE_TYPES.CONNECTED,
        clientId,
        timestamp: new Date().toISOString(),
        version: '1.0.0'
      });

      console.log(`[WS Protocol] Client connected: ${clientId}`);
      this.emit('connection', { clientId });
    });

    this.startHeartbeat();
    console.log('[WS Protocol] Bidirectional server initialized');
  }

  createClient(id, ws) {
    return {
      id,
      ws,
      connectedAt: new Date(),
      lastActivity: Date.now(),
      subscriptions: new Set([BidirectionalProtocol.CHANNELS.ALL]),
      metadata: {},
      status: 'connected'
    };
  }

  setupClientHandlers(ws, clientId) {
    ws.on('message', (rawMessage) => {
      this.handleMessage(clientId, rawMessage);
    });

    ws.on('close', () => {
      this.handleDisconnect(clientId);
    });

    ws.on('error', (error) => {
      console.error(`[WS Protocol] Client error ${clientId}:`, error.message);
      this.emit('error', { clientId, error });
    });

    ws.on('pong', () => {
      this.updateActivity(clientId);
    });
  }

  handleMessage(clientId, rawMessage) {
    try {
      const message = JSON.parse(rawMessage);
      this.updateActivity(clientId);

      console.log(`[WS Protocol] Message from ${clientId}:`, message.type);

      switch (message.type) {
        case BidirectionalProtocol.MESSAGE_TYPES.PING:
          this.sendToClient(clientId, { type: BidirectionalProtocol.MESSAGE_TYPES.PONG });
          break;
        case BidirectionalProtocol.MESSAGE_TYPES.SUBSCRIBE:
          this.handleSubscribe(clientId, message);
          break;
        case BidirectionalProtocol.MESSAGE_TYPES.UNSUBSCRIBE:
          this.handleUnsubscribe(clientId, message);
          break;
        case BidirectionalProtocol.MESSAGE_TYPES.REQUEST:
          this.handleRequest(clientId, message);
          break;
        case BidirectionalProtocol.MESSAGE_TYPES.ACK:
          this.handleAck(clientId, message);
          break;
        case BidirectionalProtocol.MESSAGE_TYPES.DISCONNECT:
          this.handleDisconnect(clientId);
          break;
        case BidirectionalProtocol.MESSAGE_TYPES.EVENT:
          this.handleEvent(clientId, message);
          break;
        default:
          console.log(`[WS Protocol] Unknown message type: ${message.type}`);
      }
    } catch (error) {
      console.error(`[WS Protocol] Parse error from ${clientId}:`, error.message);
      this.sendError(clientId, 'PARSE_ERROR', 'Invalid JSON message');
    }
  }

  handleSubscribe(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    const channels = message.channels || [message.channel];
    channels.forEach(channel => {
      client.subscriptions.add(channel);
    });

    this.sendToClient(clientId, {
      type: BidirectionalProtocol.MESSAGE_TYPES.SUBSCRIBED,
      channels,
      timestamp: new Date().toISOString()
    });

    this.emit('subscribe', { clientId, channels });
  }

  handleUnsubscribe(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client) return;

    const channels = message.channels || [message.channel];
    channels.forEach(channel => {
      client.subscriptions.delete(channel);
    });

    this.sendToClient(clientId, {
      type: BidirectionalProtocol.MESSAGE_TYPES.UNSUBSCRIBED,
      channels,
      timestamp: new Date().toISOString()
    });

    this.emit('unsubscribe', { clientId, channels });
  }

  handleRequest(clientId, message) {
    const { requestId, action, payload } = message;
    const handlers = this.handlers.get(action);

    if (!handlers || handlers.length === 0) {
      this.sendResponse(clientId, requestId, null, {
        success: false,
        error: `Unknown action: ${action}`
      });
      return;
    }

    const results = [];
    let hasError = false;
    let lastError = null;

    Promise.all(handlers.map(handler => 
      Promise.resolve()
        .then(() => handler({ clientId, payload }))
        .then(result => results.push({ success: true, result }))
        .catch(error => {
          results.push({ success: false, error: error.message });
          hasError = true;
          lastError = error;
        })
    )).then(() => {
      this.sendResponse(clientId, requestId, results, { 
        success: !hasError,
        hasErrors: hasError 
      });
    });
  }

  handleAck(clientId, message) {
    const { ackId } = message;
    const pending = this.pendingRequests.get(ackId);

    if (pending) {
      clearTimeout(pending.timeout);
      pending.resolve(message);
      this.pendingRequests.delete(ackId);
    }
  }

  handleEvent(clientId, message) {
    this.emit('event', { clientId, event: message.event, data: message.data });
  }

  handleDisconnect(clientId) {
    const client = this.clients.get(clientId);
    if (client) {
      this.clients.delete(clientId);
      console.log(`[WS Protocol] Client disconnected: ${clientId}`);
      this.emit('disconnection', { clientId, metadata: client.metadata });
    }

    this.pendingRequests.forEach((pending, ackId) => {
      if (pending.clientId === clientId) {
        clearTimeout(pending.timeout);
        pending.reject(new Error('Client disconnected'));
        this.pendingRequests.delete(ackId);
      }
    });
  }

  sendToClient(clientId, data) {
    const client = this.clients.get(clientId);
    if (client && client.ws.readyState === WebSocket.OPEN) {
      const message = JSON.stringify({
        ...data,
        timestamp: data.timestamp || new Date().toISOString()
      });
      console.log(`[WS Protocol] Sending to ${clientId}:`, data.type);
      client.ws.send(message);
    }
  }

  sendResponse(clientId, requestId, result, meta = {}) {
    this.sendToClient(clientId, {
      type: BidirectionalProtocol.MESSAGE_TYPES.RESPONSE,
      requestId,
      result,
      ...meta
    });
  }

  sendError(clientId, code, message) {
    this.sendToClient(clientId, {
      type: BidirectionalProtocol.MESSAGE_TYPES.ERROR,
      code,
      message,
      timestamp: new Date().toISOString()
    });
  }

  broadcast(data, channels = [BidirectionalProtocol.CHANNELS.ALL]) {
    const message = {
      type: BidirectionalProtocol.MESSAGE_TYPES.BROADCAST,
      ...data,
      timestamp: new Date().toISOString()
    };

    this.clients.forEach((client) => {
      const hasSubscription = channels.some(channel =>
        client.subscriptions.has(channel)
      );

      if (hasSubscription && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(JSON.stringify(message));
      }
    });
  }

  broadcastFileChange(change) {
    this.broadcast({
      type: BidirectionalProtocol.MESSAGE_TYPES.FILE_CHANGE,
      data: change
    }, [BidirectionalProtocol.CHANNELS.FILE_CHANGES, BidirectionalProtocol.CHANNELS.ALL]);
  }

  broadcastGitCommit(commit) {
    this.broadcast({
      type: BidirectionalProtocol.MESSAGE_TYPES.GIT_COMMIT,
      data: commit
    }, [BidirectionalProtocol.CHANNELS.GIT_COMMITS, BidirectionalProtocol.CHANNELS.ALL]);
  }

  broadcastStatus(status) {
    this.broadcast({
      type: BidirectionalProtocol.MESSAGE_TYPES.STATUS,
      data: status
    }, [BidirectionalProtocol.CHANNELS.STATUS, BidirectionalProtocol.CHANNELS.ALL]);
  }

  async request(clientId, action, payload, timeout = this.requestTimeout) {
    const requestId = uuidv4();
    const ackId = uuidv4();

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingRequests.delete(ackId);
        reject(new Error(`Request timeout: ${action}`));
      }, timeout);

      this.pendingRequests.set(ackId, {
        clientId,
        requestId,
        timeout: timeoutId,
        resolve,
        reject
      });

      this.sendToClient(clientId, {
        type: BidirectionalProtocol.MESSAGE_TYPES.REQUEST,
        requestId,
        ackId,
        action,
        payload
      });
    });
  }

  registerHandler(action, handler) {
    if (!this.handlers.has(action)) {
      this.handlers.set(action, []);
    }
    this.handlers.get(action).push(handler);
    return this;
  }

  unregisterHandler(action, handler) {
    const handlers = this.handlers.get(action);
    if (handlers) {
      if (handler) {
        const index = handlers.indexOf(handler);
        if (index > -1) {
          handlers.splice(index, 1);
        }
      } else {
        this.handlers.delete(action);
      }
    }
    return this;
  }

  emit(event, data) {
    if (!this.listeners) return;
    const callback = this.listeners.get(event);
    if (callback) {
      callback(data);
    }
  }

  on(event, callback) {
    if (!this.listeners) {
      this.listeners = new Map();
    }
    this.listeners.set(event, callback);
    return this;
  }

  off(event) {
    if (this.listeners) {
      this.listeners.delete(event);
    }
    return this;
  }

  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      this.clients.forEach((client, clientId) => {
        if (Date.now() - client.lastActivity > this.heartbeatTimeout) {
          client.ws.terminate();
          this.handleDisconnect(clientId);
        } else {
          client.ws.ping();
        }
      });
    }, this.heartbeatTimeout / 2);
  }

  updateActivity(clientId) {
    const client = this.clients.get(clientId);
    if (client) {
      client.lastActivity = Date.now();
    }
  }

  generateId() {
    return uuidv4();
  }

  getConnectedClients() {
    return Array.from(this.clients.entries()).map(([id, client]) => ({
      id,
      subscriptions: Array.from(client.subscriptions),
      connectedAt: client.connectedAt,
      lastActivity: client.lastActivity,
      metadata: client.metadata
    }));
  }

  getStatistics() {
    let totalHandlers = 0;
    this.handlers.forEach(handlers => totalHandlers += handlers.length);
    return {
      totalClients: this.clients.size,
      pendingRequests: this.pendingRequests.size,
      registeredHandlers: totalHandlers,
      clients: this.getConnectedClients()
    };
  }

  connectToOmnibus(omnibusUrl = 'http://localhost:3001') {
    const http = require('http');
    return new Promise((resolve, reject) => {
      const req = http.get(`${omnibusUrl}/api/projects`, (res) => {
        let data = '';
        res.on('data', chunk => data += chunk);
        res.on('end', () => {
          console.log('[WS Protocol] Connected to OmnibusRouter');
          this.omnibusConnected = true;
          this.omnibusUrl = omnibusUrl;
          this.emit('omnibus_connected', { url: omnibusUrl });
          resolve(true);
        });
      });
      req.on('error', (err) => {
        console.warn('[WS Protocol] Cannot connect to Omnibus:', err.message);
        this.omnibusConnected = false;
        resolve(false);
      });
    });
  }

  sendToOmnibus(message) {
    if (!this.omnibusConnected) return false;
    const http = require('http');
    const postData = JSON.stringify({
      from: 'codebase-monitor',
      message,
      channels: ['all']
    });
    const req = http.request(`${this.omnibusUrl}/api/broadcast`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    }, (res) => {
      res.on('data', () => {});
    });
    req.write(postData);
    req.end();
    return true;
  }

  broadcastViaOmnibus(data) {
    return this.sendToOmnibus({
      type: 'omnibus_broadcast',
      data,
      source: 'codebase-monitor',
      timestamp: new Date().toISOString()
    });
  }

  shutdown() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    if (this.omnibusWs) {
      this.omnibusWs.close();
    }

    this.clients.forEach((client) => {
      client.ws.close();
    });
    this.clients.clear();
    this.pendingRequests.clear();
    this.handlers.clear();

    if (this.wss) {
      this.wss.close();
    }

    console.log('[WS Protocol] Bidirectional server shutdown complete');
  }
}

module.exports = BidirectionalProtocol;
