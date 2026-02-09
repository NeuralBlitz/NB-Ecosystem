/**
 * NeuralBlitz WebSocket Client Example
 * Generated: 2026-02-08
 */

/**
 * NeuralBlitz WebSocket Client
 * 
 * Example usage:
 * 
 * ```typescript
 * import { NeuralBlitzWSClient } from '@neuralblitz/ws';
 * 
 * const ws = new NeuralBlitzWSClient({
 *   apiKey: 'nb_pat_xxx',
 *   channel: 'consciousness'
 * });
 * 
 * ws.onMessage((data) => {
 *   console.log('Consciousness update:', data);
 * });
 * 
 * ws.connect();
 * ```
 */

export interface WSConfig {
  apiKey: string;
  channel?: string;
  baseUrl?: string;
}

export interface WSMessage {
  type: string;
  data?: Record<string, unknown>;
  timestamp?: string;
}

export type MessageHandler = (message: WSMessage) => void;

// ============================================================================
// WebSocket Client Class
// ============================================================================

export class NeuralBlitzWSClient {
  private apiKey: string;
  private baseUrl: string;
  private channel: string;
  private socket: WebSocket | null = null;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnected = false;
  
  constructor(config: WSConfig) {
    this.apiKey = config.apiKey;
    this.baseUrl = config.baseUrl || 'ws://localhost:8000';
    this.channel = config.channel || 'general';
    
    // Initialize handler sets
    this.handlers.set('connect', new Set());
    this.handlers.set('disconnect', new Set());
    this.handlers.set('error', new Set());
    this.handlers.set('message', new Set());
    this.handlers.set('consciousness', new Set());
    this.handlers.set('agents', new Set());
    this.handlers.set('metrics', new Set());
    this.handlers.set('quantum', new Set());
  }
  
  // ========================================================================
  // Connection Management
  // ========================================================================
  
  /**
   * Connect to WebSocket endpoint
   * 
   * ```typescript
   * const ws = new NeuralBlitzWSClient({ apiKey: 'nb_pat_xxx' });
   * ws.connect();
   * ```
   */
  connect(): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      console.warn('Already connected');
      return;
    }
    
    const url = `${this.baseUrl}/api/v1/ws/stream/${this.channel}?api_key=${this.apiKey}`;
    
    try {
      this.socket = new WebSocket(url);
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to connect:', error);
      this.handleError(error as Error);
    }
  }
  
  /**
   * Disconnect from WebSocket
   * 
   * ```typescript
   * ws.disconnect();
   * ```
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }
  
  /**
   * Reconnect to WebSocket
   * 
   * ```typescript
   * ws.reconnect();
   * ```
   */
  reconnect(): void {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connect();
  }
  
  // ========================================================================
  // Event Handlers
  // ========================================================================
  
  private setupEventHandlers(): void {
    if (!this.socket) return;
    
    this.socket.onopen = () => {
      console.log('Connected to NeuralBlitz WebSocket');
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this.emit('connect', { type: 'connect' });
    };
    
    this.socket.onclose = (event: CloseEvent) => {
      console.log('Disconnected from NeuralBlitz WebSocket');
      this.isConnected = false;
      this.emit('disconnect', { type: 'disconnect', data: { code: event.code } });
      
      // Attempt reconnection
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
        setTimeout(() => this.connect(), delay);
      }
    };
    
    this.socket.onerror = (event: Event) => {
      console.error('WebSocket error:', event);
      this.handleError(new Error('WebSocket error'));
    };
    
    this.socket.onmessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data) as WSMessage;
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };
  }
  
  // ========================================================================
  // Message Handling
  // ========================================================================
  
  private handleMessage(message: WSMessage): void {
    // Emit to specific handler
    if (message.type && this.handlers.has(message.type)) {
      this.emit(message.type, message);
    }
    
    // Also emit to generic message handler
    this.emit('message', message);
    
    // Handle specific message types
    switch (message.type) {
      case 'consciousness_update':
        console.log('Consciousness Level:', (message.data as { level?: number })?.level);
        break;
      case 'quantum_update':
        console.log('Quantum Spike Rate:', (message.data as { spikeRate?: number })?.spikeRate);
        break;
      case 'agents_state':
        console.log('Active Agents:', (message.data as { active?: unknown[] })?.active?.length);
        break;
      case 'metrics_update':
        console.log('Requests/min:', (message.data as { requestsPerMinute?: number })?.requestsPerMinute);
        break;
    }
  }
  
  private handleError(error: Error): void {
    this.emit('error', { type: 'error', data: { message: error.message } });
  }
  
  // ========================================================================
  // Public API
  // ========================================================================
  
  /**
   * Subscribe to message type
   * 
   * ```typescript
   * ws.on('consciousness', (msg) => {
   *   console.log('Level:', msg.data?.level);
   * });
   * ```
   */
  on(type: string, handler: MessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }
  
  /**
   * Unsubscribe from message type
   * 
   * ```typescript
   * ws.off('consciousness', handler);
   * ```
   */
  off(type: string, handler: MessageHandler): void {
    this.handlers.get(type)?.delete(handler);
  }
  
  /**
   * Send message to server
   * 
   * ```typescript
   * ws.send({ action: 'status' });
   * ```
   */
  send(message: Record<string, unknown>): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(message));
    } else {
      console.warn('Not connected');
    }
  }
  
  /**
   * Check connection status
   * 
   * ```typescript
   * if (ws.isConnected) {
   *   console.log('Ready to receive');
   * }
   * ```
   */
  get connected(): boolean {
    return this.isConnected;
  }
  
  // ========================================================================
  // Event Emitter
  // ========================================================================
  
  private emit(type: string, message: WSMessage): void {
    this.handlers.get(type)?.forEach(handler => {
      try {
        handler(message);
      } catch (error) {
        console.error(`Handler error for ${type}:`, error);
      }
    });
  }
}

// ============================================================================
// Specialized Clients
// ============================================================================

/**
 * Consciousness WebSocket Client
 * 
 * ```typescript
 * const consciousness = new NeuralBlitzConsciousnessWS({
 *   apiKey: 'nb_pat_xxx'
 * });
 * 
 * consciousness.onUpdate((data) => {
 *   console.log('Level:', data.level);
 * });
 * 
 * consciousness.connect();
 * ```
 */
export class NeuralBlitzConsciousnessWS {
  private client: NeuralBlitzWSClient;
  
  constructor(config: { apiKey: string; baseUrl?: string }) {
    this.client = new NeuralBlitzWSClient({
      ...config,
      channel: 'consciousness'
    });
    
    this.client.on('consciousness_update', (msg) => {
      this.onUpdate(msg as WSMessage & { data: { level: number; integration: number; dimensions: number } });
    });
  }
  
  onUpdate(handler: WSMessage & { data: { level: number; integration: number; dimensions: number } }): void {
    this.client.on('consciousness_update', handler);
  }
  
  connect(): void {
    this.client.connect();
  }
  
  disconnect(): void {
    this.client.disconnect();
  }
  
  get connected(): boolean {
    return this.client.connected;
  }
}

/**
 * Metrics WebSocket Client
 * 
 * ```typescript
 * const metrics = new NeuralBlitzMetricsWS({
 *   apiKey: 'nb_pat_xxx'
 * });
 * 
 * metrics.onUpdate((data) => {
 *   console.log('RPM:', data.requestsPerMinute);
 * });
 * 
 * metrics.connect();
 * ```
 */
export class NeuralBlitzMetricsWS {
  private client: NeuralBlitzWSClient;
  
  constructor(config: { apiKey: string; baseUrl?: string }) {
    this.client = new NeuralBlitzWSClient({
      ...config,
      channel: 'metrics'
    });
    
    this.client.on('metrics_update', (msg) => {
      this.onUpdate(msg as WSMessage & { data: { requestsPerMinute: number; avgLatencyMs: number } });
    });
  }
  
  onUpdate(handler: WSMessage & { data: { requestsPerMinute: number; avgLatencyMs: number } }): void {
    this.client.on('metrics_update', handler);
  }
  
  connect(): void {
    this.client.connect();
  }
  
  disconnect(): void {
    this.client.disconnect();
  }
  
  get connected(): boolean {
    return this.client.connected;
  }
}

/**
 * Quantum WebSocket Client
 * 
 * ```typescript
 * const quantum = new NeuralBlitzQuantumWS({
 *   apiKey: 'nb_pat_xxx'
 * });
 * 
 * quantum.onUpdate((data) => {
 *   console.log('Spike Rate:', data.spikeRate);
 * });
 * 
 * quantum.connect();
 * ```
 */
export class NeuralBlitzQuantumWS {
  private client: NeuralBlitzWSClient;
  
  constructor(config: { apiKey: string; baseUrl?: string }) {
    this.client = new NeuralBlitzWSClient({
      ...config,
      channel: 'quantum'
    });
    
    this.client.on('quantum_update', (msg) => {
      this.onUpdate(msg as WSMessage & { data: { spikeRate: number; coherence: number } });
    });
  }
  
  onUpdate(handler: WSMessage & { data: { spikeRate: number; coherence: number } }): void {
    this.client.on('quantum_update', handler);
  }
  
  connect(): void {
    this.client.connect();
  }
  
  disconnect(): void {
    this.client.disconnect();
  }
  
  get connected(): boolean {
    return this.client.connected;
  }
}

// ============================================================================
// Example Usage
// ============================================================================

async function wsExample() {
  console.log('='.repeat(60));
  console.log('NeuralBlitz WebSocket Examples');
  console.log('='.repeat(60));
  
  // Example 1: Consciousness Client
  console.log('\nExample 1: Consciousness Monitoring');
  const consciousness = new NeuralBlitzConsciousnessWS({
    apiKey: 'nb_pat_xxxxxxxxxxxxxxxxxxxx'
  });
  
  consciousness.onUpdate((data) => {
    console.log(`Level: ${data.level}/8, Integration: ${(data.integration * 100).toFixed(1)}%`);
  });
  
  // consciousness.connect();
  
  // Example 2: Metrics Client
  console.log('\nExample 2: Real-Time Metrics');
  const metrics = new NeuralBlitzMetricsWS({
    apiKey: 'nb_pat_xxxxxxxxxxxxxxxxxxxx'
  });
  
  metrics.onUpdate((data) => {
    console.log(`RPM: ${data.requestsPerMinute}, Latency: ${data.avgLatencyMs}ms`);
  });
  
  // metrics.connect();
  
  // Example 3: Quantum Client
  console.log('\nExample 3: Quantum Monitoring');
  const quantum = new NeuralBlitzQuantumWS({
    apiKey: 'nb_pat_xxxxxxxxxxxxxxxxxxxx'
  });
  
  quantum.onUpdate((data) => {
    console.log(`Spike Rate: ${data.spikeRate}Hz, Coherence: ${(data.coherence * 100).toFixed(1)}%`);
  });
  
  // quantum.connect();
  
  console.log('\n' + '='.repeat(60));
  console.log('WebSocket Examples Ready!');
  console.log('='.repeat(60));
}

wsExample().catch(console.error);
