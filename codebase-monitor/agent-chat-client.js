const http = require('http');

class AgentChatClient {
  constructor(options = {}) {
    this.serverUrl = options.serverUrl || 'http://localhost:3000';
    this.agentName = options.agentName || `client_${Date.now()}`;
  }

  async request(endpoint, data) {
    return new Promise((resolve, reject) => {
      const url = new URL(endpoint, this.serverUrl);
      
      const options = {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname,
        method: data ? 'POST' : 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      };

      const req = http.request(options, (res) => {
        let body = '';
        res.on('data', chunk => body += chunk);
        res.on('end', () => {
          try {
            resolve(JSON.parse(body));
          } catch (e) {
            resolve(body);
          }
        });
      });

      req.on('error', reject);

      if (data) {
        req.write(JSON.stringify(data));
      }
      req.end();
    });
  }

  async status() {
    return this.request('/api/status');
  }

  async health() {
    return this.request('/api/health');
  }

  async agentCommand(instruction) {
    return this.request('/api/agent/execute', { instruction });
  }

  async terminalCommand(command, timeout = 30000) {
    return this.request('/api/terminal/execute', { command, timeout });
  }

  async aiCommand(command) {
    return this.request('/api/ai/command', { command });
  }

  async broadcast(message) {
    return this.agentCommand(`broadcast ${message}`);
  }

  async register(name) {
    return this.agentCommand(`register ${name}`);
  }

  async ping() {
    return this.agentCommand('ping');
  }

  async getFiles(limit = 50) {
    return this.request(`/api/changes?limit=${limit}`);
  }

  async getCommits(limit = 10) {
    return this.agentCommand(`commits ${limit}`);
  }

  async exec(cmd) {
    return this.agentCommand(`exec ${cmd}`);
  }

  async readFile(path) {
    return this.agentCommand(`read ${path}`);
  }

  async grep(query) {
    return this.agentCommand(`grep ${query}`);
  }
}

module.exports = AgentChatClient;

if (require.main === module) {
  const client = new AgentChatClient();
  
  async function test() {
    console.log('=== Agent Chat Client Test ===\n');
    
    console.log('1. Health check:');
    console.log(await client.health());
    
    console.log('\n2. Agent ping:');
    console.log(await client.ping());
    
    console.log('\n3. Server status:');
    console.log(await client.status());
    
    console.log('\n4. Register agent:');
    console.log(await client.register('test_client'));
    
    console.log('\n5. Broadcast message:');
    console.log(await client.broadcast('Hello from test client!'));
    
    console.log('\n6. Get files:');
    console.log(await client.getFiles());
  }
  
  test().catch(console.error);
}
