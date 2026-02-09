const OpenAI = require('openai');

class AICommandHandler {
  constructor(config = {}) {
    this.apiKey = config.apiKey || process.env.OPENAI_API_KEY;
    this.openai = this.apiKey ? new OpenAI({ apiKey: this.apiKey }) : null;
    this.initialized = !!this.apiKey;
    this.contextWindow = [];
    this.maxContextSize = 20;
    this.registeredTools = new Map();
    this.systemPrompt = `You are an AI coding assistant integrated into a codebase monitoring server. You have access to tools that can interact with the codebase.

Available capabilities:
- Execute bash commands
- Read files
- Write files
- Search code
- Monitor file changes
- Track git commits
- Query server status

When given a task:
1. Understand the user's intent
2. Break it down into steps
3. Execute using available tools
4. Report results clearly

Keep responses concise and actionable.`;
  }

  addTool(name, description, parameters, handler) {
    this.registeredTools.set(name, {
      name,
      description,
      parameters,
      handler
    });
    return this;
  }

  async processCommand(command, context = {}) {
    const conversationHistory = [
      { role: 'system', content: this.systemPrompt },
      ...this.contextWindow.slice(-this.maxContextSize),
      { role: 'user', content: command }
    ];

    if (!this.openai) {
      return this.fallbackProcessing(command, context);
    }

    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4',
        messages: conversationHistory,
        tools: this.buildToolsDescription(),
        tool_choice: 'auto',
        temperature: 0.3
      });

      const assistantMessage = response.choices[0].message;
      
      if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
        return this.executeToolCalls(assistantMessage.tool_calls, context);
      }

      return {
        success: true,
        response: assistantMessage.content,
        role: 'assistant'
      };
    } catch (error) {
      console.error('[AICommandHandler] OpenAI API error:', error.message);
      return this.fallbackProcessing(command, context);
    }
  }

  buildToolsDescription() {
    const tools = [];
    
    this.registeredTools.forEach((tool, name) => {
      tools.push({
        type: 'function',
        function: {
          name: name,
          description: tool.description,
          parameters: tool.parameters
        }
      });
    });

    return tools;
  }

  async executeToolCalls(toolCalls, context) {
    const results = [];

    for (const call of toolCalls) {
      const toolName = call.function.name;
      const args = JSON.parse(call.function.arguments);
      
      const tool = this.registeredTools.get(toolName);
      if (tool) {
        try {
          const result = await tool.handler(args, context);
          results.push({
            tool: toolName,
            success: true,
            result
          });
        } catch (error) {
          results.push({
            tool: toolName,
            success: false,
            error: error.message
          });
        }
      } else {
        results.push({
          tool: toolName,
          success: false,
          error: `Unknown tool: ${toolName}`
        });
      }
    }

    return {
      success: true,
      response: `Executed ${toolCalls.length} tool(s)`,
      toolResults: results,
      role: 'assistant'
    };
  }

  fallbackProcessing(command, context) {
    const lowerCommand = command.toLowerCase();
    
    const patterns = [
      { pattern: /status|health|running/i, action: 'getStatus' },
      { pattern: /files?|watching|changes?\?/i, action: 'getFiles' },
      { pattern: /git|commits?|history/i, action: 'getGitCommits' },
      { pattern: /clients?|connections?|users?\?/i, action: 'getClients' },
      { pattern: /restart|reload|reboot/i, action: 'restart' },
      { pattern: /stop|shutdown|quit|exit/i, action: 'stop' },
      { pattern: /help|what can you do/i, action: 'help' }
    ];

    for (const { pattern, action } of patterns) {
      if (pattern.test(command)) {
        const tool = this.registeredTools.get(action);
        console.log(`[AICommandHandler] Matched action: ${action}, tool exists: ${!!tool}`);
        if (tool) {
          try {
            const result = tool.handler({}, context);
            console.log(`[AICommandHandler] Handler result:`, result);
            const responseText = typeof result === 'object' 
              ? JSON.stringify(result, null, 2)
              : result;
            return {
              success: true,
              response: responseText,
              role: 'assistant'
            };
          } catch (error) {
            return {
              success: false,
              error: error.message
            };
          }
        }
      }
    }

    return {
      success: false,
      response: "I couldn't understand that command. Try 'status', 'files', 'git', 'help', or configure an OpenAI API key for advanced NLP capabilities.",
      role: 'assistant'
    };
  }

  addToContext(role, content) {
    this.contextWindow.push({ role, content });
    if (this.contextWindow.length > this.maxContextSize) {
      this.contextWindow.shift();
    }
  }

  clearContext() {
    this.contextWindow = [];
  }

  getRegisteredTools() {
    return Array.from(this.registeredTools.keys());
  }
}

module.exports = AICommandHandler;
