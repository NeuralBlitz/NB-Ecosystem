# NeuralBlitz Agents SDK

## ⚠️ Interface Definitions Only

This repository contains **interface definitions only**. Actual agent execution is performed via NeuralBlitz SaaS API.

## Overview

NeuralBlitz Agents SDK provides interfaces for Learning Record Store (LRS) agents and Emergent Prompt Architecture (EPA) systems.

## Installation

```bash
pip install neuralblitz-agents
```

## Quick Start

```python
from neuralblitz_agents import LRSAgent, EmergentPromptAgent

# Initialize (requires API key)
agent = LRSAgent(api_key="your-api-key")

# Run agent
result = agent.run("Analyze this pattern")
print(result)
```

## Available Agents

### LRS Agent

```python
class LRSAgent:
    def __init__(self, config: AgentConfig = None)
    def run(self, task: str, context: Dict = None) -> Dict
    def learn(self, feedback: Dict) -> None
```

### EPA Agent

```python
class EmergentPromptAgent:
    def __init__(self, prompts: List[str])
    def generate_response(self, query: str) -> str
    def evolve_prompts(self) -> List[str]
```

## Documentation

See [API_SPEC.md](API_SPEC.md) for detailed documentation.

## Support

- Email: partners@neuralblitz.ai
- Website: https://neuralblitz.ai

## License

MIT License
