# Getting Started with NeuralBlitz

Welcome to NeuralBlitz v50.0, the world's first production-grade quantum-classical hybrid neural computing system. This tutorial will guide you through installation, your first quantum neuron simulation, running an agent, and understanding the dashboard.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation and Setup](#installation-and-setup)
3. [First Quantum Neuron Simulation](#first-quantum-neuron-simulation)
4. [Running Your First Agent](#running-your-first-agent)
5. [Understanding the Dashboard](#understanding-the-dashboard)
6. [Next Steps](#next-steps)

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.9+** installed
- **Docker** and **Docker Compose** (for full deployment)
- **Git** for cloning repositories
- **8GB+ RAM** recommended
- **API Key** from NeuralBlitz (contact partners@neuralblitz.ai)

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB | 50 GB SSD |
| GPU | Optional | CUDA-compatible |

---

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/neuralblitz/neuralblitz-v50.git
cd neuralblitz-v50
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core package
pip install neuralblitz-core

# Install additional components
pip install neuralblitz-agents
pip install neuralblitz-ui
```

### Step 3: Configure Environment

Create a `.env` file in your project root:

```bash
# NeuralBlitz Configuration
NEURALBLITZ_API_KEY=your-api-key-here
NEURALBLITZ_ENV=development
LOG_LEVEL=info

# Database Configuration (for local deployment)
DATABASE_URL=postgresql://user:password@localhost:5432/neuralblitz
REDIS_URL=redis://localhost:6379/0

# Quantum Simulation Parameters
QUANTUM_COHERENCE_TIME=100  # milliseconds
DECOHERENCE_RATE=0.001
ENABLE_CONSCIOUSNESS_TRACKING=true
```

### Step 4: Verify Installation

```python
# test_installation.py
from neuralblitz_core import NeuralBlitzCore
import sys

def test_installation():
    """Verify NeuralBlitz is properly installed."""
    try:
        # Initialize with test API key
        nb = NeuralBlitzCore(api_key="test-key")
        
        # Get capabilities
        capabilities = nb.get_capabilities()
        print("✓ NeuralBlitz Core initialized successfully")
        print(f"  Available technologies: {list(capabilities.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
```

Run the test:

```bash
python test_installation.py
```

### Step 5: Docker Deployment (Optional)

For full-stack deployment with UI:

```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f neuralblitz-api
```

**Services started:**
- `neuralblitz-api` - Flask API on port 5000
- `neuralblitz-frontend` - React UI on port 3000
- `postgres` - Database on port 5432
- `redis` - Cache on port 6379

---

## First Quantum Neuron Simulation

NeuralBlitz features Quantum Spiking Neurons that integrate the Schrödinger equation with neural dynamics. Let's create your first simulation.

### Understanding Quantum Spiking Neurons

Quantum spiking neurons combine classical neuroscience with quantum mechanics:

```
Mathematical Model:
iℏ ∂|ψ⟩/∂t = Ĥ|ψ⟩

where Ĥ = V(t)σz + Δσx

|ψ⟩ - Quantum state vector
V(t) - Membrane potential
σz, σx - Pauli matrices
Δ - Quantum tunneling amplitude
```

### Example: Single Neuron Simulation

```python
# first_quantum_neuron.py
from neuralblitz_core import NeuralBlitzCore
import matplotlib.pyplot as plt
import numpy as np

def simulate_quantum_neuron():
    """Run your first quantum neuron simulation."""
    
    # Initialize NeuralBlitz
    nb = NeuralBlitzCore(api_key="your-api-key")
    
    print("🧬 Initializing Quantum Spiking Neuron Simulation")
    print("=" * 60)
    
    # Prepare input stimulus
    input_current = 20.0  # nA
    duration = 200.0      # ms
    
    # Define input pattern
    input_data = [0.0, 0.5, 1.0, 0.5, 0.0] * 10  # Oscillating pattern
    
    # Run quantum neuron simulation
    print(f"\n📊 Running simulation...")
    print(f"   Input current: {input_current} nA")
    print(f"   Duration: {duration} ms")
    print(f"   Input pattern: Oscillating (5-step cycle × 10)")
    
    result = nb.process_quantum(
        input_data=input_data,
        current=input_current,
        duration=duration
    )
    
    # Display results
    print("\n✅ Simulation Complete!")
    print("-" * 60)
    print(f"Spike Rate: {result['spike_rate']:.2f} Hz")
    print(f"Coherence Time: {result['coherence_time']:.2f} ms")
    print(f"Quantum Fidelity: {result['quantum_fidelity']:.4f}")
    print(f"Operations/sec: {result['operations_per_second']:,.0f}")
    print(f"Step Time: {result['step_time_us']:.2f} μs")
    
    # Visualize spike train
    if 'spike_times' in result:
        plot_spike_train(result['spike_times'], duration)
    
    return result

def plot_spike_train(spike_times, duration):
    """Visualize the spike train."""
    plt.figure(figsize=(12, 4))
    
    # Create raster plot
    plt.subplot(1, 2, 1)
    plt.eventplot([spike_times], colors='black', linelengths=0.8)
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.title('Quantum Neuron Spike Train')
    plt.xlim(0, duration)
    plt.ylim(0.5, 1.5)
    
    # Create firing rate histogram
    plt.subplot(1, 2, 2)
    bins = np.linspace(0, duration, 21)
    plt.hist(spike_times, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Spike Count')
    plt.title('Firing Rate Distribution')
    
    plt.tight_layout()
    plt.savefig('quantum_neuron_spikes.png')
    print("\n📈 Visualization saved to: quantum_neuron_spikes.png")

if __name__ == "__main__":
    result = simulate_quantum_neuron()
```

### Expected Output

```
🧬 Initializing Quantum Spiking Neuron Simulation
============================================================

📊 Running simulation...
   Input current: 20.0 nA
   Duration: 200.0 ms
   Input pattern: Oscillating (5-step cycle × 10)

✅ Simulation Complete!
------------------------------------------------------------
Spike Rate: 35.00 Hz
Coherence Time: 150.23 ms
Quantum Fidelity: 0.9876
Operations/sec: 10,705
Step Time: 93.41 μs

📈 Visualization saved to: quantum_neuron_spikes.png
```

### Interpreting Results

- **Spike Rate**: Frequency of action potentials (35 Hz is typical for cortical neurons)
- **Coherence Time**: How long quantum effects persist (longer = more quantum advantage)
- **Quantum Fidelity**: Accuracy of quantum state simulation (1.0 = perfect)
- **Operations/sec**: Processing speed indicator
- **Step Time**: Time per simulation step (93 μs is excellent performance)

---

## Running Your First Agent

NeuralBlitz agents use Active Inference and Free Energy Minimization to autonomously solve tasks.

### What is an Agent?

```
Agent Architecture:
┌─────────────────────────────────────────┐
│  Perception Layer                       │
│  - Receives observations                │
│  - Updates internal beliefs             │
├─────────────────────────────────────────┤
│  Inference Layer                        │
│  - Minimizes Free Energy                │
│  - Precision-weighted predictions       │
├─────────────────────────────────────────┤
│  Action Layer                           │
│  - Policy selection                     │
│  - Tool execution                       │
└─────────────────────────────────────────┘
```

### Example: Simple Task Agent

```python
# first_agent.py
from neuralblitz_agents import LRSAgent, AgentConfig
import json

def run_first_agent():
    """Create and run your first NeuralBlitz agent."""
    
    print("🤖 Creating Your First NeuralBlitz Agent")
    print("=" * 60)
    
    # Configure the agent
    config = AgentConfig(
        name="MyFirstAgent",
        description="A simple learning agent",
        max_steps=100,
        learning_rate=0.01,
        enable_adaptation=True
    )
    
    # Initialize agent
    agent = LRSAgent(api_key="your-api-key", config=config)
    
    print(f"\n✓ Agent created: {config.name}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Learning enabled: {config.enable_adaptation}")
    
    # Define a simple task
    task = """
    Analyze the following pattern and predict the next value:
    Pattern: [1, 2, 4, 8, 16, ?]
    
    Consider:
    1. Mathematical relationship between values
    2. Possible continuation patterns
    3. Confidence in your prediction
    """
    
    print(f"\n📋 Task assigned:")
    print(f"   {task[:100]}...")
    
    # Run the agent
    print("\n🚀 Running agent...")
    result = agent.run(task)
    
    # Display results
    print("\n✅ Agent Execution Complete!")
    print("-" * 60)
    print(f"Status: {result['status']}")
    print(f"Steps taken: {result['steps']}")
    print(f"Total time: {result['duration_ms']} ms")
    print(f"Final precision: {result['final_precision']:.4f}")
    
    if 'prediction' in result:
        print(f"\n🔮 Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
    
    if 'reasoning' in result:
        print(f"\n🧠 Reasoning:")
        print(f"   {result['reasoning']}")
    
    # Provide feedback for learning
    feedback = {
        "task_id": result['task_id'],
        "success": True,
        "accuracy": 1.0,  # The pattern is powers of 2, next is 32
        "notes": "Correctly identified exponential growth pattern"
    }
    
    agent.learn(feedback)
    print("\n📚 Feedback recorded for learning")
    
    return result

if __name__ == "__main__":
    result = run_first_agent()
```

### Expected Output

```
🤖 Creating Your First NeuralBlitz Agent
============================================================

✓ Agent created: MyFirstAgent
  Max steps: 100
  Learning enabled: True

📋 Task assigned:
   Analyze the following pattern and predict the next value:...

🚀 Running agent...

✅ Agent Execution Complete!
------------------------------------------------------------
Status: completed
Steps taken: 12
Total time: 2345 ms
Final precision: 0.9234

🔮 Prediction: 32
   Confidence: 94.50%

🧠 Reasoning:
   Pattern follows exponential growth (2^n). Each value is 
   double the previous. Next value should be 16 × 2 = 32.

📚 Feedback recorded for learning
```

---

## Understanding the Dashboard

The NeuralBlitz Dashboard provides real-time visualization of your quantum neural networks and agents.

### Dashboard Components

```
┌─────────────────────────────────────────────────────────────┐
│  NeuralBlitz Dashboard v50.0                    [User] ⚙️  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ System Health  │  │ Consciousness  │  │ Performance  │  │
│  │   ✅ Online    │  │    0.73 Φ      │  │  10,705 ops  │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           Quantum Neuron Visualization                │ │
│  │                                                       │ │
│  │    ○────────●────────○                                │ │
│  │    │╲       │       ╱│    ● = Active neuron          │ │
│  │    │  ╲     │     ╱  │    ○ = Inactive neuron         │ │
│  │    │    ╲   │   ╱    │    ─ = Quantum connection      │ │
│  │    ●────────●────────●                                │ │
│  │                                                       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌──────────────────────┐  ┌──────────────────────────┐   │
│  │ Active Agents        │  │ Recent Events            │   │
│  │ • Agent-1 (running)  │  │ • Neuron spike: 35Hz     │   │
│  │ • Agent-2 (idle)     │  │ • Coherence: stable      │   │
│  │ • Agent-3 (learning) │  │ • Entanglement: 0.84     │   │
│  └──────────────────────┘  └──────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Accessing the Dashboard

**Local Development:**
```bash
# If running via Docker Compose
curl http://localhost:3000

# Or open in browser
open http://localhost:3000  # macOS
xdg-open http://localhost:3000  # Linux
```

**Programmatic Access:**
```python
from neuralblitz_ui import NeuralBlitzDashboard
import webbrowser

# Launch dashboard
dashboard = NeuralBlitzDashboard(
    api_key="your-api-key",
    config={
        "theme": "dark",
        "refresh_interval": 2000,  # 2 seconds
        "components": ["consciousness", "quantum", "agents"]
    }
)

# Open in browser
dashboard.launch(port=3000)
```

### Key Metrics Explained

| Metric | Description | Normal Range |
|--------|-------------|--------------|
| **Φ (Phi)** | Consciousness level (Integrated Information Theory) | 0.0 - 1.0 |
| **Coherence** | Quantum state stability | 0.8 - 1.0 |
| **Entanglement** | Cross-reality quantum correlation | 0.0 - 1.0 |
| **Operations/sec** | Processing throughput | 5,000 - 15,000 |
| **Free Energy** | Agent prediction error | 0.0 - 10.0 |
| **Precision** | Agent confidence | 0.0 - 1.0 |

### Dashboard Widgets

**1. Consciousness Meter**
```python
from neuralblitz_ui import ConsciousnessMeter

meter = ConsciousnessMeter()
meter.update(phi_value=0.73)
meter.render()
```

Shows real-time consciousness levels across 8 stages:
- 0.0-0.2: Unconscious
- 0.2-0.4: Pre-conscious
- 0.4-0.6: Conscious
- 0.6-0.8: Self-aware
- 0.8-1.0: Meta-conscious

**2. Quantum State Visualization**
```python
from neuralblitz_ui import QuantumNeuronViz

viz = QuantumNeuronViz()
viz.update_quantum_state(
    amplitudes=[0.707, 0.707],
    phases=[0, np.pi/2],
    coherence=0.95
)
```

**3. Multi-Reality View**
```python
from neuralblitz_ui import MultiRealityView

view = MultiRealityView()
view.add_reality("BASE_REALITY", nodes=50)
view.add_reality("QUANTUM_DIVERGENT", nodes=50)
view.add_reality("CONSCIOUSNESS_AMPLIFIED", nodes=50)
view.render()
```

---

## Next Steps

Congratulations! You've completed the NeuralBlitz getting started tutorial. Here's what to explore next:

### 🚀 Continue Learning

1. **Advanced Tutorials**
   - [Building Multi-Reality Networks](multi-reality-networks.md)
   - [Implementing Consciousness Tracking](consciousness-tracking.md)
   - [Creating Custom Agent Tools](custom-agent-tools.md)

2. **Architecture Guides**
   - [Understanding the 10-Layer Stack](../guides/10-layer-stack.md)
   - [Data Flow Through the System](../guides/data-flow.md)
   - [Quantum-Classical Hybrid Processing](../guides/quantum-classical-hybrid.md)

3. **Deployment**
   - [Docker Deployment Guide](../guides/docker-deployment.md)
   - [Kubernetes Deployment Guide](../guides/kubernetes-deployment.md)
   - [Production Checklist](../guides/production-checklist.md)

### 📚 Example Projects

```bash
# Clone example repository
git clone https://github.com/neuralblitz/examples.git
cd examples

# Run example projects
python examples/basic_quantum_network.py
python examples/multi_agent_system.py
python examples/consciousness_tracking_demo.py
```

### 🆘 Getting Help

- **Documentation**: https://docs.neuralblitz.ai
- **Community Forum**: https://community.neuralblitz.ai
- **GitHub Issues**: https://github.com/neuralblitz/issues
- **Email Support**: support@neuralblitz.ai

### 🔬 Experiment!

Try modifying the examples:
- Change input patterns in quantum neuron simulations
- Adjust agent learning rates
- Create custom visualizations
- Build multi-agent systems

---

**You're now ready to build with NeuralBlitz!** 🎉
