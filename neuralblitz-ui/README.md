# NeuralBlitz UI SDK

## ⚠️ Interface Definitions Only

This repository contains **interface definitions only**. Actual rendering is performed via NeuralBlitz SaaS API.

## Overview

NeuralBlitz UI SDK provides React component interfaces for NeuralBlitz dashboard and visualization components.

## Installation

```bash
npm install @neuralblitz/ui
```

## Quick Start

```tsx
import { NeuralBlitzDashboard } from '@neuralblitz/ui';

<NeuralBlitzDashboard
  apiKey="your-api-key"
  config={{ theme: 'dark', components: ['consciousness', 'quantum'] }}
/>
```

## Available Components

- `NeuralBlitzDashboard` - Main dashboard
- `ConsciousnessMeter` - Consciousness level visualization
- `QuantumNeuronViz` - Quantum neuron visualization
- `MultiRealityView` - Multi-reality network view

## Documentation

See [API_SPEC.md](API_SPEC.md) for detailed documentation.

## Support

- Email: partners@neuralblitz.ai
- Website: https://neuralblitz.ai

## License

MIT License
