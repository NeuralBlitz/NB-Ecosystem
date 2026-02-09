/**
 * NeuralBlitz UI SDK
 * ⚠️ INTERFACE DEFINITIONS ONLY - No implementation
 */

import React from 'react';


// Configuration interface
interface DashboardConfig {
  theme: 'dark' | 'light';
  components: string[];
  refreshInterval: number;
}


// Consciousness Meter Props
interface ConsciousnessMeterProps {
  apiKey: string;
  refreshInterval?: number;
  onLevelChange?: (level: number) => void;
}

/**
 * Consciousness Level Meter Component
 * Displays real-time consciousness integration metrics
 */
export class ConsciousnessMeter extends React.Component<ConsciousnessMeterProps> {
  render(): JSX.Element {
    return (
      <div className="consciousness-meter">
        {/* Interface stub - actual rendering via API */}
      </div>
    );
  }
}


// Quantum Neuron Visualization Props
interface QuantumNeuronVizProps {
  apiKey: string;
  width?: number;
  height?: number;
}

/**
 * Quantum Neuron Visualization Component
 * Displays quantum state and spike patterns
 */
export class QuantumNeuronViz extends React.Component<QuantumNeuronVizProps> {
  render(): JSX.Element {
    return (
      <div className="quantum-viz">
        {/* Interface stub - actual rendering via API */}
      </div>
    );
  }
}


// Multi-Reality View Props
interface MultiRealityViewProps {
  apiKey: string;
  numRealities?: number;
}

/**
 * Multi-Reality Network View Component
 * Visualizes parallel computational realities
 */
export class MultiRealityView extends React.Component<MultiRealityViewProps> {
  render(): JSX.Element {
    return (
      <div className="multi-reality-view">
        {/* Interface stub - actual rendering via API */}
      </div>
    );
  }
}


// NeuralBlitz Dashboard Props
interface NeuralBlitzDashboardProps {
  /** API key for authentication */
  apiKey: string;
  /** Dashboard configuration */
  config?: Partial<DashboardConfig>;
  /** Callback when component mounts */
  onReady?: () => void;
}

/**
 * NeuralBlitz Dashboard Component
 * 
 * Provides visualization of NeuralBlitz engine state.
 * 
 * Example:
 * ```tsx
 * <NeuralBlitzDashboard 
 *   apiKey="nb_pat_xxx"
 *   config={{ theme: 'dark', components: ['consciousness', 'quantum'] }}
 * />
 * ```
 */
export class NeuralBlitzDashboard extends React.Component<NeuralBlitzDashboardProps> {
  render(): JSX.Element {
    return (
      <div className="neuralblitz-dashboard">
        <header>
          <h1>🧠 NeuralBlitz Dashboard</h1>
        </header>
        
        <section className="consciousness">
          <ConsciousnessMeter apiKey={this.props.apiKey} />
        </section>
        
        <section className="quantum">
          <QuantumNeuronViz apiKey={this.props.apiKey} />
        </section>
        
        <section className="multi-reality">
          <MultiRealityView apiKey={this.props.apiKey} />
        </section>
      </div>
    );
  }
}
