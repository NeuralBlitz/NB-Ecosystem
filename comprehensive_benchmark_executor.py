#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite
Task 3.1: Execute all benchmarks and collect performance metrics
"""

import sys
import time
import statistics
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the correct paths
sys.path.insert(0, "/home/runner/workspace/lrs-agents/neuralblitz-v50")
sys.path.insert(0, "/home/runner/workspace/NB-Ecosystem/lib/python3.11/site-packages")


def benchmark_quantum_neurons(
    num_neurons: int = 5, steps: int = 1000
) -> Dict[str, Any]:
    """Benchmark Quantum Spiking Neurons"""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: Quantum Spiking Neurons")
    print("=" * 70)

    try:
        from quantum_spiking_neuron import QuantumSpikingNeuron, NeuronConfiguration
        import numpy as np

        print(f"\n📊 Configuration:")
        print(f"  Number of neurons: {num_neurons}")
        print(f"  Simulation steps: {steps}")
        print(f"  Time step: 0.1 ms")

        # Create neurons
        config = NeuronConfiguration(
            resting_potential=-70.0,
            threshold_potential=-55.0,
            membrane_time_constant=20.0,
            quantum_tunneling=0.1,
            coherence_time=50.0,
            dt=0.1,
        )

        neurons = []
        init_times = []

        print("\n⚡ Initializing neurons...")
        for i in range(num_neurons):
            t0 = time.perf_counter()
            neuron = QuantumSpikingNeuron(f"neuron_{i}", config)
            t1 = time.perf_counter()
            init_times.append((t1 - t0) * 1000)
            neurons.append(neuron)

        # Benchmark step performance
        print("\n🔄 Running simulation...")
        step_times = []
        spike_counts = []

        # Generate input trace
        t = np.linspace(0, steps * 0.1, steps)
        inputs = 15.0 * np.ones_like(t)  # Constant input

        for neuron in neurons:
            neuron_step_times = []

            for i, inp in enumerate(inputs):
                t0 = time.perf_counter()
                spike, state = neuron.step(inp)
                t1 = time.perf_counter()

                step_time = (t1 - t0) * 1000  # Convert to ms
                neuron_step_times.append(step_time)

            step_times.extend(neuron_step_times)
            spike_counts.append(neuron.spike_count)

        # Calculate statistics
        mean_step_time = statistics.mean(step_times)
        median_step_time = statistics.median(step_times)
        std_step_time = statistics.stdev(step_times) if len(step_times) > 1 else 0
        min_step_time = min(step_times)
        max_step_time = max(step_times)

        # Calculate throughput
        total_steps = num_neurons * steps
        total_time = sum(step_times) / 1000  # Convert to seconds
        throughput = total_steps / total_time if total_time > 0 else 0

        # Spike rate statistics
        mean_spike_rate = statistics.mean([n.spike_rate for n in neurons])

        results = {
            "system": "Quantum Spiking Neurons",
            "num_neurons": num_neurons,
            "steps_per_neuron": steps,
            "init_time_ms": {
                "mean": round(statistics.mean(init_times), 3),
                "std": round(statistics.stdev(init_times), 3)
                if len(init_times) > 1
                else 0,
            },
            "step_time_ms": {
                "mean": round(mean_step_time, 3),
                "median": round(median_step_time, 3),
                "std": round(std_step_time, 3),
                "min": round(min_step_time, 3),
                "max": round(max_step_time, 3),
            },
            "throughput_steps_per_sec": round(throughput, 0),
            "spike_rate_hz": {
                "mean": round(mean_spike_rate, 2),
                "total_spikes": sum(spike_counts),
            },
            "quantum_coherence": round(neurons[0].quantum_state.coherence, 4),
            "status": "✅ PASSED",
        }

        print(f"\n✅ Results:")
        print(
            f"  Init time: {results['init_time_ms']['mean']:.3f} ± {results['init_time_ms']['std']:.3f} ms"
        )
        print(
            f"  Step time: {results['step_time_ms']['mean']:.3f} ± {results['step_time_ms']['std']:.3f} ms"
        )
        print(f"  Throughput: {results['throughput_steps_per_sec']:,} steps/sec")
        print(f"  Spike rate: {results['spike_rate_hz']['mean']:.2f} Hz")
        print(f"  Quantum coherence: {results['quantum_coherence']:.4f}")

        return results

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return {
            "system": "Quantum Spiking Neurons",
            "status": "❌ FAILED",
            "error": str(e),
        }


def benchmark_multi_reality_networks(configs: List[Dict]) -> List[Dict[str, Any]]:
    """Benchmark Multi-Reality Networks with different configurations"""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Multi-Reality Neural Networks")
    print("=" * 70)

    try:
        from multi_reality_nn import MultiRealityNeuralNetwork

        results = []

        for config in configs:
            num_realities = config["num_realities"]
            nodes_per_reality = config["nodes_per_reality"]
            total_nodes = num_realities * nodes_per_reality

            print(
                f"\n📊 Testing {num_realities} realities × {nodes_per_reality} nodes = {total_nodes} total nodes"
            )

            # Measure initialization time
            t0 = time.perf_counter()
            network = MultiRealityNeuralNetwork(num_realities, nodes_per_reality)
            t1 = time.perf_counter()
            init_time = (t1 - t0) * 1000  # Convert to ms

            # Measure evolution cycles
            num_cycles = 50
            t0 = time.perf_counter()
            history = network.evolve_multi_reality_network(num_cycles=num_cycles)
            t1 = time.perf_counter()
            evolve_time = (t1 - t0) * 1000  # Convert to ms

            # Calculate metrics
            cycles_per_sec = num_cycles / (evolve_time / 1000) if evolve_time > 0 else 0

            # Get final state
            state = network.get_multi_reality_state()

            # Memory usage estimation
            memory_per_node = 8 * 3  # 3 arrays of float64 per node
            estimated_memory_mb = (total_nodes * memory_per_node) / (1024 * 1024)

            config_results = {
                "system": "Multi-Reality Neural Network",
                "num_realities": num_realities,
                "nodes_per_reality": nodes_per_reality,
                "total_nodes": total_nodes,
                "init_time_ms": round(init_time, 3),
                "evolve_cycles": num_cycles,
                "evolve_time_ms": round(evolve_time, 3),
                "cycles_per_sec": round(cycles_per_sec, 2),
                "global_consciousness": round(state["global_consciousness"], 4),
                "cross_reality_coherence": round(state["cross_reality_coherence"], 4),
                "information_flow_rate": round(state["information_flow_rate"], 2),
                "reality_synchronization": round(state["reality_synchronization"], 4),
                "estimated_memory_mb": round(estimated_memory_mb, 2),
                "status": "✅ PASSED",
            }

            print(f"  ✅ Init: {config_results['init_time_ms']:.3f} ms")
            print(
                f"  ✅ Evolve: {config_results['evolve_time_ms']:.3f} ms ({config_results['cycles_per_sec']:.2f} cycles/sec)"
            )
            print(f"  📊 Consciousness: {config_results['global_consciousness']:.4f}")
            print(f"  📊 Coherence: {config_results['cross_reality_coherence']:.4f}")

            results.append(config_results)

        return results

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return [
            {
                "system": "Multi-Reality Neural Networks",
                "status": "❌ FAILED",
                "error": str(e),
            }
        ]


def generate_performance_report(qsn_results: Dict, mrnn_results: List[Dict]) -> Dict:
    """Generate comprehensive performance report"""

    report = {
        "benchmark_timestamp": datetime.utcnow().isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
        "benchmarks": {
            "quantum_spiking_neurons": qsn_results,
            "multi_reality_networks": mrnn_results,
        },
        "performance_comparison": {
            "qsn_throughput": qsn_results.get("throughput_steps_per_sec", 0),
            "qsn_latency_ms": qsn_results.get("step_time_ms", {}).get("mean", 0),
            "mrnn_best_cycles_per_sec": max(
                [
                    r.get("cycles_per_sec", 0)
                    for r in mrnn_results
                    if "cycles_per_sec" in r
                ],
                default=0,
            ),
            "mrnn_largest_network": max(
                [r.get("total_nodes", 0) for r in mrnn_results if "total_nodes" in r],
                default=0,
            ),
        },
        "bottleneck_analysis": {},
        "optimization_recommendations": [],
    }

    # Bottleneck analysis
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)

    # QSN analysis
    if qsn_results.get("status") == "✅ PASSED":
        step_time = qsn_results["step_time_ms"]["mean"]
        if step_time < 0.1:
            report["bottleneck_analysis"]["quantum_neurons"] = (
                "EXCELLENT - Sub-100μs step time"
            )
        elif step_time < 1.0:
            report["bottleneck_analysis"]["quantum_neurons"] = (
                "GOOD - Sub-millisecond step time"
            )
        else:
            report["bottleneck_analysis"]["quantum_neurons"] = (
                "NEEDS OPTIMIZATION - Step time > 1ms"
            )

    # MRNN analysis
    for result in mrnn_results:
        if result.get("status") == "✅ PASSED":
            total_nodes = result["total_nodes"]
            cycles_per_sec = result["cycles_per_sec"]

            if cycles_per_sec > 2000:
                perf = "EXCELLENT"
            elif cycles_per_sec > 1000:
                perf = "GOOD"
            elif cycles_per_sec > 500:
                perf = "ACCEPTABLE"
            else:
                perf = "NEEDS OPTIMIZATION"

            report["bottleneck_analysis"][f"mrnn_{total_nodes}_nodes"] = (
                f"{perf} - {cycles_per_sec:.2f} cycles/sec"
            )

    # Recommendations
    print("\n" + "=" * 70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []

    # QSN recommendations
    if qsn_results.get("status") == "✅ PASSED":
        if (
            qsn_results["step_time_ms"]["std"]
            > qsn_results["step_time_ms"]["mean"] * 0.5
        ):
            recommendations.append(
                {
                    "system": "Quantum Spiking Neurons",
                    "issue": "High variance in step time",
                    "recommendation": "Consider optimizing memory allocation or using batch processing",
                    "priority": "MEDIUM",
                }
            )

    # MRNN recommendations
    for result in mrnn_results:
        if result.get("status") == "✅ PASSED":
            if result["cycles_per_sec"] < 1000:
                recommendations.append(
                    {
                        "system": f"Multi-Reality Network ({result['total_nodes']} nodes)",
                        "issue": f"Low evolution rate: {result['cycles_per_sec']:.2f} cycles/sec",
                        "recommendation": "Consider parallelizing cross-reality signal processing or reducing connection density",
                        "priority": "HIGH" if result["total_nodes"] < 200 else "MEDIUM",
                    }
                )

    report["optimization_recommendations"] = recommendations

    for rec in recommendations:
        print(f"\n🎯 {rec['priority']} Priority - {rec['system']}")
        print(f"   Issue: {rec['issue']}")
        print(f"   Recommendation: {rec['recommendation']}")

    return report


def main():
    """Run comprehensive benchmark suite"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK SUITE")
    print("Task 3.1: Execute all benchmarks and collect performance metrics")
    print("=" * 70)

    # Benchmark 1: Quantum Spiking Neurons
    qsn_results = benchmark_quantum_neurons(num_neurons=5, steps=1000)

    # Benchmark 2: Multi-Reality Networks
    mrnn_configs = [
        {"num_realities": 4, "nodes_per_reality": 20},  # 80 nodes
        {"num_realities": 4, "nodes_per_reality": 50},  # 200 nodes
        {"num_realities": 8, "nodes_per_reality": 50},  # 400 nodes
        {"num_realities": 8, "nodes_per_reality": 100},  # 800 nodes
    ]

    mrnn_results = benchmark_multi_reality_networks(mrnn_configs)

    # Generate comprehensive report
    report = generate_performance_report(qsn_results, mrnn_results)

    # Save report
    report_filename = f"comprehensive_benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ BENCHMARK COMPLETE - Report saved to {report_filename}")
    print("=" * 70)

    # Print final summary
    print("\n📊 FINAL SUMMARY TABLE")
    print("-" * 70)
    print(f"{'System':<30} {'Metric':<20} {'Value':<20}")
    print("-" * 70)

    if qsn_results.get("status") == "✅ PASSED":
        print(
            f"{'Quantum Neurons':<30} {'Step Time':<20} {qsn_results['step_time_ms']['mean']:.3f} ms"
        )
        print(
            f"{'':<30} {'Throughput':<20} {qsn_results['throughput_steps_per_sec']:,.0f} steps/sec"
        )
        print(
            f"{'':<30} {'Spike Rate':<20} {qsn_results['spike_rate_hz']['mean']:.2f} Hz"
        )

    for result in mrnn_results:
        if result.get("status") == "✅ PASSED":
            nodes = result["total_nodes"]
            print(
                f"{'Multi-Reality (' + str(nodes) + ' nodes)':<30} {'Init Time':<20} {result['init_time_ms']:.3f} ms"
            )
            print(f"{'':<30} {'Cycles/sec':<20} {result['cycles_per_sec']:.2f}")
            print(
                f"{'':<30} {'Consciousness':<20} {result['global_consciousness']:.4f}"
            )
            print("-" * 70)

    return report


if __name__ == "__main__":
    report = main()
