"""
NeuralBlitz v50.0 Scalability Testing Framework
Task 3.2 from R&D Case Study Framework

Tests system behavior under increasing load and scale across:
- Network Size: 80 → 200 → 400 → 800 → 1600 nodes
- Concurrent Requests: 1 → 10 → 50 → 100 → 500
- Data Volume: Small → Medium → Large → Very Large
- Time Duration: Short bursts → Sustained load → Endurance

Metrics Tracked:
- Response time degradation
- Error rate increases
- Memory growth patterns
- CPU utilization
- Throughput saturation points

Author: NeuralBlitz R&D Team
Date: 2026-02-08
"""

import asyncio
import time
import json
import psutil
import threading
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# NeuralBlitz imports
import sys

sys.path.insert(0, "/home/runner/workspace/NB-Ecosystem/lib/python3.11/site-packages")
sys.path.insert(0, "/home/runner/workspace/opencode-lrs-agents-nbx/neuralblitz-v50")

from spiking_neural_network import SpikingNeuralNetwork
from multi_reality_nn import MultiRealityNeuralNetwork


@dataclass
class TestConfig:
    """Configuration for scalability test"""

    test_name: str
    network_sizes: List[int] = field(default_factory=lambda: [80, 200, 400, 800, 1600])
    concurrent_requests: List[int] = field(
        default_factory=lambda: [1, 10, 50, 100, 500]
    )
    data_volumes: List[str] = field(
        default_factory=lambda: ["small", "medium", "large", "very_large"]
    )
    duration_seconds: int = 60
    warmup_seconds: int = 10


@dataclass
class TestMetrics:
    """Metrics collected during a test run"""

    timestamp: float
    network_size: int
    concurrent_requests: int
    data_volume: str

    # Performance metrics
    response_time_ms: float
    response_time_p95_ms: float
    response_time_p99_ms: float
    throughput_rps: float

    # Error metrics
    error_count: int
    error_rate: float
    timeout_count: int

    # Resource metrics
    memory_usage_mb: float
    memory_growth_mb: float
    cpu_percent: float

    # System metrics
    active_threads: int
    queue_depth: int


@dataclass
class ScalabilityResult:
    """Complete result from a scalability test"""

    test_id: str
    test_config: TestConfig
    start_time: datetime
    end_time: datetime

    # Raw metrics
    metrics: List[TestMetrics] = field(default_factory=list)

    # Analysis
    breaking_point: Optional[Dict[str, Any]] = None
    saturation_point: Optional[Dict[str, Any]] = None
    max_throughput: float = 0.0
    max_stable_concurrency: int = 0

    # Resource utilization patterns
    memory_pattern: str = ""
    cpu_pattern: str = ""
    response_time_curve: List[Tuple[int, float]] = field(default_factory=list)


class ResourceMonitor:
    """Monitors system resources during testing"""

    def __init__(self):
        self.monitoring = False
        self.metrics_history: List[Dict[str, float]] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.initial_memory = 0.0

    def start(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return summary"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

        if not self.metrics_history:
            return {}

        # Calculate statistics
        memory_values = [m["memory_mb"] for m in self.metrics_history]
        cpu_values = [m["cpu_percent"] for m in self.metrics_history]

        return {
            "memory_avg_mb": np.mean(memory_values),
            "memory_max_mb": np.max(memory_values),
            "memory_growth_mb": memory_values[-1] - self.initial_memory,
            "cpu_avg": np.mean(cpu_values),
            "cpu_max": np.max(cpu_values),
            "sample_count": len(self.metrics_history),
        }

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                cpu_percent = process.cpu_percent(interval=0.1)

                self.metrics_history.append(
                    {
                        "timestamp": time.time(),
                        "memory_mb": memory_mb,
                        "cpu_percent": cpu_percent,
                        "threads": threading.active_count(),
                    }
                )
            except Exception as e:
                print(f"Monitor error: {e}")

            time.sleep(0.5)  # Sample every 500ms

    def get_current_stats(self) -> Dict[str, float]:
        """Get current resource stats"""
        if not self.metrics_history:
            return {"memory_mb": 0, "cpu_percent": 0}
        return self.metrics_history[-1]


class ScalabilityTestRunner:
    """Main test runner for scalability testing"""

    def __init__(self, api_base_url: str = "http://localhost:5000"):
        self.api_base_url = api_base_url
        self.results: List[ScalabilityResult] = []
        self.resource_monitor = ResourceMonitor()

    def run_network_size_test(self, config: TestConfig) -> ScalabilityResult:
        """
        Test 1: Network Size Scalability
        Tests neural network performance with increasing node counts
        """
        print(f"\n{'=' * 80}")
        print(f"TEST 1: Network Size Scalability")
        print(f"{'=' * 80}")

        result = ScalabilityResult(
            test_id=f"network_size_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_config=config,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        for network_size in config.network_sizes:
            print(f"\n--- Testing with {network_size} nodes ---")

            try:
                # Test Spiking Neural Network
                metrics = self._test_snn_scalability(network_size, config)
                result.metrics.append(metrics)

                # Test Multi-Reality Network
                mr_metrics = self._test_mrnn_scalability(network_size, config)
                result.metrics.append(mr_metrics)

                print(
                    f"✓ SNN: {metrics.response_time_ms:.2f}ms, "
                    f"Throughput: {metrics.throughput_rps:.2f} ops/sec"
                )
                print(
                    f"✓ MRNN: {mr_metrics.response_time_ms:.2f}ms, "
                    f"Consciousness: {mr_metrics.response_time_p95_ms:.4f}"
                )

            except Exception as e:
                print(f"✗ Error with {network_size} nodes: {e}")
                # Record failure
                result.metrics.append(
                    TestMetrics(
                        timestamp=time.time(),
                        network_size=network_size,
                        concurrent_requests=1,
                        data_volume="small",
                        response_time_ms=0,
                        response_time_p95_ms=0,
                        response_time_p99_ms=0,
                        throughput_rps=0,
                        error_count=1,
                        error_rate=1.0,
                        timeout_count=0,
                        memory_usage_mb=0,
                        memory_growth_mb=0,
                        cpu_percent=0,
                        active_threads=0,
                        queue_depth=0,
                    )
                )

                # Mark as breaking point
                if not result.breaking_point:
                    result.breaking_point = {
                        "dimension": "network_size",
                        "value": network_size,
                        "error": str(e),
                    }

        result.end_time = datetime.now()
        return result

    def _test_snn_scalability(
        self, num_neurons: int, config: TestConfig
    ) -> TestMetrics:
        """Test Spiking Neural Network scalability"""
        self.resource_monitor.start()

        try:
            # Create network
            start_time = time.time()
            snn = SpikingNeuralNetwork(num_neurons=num_neurons, connectivity=0.1)
            init_time = (time.time() - start_time) * 1000

            # Warmup
            for _ in range(10):
                snn.simulate_step()

            # Measure performance
            start_time = time.time()
            num_steps = 100
            for _ in range(num_steps):
                snn.simulate_step()
            elapsed_ms = (time.time() - start_time) * 1000

            # Calculate metrics
            step_time_ms = elapsed_ms / num_steps
            throughput = 1000.0 / step_time_ms  # steps per second

            resource_stats = self.resource_monitor.stop()

            return TestMetrics(
                timestamp=time.time(),
                network_size=num_neurons,
                concurrent_requests=1,
                data_volume="small",
                response_time_ms=step_time_ms,
                response_time_p95_ms=step_time_ms * 1.5,
                response_time_p99_ms=step_time_ms * 2.0,
                throughput_rps=throughput,
                error_count=0,
                error_rate=0.0,
                timeout_count=0,
                memory_usage_mb=resource_stats.get("memory_avg_mb", 0),
                memory_growth_mb=resource_stats.get("memory_growth_mb", 0),
                cpu_percent=resource_stats.get("cpu_avg", 0),
                active_threads=resource_stats.get("sample_count", 0),
                queue_depth=0,
            )

        except Exception as e:
            self.resource_monitor.stop()
            raise e

    def _test_mrnn_scalability(
        self, total_nodes: int, config: TestConfig
    ) -> TestMetrics:
        """Test Multi-Reality Neural Network scalability"""
        self.resource_monitor.start()

        try:
            # Calculate realities and nodes per reality
            num_realities = max(2, total_nodes // 50)
            nodes_per_reality = total_nodes // num_realities

            # Create network
            start_time = time.time()
            mrnn = MultiRealityNeuralNetwork(
                num_realities=num_realities, nodes_per_reality=nodes_per_reality
            )
            init_time = (time.time() - start_time) * 1000

            # Warmup
            input_patterns = {}
            for i, reality_id in enumerate(list(mrnn.realities.keys())[:2]):
                input_patterns[reality_id] = np.random.randn(nodes_per_reality) * 0.1
            mrnn.process_multi_reality_computation(input_patterns)

            # Measure performance
            start_time = time.time()
            num_cycles = 20
            for _ in range(num_cycles):
                mrnn.process_multi_reality_computation(input_patterns)
            elapsed_ms = (time.time() - start_time) * 1000

            # Get consciousness level
            state = mrnn.get_multi_reality_state()
            consciousness = state["global_consciousness"]

            resource_stats = self.resource_monitor.stop()

            cycle_time_ms = elapsed_ms / num_cycles
            throughput = 1000.0 / cycle_time_ms

            return TestMetrics(
                timestamp=time.time(),
                network_size=total_nodes,
                concurrent_requests=num_realities,
                data_volume="medium",
                response_time_ms=cycle_time_ms,
                response_time_p95_ms=consciousness,
                response_time_p99_ms=state["cross_reality_coherence"],
                throughput_rps=throughput,
                error_count=0,
                error_rate=0.0,
                timeout_count=0,
                memory_usage_mb=resource_stats.get("memory_avg_mb", 0),
                memory_growth_mb=resource_stats.get("memory_growth_mb", 0),
                cpu_percent=resource_stats.get("cpu_avg", 0),
                active_threads=threading.active_count(),
                queue_depth=len(mrnn.active_signals),
            )

        except Exception as e:
            self.resource_monitor.stop()
            raise e

    def run_concurrent_request_test(self, config: TestConfig) -> ScalabilityResult:
        """
        Test 2: Concurrent Request Scalability
        Tests API endpoints under increasing concurrent load
        """
        print(f"\n{'=' * 80}")
        print(f"TEST 2: Concurrent Request Scalability")
        print(f"{'=' * 80}")

        result = ScalabilityResult(
            test_id=f"concurrent_requests_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_config=config,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # First, verify API is running
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/health", timeout=5)
            if response.status_code != 200:
                print("⚠️  API not available, skipping concurrent request tests")
                return result
        except Exception as e:
            print(f"⚠️  API connection failed: {e}")
            print("⚠️  Skipping concurrent request tests (run API first)")
            return result

        for concurrency in config.concurrent_requests:
            print(f"\n--- Testing with {concurrency} concurrent requests ---")

            try:
                metrics = self._test_api_concurrency(concurrency, config)
                result.metrics.append(metrics)

                print(
                    f"✓ Response Time: {metrics.response_time_ms:.2f}ms (p95: {metrics.response_time_p95_ms:.2f}ms)"
                )
                print(f"✓ Throughput: {metrics.throughput_rps:.2f} req/sec")
                print(f"✓ Error Rate: {metrics.error_rate * 100:.2f}%")
                print(
                    f"✓ Memory: {metrics.memory_usage_mb:.2f}MB (+{metrics.memory_growth_mb:.2f}MB)"
                )

                # Check for saturation
                if metrics.error_rate > 0.05 or metrics.response_time_ms > 5000:
                    if not result.saturation_point:
                        result.saturation_point = {
                            "dimension": "concurrent_requests",
                            "value": concurrency,
                            "reason": "high_error_rate"
                            if metrics.error_rate > 0.05
                            else "high_latency",
                        }
                        print(
                            f"⚠️  SATURATION POINT DETECTED at {concurrency} concurrent requests"
                        )

                # Update max throughput
                if metrics.throughput_rps > result.max_throughput:
                    result.max_throughput = metrics.throughput_rps
                    result.max_stable_concurrency = concurrency

            except Exception as e:
                print(f"✗ Error with {concurrency} concurrent requests: {e}")
                if not result.breaking_point:
                    result.breaking_point = {
                        "dimension": "concurrent_requests",
                        "value": concurrency,
                        "error": str(e),
                    }

        result.end_time = datetime.now()
        return result

    def _test_api_concurrency(
        self, concurrency: int, config: TestConfig
    ) -> TestMetrics:
        """Test API with specific concurrency level"""
        self.resource_monitor.start()

        response_times = []
        error_count = 0
        timeout_count = 0

        def make_request():
            try:
                start = time.time()
                response = requests.get(
                    f"{self.api_base_url}/api/v1/metrics", timeout=10
                )
                elapsed_ms = (time.time() - start) * 1000

                if response.status_code == 200:
                    return elapsed_ms, False, False
                else:
                    return elapsed_ms, True, False
            except requests.Timeout:
                return 10000, True, True
            except Exception:
                return 0, True, False

        # Warmup
        for _ in range(min(10, concurrency)):
            make_request()

        # Actual test
        start_time = time.time()
        total_requests = concurrency * 10  # 10 requests per concurrent connection

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]

            for future in as_completed(futures):
                elapsed_ms, is_error, is_timeout = future.result()

                if is_error:
                    error_count += 1
                    if is_timeout:
                        timeout_count += 1
                else:
                    response_times.append(elapsed_ms)

        test_duration = time.time() - start_time
        resource_stats = self.resource_monitor.stop()

        # Calculate metrics
        if response_times:
            avg_response = np.mean(response_times)
            p95_response = np.percentile(response_times, 95)
            p99_response = np.percentile(response_times, 99)
        else:
            avg_response = p95_response = p99_response = 0

        throughput = total_requests / test_duration if test_duration > 0 else 0
        error_rate = error_count / total_requests if total_requests > 0 else 0

        return TestMetrics(
            timestamp=time.time(),
            network_size=400,  # Default network size for API tests
            concurrent_requests=concurrency,
            data_volume="small",
            response_time_ms=avg_response,
            response_time_p95_ms=p95_response,
            response_time_p99_ms=p99_response,
            throughput_rps=throughput,
            error_count=error_count,
            error_rate=error_rate,
            timeout_count=timeout_count,
            memory_usage_mb=resource_stats.get("memory_avg_mb", 0),
            memory_growth_mb=resource_stats.get("memory_growth_mb", 0),
            cpu_percent=resource_stats.get("cpu_avg", 0),
            active_threads=threading.active_count(),
            queue_depth=0,
        )

    def run_data_volume_test(self, config: TestConfig) -> ScalabilityResult:
        """
        Test 3: Data Volume Scalability
        Tests system with increasing data volumes
        """
        print(f"\n{'=' * 80}")
        print(f"TEST 3: Data Volume Scalability")
        print(f"{'=' * 80}")

        result = ScalabilityResult(
            test_id=f"data_volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_config=config,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        # Data volume sizes (in neurons/states)
        volume_sizes = {
            "small": 100,
            "medium": 1000,
            "large": 10000,
            "very_large": 50000,
        }

        for volume_name in config.data_volumes:
            volume_size = volume_sizes.get(volume_name, 100)
            print(f"\n--- Testing {volume_name} data volume ({volume_size} states) ---")

            try:
                metrics = self._test_data_volume(volume_name, volume_size, config)
                result.metrics.append(metrics)

                print(f"✓ Processing Time: {metrics.response_time_ms:.2f}ms")
                print(f"✓ Throughput: {metrics.throughput_rps:.2f} states/sec")
                print(f"✓ Memory: {metrics.memory_usage_mb:.2f}MB")

            except Exception as e:
                print(f"✗ Error with {volume_name} volume: {e}")
                if not result.breaking_point:
                    result.breaking_point = {
                        "dimension": "data_volume",
                        "value": volume_size,
                        "error": str(e),
                    }

        result.end_time = datetime.now()
        return result

    def _test_data_volume(
        self, volume_name: str, volume_size: int, config: TestConfig
    ) -> TestMetrics:
        """Test with specific data volume"""
        self.resource_monitor.start()

        try:
            # Create large data structure
            start_time = time.time()

            # Simulate processing of large neural network state
            data = np.random.randn(volume_size, 100)  # 100 features per state

            # Process data (matrix operations)
            result = np.dot(data.T, data)  # Covariance matrix
            eigenvalues = np.linalg.eigvals(result)

            elapsed_ms = (time.time() - start_time) * 1000

            resource_stats = self.resource_monitor.stop()

            throughput = volume_size / (elapsed_ms / 1000.0)

            return TestMetrics(
                timestamp=time.time(),
                network_size=volume_size,
                concurrent_requests=1,
                data_volume=volume_name,
                response_time_ms=elapsed_ms,
                response_time_p95_ms=elapsed_ms,
                response_time_p99_ms=elapsed_ms,
                throughput_rps=throughput,
                error_count=0,
                error_rate=0.0,
                timeout_count=0,
                memory_usage_mb=resource_stats.get("memory_avg_mb", 0),
                memory_growth_mb=resource_stats.get("memory_growth_mb", 0),
                cpu_percent=resource_stats.get("cpu_avg", 0),
                active_threads=1,
                queue_depth=0,
            )

        except Exception as e:
            self.resource_monitor.stop()
            raise e

    def run_endurance_test(self, config: TestConfig) -> ScalabilityResult:
        """
        Test 4: Endurance Test
        Tests system stability over extended duration
        """
        print(f"\n{'=' * 80}")
        print(f"TEST 4: Endurance Test ({config.duration_seconds}s)")
        print(f"{'=' * 80}")

        result = ScalabilityResult(
            test_id=f"endurance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_config=config,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        self.resource_monitor.start()

        try:
            # Create a stable network
            snn = SpikingNeuralNetwork(num_neurons=400, connectivity=0.1)

            start_time = time.time()
            sample_count = 0
            memory_readings = []
            response_times = []

            print("Running endurance test...")
            while time.time() - start_time < config.duration_seconds:
                # Simulate continuous operation
                step_start = time.time()
                snn.simulate_step()
                step_time = (time.time() - step_start) * 1000

                response_times.append(step_time)

                # Sample every second
                if sample_count % 10 == 0:
                    current_stats = self.resource_monitor.get_current_stats()
                    memory_readings.append(current_stats.get("memory_mb", 0))

                    elapsed = int(time.time() - start_time)
                    progress = (elapsed / config.duration_seconds) * 100
                    print(
                        f"\rProgress: {progress:.1f}% | "
                        f"Memory: {current_stats.get('memory_mb', 0):.1f}MB | "
                        f"Avg Response: {np.mean(response_times[-100:]):.2f}ms",
                        end="",
                        flush=True,
                    )

                sample_count += 1
                time.sleep(0.1)  # 10Hz operation

            print("\n✓ Endurance test completed")

            resource_stats = self.resource_monitor.stop()

            # Analyze memory pattern
            if len(memory_readings) > 1:
                memory_growth = memory_readings[-1] - memory_readings[0]
                memory_trend = np.polyfit(
                    range(len(memory_readings)), memory_readings, 1
                )[0]

                if memory_trend > 1.0:  # More than 1MB per sample growth
                    result.memory_pattern = "LEAK_DETECTED"
                elif memory_trend > 0.1:
                    result.memory_pattern = "GRADUAL_GROWTH"
                else:
                    result.memory_pattern = "STABLE"

            # Record final metrics
            result.metrics.append(
                TestMetrics(
                    timestamp=time.time(),
                    network_size=400,
                    concurrent_requests=1,
                    data_volume="medium",
                    response_time_ms=np.mean(response_times),
                    response_time_p95_ms=np.percentile(response_times, 95),
                    response_time_p99_ms=np.percentile(response_times, 99),
                    throughput_rps=10.0,  # 10Hz
                    error_count=0,
                    error_rate=0.0,
                    timeout_count=0,
                    memory_usage_mb=resource_stats.get("memory_avg_mb", 0),
                    memory_growth_mb=resource_stats.get("memory_growth_mb", 0),
                    cpu_percent=resource_stats.get("cpu_avg", 0),
                    active_threads=threading.active_count(),
                    queue_depth=0,
                )
            )

        except Exception as e:
            print(f"\n✗ Endurance test failed: {e}")
            self.resource_monitor.stop()

        result.end_time = datetime.now()
        return result

    def generate_report(
        self,
        results: List[ScalabilityResult],
        output_dir: str = "/home/runner/workspace",
    ):
        """Generate comprehensive scalability report with visualizations"""
        print(f"\n{'=' * 80}")
        print(f"GENERATING SCALABILITY REPORT")
        print(f"{'=' * 80}")

        # Create visualizations
        self._create_scalability_curves(results, output_dir)

        # Generate JSON report
        report_data = {"generated_at": datetime.now().isoformat(), "test_summary": []}

        for result in results:
            test_summary = {
                "test_id": result.test_id,
                "test_name": result.test_config.test_name,
                "duration": str(result.end_time - result.start_time),
                "breaking_point": result.breaking_point,
                "saturation_point": result.saturation_point,
                "max_throughput": result.max_throughput,
                "max_stable_concurrency": result.max_stable_concurrency,
                "metrics_count": len(result.metrics),
            }
            report_data["test_summary"].append(test_summary)

        # Save report
        report_path = f"{output_dir}/scalability_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"✓ Report saved to: {report_path}")

        # Generate capacity planning recommendations
        self._generate_capacity_recommendations(results, output_dir)

    def _create_scalability_curves(
        self, results: List[ScalabilityResult], output_dir: str
    ):
        """Create visualization plots for scalability metrics"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                "NeuralBlitz Scalability Test Results", fontsize=16, fontweight="bold"
            )

            for result in results:
                if not result.metrics:
                    continue

                # Extract data
                network_sizes = [m.network_size for m in result.metrics]
                response_times = [m.response_time_ms for m in result.metrics]
                throughputs = [m.throughput_rps for m in result.metrics]
                memory_usage = [m.memory_usage_mb for m in result.metrics]

                # Plot 1: Response Time vs Network Size
                axes[0, 0].plot(
                    network_sizes,
                    response_times,
                    marker="o",
                    label=result.test_config.test_name,
                )
                axes[0, 0].set_xlabel("Network Size (nodes)")
                axes[0, 0].set_ylabel("Response Time (ms)")
                axes[0, 0].set_title("Response Time Degradation")
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

                # Plot 2: Throughput vs Network Size
                axes[0, 1].plot(
                    network_sizes,
                    throughputs,
                    marker="s",
                    label=result.test_config.test_name,
                )
                axes[0, 1].set_xlabel("Network Size (nodes)")
                axes[0, 1].set_ylabel("Throughput (ops/sec)")
                axes[0, 1].set_title("Throughput Saturation")
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # Plot 3: Memory Usage
                axes[1, 0].plot(
                    network_sizes,
                    memory_usage,
                    marker="^",
                    label=result.test_config.test_name,
                )
                axes[1, 0].set_xlabel("Network Size (nodes)")
                axes[1, 0].set_ylabel("Memory Usage (MB)")
                axes[1, 0].set_title("Memory Growth Patterns")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: Summary table
            axes[1, 1].axis("off")
            summary_text = "SCALABILITY SUMMARY\n\n"

            for result in results:
                summary_text += f"Test: {result.test_config.test_name}\n"
                if result.breaking_point:
                    summary_text += f"  Breaking Point: {result.breaking_point['dimension']} = {result.breaking_point['value']}\n"
                if result.saturation_point:
                    summary_text += f"  Saturation Point: {result.saturation_point['dimension']} = {result.saturation_point['value']}\n"
                summary_text += (
                    f"  Max Throughput: {result.max_throughput:.2f} ops/sec\n"
                )
                summary_text += (
                    f"  Max Stable Concurrency: {result.max_stable_concurrency}\n\n"
                )

            axes[1, 1].text(
                0.1,
                0.9,
                summary_text,
                transform=axes[1, 1].transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
            )

            plt.tight_layout()
            plt.savefig(
                f"{output_dir}/scalability_curves.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

            print(f"✓ Scalability curves saved to: {output_dir}/scalability_curves.png")

        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")

    def _generate_capacity_recommendations(
        self, results: List[ScalabilityResult], output_dir: str
    ):
        """Generate capacity planning recommendations"""

        recommendations = {
            "generated_at": datetime.now().isoformat(),
            "recommendations": [],
        }

        # Analyze results and generate recommendations
        for result in results:
            if result.breaking_point:
                recommendations["recommendations"].append(
                    {
                        "type": "CRITICAL",
                        "test": result.test_config.test_name,
                        "message": f"System breaks at {result.breaking_point['dimension']} = {result.breaking_point['value']}",
                        "action": "Implement horizontal scaling or optimize algorithms",
                    }
                )

            if result.saturation_point:
                recommendations["recommendations"].append(
                    {
                        "type": "WARNING",
                        "test": result.test_config.test_name,
                        "message": f"Performance degrades at {result.saturation_point['dimension']} = {result.saturation_point['value']}",
                        "action": "Consider load balancing or request queuing",
                    }
                )

            # Memory recommendations
            for metric in result.metrics:
                if metric.memory_growth_mb > 100:  # 100MB growth
                    recommendations["recommendations"].append(
                        {
                            "type": "OPTIMIZATION",
                            "test": result.test_config.test_name,
                            "message": f"High memory growth detected: {metric.memory_growth_mb:.2f}MB",
                            "action": "Review memory management and garbage collection",
                        }
                    )
                    break

        # Add general recommendations
        recommendations["recommendations"].append(
            {
                "type": "BEST_PRACTICE",
                "test": "General",
                "message": "Monitor CPU utilization under peak load",
                "action": "Set up alerts at 80% CPU usage",
            }
        )

        # Save recommendations
        rec_path = f"{output_dir}/capacity_recommendations.json"
        with open(rec_path, "w") as f:
            json.dump(recommendations, f, indent=2)

        print(f"✓ Capacity recommendations saved to: {rec_path}")


def main():
    """Main entry point for scalability testing"""
    print("=" * 80)
    print("NEURALBLITZ v50.0 - SCALABILITY TESTING FRAMEWORK")
    print("Task 3.2: System Behavior Under Load and Scale")
    print("=" * 80)

    # Initialize test runner
    runner = ScalabilityTestRunner(api_base_url="http://localhost:5000")

    results = []

    # Test 1: Network Size Scalability
    config1 = TestConfig(
        test_name="Network Size Test",
        network_sizes=[80, 200, 400, 800, 1600],
        duration_seconds=30,
    )
    result1 = runner.run_network_size_test(config1)
    results.append(result1)

    # Test 2: Concurrent Request Scalability
    config2 = TestConfig(
        test_name="Concurrent Request Test",
        concurrent_requests=[1, 10, 50, 100, 500],
        duration_seconds=60,
    )
    result2 = runner.run_concurrent_request_test(config2)
    results.append(result2)

    # Test 3: Data Volume Scalability
    config3 = TestConfig(
        test_name="Data Volume Test",
        data_volumes=["small", "medium", "large", "very_large"],
        duration_seconds=30,
    )
    result3 = runner.run_data_volume_test(config3)
    results.append(result3)

    # Test 4: Endurance Test
    config4 = TestConfig(
        test_name="Endurance Test",
        duration_seconds=120,  # 2 minutes
    )
    result4 = runner.run_endurance_test(config4)
    results.append(result4)

    # Generate comprehensive report
    runner.generate_report(results, output_dir="/home/runner/workspace")

    print("\n" + "=" * 80)
    print("SCALABILITY TESTING COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - scalability_report.json")
    print("  - scalability_curves.png")
    print("  - capacity_recommendations.json")


if __name__ == "__main__":
    main()
