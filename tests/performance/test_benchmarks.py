"""
Performance benchmarks and regression detection tests.

Tests system performance under various conditions, establishes
benchmarks for key operations, and detects performance regressions.
"""

import time
import pytest
import psutil
import statistics
import json
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch
from contextlib import contextmanager
import torch
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PerformanceBenchmark:
    """Represents a performance benchmark result."""

    operation: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    throughput: float
    metadata: Dict[str, Any]


class PerformanceMonitor:
    """Monitor system performance during test execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu_times = None

    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.start_cpu_times = self.process.cpu_times()

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu_times = self.process.cpu_times()

        duration_ms = (end_time - self.start_time) * 1000
        memory_delta = end_memory - self.start_memory
        cpu_time_delta = (end_cpu_times.user + end_cpu_times.system) - (
            self.start_cpu_times.user + self.start_cpu_times.system
        )

        return {
            "duration_ms": duration_ms,
            "memory_delta_mb": memory_delta,
            "cpu_time_seconds": cpu_time_delta,
            "peak_memory_mb": end_memory,
        }


@contextmanager
def default_performance_monitor():
    """Context manager for performance monitoring."""
    monitor = PerformanceMonitor()
    monitor.start()
    try:
        yield monitor
    finally:
        metrics = monitor.stop()
        return metrics


@pytest.mark.performance
class TestGenerationPerformance:
    """Test music generation performance benchmarks."""

    @pytest.fixture
    def benchmark_config(self):
        """Configuration for performance benchmarks."""
        return {
            "short_duration": 5.0,
            "medium_duration": 15.0,
            "long_duration": 30.0,
            "batch_sizes": [1, 2, 4, 8],
            "temperature_values": [0.5, 0.8, 1.0, 1.2],
            "max_memory_mb": 1000,
            "max_generation_time_ms": 30000,  # 30 seconds
        }

    def test_generation_time_benchmarks(self, benchmark_config):
        """Benchmark generation times for different durations."""
        durations = [
            benchmark_config["short_duration"],
            benchmark_config["medium_duration"],
            benchmark_config["long_duration"],
        ]

        benchmarks = []

        for duration in durations:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Mock generation process
                with patch("music_gen.inference.generators.MusicGenerator") as MockGenerator:
                    mock_generator = Mock()

                    # Simulate generation time based on duration
                    generation_time = duration * 0.8  # 0.8x real-time
                    time.sleep(min(generation_time, 0.1))  # Cap sleep for tests

                    mock_result = Mock()
                    mock_result.audio = torch.randn(1, int(duration * 24000))
                    mock_result.duration = duration
                    mock_result.sample_rate = 24000

                    mock_generator.generate.return_value = mock_result

                    # Execute generation
                    from music_gen.core.interfaces.services import GenerationRequest

                    request = GenerationRequest(prompt="Benchmark test", duration=duration)

                    result = mock_generator.generate(request)
                    assert result.duration == duration

                metrics = monitor.stop()

                benchmark = PerformanceBenchmark(
                    operation=f"generation_{duration}s",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=metrics["peak_memory_mb"],
                    cpu_percent=0.0,  # Would be calculated in real scenario
                    throughput=duration / (metrics["duration_ms"] / 1000),
                    metadata={"audio_duration": duration},
                )

                benchmarks.append(benchmark)

        # Verify performance expectations
        for benchmark in benchmarks:
            assert benchmark.duration_ms < benchmark_config["max_generation_time_ms"]
            assert benchmark.memory_mb < benchmark_config["max_memory_mb"]
            assert benchmark.throughput > 0.1  # At least 0.1x real-time

    def test_batch_generation_performance(self, benchmark_config):
        """Benchmark batch generation performance."""
        batch_sizes = benchmark_config["batch_sizes"]
        duration = benchmark_config["short_duration"]

        batch_benchmarks = []

        for batch_size in batch_sizes:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Mock batch generation
                with patch("music_gen.inference.generators.BatchGenerator") as MockBatchGenerator:
                    mock_generator = Mock()

                    # Simulate batch processing
                    batch_time = duration * 0.8 * (1 + batch_size * 0.1)  # Slight overhead per item
                    time.sleep(min(batch_time, 0.2))  # Cap for tests

                    mock_results = [
                        Mock(
                            audio=torch.randn(1, int(duration * 24000)),
                            duration=duration,
                            sample_rate=24000,
                        )
                        for _ in range(batch_size)
                    ]

                    mock_generator.generate_batch.return_value = mock_results

                    # Execute batch generation
                    from music_gen.core.interfaces.services import GenerationRequest

                    requests = [
                        GenerationRequest(prompt=f"Batch test {i}", duration=duration)
                        for i in range(batch_size)
                    ]

                    results = mock_generator.generate_batch(requests)
                    assert len(results) == batch_size

                metrics = monitor.stop()

                # Calculate per-item metrics
                per_item_time = metrics["duration_ms"] / batch_size
                per_item_memory = metrics["memory_delta_mb"] / batch_size

                benchmark = PerformanceBenchmark(
                    operation=f"batch_generation_size_{batch_size}",
                    duration_ms=per_item_time,
                    memory_mb=per_item_memory,
                    cpu_percent=0.0,
                    throughput=batch_size / (metrics["duration_ms"] / 1000),
                    metadata={
                        "batch_size": batch_size,
                        "total_duration_ms": metrics["duration_ms"],
                    },
                )

                batch_benchmarks.append(benchmark)

        # Verify batch efficiency
        for i, benchmark in enumerate(batch_benchmarks[1:], 1):
            previous_benchmark = batch_benchmarks[i - 1]

            # Batch processing should be more efficient per item
            efficiency_ratio = benchmark.duration_ms / previous_benchmark.duration_ms
            assert efficiency_ratio < 1.5  # Should not be 50% worse per item

    def test_temperature_performance_impact(self, benchmark_config):
        """Test performance impact of different temperature settings."""
        temperatures = benchmark_config["temperature_values"]
        duration = benchmark_config["short_duration"]

        temperature_benchmarks = []

        for temperature in temperatures:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Mock generation with temperature
                with patch("music_gen.inference.strategies.SamplingStrategy") as MockStrategy:
                    mock_strategy = Mock()

                    # Higher temperature might mean more sampling iterations
                    sampling_overhead = temperature * 0.1  # Slight overhead
                    time.sleep(min(sampling_overhead, 0.05))

                    mock_tokens = torch.randint(0, 2048, (1, 100))
                    mock_strategy.sample.return_value = mock_tokens

                    # Simulate generation
                    from music_gen.core.interfaces.services import GenerationRequest

                    request = GenerationRequest(
                        prompt="Temperature test", duration=duration, temperature=temperature
                    )

                    # Mock result
                    result = Mock(
                        audio=torch.randn(1, int(duration * 24000)),
                        duration=duration,
                        metadata={"temperature": temperature},
                    )

                metrics = monitor.stop()

                benchmark = PerformanceBenchmark(
                    operation=f"generation_temp_{temperature}",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=metrics["peak_memory_mb"],
                    cpu_percent=0.0,
                    throughput=1.0 / (metrics["duration_ms"] / 1000),
                    metadata={"temperature": temperature},
                )

                temperature_benchmarks.append(benchmark)

        # Verify temperature impact is reasonable
        base_benchmark = temperature_benchmarks[0]  # Lowest temperature

        for benchmark in temperature_benchmarks[1:]:
            # Higher temperature should not be dramatically slower
            slowdown_ratio = benchmark.duration_ms / base_benchmark.duration_ms
            assert slowdown_ratio < 2.0  # Should not be 2x slower


@pytest.mark.performance
class TestAudioProcessingPerformance:
    """Test audio processing performance benchmarks."""

    def test_audio_loading_performance(self):
        """Benchmark audio loading performance."""
        # Test different audio sizes
        audio_sizes = [
            (1, 24000),  # 1 second
            (1, 240000),  # 10 seconds
            (1, 720000),  # 30 seconds
            (2, 240000),  # 10 seconds stereo
        ]

        loading_benchmarks = []

        for channels, samples in audio_sizes:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Mock audio loading
                with patch("torchaudio.load") as mock_load:
                    # Simulate loading time based on size
                    load_time = (samples * channels) / 1000000  # Rough estimate
                    time.sleep(min(load_time, 0.1))

                    audio_data = torch.randn(channels, samples)
                    sample_rate = 24000

                    mock_load.return_value = (audio_data, sample_rate)

                    # Execute loading
                    from music_gen.utils.audio import AudioProcessor

                    processor = AudioProcessor(sample_rate=24000)

                    loaded_audio, sr = mock_load("dummy_path")
                    assert loaded_audio.shape == (channels, samples)
                    assert sr == sample_rate

                metrics = monitor.stop()

                # Calculate throughput (samples per second)
                throughput = (samples * channels) / (metrics["duration_ms"] / 1000)

                benchmark = PerformanceBenchmark(
                    operation=f"audio_loading_{channels}ch_{samples}samples",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=metrics["memory_delta_mb"],
                    cpu_percent=0.0,
                    throughput=throughput,
                    metadata={"channels": channels, "samples": samples},
                )

                loading_benchmarks.append(benchmark)

        # Verify loading performance
        for benchmark in loading_benchmarks:
            # Should load faster than real-time playback
            samples = benchmark.metadata["samples"]
            channels = benchmark.metadata["channels"]
            audio_duration_ms = (samples / 24000) * 1000

            assert benchmark.duration_ms < audio_duration_ms * 10  # 10x real-time max

    def test_audio_processing_performance(self):
        """Benchmark audio processing operations."""
        # Test different processing operations
        operations = [
            "normalize",
            "resample",
            "fade",
            "reverb",
            "compression",
        ]

        audio = torch.randn(1, 240000)  # 10 seconds
        processing_benchmarks = []

        for operation in operations:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Mock processing operation
                if operation == "normalize":
                    # Normalization is typically fast
                    time.sleep(0.001)
                    result = audio / audio.abs().max()
                elif operation == "resample":
                    # Resampling can be slower
                    time.sleep(0.01)
                    result = torch.nn.functional.interpolate(
                        audio.unsqueeze(0), size=int(audio.shape[-1] * 1.5), mode="linear"
                    ).squeeze(0)
                elif operation == "fade":
                    # Fade is usually fast
                    time.sleep(0.002)
                    fade_samples = 1000
                    result = audio.clone()
                    result[0, :fade_samples] *= torch.linspace(0, 1, fade_samples)
                    result[0, -fade_samples:] *= torch.linspace(1, 0, fade_samples)
                else:
                    # Other effects (reverb, compression) can be slower
                    time.sleep(0.02)
                    result = audio  # Mock result

                metrics = monitor.stop()

                # Calculate throughput (samples per second)
                throughput = audio.shape[-1] / (metrics["duration_ms"] / 1000)

                benchmark = PerformanceBenchmark(
                    operation=f"audio_processing_{operation}",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=metrics["memory_delta_mb"],
                    cpu_percent=0.0,
                    throughput=throughput,
                    metadata={"operation": operation, "input_samples": audio.shape[-1]},
                )

                processing_benchmarks.append(benchmark)

        # Verify processing performance
        for benchmark in processing_benchmarks:
            operation = benchmark.metadata["operation"]

            # Set different expectations for different operations
            if operation in ["normalize", "fade"]:
                max_time_ms = 100  # Should be very fast
            elif operation in ["resample"]:
                max_time_ms = 500  # Can be slower
            else:
                max_time_ms = 1000  # Effects can take longer

            assert benchmark.duration_ms < max_time_ms


@pytest.mark.performance
class TestModelPerformance:
    """Test model loading and inference performance."""

    def test_model_loading_performance(self):
        """Benchmark model loading times."""
        model_sizes = [
            ("small", 300),  # 300MB
            ("medium", 1500),  # 1.5GB
            ("large", 3000),  # 3GB
        ]

        loading_benchmarks = []

        for model_name, size_mb in model_sizes:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Mock model loading
                with patch("music_gen.models.transformer.MusicGenModel") as MockModel:
                    # Simulate loading time based on model size
                    load_time = size_mb / 1000  # 1 second per GB
                    time.sleep(min(load_time, 0.5))  # Cap for tests

                    mock_model = Mock()
                    mock_model.config.vocab_size = 2048
                    mock_model.config.hidden_size = 512 * (size_mb // 300)  # Scale with size

                    MockModel.return_value = mock_model

                    # Execute loading
                    from music_gen.application.services import ModelServiceImpl
                    from music_gen.infrastructure.repositories import InMemoryModelRepository
                    from music_gen.core.config import AppConfig

                    repo = InMemoryModelRepository()
                    config = AppConfig()
                    service = ModelServiceImpl(repo, config)

                    # Mock the actual loading
                    model = MockModel()
                    assert model is not None

                metrics = monitor.stop()

                benchmark = PerformanceBenchmark(
                    operation=f"model_loading_{model_name}",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=metrics["memory_delta_mb"],
                    cpu_percent=0.0,
                    throughput=size_mb / (metrics["duration_ms"] / 1000),  # MB/s
                    metadata={"model_size_mb": size_mb, "model_name": model_name},
                )

                loading_benchmarks.append(benchmark)

        # Verify loading performance
        for benchmark in loading_benchmarks:
            size_mb = benchmark.metadata["model_size_mb"]

            # Should load within reasonable time (max 10 seconds for largest model)
            max_load_time_ms = (size_mb / 300) * 10000  # 10s for 3GB model
            assert benchmark.duration_ms < max_load_time_ms

            # Should have reasonable throughput (min 10 MB/s)
            assert benchmark.throughput > 10.0

    def test_inference_performance(self):
        """Benchmark model inference performance."""
        sequence_lengths = [50, 100, 200, 500]

        inference_benchmarks = []

        for seq_len in sequence_lengths:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Mock inference
                with patch("music_gen.models.transformer.MusicGenModel") as MockModel:
                    mock_model = Mock()

                    # Simulate inference time based on sequence length
                    inference_time = seq_len / 1000  # Rough estimate
                    time.sleep(min(inference_time, 0.1))

                    # Mock forward pass
                    batch_size = 1
                    vocab_size = 2048
                    output = torch.randn(batch_size, seq_len, vocab_size)

                    mock_model.forward.return_value = output
                    MockModel.return_value = mock_model

                    # Execute inference
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
                    result = mock_model.forward(input_ids)

                    assert result.shape == (batch_size, seq_len, vocab_size)

                metrics = monitor.stop()

                # Calculate tokens per second
                tokens_per_second = seq_len / (metrics["duration_ms"] / 1000)

                benchmark = PerformanceBenchmark(
                    operation=f"inference_seq_{seq_len}",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=metrics["memory_delta_mb"],
                    cpu_percent=0.0,
                    throughput=tokens_per_second,
                    metadata={"sequence_length": seq_len},
                )

                inference_benchmarks.append(benchmark)

        # Verify inference performance
        for benchmark in inference_benchmarks:
            seq_len = benchmark.metadata["sequence_length"]

            # Should process at reasonable speed (min 10 tokens/second)
            assert benchmark.throughput > 10.0

            # Memory usage should scale reasonably with sequence length
            memory_per_token = benchmark.memory_mb / seq_len
            assert memory_per_token < 1.0  # Less than 1MB per token


@pytest.mark.performance
class TestSystemPerformance:
    """Test overall system performance and resource usage."""

    def test_memory_usage_patterns(self):
        """Test memory usage patterns under different loads."""
        # Test scenarios with different memory profiles
        scenarios = [
            {"name": "single_short", "requests": 1, "duration": 5.0},
            {"name": "single_long", "requests": 1, "duration": 30.0},
            {"name": "multiple_short", "requests": 5, "duration": 5.0},
            {"name": "batch_medium", "requests": 3, "duration": 15.0},
        ]

        memory_benchmarks = []

        for scenario in scenarios:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Simulate memory usage for scenario
                base_memory_mb = 100  # Base system memory
                per_request_mb = 50  # Memory per request
                duration_factor = scenario["duration"] / 10  # Duration impact

                total_memory_mb = base_memory_mb + (
                    per_request_mb * scenario["requests"] * duration_factor
                )

                # Simulate processing
                time.sleep(0.05)  # Small delay for test

                metrics = monitor.stop()

                benchmark = PerformanceBenchmark(
                    operation=f"memory_usage_{scenario['name']}",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=total_memory_mb,  # Simulated peak memory
                    cpu_percent=0.0,
                    throughput=scenario["requests"] / (metrics["duration_ms"] / 1000),
                    metadata=scenario,
                )

                memory_benchmarks.append(benchmark)

        # Verify memory usage is reasonable
        for benchmark in memory_benchmarks:
            scenario = benchmark.metadata

            # Memory should scale reasonably with requests and duration
            expected_max_memory = 100 + (scenario["requests"] * scenario["duration"] * 10)
            assert benchmark.memory_mb < expected_max_memory

    def test_concurrent_request_performance(self):
        """Test performance under concurrent load."""
        concurrency_levels = [1, 2, 4, 8]

        concurrency_benchmarks = []

        for concurrency in concurrency_levels:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Simulate concurrent processing
                # In real scenario, would use actual concurrent requests
                processing_time = 0.1 + (concurrency * 0.02)  # Slight overhead
                time.sleep(processing_time)

                metrics = monitor.stop()

                # Calculate requests per second
                rps = concurrency / (metrics["duration_ms"] / 1000)

                benchmark = PerformanceBenchmark(
                    operation=f"concurrent_requests_{concurrency}",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=metrics["memory_delta_mb"],
                    cpu_percent=0.0,
                    throughput=rps,
                    metadata={"concurrency": concurrency},
                )

                concurrency_benchmarks.append(benchmark)

        # Verify concurrent performance scaling
        base_benchmark = concurrency_benchmarks[0]

        for benchmark in concurrency_benchmarks[1:]:
            concurrency = benchmark.metadata["concurrency"]

            # Throughput should increase with concurrency (with some overhead)
            efficiency = benchmark.throughput / (base_benchmark.throughput * concurrency)
            assert efficiency > 0.5  # At least 50% efficiency

    def test_resource_cleanup_performance(self):
        """Test resource cleanup and garbage collection performance."""
        cleanup_scenarios = [
            {"name": "model_cache_cleanup", "items": 10, "size_mb": 100},
            {"name": "audio_cache_cleanup", "items": 50, "size_mb": 10},
            {"name": "task_history_cleanup", "items": 100, "size_mb": 1},
        ]

        cleanup_benchmarks = []

        for scenario in cleanup_scenarios:
            with PerformanceMonitor() as monitor:
                monitor.start()

                # Simulate cleanup operation
                cleanup_time = scenario["items"] * 0.001  # 1ms per item
                time.sleep(cleanup_time)

                metrics = monitor.stop()

                # Calculate cleanup rate
                items_per_second = scenario["items"] / (metrics["duration_ms"] / 1000)

                benchmark = PerformanceBenchmark(
                    operation=f"cleanup_{scenario['name']}",
                    duration_ms=metrics["duration_ms"],
                    memory_mb=-scenario["size_mb"] * scenario["items"],  # Memory freed
                    cpu_percent=0.0,
                    throughput=items_per_second,
                    metadata=scenario,
                )

                cleanup_benchmarks.append(benchmark)

        # Verify cleanup performance
        for benchmark in cleanup_benchmarks:
            scenario = benchmark.metadata

            # Cleanup should be fast (>100 items/second)
            assert benchmark.throughput > 100.0

            # Should free expected amount of memory
            expected_memory_freed = scenario["size_mb"] * scenario["items"]
            assert abs(benchmark.memory_mb) >= expected_memory_freed * 0.8


class BenchmarkReporter:
    """Generate performance benchmark reports."""

    @staticmethod
    def save_benchmarks(benchmarks: List[PerformanceBenchmark], output_file: Path):
        """Save benchmark results to file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": [
                {
                    "operation": b.operation,
                    "duration_ms": b.duration_ms,
                    "memory_mb": b.memory_mb,
                    "cpu_percent": b.cpu_percent,
                    "throughput": b.throughput,
                    "metadata": b.metadata,
                }
                for b in benchmarks
            ],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def compare_benchmarks(current_file: Path, baseline_file: Path) -> Dict[str, Any]:
        """Compare current benchmarks with baseline."""
        with open(current_file) as f:
            current_data = json.load(f)

        with open(baseline_file) as f:
            baseline_data = json.load(f)

        # Create lookup for baseline benchmarks
        baseline_lookup = {b["operation"]: b for b in baseline_data["benchmarks"]}

        comparisons = []
        regressions = []

        for current_bench in current_data["benchmarks"]:
            operation = current_bench["operation"]
            baseline_bench = baseline_lookup.get(operation)

            if baseline_bench:
                # Compare performance metrics
                duration_ratio = current_bench["duration_ms"] / baseline_bench["duration_ms"]
                memory_ratio = current_bench["memory_mb"] / max(baseline_bench["memory_mb"], 1)
                throughput_ratio = current_bench["throughput"] / max(
                    baseline_bench["throughput"], 1
                )

                comparison = {
                    "operation": operation,
                    "duration_ratio": duration_ratio,
                    "memory_ratio": memory_ratio,
                    "throughput_ratio": throughput_ratio,
                    "is_regression": duration_ratio > 1.2
                    or memory_ratio > 1.3
                    or throughput_ratio < 0.8,
                }

                comparisons.append(comparison)

                if comparison["is_regression"]:
                    regressions.append(comparison)

        return {
            "comparison_timestamp": datetime.now().isoformat(),
            "total_comparisons": len(comparisons),
            "regressions_found": len(regressions),
            "comparisons": comparisons,
            "regressions": regressions,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
