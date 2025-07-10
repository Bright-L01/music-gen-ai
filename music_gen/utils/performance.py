"""
Performance optimization utilities for Music Gen AI.

This module provides utilities for optimizing model performance,
memory usage, and inference speed.
"""

import functools
import gc
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiler for tracking execution times and memory usage."""

    def __init__(self):
        self.timings: Dict[str, list] = {}
        self.memory_usage: Dict[str, list] = {}

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            # Record timing
            duration = end_time - start_time
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(duration)

            # Record memory usage
            memory_diff = end_memory - start_memory
            if name not in self.memory_usage:
                self.memory_usage[name] = []
            self.memory_usage[name].append(memory_diff)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary statistics."""
        summary = {}

        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times),
                    "count": len(times),
                }

                if name in self.memory_usage:
                    memory_values = self.memory_usage[name]
                    summary[name].update(
                        {
                            "avg_memory": sum(memory_values) / len(memory_values),
                            "max_memory": max(memory_values),
                            "min_memory": min(memory_values),
                        }
                    )

        return summary

    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        for name, stats in summary.items():
            print(f"\n{name}:")
            print(f"  Average Time: {stats['avg_time']:.3f}s")
            print(f"  Min/Max Time: {stats['min_time']:.3f}s / {stats['max_time']:.3f}s")
            print(f"  Total Time:   {stats['total_time']:.3f}s")
            print(f"  Call Count:   {stats['count']}")

            if "avg_memory" in stats:
                print(f"  Avg Memory:   {stats['avg_memory']:.1f} MB")
                print(f"  Max Memory:   {stats['max_memory']:.1f} MB")


def profile_function(name: Optional[str] = None):
    """Decorator for profiling function execution."""

    def decorator(func: Callable) -> Callable:
        func_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "_profiler"):
                wrapper._profiler = PerformanceProfiler()

            with wrapper._profiler.profile(func_name):
                return func(*args, **kwargs)

        wrapper.get_profile_summary = lambda: wrapper._profiler.get_summary()
        wrapper.print_profile_summary = lambda: wrapper._profiler.print_summary()

        return wrapper

    return decorator


@contextmanager
def memory_efficient_context():
    """Context manager for memory-efficient operations."""
    # Clear cache before operation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    try:
        yield
    finally:
        # Clear cache after operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def optimize_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    max_memory_gb: float = 8.0,
    device: str = "cuda",
) -> int:
    """Find optimal batch size for given model and memory constraints."""

    if not torch.cuda.is_available() or device == "cpu":
        # Conservative batch size for CPU
        return 4

    model_device = torch.device(device)
    model = model.to(model_device)
    model.eval()

    # Start with batch size 1 and increase until memory limit
    batch_size = 1
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024

    with torch.no_grad():
        while batch_size <= 64:  # Reasonable upper limit
            try:
                # Create dummy input with current batch size
                dummy_input = torch.randn(batch_size, *input_shape[1:], device=model_device)

                # Clear cache and measure initial memory
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()

                # Forward pass
                _ = model(dummy_input)

                # Measure peak memory
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = peak_memory - initial_memory

                logger.debug(f"Batch size {batch_size}: {memory_used / 1024 / 1024:.1f} MB")

                # Check if we're approaching memory limit
                if memory_used > max_memory_bytes * 0.8:  # 80% threshold
                    break

                batch_size *= 2

            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size //= 2  # Go back to last working size
                    break
                else:
                    raise
            finally:
                torch.cuda.empty_cache()

    logger.info(f"Optimal batch size: {batch_size}")
    return max(1, batch_size)


def benchmark_inference(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    num_warmup: int = 5,
    num_iterations: int = 20,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark model inference performance."""

    model_device = torch.device(device)
    model = model.to(model_device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape[1:], device=model_device)

    # Warmup
    logger.info(f"Warming up for {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Benchmark
    logger.info(f"Benchmarking for {num_iterations} iterations...")
    times = []

    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()

            _ = model(dummy_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

            if i % 5 == 0:
                logger.debug(f"Iteration {i+1}/{num_iterations}: {times[-1]:.3f}s")

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    throughput = batch_size / avg_time  # samples per second

    results = {
        "avg_latency": avg_time,
        "min_latency": min_time,
        "max_latency": max_time,
        "throughput": throughput,
        "batch_size": batch_size,
    }

    logger.info(f"Benchmark Results:")
    logger.info(f"  Average Latency: {avg_time:.3f}s")
    logger.info(f"  Min/Max Latency: {min_time:.3f}s / {max_time:.3f}s")
    logger.info(f"  Throughput: {throughput:.1f} samples/sec")

    return results


class TensorCoreOptimizer:
    """Optimizer for Tensor Core usage on modern GPUs."""

    @staticmethod
    def is_tensor_core_available() -> bool:
        """Check if Tensor Cores are available."""
        if not torch.cuda.is_available():
            return False

        # Check for Tensor Core capable GPU (compute capability >= 7.0)
        device_capability = torch.cuda.get_device_capability()
        return device_capability[0] >= 7

    @staticmethod
    def optimize_tensor_shapes(tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor shapes for Tensor Core usage."""
        if not TensorCoreOptimizer.is_tensor_core_available():
            return tensor

        # Tensor Cores work optimally with dimensions divisible by 8 (for FP16)
        # or 16 (for INT8)
        shape = list(tensor.shape)

        for i in range(len(shape)):
            if shape[i] % 8 != 0:
                # Pad to next multiple of 8
                new_size = ((shape[i] + 7) // 8) * 8
                padding = [(0, 0)] * len(shape)
                padding[i] = (0, new_size - shape[i])
                tensor = torch.nn.functional.pad(
                    tensor, [item for sublist in reversed(padding) for item in sublist]
                )
                shape[i] = new_size

        return tensor


def enable_torch_optimizations():
    """Enable various PyTorch optimizations."""
    logger.info("Enabling PyTorch optimizations...")

    # Enable cuDNN benchmarking for consistent input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("Enabled cuDNN benchmark mode")

    # Enable optimized attention if available
    if hasattr(torch.backends, "opt_einsum"):
        torch.backends.opt_einsum.enabled = True
        logger.info("Enabled optimized einsum")

    # Set optimal number of threads
    if torch.get_num_threads() != torch.get_num_interop_threads():
        optimal_threads = min(torch.get_num_threads(), 8)  # Prevent oversubscription
        torch.set_num_threads(optimal_threads)
        logger.info(f"Set number of threads to {optimal_threads}")


class CacheManager:
    """Intelligent cache management for models and data."""

    def __init__(self, max_cache_size_gb: float = 4.0):
        self.max_cache_size = max_cache_size_gb * 1024 * 1024 * 1024
        self.cache = {}
        self.access_times = {}
        self.cache_sizes = {}

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, key: str, value: Any, size_bytes: Optional[int] = None) -> None:
        """Put item in cache with LRU eviction."""
        if size_bytes is None:
            # Estimate size
            size_bytes = self._estimate_size(value)

        # Check if we need to evict items
        while self._get_total_size() + size_bytes > self.max_cache_size:
            self._evict_lru()

        self.cache[key] = value
        self.cache_sizes[key] = size_bytes
        self.access_times[key] = time.time()

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of an object."""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        else:
            # Rough estimate for other objects
            return 1024  # 1KB default

    def _get_total_size(self) -> int:
        """Get total cache size."""
        return sum(self.cache_sizes.values())

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.cache_sizes[lru_key]

        logger.debug(f"Evicted {lru_key} from cache")


# Global instances
global_profiler = PerformanceProfiler()
global_cache = CacheManager()
