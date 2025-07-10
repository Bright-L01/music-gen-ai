"""
Memory optimization strategies for Music Gen AI.

This module provides memory-efficient patterns including lazy loading,
object pooling, and automatic memory management.
"""

import asyncio
import gc
import logging
import sys
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MemoryProfile:
    """Memory usage profile."""

    total_memory: float  # Total system memory in GB
    available_memory: float  # Available memory in GB
    used_memory: float  # Used memory in GB
    memory_percent: float  # Memory usage percentage
    gpu_memory_allocated: float  # GPU memory allocated in GB
    gpu_memory_reserved: float  # GPU memory reserved in GB
    process_memory: float  # Current process memory in GB


class MemoryMonitor:
    """Monitors and reports memory usage."""

    @staticmethod
    def get_memory_profile() -> MemoryProfile:
        """Get current memory usage profile."""

        # System memory
        memory = psutil.virtual_memory()

        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / (1024**3)

        # GPU memory if available
        gpu_allocated = 0.0
        gpu_reserved = 0.0

        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)

        return MemoryProfile(
            total_memory=memory.total / (1024**3),
            available_memory=memory.available / (1024**3),
            used_memory=memory.used / (1024**3),
            memory_percent=memory.percent,
            gpu_memory_allocated=gpu_allocated,
            gpu_memory_reserved=gpu_reserved,
            process_memory=process_memory,
        )

    @staticmethod
    def log_memory_usage(prefix: str = ""):
        """Log current memory usage."""

        profile = MemoryMonitor.get_memory_profile()

        logger.info(
            f"{prefix} Memory Usage - "
            f"System: {profile.used_memory:.1f}/{profile.total_memory:.1f}GB ({profile.memory_percent:.1f}%), "
            f"Process: {profile.process_memory:.1f}GB, "
            f"GPU: {profile.gpu_memory_allocated:.1f}/{profile.gpu_memory_reserved:.1f}GB"
        )


class LazyLoader:
    """Lazy loading wrapper for expensive objects."""

    def __init__(self, loader_func: Callable[[], T]):
        self._loader_func = loader_func
        self._value: Optional[T] = None
        self._loaded = False

    def get(self) -> T:
        """Get the value, loading it if necessary."""
        if not self._loaded:
            self._value = self._loader_func()
            self._loaded = True
        return self._value

    def is_loaded(self) -> bool:
        """Check if value has been loaded."""
        return self._loaded

    def unload(self):
        """Unload the value to free memory."""
        self._value = None
        self._loaded = False
        gc.collect()


class ObjectPool:
    """Object pool for reusing expensive objects."""

    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        reset_func: Optional[Callable[[T], None]] = None,
    ):
        self._factory = factory
        self._max_size = max_size
        self._reset_func = reset_func
        self._available: List[T] = []
        self._in_use: weakref.WeakSet = weakref.WeakSet()

    def acquire(self) -> T:
        """Acquire an object from the pool."""

        if self._available:
            obj = self._available.pop()
        else:
            obj = self._factory()

        self._in_use.add(obj)
        return obj

    def release(self, obj: T):
        """Release an object back to the pool."""

        if obj in self._in_use:
            self._in_use.remove(obj)

            if self._reset_func:
                self._reset_func(obj)

            if len(self._available) < self._max_size:
                self._available.append(obj)
            else:
                # Let garbage collector handle it
                del obj

    @contextmanager
    def get(self):
        """Context manager for acquiring and releasing objects."""
        obj = self.acquire()
        try:
            yield obj
        finally:
            self.release(obj)

    def clear(self):
        """Clear the pool."""
        self._available.clear()
        gc.collect()


class LRUCache:
    """Memory-aware LRU cache with size limits."""

    def __init__(self, max_size: int = 100, max_memory_mb: float = 1024):
        self._max_size = max_size
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict = OrderedDict()
        self._sizes: Dict[Any, int] = {}
        self._total_size = 0

    def _get_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""

        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, (list, tuple)):
            return sum(self._get_size(item) for item in obj)
        elif isinstance(obj, dict):
            return sum(self._get_size(k) + self._get_size(v) for k, v in obj.items())
        else:
            return sys.getsizeof(obj)

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: Any, value: Any):
        """Put value in cache."""

        # Calculate size
        size = self._get_size(value)

        # Remove items if necessary
        while (
            len(self._cache) >= self._max_size or self._total_size + size > self._max_memory_bytes
        ) and len(self._cache) > 0:
            oldest_key = next(iter(self._cache))
            self._remove(oldest_key)

        # Add new item
        self._cache[key] = value
        self._sizes[key] = size
        self._total_size += size

        # Move to end (most recently used)
        self._cache.move_to_end(key)

    def _remove(self, key: Any):
        """Remove item from cache."""

        if key in self._cache:
            del self._cache[key]
            self._total_size -= self._sizes.get(key, 0)
            self._sizes.pop(key, None)

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._sizes.clear()
        self._total_size = 0
        gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "memory_mb": self._total_size / (1024 * 1024),
            "max_size": self._max_size,
            "max_memory_mb": self._max_memory_bytes / (1024 * 1024),
        }


class TensorMemoryManager:
    """Manages PyTorch tensor memory efficiently."""

    @staticmethod
    def optimize_model_memory(model: torch.nn.Module, enable_gradient_checkpointing: bool = True):
        """Optimize model memory usage."""

        # Enable gradient checkpointing
        if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

        # Use mixed precision
        model = model.half()  # Convert to fp16
        logger.info("Converted model to half precision")

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    @staticmethod
    @contextmanager
    def low_memory_mode():
        """Context manager for low memory operations."""

        # Store original settings
        original_num_threads = torch.get_num_threads()

        try:
            # Reduce number of threads
            torch.set_num_threads(1)

            # Clear caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Force garbage collection
            gc.collect()

            yield

        finally:
            # Restore settings
            torch.set_num_threads(original_num_threads)

            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    @staticmethod
    def move_to_cpu_on_low_memory(tensor: torch.Tensor, threshold_gb: float = 1.0) -> torch.Tensor:
        """Move tensor to CPU if GPU memory is low."""

        if tensor.is_cuda and torch.cuda.is_available():
            free_memory = (
                torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            )
            free_memory_gb = free_memory / (1024**3)

            if free_memory_gb < threshold_gb:
                logger.warning(f"Low GPU memory ({free_memory_gb:.1f}GB), moving tensor to CPU")
                return tensor.cpu()

        return tensor


def memory_efficient_decorator(
    max_memory_gb: float = 4.0, cleanup_after: bool = True, monitor: bool = True
):
    """Decorator for memory-efficient function execution."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if monitor:
                MemoryMonitor.log_memory_usage(f"Before {func.__name__}")

            # Check available memory
            profile = MemoryMonitor.get_memory_profile()
            if profile.available_memory < max_memory_gb:
                logger.warning(
                    f"Low memory warning: {profile.available_memory:.1f}GB available, "
                    f"{max_memory_gb:.1f}GB recommended for {func.__name__}"
                )

            try:
                # Execute function
                result = await func(*args, **kwargs)

                return result

            finally:
                if cleanup_after:
                    # Force garbage collection
                    gc.collect()

                    # Clear PyTorch cache if available
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if monitor:
                    MemoryMonitor.log_memory_usage(f"After {func.__name__}")

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if monitor:
                MemoryMonitor.log_memory_usage(f"Before {func.__name__}")

            # Check available memory
            profile = MemoryMonitor.get_memory_profile()
            if profile.available_memory < max_memory_gb:
                logger.warning(
                    f"Low memory warning: {profile.available_memory:.1f}GB available, "
                    f"{max_memory_gb:.1f}GB recommended for {func.__name__}"
                )

            try:
                # Execute function
                result = func(*args, **kwargs)

                return result

            finally:
                if cleanup_after:
                    # Force garbage collection
                    gc.collect()

                    # Clear PyTorch cache if available
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if monitor:
                    MemoryMonitor.log_memory_usage(f"After {func.__name__}")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class StreamingDataLoader:
    """Memory-efficient data loader that streams data instead of loading all at once."""

    def __init__(
        self, data_source: Union[Path, List[Path]], batch_size: int = 32, buffer_size: int = 3
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._buffer = asyncio.Queue(maxsize=buffer_size)
        self._producer_task = None

    async def __aiter__(self):
        """Async iterator for streaming data."""

        # Start producer task
        self._producer_task = asyncio.create_task(self._produce_data())

        try:
            while True:
                batch = await self._buffer.get()
                if batch is None:  # Sentinel value
                    break
                yield batch

        finally:
            # Cleanup
            if self._producer_task:
                self._producer_task.cancel()
                try:
                    await self._producer_task
                except asyncio.CancelledError:
                    pass

    async def _produce_data(self):
        """Producer task that loads data and puts it in the buffer."""

        try:
            # Implementation depends on data source type
            if isinstance(self.data_source, Path):
                await self._load_from_file()
            else:
                await self._load_from_files()

        finally:
            # Signal end of data
            await self._buffer.put(None)

    async def _load_from_file(self):
        """Load data from a single file."""

        # Example implementation - adjust based on actual data format
        batch = []

        # Read file in chunks
        async with aiofiles.open(self.data_source, "r") as f:
            async for line in f:
                batch.append(line.strip())

                if len(batch) >= self.batch_size:
                    await self._buffer.put(batch)
                    batch = []

            # Put remaining data
            if batch:
                await self._buffer.put(batch)

    async def _load_from_files(self):
        """Load data from multiple files."""

        for file_path in self.data_source:
            self.data_source = file_path
            await self._load_from_file()


# Global memory manager instance
_memory_cache = LRUCache(max_size=100, max_memory_mb=2048)
_tensor_manager = TensorMemoryManager()


# Utility functions
def clear_memory_caches():
    """Clear all memory caches."""

    _memory_cache.clear()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Cleared memory caches")


def get_memory_usage_summary() -> Dict[str, Any]:
    """Get comprehensive memory usage summary."""

    profile = MemoryMonitor.get_memory_profile()

    return {
        "system": {
            "total_gb": profile.total_memory,
            "used_gb": profile.used_memory,
            "available_gb": profile.available_memory,
            "percent": profile.memory_percent,
        },
        "process": {"memory_gb": profile.process_memory},
        "gpu": {
            "allocated_gb": profile.gpu_memory_allocated,
            "reserved_gb": profile.gpu_memory_reserved,
            "available": torch.cuda.is_available(),
        },
        "cache": _memory_cache.get_stats(),
    }
