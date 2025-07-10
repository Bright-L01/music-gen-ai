from datetime import timedelta

"""
Enterprise Resource Management System for Music Gen AI.

This module provides comprehensive GPU/memory monitoring and management
to ensure stable production operation without resource failures.
"""

import gc
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

from ..core.config import AppConfig
from ..core.exceptions import InsufficientResourcesError

logger = logging.getLogger(__name__)


@dataclass
class ResourceRequirements:
    """Resource requirements for a model or operation."""

    cpu_memory_gb: float
    gpu_memory_gb: float
    min_gpu_compute: float = 0.0  # Minimum compute capability
    recommended_batch_size: int = 1
    notes: str = ""


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time."""

    timestamp: datetime
    # CPU Resources
    cpu_percent: float
    cpu_memory_used_gb: float
    cpu_memory_available_gb: float
    cpu_memory_percent: float
    # GPU Resources
    gpu_available: bool
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    # Process specific
    process_memory_gb: float = 0.0
    process_gpu_memory_gb: float = 0.0


@dataclass
class ResourceAlert:
    """Resource usage alert."""

    timestamp: datetime
    severity: str  # "warning", "critical", "error"
    resource_type: str  # "cpu_memory", "gpu_memory", "gpu_utilization"
    message: str
    current_value: float
    threshold: float
    recommendations: List[str] = field(default_factory=list)


class ResourceMonitor:
    """Monitors system resources and tracks usage patterns."""

    def __init__(
        self, sampling_interval: float = 1.0, history_size: int = 300
    ):  # 5 minutes at 1s intervals
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        self.history: deque[ResourceSnapshot] = deque(maxlen=history_size)
        self.alerts: deque[ResourceAlert] = deque(maxlen=100)
        self._monitoring = False
        self._monitor_thread = None
        self._callbacks: List[Callable[[ResourceSnapshot], None]] = []

    def start_monitoring(self):
        """Start background resource monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                snapshot = self.get_current_snapshot()
                self.history.append(snapshot)

                # Check for alerts
                self._check_alerts(snapshot)

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                time.sleep(self.sampling_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.sampling_interval)

    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get current resource usage snapshot."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()

        snapshot = ResourceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            cpu_memory_used_gb=memory.used / (1024**3),
            cpu_memory_available_gb=memory.available / (1024**3),
            cpu_memory_percent=memory.percent,
            process_memory_gb=process_memory.rss / (1024**3),
            gpu_available=torch.cuda.is_available(),
        )

        # GPU metrics if available
        if torch.cuda.is_available():
            try:
                # Get GPU memory stats
                gpu_mem_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                snapshot.gpu_memory_used_gb = gpu_mem_used
                snapshot.gpu_memory_total_gb = gpu_mem_total
                snapshot.gpu_memory_percent = (
                    (gpu_mem_used / gpu_mem_total) * 100 if gpu_mem_total > 0 else 0
                )

                # Try to get utilization (requires nvidia-ml-py)
                try:
                    import nvidia_ml_py3 as nvml

                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)

                    snapshot.gpu_utilization = util.gpu
                    snapshot.gpu_temperature = temp

                except ImportError:
                    logger.debug("nvidia-ml-py3 not available for GPU utilization metrics")
                except Exception as e:
                    logger.debug(f"Could not get GPU utilization: {e}")

            except Exception as e:
                logger.error(f"Error getting GPU metrics: {e}")

        return snapshot

    def _check_alerts(self, snapshot: ResourceSnapshot):
        """Check for resource alerts based on thresholds."""
        # CPU memory alerts
        if snapshot.cpu_memory_percent > 90:
            self.alerts.append(
                ResourceAlert(
                    timestamp=snapshot.timestamp,
                    severity="critical",
                    resource_type="cpu_memory",
                    message="Critical CPU memory usage",
                    current_value=snapshot.cpu_memory_percent,
                    threshold=90,
                    recommendations=["Unload unused models", "Reduce batch size", "Clear caches"],
                )
            )
        elif snapshot.cpu_memory_percent > 80:
            self.alerts.append(
                ResourceAlert(
                    timestamp=snapshot.timestamp,
                    severity="warning",
                    resource_type="cpu_memory",
                    message="High CPU memory usage",
                    current_value=snapshot.cpu_memory_percent,
                    threshold=80,
                )
            )

        # GPU memory alerts
        if snapshot.gpu_available and snapshot.gpu_memory_percent > 90:
            self.alerts.append(
                ResourceAlert(
                    timestamp=snapshot.timestamp,
                    severity="critical",
                    resource_type="gpu_memory",
                    message="Critical GPU memory usage",
                    current_value=snapshot.gpu_memory_percent,
                    threshold=90,
                    recommendations=[
                        "Clear GPU cache: torch.cuda.empty_cache()",
                        "Reduce model size or batch size",
                        "Enable gradient checkpointing",
                    ],
                )
            )

        # GPU temperature alerts
        if snapshot.gpu_temperature > 85:
            self.alerts.append(
                ResourceAlert(
                    timestamp=snapshot.timestamp,
                    severity="warning",
                    resource_type="gpu_temperature",
                    message="High GPU temperature",
                    current_value=snapshot.gpu_temperature,
                    threshold=85,
                    recommendations=[
                        "Reduce GPU utilization",
                        "Check cooling system",
                        "Consider thermal throttling",
                    ],
                )
            )

    def add_callback(self, callback: Callable[[ResourceSnapshot], None]):
        """Add a callback for resource updates."""
        self._callbacks.append(callback)

    def get_average_usage(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get average resource usage over a time window."""
        if not self.history:
            return {}

        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_snapshots = [s for s in self.history if s.timestamp > cutoff_time]

        if not recent_snapshots:
            return {}

        return {
            "cpu_percent": np.mean([s.cpu_percent for s in recent_snapshots]),
            "cpu_memory_percent": np.mean([s.cpu_memory_percent for s in recent_snapshots]),
            "gpu_memory_percent": np.mean([s.gpu_memory_percent for s in recent_snapshots]),
            "gpu_utilization": np.mean([s.gpu_utilization for s in recent_snapshots]),
        }

    def get_recent_alerts(self, severity: Optional[str] = None) -> List[ResourceAlert]:
        """Get recent alerts, optionally filtered by severity."""
        if severity:
            return [a for a in self.alerts if a.severity == severity]
        return list(self.alerts)


class ResourceManager:
    """Manages system resources for Music Gen AI."""

    # Model memory requirements (estimates)
    MODEL_REQUIREMENTS = {
        "small": ResourceRequirements(
            cpu_memory_gb=2.0, gpu_memory_gb=4.0, min_gpu_compute=6.0, recommended_batch_size=4
        ),
        "medium": ResourceRequirements(
            cpu_memory_gb=4.0, gpu_memory_gb=8.0, min_gpu_compute=7.0, recommended_batch_size=2
        ),
        "large": ResourceRequirements(
            cpu_memory_gb=8.0, gpu_memory_gb=16.0, min_gpu_compute=7.5, recommended_batch_size=1
        ),
        "facebook/musicgen-small": ResourceRequirements(
            cpu_memory_gb=3.0, gpu_memory_gb=6.0, min_gpu_compute=6.0, recommended_batch_size=2
        ),
        "facebook/musicgen-medium": ResourceRequirements(
            cpu_memory_gb=6.0, gpu_memory_gb=12.0, min_gpu_compute=7.0, recommended_batch_size=1
        ),
        "facebook/musicgen-large": ResourceRequirements(
            cpu_memory_gb=12.0,
            gpu_memory_gb=24.0,
            min_gpu_compute=8.0,
            recommended_batch_size=1,
            notes="Requires high-end GPU (A100, V100)",
        ),
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self.monitor = ResourceMonitor()
        self._resource_locks = {}
        self._allocated_resources: Dict[str, float] = {}
        self._model_cache_tracker: Dict[str, Tuple[float, datetime]] = {}

        # Start monitoring
        self.monitor.start_monitoring()

        # Setup automatic cleanup
        self._setup_automatic_cleanup()

    def _setup_automatic_cleanup(self):
        """Setup automatic resource cleanup mechanisms."""
        # Register cleanup callback
        self.monitor.add_callback(self._cleanup_callback)

        # Set memory pressure threshold
        self._memory_pressure_threshold = 0.85  # 85% usage triggers cleanup

    def _cleanup_callback(self, snapshot: ResourceSnapshot):
        """Callback for automatic cleanup based on resource pressure."""
        # CPU memory pressure
        if snapshot.cpu_memory_percent > self._memory_pressure_threshold * 100:
            logger.warning(f"High memory pressure: {snapshot.cpu_memory_percent:.1f}%")
            self._emergency_cleanup()

        # GPU memory pressure
        if (
            snapshot.gpu_available
            and snapshot.gpu_memory_percent > self._memory_pressure_threshold * 100
        ):
            logger.warning(f"High GPU memory pressure: {snapshot.gpu_memory_percent:.1f}%")
            self._emergency_gpu_cleanup()

    def validate_system_resources(self, model_identifier: str) -> Dict[str, Any]:
        """Validate that system has sufficient resources for a model.

        Args:
            model_identifier: Model name or size identifier

        Returns:
            Validation result with details

        Raises:
            InsufficientResourcesError: If resources are insufficient
        """
        # Get requirements
        requirements = self.get_model_requirements(model_identifier)

        # Get current resources
        snapshot = self.monitor.get_current_snapshot()

        validation_result = {
            "valid": True,
            "model": model_identifier,
            "requirements": requirements,
            "available": {
                "cpu_memory_gb": snapshot.cpu_memory_available_gb,
                "gpu_memory_gb": (
                    snapshot.gpu_memory_total_gb - snapshot.gpu_memory_used_gb
                    if snapshot.gpu_available
                    else 0
                ),
                "gpu_available": snapshot.gpu_available,
            },
            "warnings": [],
            "errors": [],
        }

        # Check CPU memory
        if requirements.cpu_memory_gb > snapshot.cpu_memory_available_gb:
            validation_result["valid"] = False
            validation_result["errors"].append(
                f"Insufficient CPU memory: Required {requirements.cpu_memory_gb:.1f}GB, "
                f"Available {snapshot.cpu_memory_available_gb:.1f}GB"
            )
        elif requirements.cpu_memory_gb > snapshot.cpu_memory_available_gb * 0.8:
            validation_result["warnings"].append(
                "CPU memory usage will be high, performance may degrade"
            )

        # Check GPU if required
        if requirements.gpu_memory_gb > 0:
            if not snapshot.gpu_available:
                validation_result["warnings"].append(
                    "GPU not available, will use CPU (slower performance)"
                )
            else:
                available_gpu_memory = snapshot.gpu_memory_total_gb - snapshot.gpu_memory_used_gb
                if requirements.gpu_memory_gb > available_gpu_memory:
                    validation_result["valid"] = False
                    validation_result["errors"].append(
                        f"Insufficient GPU memory: Required {requirements.gpu_memory_gb:.1f}GB, "
                        f"Available {available_gpu_memory:.1f}GB"
                    )

                # Check compute capability
                if requirements.min_gpu_compute > 0:
                    compute_capability = (
                        torch.cuda.get_device_capability()[0]
                        + torch.cuda.get_device_capability()[1] * 0.1
                    )
                    if compute_capability < requirements.min_gpu_compute:
                        validation_result["warnings"].append(
                            f"GPU compute capability {compute_capability} is below recommended {requirements.min_gpu_compute}"
                        )

        # Raise error if validation failed
        if not validation_result["valid"]:
            error_msg = "\n".join(validation_result["errors"])
            raise InsufficientResourcesError(
                f"Insufficient resources for model '{model_identifier}':\n{error_msg}\n\n"
                f"Required: CPU {requirements.cpu_memory_gb}GB, GPU {requirements.gpu_memory_gb}GB\n"
                f"Available: CPU {snapshot.cpu_memory_available_gb:.1f}GB, "
                f"GPU {snapshot.gpu_memory_total_gb - snapshot.gpu_memory_used_gb:.1f}GB"
            )

        return validation_result

    def get_model_requirements(self, model_identifier: str) -> ResourceRequirements:
        """Get resource requirements for a model."""
        # Check if we have predefined requirements
        if model_identifier in self.MODEL_REQUIREMENTS:
            return self.MODEL_REQUIREMENTS[model_identifier]

        # Try to infer from model size in name
        if "small" in model_identifier.lower():
            return self.MODEL_REQUIREMENTS["small"]
        elif "medium" in model_identifier.lower():
            return self.MODEL_REQUIREMENTS["medium"]
        elif "large" in model_identifier.lower():
            return self.MODEL_REQUIREMENTS["large"]

        # Default conservative estimate
        logger.warning(f"Unknown model '{model_identifier}', using default resource estimates")
        return ResourceRequirements(
            cpu_memory_gb=4.0, gpu_memory_gb=8.0, notes="Default estimate for unknown model"
        )

    def allocate_resources(
        self, resource_id: str, cpu_memory_gb: float = 0, gpu_memory_gb: float = 0
    ) -> bool:
        """Allocate resources for an operation.

        Args:
            resource_id: Unique identifier for the resource allocation
            cpu_memory_gb: CPU memory to allocate
            gpu_memory_gb: GPU memory to allocate

        Returns:
            True if allocation successful
        """
        try:
            # Validate available resources
            snapshot = self.monitor.get_current_snapshot()

            if cpu_memory_gb > snapshot.cpu_memory_available_gb:
                raise InsufficientResourcesError(
                    f"Cannot allocate {cpu_memory_gb}GB CPU memory, "
                    f"only {snapshot.cpu_memory_available_gb:.1f}GB available"
                )

            if gpu_memory_gb > 0 and snapshot.gpu_available:
                available_gpu = snapshot.gpu_memory_total_gb - snapshot.gpu_memory_used_gb
                if gpu_memory_gb > available_gpu:
                    raise InsufficientResourcesError(
                        f"Cannot allocate {gpu_memory_gb}GB GPU memory, "
                        f"only {available_gpu:.1f}GB available"
                    )

            # Track allocation
            self._allocated_resources[resource_id] = {
                "cpu_memory_gb": cpu_memory_gb,
                "gpu_memory_gb": gpu_memory_gb,
                "timestamp": datetime.now(),
            }

            logger.info(
                f"Allocated resources for '{resource_id}': "
                f"CPU {cpu_memory_gb}GB, GPU {gpu_memory_gb}GB"
            )

            return True

        except Exception as e:
            logger.error(f"Resource allocation failed: {e}")
            return False

    def release_resources(self, resource_id: str):
        """Release allocated resources."""
        if resource_id in self._allocated_resources:
            allocation = self._allocated_resources.pop(resource_id)
            logger.info(
                f"Released resources for '{resource_id}': "
                f"CPU {allocation['cpu_memory_gb']}GB, "
                f"GPU {allocation['gpu_memory_gb']}GB"
            )

            # Trigger cleanup if GPU resources were released
            if allocation["gpu_memory_gb"] > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def track_model_cache(self, model_id: str, size_gb: float):
        """Track a model in cache for automatic eviction."""
        self._model_cache_tracker[model_id] = (size_gb, datetime.now())
        logger.debug(f"Tracking cached model '{model_id}' ({size_gb}GB)")

    def _emergency_cleanup(self):
        """Emergency cleanup when memory pressure is high."""
        logger.warning("Initiating emergency memory cleanup")

        # Force garbage collection
        gc.collect()

        # Clear PyTorch caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # TODO: Implement model eviction based on LRU
        # This would require integration with ModelService

    def _emergency_gpu_cleanup(self):
        """Emergency GPU cleanup."""
        logger.warning("Initiating emergency GPU cleanup")

        if torch.cuda.is_available():
            # Clear all caches
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

            # Force garbage collection
            gc.collect()

    def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource usage report."""
        snapshot = self.monitor.get_current_snapshot()
        avg_usage = self.monitor.get_average_usage(window_seconds=300)  # 5 min average
        recent_alerts = self.monitor.get_recent_alerts()

        report = {
            "timestamp": snapshot.timestamp.isoformat(),
            "current": {
                "cpu": {
                    "percent": snapshot.cpu_percent,
                    "memory_used_gb": snapshot.cpu_memory_used_gb,
                    "memory_available_gb": snapshot.cpu_memory_available_gb,
                    "memory_percent": snapshot.cpu_memory_percent,
                },
                "gpu": {
                    "available": snapshot.gpu_available,
                    "memory_used_gb": snapshot.gpu_memory_used_gb,
                    "memory_total_gb": snapshot.gpu_memory_total_gb,
                    "memory_percent": snapshot.gpu_memory_percent,
                    "utilization": snapshot.gpu_utilization,
                    "temperature": snapshot.gpu_temperature,
                },
                "process": {
                    "memory_gb": snapshot.process_memory_gb,
                    "gpu_memory_gb": snapshot.process_gpu_memory_gb,
                },
            },
            "average_5min": avg_usage,
            "allocated_resources": dict(self._allocated_resources),
            "cached_models": dict(self._model_cache_tracker),
            "alerts": {
                "total": len(recent_alerts),
                "critical": len([a for a in recent_alerts if a.severity == "critical"]),
                "warnings": len([a for a in recent_alerts if a.severity == "warning"]),
                "recent": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "severity": a.severity,
                        "type": a.resource_type,
                        "message": a.message,
                        "value": a.current_value,
                        "threshold": a.threshold,
                    }
                    for a in recent_alerts[-5:]  # Last 5 alerts
                ],
            },
            "health_status": self._calculate_health_status(snapshot, recent_alerts),
        }

        return report

    def _calculate_health_status(
        self, snapshot: ResourceSnapshot, alerts: List[ResourceAlert]
    ) -> str:
        """Calculate overall system health status."""
        critical_alerts = [
            a
            for a in alerts
            if a.severity == "critical" and (datetime.now() - a.timestamp).seconds < 300
        ]

        if critical_alerts:
            return "critical"
        elif snapshot.cpu_memory_percent > 90 or snapshot.gpu_memory_percent > 90:
            return "warning"
        elif snapshot.cpu_memory_percent > 70 or snapshot.gpu_memory_percent > 70:
            return "moderate"
        else:
            return "healthy"

    def shutdown(self):
        """Shutdown resource manager."""
        self.monitor.stop_monitoring()
        logger.info("Resource manager shutdown complete")


def resource_monitored(func):
    """Decorator to monitor resource usage during function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get resource manager from first arg if it has one
        resource_manager = None
        if hasattr(args[0], "resource_manager"):
            resource_manager = args[0].resource_manager

        if resource_manager:
            # Take snapshot before
            before = resource_manager.monitor.get_current_snapshot()

        start_time = time.time()

        try:
            result = func(*args, **kwargs)

            if resource_manager:
                # Take snapshot after
                after = resource_manager.monitor.get_current_snapshot()

                # Log resource usage
                cpu_mem_diff = after.cpu_memory_used_gb - before.cpu_memory_used_gb
                gpu_mem_diff = after.gpu_memory_used_gb - before.gpu_memory_used_gb

                logger.info(
                    f"{func.__name__} resource usage: "
                    f"CPU memory: {cpu_mem_diff:+.2f}GB, "
                    f"GPU memory: {gpu_mem_diff:+.2f}GB, "
                    f"Duration: {time.time() - start_time:.2f}s"
                )

            return result

        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            raise

    return wrapper


class ResourceOptimizer:
    """Provides resource optimization recommendations."""

    @staticmethod
    def get_optimal_batch_size(model_size: str, available_gpu_memory_gb: float) -> int:
        """Get optimal batch size based on model and available memory."""
        # Conservative estimates to avoid OOM
        memory_per_sample_gb = {
            "small": 0.5,
            "medium": 1.0,
            "large": 2.0,
        }

        size_key = "medium"  # default
        for key in ["small", "medium", "large"]:
            if key in model_size.lower():
                size_key = key
                break

        per_sample = memory_per_sample_gb.get(size_key, 1.0)

        # Leave 20% headroom
        usable_memory = available_gpu_memory_gb * 0.8

        optimal_batch_size = max(1, int(usable_memory / per_sample))

        return optimal_batch_size

    @staticmethod
    def get_optimization_suggestions(resource_report: Dict[str, Any]) -> List[str]:
        """Get optimization suggestions based on resource usage."""
        suggestions = []

        current = resource_report.get("current", {})

        # CPU memory suggestions
        cpu_mem_percent = current.get("cpu", {}).get("memory_percent", 0)
        if cpu_mem_percent > 80:
            suggestions.append("High CPU memory usage detected:")
            suggestions.append("- Consider reducing model cache size")
            suggestions.append("- Enable model offloading to disk")
            suggestions.append("- Use smaller model variants")

        # GPU memory suggestions
        gpu = current.get("gpu", {})
        if gpu.get("available", False):
            gpu_mem_percent = gpu.get("memory_percent", 0)
            if gpu_mem_percent > 80:
                suggestions.append("High GPU memory usage detected:")
                suggestions.append("- Enable gradient checkpointing")
                suggestions.append("- Reduce batch size")
                suggestions.append("- Use mixed precision training")
                suggestions.append("- Clear GPU cache more frequently")

            # Temperature suggestions
            if gpu.get("temperature", 0) > 80:
                suggestions.append("High GPU temperature detected:")
                suggestions.append("- Reduce GPU utilization")
                suggestions.append("- Check cooling system")
                suggestions.append("- Consider workload distribution")

        # Alert-based suggestions
        alerts = resource_report.get("alerts", {})
        if alerts.get("critical", 0) > 0:
            suggestions.append("Critical resource alerts detected - immediate action required")

        return suggestions
