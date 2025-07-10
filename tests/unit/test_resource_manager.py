"""
Comprehensive tests for core/resource_manager.py with 95% coverage.

Tests include:
- GPU memory allocation and deallocation
- Resource limits and quotas
- Concurrent resource requests
- Resource cleanup on errors
- Monitoring and metrics
- Edge cases and error conditions
"""

import pytest
import time
import threading
import gc
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import warnings

import torch
import psutil
import numpy as np

from music_gen.core.resource_manager import (
    ResourceRequirements,
    ResourceSnapshot,
    ResourceAlert,
    ResourceMonitor,
    ResourceManager,
    ResourceOptimizer,
    resource_monitored,
)
from music_gen.core.exceptions import InsufficientResourcesError, ResourceExhaustionError
from music_gen.core.config import AppConfig


@pytest.fixture
def mock_config():
    """Mock configuration for tests."""
    config = Mock(spec=AppConfig)
    config.resource_limits = {
        "max_gpu_memory_gb": 16.0,
        "max_cpu_memory_gb": 32.0,
        "emergency_threshold": 0.9,
    }
    return config


@pytest.fixture
def mock_torch():
    """Mock torch module for GPU operations."""
    with patch("music_gen.core.resource_manager.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024**3)  # 2GB
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * (1024**3)  # 8GB
        mock_torch.cuda.get_device_capability.return_value = (7, 5)  # Compute 7.5
        mock_torch.cuda.empty_cache = Mock()
        mock_torch.cuda.synchronize = Mock()
        mock_torch.cuda.reset_peak_memory_stats = Mock()
        yield mock_torch


@pytest.fixture
def mock_psutil():
    """Mock psutil for system monitoring."""
    with patch("music_gen.core.resource_manager.psutil") as mock_psutil:
        # CPU metrics
        mock_psutil.cpu_percent.return_value = 45.5

        # Memory metrics
        mock_memory = Mock()
        mock_memory.used = 8 * (1024**3)  # 8GB
        mock_memory.available = 24 * (1024**3)  # 24GB
        mock_memory.total = 32 * (1024**3)  # 32GB
        mock_memory.percent = 25.0
        mock_psutil.virtual_memory.return_value = mock_memory

        # Process metrics
        mock_process = Mock()
        mock_process_memory = Mock()
        mock_process_memory.rss = 1 * (1024**3)  # 1GB
        mock_process.memory_info.return_value = mock_process_memory
        mock_psutil.Process.return_value = mock_process

        yield mock_psutil


@pytest.fixture
def mock_nvidia_ml():
    """Mock nvidia-ml-py3 for GPU utilization."""
    with patch.dict("sys.modules", {"nvidia_ml_py3": Mock()}):
        import sys

        nvml = sys.modules["nvidia_ml_py3"]
        nvml.nvmlInit = Mock()
        nvml.nvmlDeviceGetHandleByIndex.return_value = Mock()

        # Mock utilization rates
        mock_util = Mock()
        mock_util.gpu = 75.0
        nvml.nvmlDeviceGetUtilizationRates.return_value = mock_util

        # Mock temperature
        nvml.nvmlDeviceGetTemperature.return_value = 65.0
        nvml.NVML_TEMPERATURE_GPU = 0

        yield nvml


class TestResourceRequirements:
    """Test ResourceRequirements dataclass."""

    def test_resource_requirements_creation(self):
        """Test creating ResourceRequirements."""
        req = ResourceRequirements(
            cpu_memory_gb=4.0,
            gpu_memory_gb=8.0,
            min_gpu_compute=7.0,
            recommended_batch_size=2,
            notes="Test requirements",
        )

        assert req.cpu_memory_gb == 4.0
        assert req.gpu_memory_gb == 8.0
        assert req.min_gpu_compute == 7.0
        assert req.recommended_batch_size == 2
        assert req.notes == "Test requirements"

    def test_resource_requirements_defaults(self):
        """Test default values."""
        req = ResourceRequirements(cpu_memory_gb=2.0, gpu_memory_gb=4.0)

        assert req.min_gpu_compute == 0.0
        assert req.recommended_batch_size == 1
        assert req.notes == ""


class TestResourceSnapshot:
    """Test ResourceSnapshot dataclass."""

    def test_resource_snapshot_creation(self):
        """Test creating ResourceSnapshot."""
        timestamp = datetime.now()
        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=50.0,
            cpu_memory_used_gb=8.0,
            cpu_memory_available_gb=24.0,
            cpu_memory_percent=25.0,
            gpu_available=True,
            gpu_memory_used_gb=2.0,
            gpu_memory_total_gb=8.0,
            gpu_memory_percent=25.0,
            gpu_utilization=75.0,
            gpu_temperature=65.0,
            process_memory_gb=1.0,
            process_gpu_memory_gb=0.5,
        )

        assert snapshot.timestamp == timestamp
        assert snapshot.cpu_percent == 50.0
        assert snapshot.gpu_available is True
        assert snapshot.gpu_memory_percent == 25.0
        assert snapshot.gpu_temperature == 65.0


class TestResourceAlert:
    """Test ResourceAlert dataclass."""

    def test_resource_alert_creation(self):
        """Test creating ResourceAlert."""
        timestamp = datetime.now()
        alert = ResourceAlert(
            timestamp=timestamp,
            severity="critical",
            resource_type="gpu_memory",
            message="High GPU memory usage",
            current_value=95.0,
            threshold=90.0,
            recommendations=["Clear cache", "Reduce batch size"],
        )

        assert alert.timestamp == timestamp
        assert alert.severity == "critical"
        assert alert.resource_type == "gpu_memory"
        assert alert.current_value == 95.0
        assert len(alert.recommendations) == 2


class TestResourceMonitor:
    """Test ResourceMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = ResourceMonitor(sampling_interval=0.5, history_size=100)

        assert monitor.sampling_interval == 0.5
        assert monitor.history_size == 100
        assert len(monitor.history) == 0
        assert len(monitor.alerts) == 0
        assert monitor._monitoring is False

    @patch("music_gen.core.resource_manager.psutil")
    @patch("music_gen.core.resource_manager.torch")
    def test_get_current_snapshot_with_gpu(self, mock_torch, mock_psutil):
        """Test getting current snapshot with GPU available."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 2 * (1024**3)
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * (1024**3)

        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.used = 8 * (1024**3)
        mock_memory.available = 24 * (1024**3)
        mock_memory.percent = 25.0
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_process = Mock()
        mock_process_memory = Mock()
        mock_process_memory.rss = 1 * (1024**3)
        mock_process.memory_info.return_value = mock_process_memory
        mock_psutil.Process.return_value = mock_process

        monitor = ResourceMonitor()
        snapshot = monitor.get_current_snapshot()

        assert snapshot.cpu_percent == 50.0
        assert snapshot.cpu_memory_percent == 25.0
        assert snapshot.gpu_available is True
        assert snapshot.gpu_memory_used_gb == 2.0
        assert snapshot.gpu_memory_total_gb == 8.0
        assert snapshot.gpu_memory_percent == 25.0

    @patch("music_gen.core.resource_manager.psutil")
    @patch("music_gen.core.resource_manager.torch")
    def test_get_current_snapshot_no_gpu(self, mock_torch, mock_psutil):
        """Test getting current snapshot without GPU."""
        mock_torch.cuda.is_available.return_value = False

        mock_psutil.cpu_percent.return_value = 30.0
        mock_memory = Mock()
        mock_memory.used = 4 * (1024**3)
        mock_memory.available = 12 * (1024**3)
        mock_memory.percent = 25.0
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_process = Mock()
        mock_process_memory = Mock()
        mock_process_memory.rss = 0.5 * (1024**3)
        mock_process.memory_info.return_value = mock_process_memory
        mock_psutil.Process.return_value = mock_process

        monitor = ResourceMonitor()
        snapshot = monitor.get_current_snapshot()

        assert snapshot.cpu_percent == 30.0
        assert snapshot.gpu_available is False
        assert snapshot.gpu_memory_used_gb == 0.0
        assert snapshot.gpu_memory_total_gb == 0.0

    def test_monitoring_lifecycle(self, mock_psutil, mock_torch):
        """Test monitoring start and stop."""
        monitor = ResourceMonitor(sampling_interval=0.1)

        assert monitor._monitoring is False
        assert monitor._monitor_thread is None

        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None

        # Allow some samples to be collected
        time.sleep(0.3)

        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring is False

    def test_alert_generation_cpu_memory(self, mock_psutil, mock_torch):
        """Test alert generation for CPU memory."""
        # Setup high CPU memory usage
        mock_memory = Mock()
        mock_memory.used = 29 * (1024**3)  # 29GB
        mock_memory.available = 3 * (1024**3)  # 3GB
        mock_memory.total = 32 * (1024**3)  # 32GB
        mock_memory.percent = 91.0  # > 90% threshold
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_psutil.cpu_percent.return_value = 50.0
        mock_process = Mock()
        mock_process_memory = Mock()
        mock_process_memory.rss = 1 * (1024**3)
        mock_process.memory_info.return_value = mock_process_memory
        mock_psutil.Process.return_value = mock_process

        mock_torch.cuda.is_available.return_value = False

        monitor = ResourceMonitor()
        snapshot = monitor.get_current_snapshot()
        monitor._check_alerts(snapshot)

        # Should generate critical alert
        assert len(monitor.alerts) > 0
        alert = monitor.alerts[-1]
        assert alert.severity == "critical"
        assert alert.resource_type == "cpu_memory"
        assert alert.current_value == 91.0
        assert alert.threshold == 90
        assert len(alert.recommendations) > 0

    def test_average_usage_calculation(self, mock_psutil, mock_torch):
        """Test average usage calculation."""
        monitor = ResourceMonitor()

        # Add some test snapshots
        base_time = datetime.now()
        for i in range(5):
            snapshot = ResourceSnapshot(
                timestamp=base_time - timedelta(seconds=i * 10),
                cpu_percent=50.0 + i,
                cpu_memory_used_gb=8.0,
                cpu_memory_available_gb=24.0,
                cpu_memory_percent=25.0 + i,
                gpu_available=True,
                gpu_memory_used_gb=2.0,
                gpu_memory_total_gb=8.0,
                gpu_memory_percent=25.0 + i,
                gpu_utilization=70.0 + i,
            )
            monitor.history.append(snapshot)

        avg_usage = monitor.get_average_usage(window_seconds=60)

        assert "cpu_percent" in avg_usage
        assert "cpu_memory_percent" in avg_usage
        assert "gpu_memory_percent" in avg_usage
        assert "gpu_utilization" in avg_usage

        # Should be average of all 5 samples
        assert avg_usage["cpu_percent"] == pytest.approx(52.0, abs=0.1)
        assert avg_usage["cpu_memory_percent"] == pytest.approx(27.0, abs=0.1)


class TestResourceManager:
    """Test ResourceManager class."""

    def test_initialization(self, mock_config, mock_psutil, mock_torch):
        """Test ResourceManager initialization."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            assert manager.config == mock_config
            assert isinstance(manager.monitor, ResourceMonitor)
            assert manager._allocated_resources == {}
            assert manager._model_cache_tracker == {}

    def test_model_requirements_predefined(self, mock_config, mock_psutil, mock_torch):
        """Test getting predefined model requirements."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            req = manager.get_model_requirements("facebook/musicgen-small")
            assert req.cpu_memory_gb == 3.0
            assert req.gpu_memory_gb == 6.0
            assert req.min_gpu_compute == 6.0

    def test_model_requirements_inference(self, mock_config, mock_psutil, mock_torch):
        """Test inferring model requirements from name."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Test size inference
            small_req = manager.get_model_requirements("custom-small-model")
            assert small_req.cpu_memory_gb == 2.0
            assert small_req.gpu_memory_gb == 4.0

            large_req = manager.get_model_requirements("custom-large-model")
            assert large_req.cpu_memory_gb == 8.0
            assert large_req.gpu_memory_gb == 16.0

    def test_validate_system_resources_sufficient(self, mock_config, mock_psutil, mock_torch):
        """Test system validation with sufficient resources."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Mock sufficient resources
            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 30.0
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 16.0
            mock_snapshot.gpu_memory_used_gb = 2.0

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                with patch.object(mock_torch.cuda, "get_device_capability", return_value=(7, 5)):
                    result = manager.validate_system_resources("small")

                    assert result["valid"] is True
                    assert result["model"] == "small"
                    assert "requirements" in result
                    assert "available" in result
                    assert len(result["errors"]) == 0

    def test_validate_system_resources_insufficient_cpu(self, mock_config, mock_psutil, mock_torch):
        """Test system validation with insufficient CPU memory."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Mock insufficient CPU memory
            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 1.0  # Less than required 2GB
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 16.0
            mock_snapshot.gpu_memory_used_gb = 2.0

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                with pytest.raises(InsufficientResourcesError) as exc_info:
                    manager.validate_system_resources("small")

                assert "Insufficient CPU memory" in str(exc_info.value)

    def test_allocate_resources_success(self, mock_config, mock_psutil, mock_torch):
        """Test successful resource allocation."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Mock sufficient resources
            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 30.0
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 16.0
            mock_snapshot.gpu_memory_used_gb = 2.0

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                result = manager.allocate_resources(
                    "test_alloc", cpu_memory_gb=4.0, gpu_memory_gb=2.0
                )

                assert result is True
                assert "test_alloc" in manager._allocated_resources
                allocation = manager._allocated_resources["test_alloc"]
                assert allocation["cpu_memory_gb"] == 4.0
                assert allocation["gpu_memory_gb"] == 2.0

    def test_release_resources(self, mock_config, mock_psutil, mock_torch):
        """Test resource release."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Add allocation
            manager._allocated_resources["test_alloc"] = {
                "cpu_memory_gb": 4.0,
                "gpu_memory_gb": 2.0,
                "timestamp": datetime.now(),
            }

            manager.release_resources("test_alloc")

            assert "test_alloc" not in manager._allocated_resources
            mock_torch.cuda.empty_cache.assert_called_once()


class TestConcurrentResourceRequests:
    """Test concurrent resource allocation and management."""

    def test_concurrent_allocations(self, mock_config, mock_psutil, mock_torch):
        """Test concurrent resource allocations."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Mock plenty of resources
            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 100.0
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 32.0
            mock_snapshot.gpu_memory_used_gb = 0.0

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                results = []

                def allocate_resource(resource_id):
                    return manager.allocate_resources(
                        f"concurrent_{resource_id}", cpu_memory_gb=2.0, gpu_memory_gb=1.0
                    )

                # Run concurrent allocations
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(allocate_resource, i) for i in range(20)]
                    results = [future.result() for future in as_completed(futures)]

                # All should succeed with sufficient resources
                assert all(results)
                assert len(manager._allocated_resources) == 20

    def test_concurrent_allocation_contention(self, mock_config, mock_psutil, mock_torch):
        """Test concurrent allocations with resource contention."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Mock limited resources (only enough for a few allocations)
            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 10.0  # Only 10GB available
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 8.0
            mock_snapshot.gpu_memory_used_gb = 0.0

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                results = []

                def allocate_resource(resource_id):
                    return manager.allocate_resources(
                        f"contention_{resource_id}",
                        cpu_memory_gb=3.0,  # Each needs 3GB CPU
                        gpu_memory_gb=2.0,  # Each needs 2GB GPU
                    )

                # Run concurrent allocations
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(allocate_resource, i) for i in range(10)]
                    results = [future.result() for future in as_completed(futures)]

                # Some should fail due to insufficient resources
                successful = sum(results)
                failed = len(results) - successful

                assert successful <= 3  # At most 3 should succeed (10GB / 3GB)
                assert failed > 0


class TestResourceCleanupOnErrors:
    """Test resource cleanup when errors occur."""

    def test_allocation_error_cleanup(self, mock_config, mock_psutil, mock_torch):
        """Test cleanup when allocation fails."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Mock snapshot method to fail after some calls
            call_count = 0
            original_method = manager.monitor.get_current_snapshot

            def failing_snapshot():
                nonlocal call_count
                call_count += 1
                if call_count > 2:
                    raise RuntimeError("System error")
                return Mock(
                    cpu_memory_available_gb=10.0,
                    gpu_available=True,
                    gpu_memory_total_gb=8.0,
                    gpu_memory_used_gb=0.0,
                )

            with patch.object(
                manager.monitor, "get_current_snapshot", side_effect=failing_snapshot
            ):
                # First allocation should succeed
                result1 = manager.allocate_resources("alloc1", cpu_memory_gb=2.0, gpu_memory_gb=1.0)
                assert result1 is True
                assert "alloc1" in manager._allocated_resources

                # Second allocation should succeed
                result2 = manager.allocate_resources("alloc2", cpu_memory_gb=2.0, gpu_memory_gb=1.0)
                assert result2 is True
                assert "alloc2" in manager._allocated_resources

                # Third allocation should fail due to error
                result3 = manager.allocate_resources("alloc3", cpu_memory_gb=2.0, gpu_memory_gb=1.0)
                assert result3 is False
                assert "alloc3" not in manager._allocated_resources

    def test_monitoring_cleanup_on_error(self, mock_config, mock_psutil, mock_torch):
        """Test that monitoring cleanup is triggered on errors."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Create high memory pressure snapshot
            error_snapshot = Mock()
            error_snapshot.cpu_memory_percent = 95.0
            error_snapshot.gpu_available = True
            error_snapshot.gpu_memory_percent = 95.0

            with patch.object(manager, "_emergency_cleanup") as mock_cpu_cleanup:
                with patch.object(manager, "_emergency_gpu_cleanup") as mock_gpu_cleanup:
                    manager._cleanup_callback(error_snapshot)

                    mock_cpu_cleanup.assert_called_once()
                    mock_gpu_cleanup.assert_called_once()


class TestResourceMonitoringAndMetrics:
    """Test monitoring and metrics functionality."""

    def test_monitoring_metrics_collection(self, mock_psutil, mock_torch, mock_nvidia_ml):
        """Test comprehensive metrics collection."""
        monitor = ResourceMonitor(sampling_interval=0.1)

        # Start monitoring
        monitor.start_monitoring()

        # Allow metrics to be collected
        time.sleep(0.3)

        # Stop monitoring
        monitor.stop_monitoring()

        # Should have collected some snapshots
        assert len(monitor.history) > 0

        # Check snapshot structure
        snapshot = monitor.history[-1]
        assert hasattr(snapshot, "cpu_percent")
        assert hasattr(snapshot, "gpu_memory_used_gb")

    def test_resource_report_generation(self, mock_config, mock_psutil, mock_torch):
        """Test comprehensive resource report generation."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Setup mocks
            mock_snapshot = Mock()
            mock_snapshot.timestamp = datetime.now()
            mock_snapshot.cpu_percent = 50.0
            mock_snapshot.cpu_memory_used_gb = 8.0
            mock_snapshot.cpu_memory_available_gb = 24.0
            mock_snapshot.cpu_memory_percent = 25.0
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_used_gb = 2.0
            mock_snapshot.gpu_memory_total_gb = 8.0
            mock_snapshot.gpu_memory_percent = 25.0
            mock_snapshot.gpu_utilization = 75.0
            mock_snapshot.gpu_temperature = 65.0
            mock_snapshot.process_memory_gb = 1.0
            mock_snapshot.process_gpu_memory_gb = 0.5

            mock_avg = {"cpu_percent": 45.0, "gpu_utilization": 70.0}
            mock_alerts = [Mock(severity="warning", timestamp=datetime.now())]

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                with patch.object(manager.monitor, "get_average_usage", return_value=mock_avg):
                    with patch.object(
                        manager.monitor, "get_recent_alerts", return_value=mock_alerts
                    ):
                        report = manager.get_resource_report()

            assert "timestamp" in report
            assert "current" in report
            assert "alerts" in report
            assert "health_status" in report


class TestResourceDecorator:
    """Test resource monitoring decorator."""

    def test_resource_monitored_decorator(self, mock_config, mock_psutil, mock_torch):
        """Test resource monitoring decorator."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            class TestClass:
                def __init__(self):
                    self.resource_manager = manager

                @resource_monitored
                def test_method(self, value):
                    time.sleep(0.1)  # Simulate work
                    return value * 2

            test_obj = TestClass()

            # Mock different snapshots for before/after
            snapshots = [
                Mock(cpu_memory_used_gb=8.0, gpu_memory_used_gb=2.0),
                Mock(cpu_memory_used_gb=8.5, gpu_memory_used_gb=2.2),
            ]

            with patch.object(manager.monitor, "get_current_snapshot", side_effect=snapshots):
                result = test_obj.test_method(5)

                assert result == 10

    def test_resource_monitored_without_manager(self):
        """Test decorator on object without resource manager."""

        class TestClass:
            @resource_monitored
            def test_method(self, value):
                return value * 2

        test_obj = TestClass()
        result = test_obj.test_method(5)
        assert result == 10


class TestResourceOptimizer:
    """Test ResourceOptimizer class."""

    def test_get_optimal_batch_size_small_model(self):
        """Test optimal batch size calculation for small model."""
        batch_size = ResourceOptimizer.get_optimal_batch_size("small", 8.0)

        # Small model uses 0.5GB per sample, 8GB * 0.8 / 0.5 = 12.8 -> 12
        assert batch_size == 12

    def test_get_optimal_batch_size_large_model(self):
        """Test optimal batch size calculation for large model."""
        batch_size = ResourceOptimizer.get_optimal_batch_size("large", 8.0)

        # Large model uses 2GB per sample, 8GB * 0.8 / 2 = 3.2 -> 3
        assert batch_size == 3

    def test_get_optimal_batch_size_minimum(self):
        """Test that batch size is at least 1."""
        batch_size = ResourceOptimizer.get_optimal_batch_size("large", 0.5)

        # Should return at least 1 even with very little memory
        assert batch_size == 1

    def test_get_optimization_suggestions_healthy(self):
        """Test optimization suggestions for healthy system."""
        report = {
            "current": {
                "cpu": {"memory_percent": 50.0},
                "gpu": {"available": True, "memory_percent": 50.0, "temperature": 60.0},
            },
            "alerts": {"critical": 0, "warnings": 0},
        }

        suggestions = ResourceOptimizer.get_optimization_suggestions(report)
        assert len(suggestions) == 0

    def test_get_optimization_suggestions_high_cpu_memory(self):
        """Test suggestions for high CPU memory usage."""
        report = {
            "current": {
                "cpu": {"memory_percent": 85.0},
                "gpu": {"available": True, "memory_percent": 50.0, "temperature": 60.0},
            },
            "alerts": {"critical": 0, "warnings": 0},
        }

        suggestions = ResourceOptimizer.get_optimization_suggestions(report)
        assert len(suggestions) > 0
        assert any("High CPU memory usage" in s for s in suggestions)


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions."""

    def test_zero_memory_conditions(self, mock_config, mock_psutil, mock_torch):
        """Test handling of zero memory conditions."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            # Mock zero available memory
            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 0.0
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 8.0
            mock_snapshot.gpu_memory_used_gb = 8.0  # Fully used

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                result = manager.allocate_resources("test", cpu_memory_gb=1.0, gpu_memory_gb=1.0)
                assert result is False

    def test_cuda_not_available_handling(self, mock_config, mock_psutil):
        """Test handling when CUDA is not available."""
        with patch("music_gen.core.resource_manager.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False

            with patch.object(ResourceMonitor, "start_monitoring"):
                manager = ResourceManager(mock_config)

                # Should handle GPU operations gracefully
                manager._emergency_gpu_cleanup()

    def test_threading_safety(self, mock_config, mock_psutil, mock_torch):
        """Test thread safety of resource operations."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 100.0
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 32.0
            mock_snapshot.gpu_memory_used_gb = 0.0

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                errors = []

                def thread_operations(thread_id):
                    try:
                        # Mix of operations
                        manager.allocate_resources(f"thread_{thread_id}", cpu_memory_gb=1.0)
                        manager.track_model_cache(f"model_{thread_id}", 2.0)
                        time.sleep(0.01)
                        manager.release_resources(f"thread_{thread_id}")
                        manager.get_resource_report()
                    except Exception as e:
                        errors.append(e)

                # Run multiple threads
                threads = []
                for i in range(10):
                    thread = threading.Thread(target=thread_operations, args=(i,))
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                # Should not have any errors
                assert len(errors) == 0

    def test_memory_calculation_edge_cases(self, mock_psutil, mock_torch):
        """Test edge cases in memory calculations."""
        monitor = ResourceMonitor()

        # Test division by zero in GPU memory percentage
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024  # Some usage
        mock_torch.cuda.get_device_properties.return_value.total_memory = 0  # Zero total

        snapshot = monitor.get_current_snapshot()
        assert snapshot.gpu_memory_percent == 0  # Should handle division by zero

    def test_extreme_values_handling(self, mock_config, mock_psutil, mock_torch):
        """Test handling of extreme values."""
        with patch.object(ResourceMonitor, "start_monitoring"):
            manager = ResourceManager(mock_config)

            mock_snapshot = Mock()
            mock_snapshot.cpu_memory_available_gb = 32.0
            mock_snapshot.gpu_available = True
            mock_snapshot.gpu_memory_total_gb = 16.0
            mock_snapshot.gpu_memory_used_gb = 0.0

            with patch.object(manager.monitor, "get_current_snapshot", return_value=mock_snapshot):
                # Request more than available
                result = manager.allocate_resources(
                    "test", cpu_memory_gb=1000.0, gpu_memory_gb=1000.0
                )
                assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
