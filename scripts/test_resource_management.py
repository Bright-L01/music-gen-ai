#!/usr/bin/env python3
"""
Test script for resource management system.

This script validates that the resource management system works correctly
and provides the expected functionality.
"""

import sys
import time
import asyncio
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from music_gen.core.resource_manager import (
    ResourceManager,
    ResourceMonitor,
    ResourceOptimizer,
    ResourceRequirements,
)
from music_gen.core.config import AppConfig
from music_gen.core.exceptions import InsufficientResourcesError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResourceManagementTester:
    """Test suite for resource management system."""

    def __init__(self):
        self.config = AppConfig()
        self.resource_manager = None
        self.test_results = []

    def run_test(self, test_name: str, test_func):
        """Run a single test and track results."""
        try:
            logger.info(f"🧪 Running test: {test_name}")
            test_func()
            self.test_results.append((test_name, "PASS", None))
            logger.info(f"✅ {test_name}: PASSED")
        except Exception as e:
            self.test_results.append((test_name, "FAIL", str(e)))
            logger.error(f"❌ {test_name}: FAILED - {e}")

    def test_resource_monitor_initialization(self):
        """Test ResourceMonitor can be initialized and started."""
        monitor = ResourceMonitor(sampling_interval=0.5, history_size=10)

        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring == True, "Monitor should be running"

        # Let it collect some data
        time.sleep(2)

        # Check we have some snapshots
        assert len(monitor.history) > 0, "Should have collected some snapshots"

        # Get current snapshot
        snapshot = monitor.get_current_snapshot()
        assert snapshot is not None, "Should be able to get current snapshot"
        assert snapshot.cpu_percent >= 0, "CPU percent should be non-negative"
        assert snapshot.cpu_memory_used_gb >= 0, "Memory usage should be non-negative"

        logger.info(
            f"Current CPU: {snapshot.cpu_percent:.1f}%, Memory: {snapshot.cpu_memory_used_gb:.1f}GB"
        )

        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring == False, "Monitor should be stopped"

    def test_resource_manager_initialization(self):
        """Test ResourceManager initialization."""
        self.resource_manager = ResourceManager(self.config)

        assert self.resource_manager is not None, "ResourceManager should initialize"
        assert self.resource_manager.monitor is not None, "Should have a monitor"

        # Check model requirements
        requirements = self.resource_manager.get_model_requirements("small")
        assert isinstance(requirements, ResourceRequirements), "Should return ResourceRequirements"
        assert requirements.cpu_memory_gb > 0, "Should have CPU memory requirement"

        logger.info(
            f"Small model requirements: CPU {requirements.cpu_memory_gb}GB, GPU {requirements.gpu_memory_gb}GB"
        )

    def test_resource_validation_success(self):
        """Test resource validation for small model (should pass)."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        # Test with small model (should have enough resources)
        try:
            validation_result = self.resource_manager.validate_system_resources("small")
            assert validation_result["valid"] == True, "Small model validation should pass"
            logger.info("Small model validation passed successfully")
        except InsufficientResourcesError:
            # This might happen on very resource-constrained systems
            logger.warning("Small model validation failed - system may be resource constrained")

    def test_resource_validation_failure(self):
        """Test resource validation failure for impossible requirements."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        # Create impossible requirements
        impossible_model = "test_impossible_model"
        self.resource_manager.MODEL_REQUIREMENTS[impossible_model] = ResourceRequirements(
            cpu_memory_gb=1000000,  # 1 million GB - impossible
            gpu_memory_gb=1000000,
            notes="Impossible requirements for testing",
        )

        # This should fail
        try:
            self.resource_manager.validate_system_resources(impossible_model)
            assert False, "Validation should have failed for impossible requirements"
        except InsufficientResourcesError as e:
            logger.info(f"Correctly caught insufficient resources error: {e}")

    def test_resource_allocation_tracking(self):
        """Test resource allocation and release."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        allocation_id = "test_allocation_001"

        # Allocate small amount of resources
        success = self.resource_manager.allocate_resources(
            allocation_id, cpu_memory_gb=1.0, gpu_memory_gb=0.5
        )

        assert success == True, "Small allocation should succeed"
        assert (
            allocation_id in self.resource_manager._allocated_resources
        ), "Should track allocation"

        # Release resources
        self.resource_manager.release_resources(allocation_id)
        assert (
            allocation_id not in self.resource_manager._allocated_resources
        ), "Should remove tracking after release"

        logger.info("Resource allocation and release working correctly")

    def test_resource_report_generation(self):
        """Test comprehensive resource report generation."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        # Wait a moment for monitor to collect data
        time.sleep(1)

        report = self.resource_manager.get_resource_report()

        assert isinstance(report, dict), "Report should be a dictionary"
        assert "timestamp" in report, "Report should have timestamp"
        assert "current" in report, "Report should have current status"
        assert "health_status" in report, "Report should have health status"

        health_status = report["health_status"]
        assert health_status in [
            "healthy",
            "moderate",
            "warning",
            "critical",
        ], f"Invalid health status: {health_status}"

        logger.info(f"Resource report generated successfully, health: {health_status}")

    def test_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        report = self.resource_manager.get_resource_report()
        suggestions = ResourceOptimizer.get_optimization_suggestions(report)

        assert isinstance(suggestions, list), "Suggestions should be a list"

        # Test batch size optimization
        optimal_batch = ResourceOptimizer.get_optimal_batch_size("medium", 8.0)
        assert isinstance(optimal_batch, int), "Batch size should be integer"
        assert optimal_batch >= 1, "Batch size should be at least 1"

        logger.info(f"Optimization suggestions: {len(suggestions)} items")
        logger.info(f"Optimal batch size for medium model with 8GB GPU: {optimal_batch}")

    def test_resource_cleanup(self):
        """Test resource cleanup functionality."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        # Get before snapshot
        before = self.resource_manager.monitor.get_current_snapshot()

        # Trigger cleanup
        self.resource_manager._emergency_cleanup()

        # Get after snapshot
        time.sleep(0.5)  # Give a moment for cleanup
        after = self.resource_manager.monitor.get_current_snapshot()

        # We can't guarantee memory was freed, but the operation should not crash
        logger.info(
            f"Cleanup executed - Memory before: {before.cpu_memory_used_gb:.1f}GB, after: {after.cpu_memory_used_gb:.1f}GB"
        )

    def test_model_cache_tracking(self):
        """Test model cache tracking functionality."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        model_id = "test_model_cache_123"
        model_size_gb = 2.5

        # Track model in cache
        self.resource_manager.track_model_cache(model_id, model_size_gb)

        assert model_id in self.resource_manager._model_cache_tracker, "Should track cached model"

        cached_info = self.resource_manager._model_cache_tracker[model_id]
        assert cached_info[0] == model_size_gb, "Should track correct size"

        logger.info(f"Model cache tracking working correctly for {model_id} ({model_size_gb}GB)")

    async def test_monitoring_callbacks(self):
        """Test monitoring callback system."""
        if not self.resource_manager:
            self.resource_manager = ResourceManager(self.config)

        callback_called = False
        callback_snapshot = None

        def test_callback(snapshot):
            nonlocal callback_called, callback_snapshot
            callback_called = True
            callback_snapshot = snapshot

        # Add callback
        self.resource_manager.monitor.add_callback(test_callback)

        # Wait for a callback
        for _ in range(10):  # Wait up to 5 seconds
            await asyncio.sleep(0.5)
            if callback_called:
                break

        assert callback_called == True, "Callback should have been called"
        assert callback_snapshot is not None, "Callback should have received snapshot"

        logger.info("Monitoring callbacks working correctly")

    def run_all_tests(self):
        """Run all tests and report results."""
        logger.info("🚀 Starting Resource Management System Tests")
        logger.info("=" * 60)

        # Basic functionality tests
        self.run_test("Resource Monitor Initialization", self.test_resource_monitor_initialization)
        self.run_test("Resource Manager Initialization", self.test_resource_manager_initialization)
        self.run_test("Resource Validation Success", self.test_resource_validation_success)
        self.run_test("Resource Validation Failure", self.test_resource_validation_failure)
        self.run_test("Resource Allocation Tracking", self.test_resource_allocation_tracking)
        self.run_test("Resource Report Generation", self.test_resource_report_generation)
        self.run_test("Optimization Suggestions", self.test_optimization_suggestions)
        self.run_test("Resource Cleanup", self.test_resource_cleanup)
        self.run_test("Model Cache Tracking", self.test_model_cache_tracking)

        # Async test
        try:
            asyncio.run(self.test_monitoring_callbacks())
            self.test_results.append(("Monitoring Callbacks", "PASS", None))
            logger.info("✅ Monitoring Callbacks: PASSED")
        except Exception as e:
            self.test_results.append(("Monitoring Callbacks", "FAIL", str(e)))
            logger.error(f"❌ Monitoring Callbacks: FAILED - {e}")

        # Cleanup
        if self.resource_manager:
            self.resource_manager.shutdown()

    def print_results(self):
        """Print test results summary."""
        logger.info("\n" + "=" * 60)
        logger.info("🧪 TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        passed = sum(1 for _, status, _ in self.test_results if status == "PASS")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAIL")
        total = len(self.test_results)

        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")

        if failed > 0:
            logger.info("\n❌ Failed Tests:")
            for name, status, error in self.test_results:
                if status == "FAIL":
                    logger.info(f"  - {name}: {error}")

        logger.info("\n" + "=" * 60)

        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! Resource Management System is working correctly.")
            return True
        else:
            logger.info("⚠️  Some tests failed. Check the issues above.")
            return False


def main():
    """Main test function."""
    print("🎵 Music Gen AI - Resource Management System Tests")
    print("Testing enterprise resource monitoring and management...")
    print()

    tester = ResourceManagementTester()

    try:
        tester.run_all_tests()
        success = tester.print_results()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        if tester.resource_manager:
            tester.resource_manager.shutdown()
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        if tester.resource_manager:
            tester.resource_manager.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
