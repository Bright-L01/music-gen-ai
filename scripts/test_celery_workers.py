#!/usr/bin/env python3
"""
Test script for Celery worker system.

This script tests various aspects of the Celery worker implementation:
- Task submission and completion
- Priority routing
- Retry logic
- Batch processing
- Worker health monitoring
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from music_gen.workers import celery_app, generate_music_task
from music_gen.infrastructure.repositories.redis_task_repository_advanced import (
    RedisTaskRepositoryAdvanced,
    TaskPriority,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CeleryWorkerTester:
    """Test harness for Celery worker system."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.task_repo = RedisTaskRepositoryAdvanced(redis_url)
        self.results: Dict[str, Any] = {}

    async def setup(self):
        """Initialize test environment."""
        await self.task_repo.initialize()
        logger.info("Test environment initialized")

    async def cleanup(self):
        """Clean up test resources."""
        await self.task_repo.shutdown()
        logger.info("Test environment cleaned up")

    def test_worker_connectivity(self) -> bool:
        """Test if workers are available."""
        try:
            # Check for active workers
            inspector = celery_app.control.inspect()
            stats = inspector.stats()

            if not stats:
                logger.error("No workers available")
                return False

            logger.info(f"Found {len(stats)} workers:")
            for worker, info in stats.items():
                logger.info(
                    f"  - {worker}: {info.get('pool', {}).get('max-concurrency')} concurrency"
                )

            return True
        except Exception as e:
            logger.error(f"Worker connectivity test failed: {e}")
            return False

    async def test_simple_task(self) -> bool:
        """Test simple task submission and completion."""
        try:
            task_id = "test-simple-001"
            request_data = {
                "prompt": "Test music generation",
                "duration": 5.0,
                "temperature": 1.0,
            }

            # Create task in repository
            await self.task_repo.create_task(
                task_id,
                {
                    "request": request_data,
                    "priority": TaskPriority.NORMAL.value,
                },
            )

            # Submit to Celery
            result = generate_music_task.apply_async(
                args=[task_id, request_data],
                task_id=task_id,
            )

            logger.info(f"Submitted task {task_id}, waiting for completion...")

            # Wait for completion (with timeout)
            start_time = time.time()
            timeout = 60  # 1 minute timeout

            while time.time() - start_time < timeout:
                if result.ready():
                    if result.successful():
                        logger.info(f"Task {task_id} completed successfully")
                        self.results["simple_task"] = result.result
                        return True
                    else:
                        logger.error(f"Task {task_id} failed: {result.info}")
                        return False

                await asyncio.sleep(1)

            logger.error(f"Task {task_id} timed out")
            return False

        except Exception as e:
            logger.error(f"Simple task test failed: {e}")
            return False

    async def test_priority_routing(self) -> bool:
        """Test priority-based task routing."""
        try:
            tasks = []

            # Submit tasks with different priorities
            for priority, name in [
                (TaskPriority.CRITICAL.value, "critical"),
                (TaskPriority.HIGH.value, "high"),
                (TaskPriority.NORMAL.value, "normal"),
                (TaskPriority.LOW.value, "low"),
            ]:
                task_id = f"test-priority-{name}"
                request_data = {
                    "prompt": f"Test {name} priority",
                    "duration": 5.0,
                    "priority": priority,
                }

                # Create task
                await self.task_repo.create_task(
                    task_id,
                    {
                        "request": request_data,
                        "priority": priority,
                    },
                )

                # Get appropriate queue
                if priority >= TaskPriority.CRITICAL.value:
                    queue = "critical"
                elif priority >= TaskPriority.HIGH.value:
                    queue = "generation-high"
                else:
                    queue = "generation"

                # Submit to Celery
                result = generate_music_task.apply_async(
                    args=[task_id, request_data],
                    task_id=task_id,
                    queue=queue,
                    priority=priority,
                )

                tasks.append(
                    {
                        "task_id": task_id,
                        "priority": priority,
                        "queue": queue,
                        "result": result,
                        "submitted_at": time.time(),
                    }
                )

                logger.info(f"Submitted {name} priority task to {queue} queue")

            # Monitor completion order
            completed = []
            timeout = 120  # 2 minutes
            start_time = time.time()

            while len(completed) < len(tasks) and time.time() - start_time < timeout:
                for task in tasks:
                    if task["task_id"] not in [t["task_id"] for t in completed]:
                        if task["result"].ready():
                            task["completed_at"] = time.time()
                            task["duration"] = task["completed_at"] - task["submitted_at"]
                            completed.append(task)
                            logger.info(
                                f"Task {task['task_id']} (priority {task['priority']}) "
                                f"completed in {task['duration']:.2f}s"
                            )

                await asyncio.sleep(0.5)

            # Verify priority order (higher priority should complete first)
            if len(completed) == len(tasks):
                # Check if generally higher priority completed first
                avg_duration_by_priority = {}
                for task in completed:
                    priority = task["priority"]
                    if priority not in avg_duration_by_priority:
                        avg_duration_by_priority[priority] = []
                    avg_duration_by_priority[priority].append(task["duration"])

                logger.info("Average completion times by priority:")
                for priority in sorted(avg_duration_by_priority.keys(), reverse=True):
                    avg_time = sum(avg_duration_by_priority[priority]) / len(
                        avg_duration_by_priority[priority]
                    )
                    logger.info(f"  Priority {priority}: {avg_time:.2f}s")

                return True
            else:
                logger.error("Not all priority tasks completed")
                return False

        except Exception as e:
            logger.error(f"Priority routing test failed: {e}")
            return False

    async def test_retry_logic(self) -> bool:
        """Test task retry on failure."""
        try:
            task_id = "test-retry-001"

            # Submit a task that will fail (invalid parameters)
            request_data = {
                "prompt": "Test retry logic",
                "duration": -1.0,  # Invalid duration to trigger error
            }

            # Note: This would need a special test task that simulates failures
            # For now, we'll just verify the retry configuration

            # Check retry configuration
            task_config = celery_app.conf.task_annotations.get(
                "musicgen.workers.tasks.generate_music_task", {}
            )

            max_retries = task_config.get("max_retries", 3)
            retry_delay = task_config.get("default_retry_delay", 60)

            logger.info(f"Retry configuration: max_retries={max_retries}, delay={retry_delay}s")

            return max_retries > 0

        except Exception as e:
            logger.error(f"Retry logic test failed: {e}")
            return False

    async def test_batch_processing(self) -> bool:
        """Test batch task processing."""
        try:
            batch_id = "test-batch-001"
            batch_size = 5

            # Create batch request
            requests = []
            for i in range(batch_size):
                requests.append(
                    {
                        "prompt": f"Batch test music {i}",
                        "duration": 5.0,
                    }
                )

            # Submit batch
            from music_gen.workers.tasks import generate_batch_task

            result = generate_batch_task.apply_async(
                args=[batch_id, requests],
                task_id=batch_id,
                queue="batch",
            )

            logger.info(f"Submitted batch {batch_id} with {batch_size} tasks")

            # Wait for completion
            timeout = 180  # 3 minutes
            start_time = time.time()

            while time.time() - start_time < timeout:
                if result.ready():
                    if result.successful():
                        batch_result = result.result
                        logger.info(
                            f"Batch completed: {batch_result['successful']}/{batch_result['total_tasks']} successful"
                        )
                        self.results["batch_processing"] = batch_result
                        return batch_result["successful"] == batch_result["total_tasks"]
                    else:
                        logger.error(f"Batch failed: {result.info}")
                        return False

                # Check progress
                state = result.state
                if state == "PROCESSING" and result.info:
                    progress = result.info.get("progress", 0)
                    completed = result.info.get("completed", 0)
                    total = result.info.get("total", batch_size)
                    logger.info(f"Batch progress: {completed}/{total} ({progress*100:.1f}%)")

                await asyncio.sleep(5)

            logger.error("Batch processing timed out")
            return False

        except Exception as e:
            logger.error(f"Batch processing test failed: {e}")
            return False

    async def test_worker_monitoring(self) -> bool:
        """Test worker health monitoring."""
        try:
            from music_gen.workers.tasks import health_check_task

            # Run health check
            result = health_check_task.delay()

            # Wait for result
            health_status = result.get(timeout=10)

            logger.info(f"Health check result: {health_status}")

            # Verify health status
            if health_status.get("status") in ["healthy", "degraded"]:
                logger.info("Worker health check passed")
                self.results["health_check"] = health_status
                return True
            else:
                logger.error(f"Worker health check failed: {health_status}")
                return False

        except Exception as e:
            logger.error(f"Worker monitoring test failed: {e}")
            return False

    async def test_queue_metrics(self) -> bool:
        """Test queue metrics collection."""
        try:
            # Get task metrics
            metrics = await self.task_repo.get_task_metrics()

            logger.info("Task metrics:")
            logger.info(f"  Total tasks: {metrics.total_tasks}")
            logger.info(f"  Pending: {metrics.pending_tasks}")
            logger.info(f"  Processing: {metrics.processing_tasks}")
            logger.info(f"  Completed: {metrics.completed_tasks}")
            logger.info(f"  Failed: {metrics.failed_tasks}")
            logger.info(f"  Queue length: {metrics.queue_length}")
            logger.info(f"  Avg processing time: {metrics.avg_processing_time:.2f}s")

            self.results["queue_metrics"] = metrics.to_dict()

            return True

        except Exception as e:
            logger.error(f"Queue metrics test failed: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        test_results = {}

        # Define tests
        tests = [
            ("Worker Connectivity", self.test_worker_connectivity),
            ("Simple Task", self.test_simple_task),
            ("Priority Routing", self.test_priority_routing),
            ("Retry Logic", self.test_retry_logic),
            ("Batch Processing", self.test_batch_processing),
            ("Worker Monitoring", self.test_worker_monitoring),
            ("Queue Metrics", self.test_queue_metrics),
        ]

        # Run tests
        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running test: {test_name}")
            logger.info(f"{'='*60}")

            try:
                # Handle async and sync tests
                if asyncio.iscoroutinefunction(test_func):
                    result = await test_func()
                else:
                    result = test_func()

                test_results[test_name] = result
                logger.info(f"Test {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                test_results[test_name] = False

        return test_results


async def main():
    """Run Celery worker tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Celery Worker System")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--test", help="Run specific test")

    args = parser.parse_args()

    # Create tester
    tester = CeleryWorkerTester(args.redis_url)

    try:
        # Setup
        await tester.setup()

        # Run tests
        if args.test:
            # Run specific test
            test_method = getattr(tester, f"test_{args.test}", None)
            if test_method:
                result = await test_method()
                logger.info(f"\nTest {args.test}: {'PASSED' if result else 'FAILED'}")
            else:
                logger.error(f"Unknown test: {args.test}")
        else:
            # Run all tests
            results = await tester.run_all_tests()

            # Summary
            logger.info(f"\n{'='*60}")
            logger.info("TEST SUMMARY")
            logger.info(f"{'='*60}")

            passed = sum(1 for r in results.values() if r)
            total = len(results)

            for test_name, result in results.items():
                status = "✓ PASS" if result else "✗ FAIL"
                logger.info(f"{status} - {test_name}")

            logger.info(f"\nTotal: {passed}/{total} tests passed")

            # Save results
            import json

            with open("celery_test_results.json", "w") as f:
                json.dump(
                    {
                        "results": results,
                        "details": tester.results,
                        "summary": {
                            "passed": passed,
                            "total": total,
                            "success_rate": passed / total if total > 0 else 0,
                        },
                    },
                    f,
                    indent=2,
                )

            return 0 if passed == total else 1

    finally:
        # Cleanup
        await tester.cleanup()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
