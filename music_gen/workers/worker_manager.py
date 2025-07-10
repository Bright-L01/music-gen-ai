"""
Worker manager for scaling and monitoring Celery workers.

This module provides utilities for managing worker lifecycle, scaling,
and monitoring worker health.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from celery import Celery
from celery.app.control import Control

logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for a worker instance."""

    name: str
    queues: List[str]
    concurrency: int = 1
    pool: str = "prefork"  # 'prefork', 'eventlet', 'gevent'
    max_tasks_per_child: int = 50
    autoscale: Optional[tuple] = None  # (max, min) workers
    priority_bias: Optional[Dict[str, float]] = None


class WorkerManager:
    """Manages Celery worker processes with scaling and monitoring."""

    def __init__(self, app: Celery, redis_url: str):
        """
        Initialize worker manager.

        Args:
            app: Celery application instance
            redis_url: Redis connection URL
        """
        self.app = app
        self.redis_url = redis_url
        self.control: Control = app.control
        self.workers: Dict[str, subprocess.Popen] = {}
        self.worker_configs: Dict[str, WorkerConfig] = {}
        self._running = False

    def add_worker_config(self, config: WorkerConfig):
        """Add worker configuration."""
        self.worker_configs[config.name] = config

    def start_worker(self, worker_name: str) -> bool:
        """
        Start a single worker process.

        Args:
            worker_name: Name of worker to start

        Returns:
            True if started successfully
        """
        if worker_name not in self.worker_configs:
            logger.error(f"Worker configuration not found: {worker_name}")
            return False

        if worker_name in self.workers:
            logger.warning(f"Worker already running: {worker_name}")
            return False

        config = self.worker_configs[worker_name]

        # Build celery worker command
        cmd = [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "music_gen.workers.celery_app",
            "worker",
            "--hostname",
            f"{worker_name}@%h",
            "--queues",
            ",".join(config.queues),
            "--concurrency",
            str(config.concurrency),
            "--pool",
            config.pool,
            "--max-tasks-per-child",
            str(config.max_tasks_per_child),
            "--loglevel",
            "INFO",
            "--without-gossip",  # Reduce overhead
            "--without-mingle",  # Reduce startup time
            "--without-heartbeat",  # Use Redis for heartbeat
        ]

        # Add autoscaling if configured
        if config.autoscale:
            cmd.extend(["--autoscale", f"{config.autoscale[0]},{config.autoscale[1]}"])

        # Start worker process
        try:
            env = os.environ.copy()
            env["CELERY_BROKER_URL"] = self.redis_url
            env["CELERY_RESULT_BACKEND"] = self.redis_url

            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.workers[worker_name] = process
            logger.info(f"Started worker: {worker_name} (PID: {process.pid})")

            return True

        except Exception as e:
            logger.error(f"Failed to start worker {worker_name}: {e}")
            return False

    def stop_worker(self, worker_name: str, timeout: int = 30) -> bool:
        """
        Stop a worker gracefully.

        Args:
            worker_name: Name of worker to stop
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if stopped successfully
        """
        if worker_name not in self.workers:
            logger.warning(f"Worker not running: {worker_name}")
            return False

        process = self.workers[worker_name]

        try:
            # Send warm shutdown signal
            self.control.shutdown(destination=[f"{worker_name}@*"])

            # Wait for graceful shutdown
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                logger.warning(f"Force killing worker {worker_name}")
                process.kill()
                process.wait()

            del self.workers[worker_name]
            logger.info(f"Stopped worker: {worker_name}")

            return True

        except Exception as e:
            logger.error(f"Failed to stop worker {worker_name}: {e}")
            return False

    def restart_worker(self, worker_name: str) -> bool:
        """Restart a worker."""
        logger.info(f"Restarting worker: {worker_name}")

        if self.stop_worker(worker_name):
            time.sleep(2)  # Brief pause
            return self.start_worker(worker_name)

        return False

    def scale_workers(self, worker_type: str, count: int):
        """
        Scale workers of a specific type.

        Args:
            worker_type: Type of worker (e.g., 'generation', 'processing')
            count: Desired number of workers
        """
        current_workers = [
            name for name, config in self.worker_configs.items() if worker_type in name
        ]

        current_count = len(current_workers)

        if count > current_count:
            # Scale up
            for i in range(current_count, count):
                worker_name = f"{worker_type}-{i}"

                # Create config if needed
                if worker_name not in self.worker_configs:
                    base_config = self._get_base_config(worker_type)
                    self.add_worker_config(WorkerConfig(name=worker_name, **base_config))

                self.start_worker(worker_name)

        elif count < current_count:
            # Scale down
            for i in range(count, current_count):
                worker_name = f"{worker_type}-{i}"
                self.stop_worker(worker_name)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get statistics for all workers."""
        stats = {}

        # Get Celery stats
        celery_stats = self.control.inspect().stats()

        for worker_name, process in self.workers.items():
            try:
                # Get process info
                proc = psutil.Process(process.pid)

                stats[worker_name] = {
                    "pid": process.pid,
                    "status": proc.status(),
                    "cpu_percent": proc.cpu_percent(interval=1),
                    "memory_mb": proc.memory_info().rss / 1024 / 1024,
                    "num_threads": proc.num_threads(),
                    "create_time": datetime.fromtimestamp(proc.create_time()),
                }

                # Add Celery stats if available
                hostname = f"{worker_name}@{os.uname().nodename}"
                if celery_stats and hostname in celery_stats:
                    stats[worker_name]["celery"] = celery_stats[hostname]

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                stats[worker_name] = {"status": "dead"}

        return stats

    def monitor_workers(self, interval: int = 60):
        """
        Monitor workers and restart if needed.

        Args:
            interval: Check interval in seconds
        """
        logger.info("Starting worker monitor")

        while self._running:
            try:
                # Check worker health
                dead_workers = []
                stats = self.get_worker_stats()

                for worker_name, info in stats.items():
                    if info.get("status") == "dead":
                        dead_workers.append(worker_name)
                    elif info.get("status") == "zombie":
                        dead_workers.append(worker_name)
                    elif info.get("memory_mb", 0) > 2048:  # 2GB limit
                        logger.warning(f"Worker {worker_name} using too much memory")
                        dead_workers.append(worker_name)

                # Restart dead workers
                for worker_name in dead_workers:
                    logger.warning(f"Worker {worker_name} is unhealthy, restarting")
                    self.restart_worker(worker_name)

                # Check queue lengths and scale if needed
                self._check_autoscaling()

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(interval)

    def _check_autoscaling(self):
        """Check queue lengths and scale workers accordingly."""
        try:
            # Get queue lengths from Redis
            import redis

            r = redis.from_url(self.redis_url)

            queue_lengths = {}
            for queue in ["critical", "generation-high", "generation", "processing"]:
                length = r.llen(f"celery:{queue}")
                queue_lengths[queue] = length

            # Scale generation workers based on queue length
            gen_queue_length = (
                queue_lengths.get("critical", 0)
                + queue_lengths.get("generation-high", 0)
                + queue_lengths.get("generation", 0)
            )

            if gen_queue_length > 100:
                desired_workers = min(10, gen_queue_length // 20)
            elif gen_queue_length > 50:
                desired_workers = 5
            elif gen_queue_length > 20:
                desired_workers = 3
            else:
                desired_workers = 2

            current_gen_workers = len([w for w in self.workers if "generation" in w])

            if desired_workers != current_gen_workers:
                logger.info(
                    f"Scaling generation workers: {current_gen_workers} -> {desired_workers}"
                )
                self.scale_workers("generation", desired_workers)

        except Exception as e:
            logger.error(f"Autoscaling error: {e}")

    def _get_base_config(self, worker_type: str) -> Dict[str, Any]:
        """Get base configuration for worker type."""
        configs = {
            "generation": {
                "queues": ["critical", "generation-high", "generation"],
                "concurrency": 2,
                "pool": "prefork",
                "max_tasks_per_child": 10,  # Restart after 10 tasks (memory management)
            },
            "processing": {
                "queues": ["processing"],
                "concurrency": 4,
                "pool": "prefork",
                "max_tasks_per_child": 50,
            },
            "batch": {
                "queues": ["batch"],
                "concurrency": 1,
                "pool": "prefork",
                "max_tasks_per_child": 5,
            },
            "monitoring": {
                "queues": ["monitoring"],
                "concurrency": 1,
                "pool": "eventlet",  # Lightweight for monitoring
                "max_tasks_per_child": 1000,
            },
        }

        return configs.get(worker_type, configs["generation"])

    def start_all(self):
        """Start all configured workers."""
        self._running = True

        for worker_name in self.worker_configs:
            self.start_worker(worker_name)

    def stop_all(self):
        """Stop all workers."""
        self._running = False

        worker_names = list(self.workers.keys())
        for worker_name in worker_names:
            self.stop_worker(worker_name)

    def handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down workers")
        self.stop_all()
        sys.exit(0)


def create_default_workers() -> List[WorkerConfig]:
    """Create default worker configurations."""
    return [
        # Critical priority worker
        WorkerConfig(
            name="critical-0",
            queues=["critical"],
            concurrency=2,
            pool="prefork",
            max_tasks_per_child=10,
        ),
        # High priority generation workers
        WorkerConfig(
            name="generation-high-0",
            queues=["generation-high", "generation"],
            concurrency=2,
            pool="prefork",
            max_tasks_per_child=10,
            autoscale=(4, 2),
        ),
        # Normal generation workers
        WorkerConfig(
            name="generation-0",
            queues=["generation"],
            concurrency=2,
            pool="prefork",
            max_tasks_per_child=10,
        ),
        WorkerConfig(
            name="generation-1",
            queues=["generation"],
            concurrency=2,
            pool="prefork",
            max_tasks_per_child=10,
        ),
        # Audio processing workers
        WorkerConfig(
            name="processing-0",
            queues=["processing"],
            concurrency=4,
            pool="prefork",
            max_tasks_per_child=50,
        ),
        # Batch worker
        WorkerConfig(
            name="batch-0",
            queues=["batch"],
            concurrency=1,
            pool="prefork",
            max_tasks_per_child=5,
        ),
        # Monitoring worker
        WorkerConfig(
            name="monitoring-0",
            queues=["monitoring", "dead-letter"],
            concurrency=1,
            pool="eventlet",
            max_tasks_per_child=1000,
        ),
    ]


def main():
    """Run worker manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Music Gen Worker Manager")
    parser.add_argument("--redis-url", default="redis://localhost:6379/0")
    parser.add_argument("--workers", nargs="+", help="Worker types to start")
    parser.add_argument("--scale", nargs=2, metavar=("TYPE", "COUNT"))
    parser.add_argument("--monitor", action="store_true")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create Celery app
    from music_gen.workers import celery_app

    # Create manager
    manager = WorkerManager(celery_app, args.redis_url)

    # Set up signal handlers
    signal.signal(signal.SIGTERM, manager.handle_signal)
    signal.signal(signal.SIGINT, manager.handle_signal)

    # Configure workers
    for config in create_default_workers():
        manager.add_worker_config(config)

    if args.scale:
        # Scale specific worker type
        worker_type, count = args.scale
        manager.scale_workers(worker_type, int(count))
    else:
        # Start all workers
        manager.start_all()

    if args.monitor:
        # Start monitoring
        manager.monitor_workers()
    else:
        # Wait for shutdown
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_all()


if __name__ == "__main__":
    main()
