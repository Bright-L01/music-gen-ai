from datetime import timedelta

"""
Advanced Redis-based task repository with Streams, priority, and TTL support.

This implementation provides production-ready task management with:
- Redis Streams for task queue
- Priority-based task ordering
- TTL and automatic cleanup
- Task recovery and persistence
- Monitoring capabilities
"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from music_gen.core.exceptions import TaskNotFoundError
from music_gen.core.interfaces.repositories import TaskRepository
from music_gen.utils.optional_imports import optional_import

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class TaskStatus(Enum):
    """Task status values."""

    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class TaskMetrics:
    """Task metrics for monitoring."""

    total_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_processing_time: float = 0.0
    queue_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RedisTaskRepositoryAdvanced(TaskRepository):
    """Advanced Redis-based task repository with Streams support."""

    def __init__(
        self,
        redis_url: str,
        stream_key: str = "musicgen:task:stream",
        task_prefix: str = "musicgen:task:",
        queue_prefix: str = "musicgen:queue:",
        default_ttl: int = 86400,  # 24 hours
        max_retries: int = 3,
        cleanup_interval: int = 300,
    ):  # 5 minutes
        """
        Initialize advanced Redis task repository.

        Args:
            redis_url: Redis connection URL
            stream_key: Redis stream key for task queue
            task_prefix: Prefix for task data keys
            queue_prefix: Prefix for priority queues
            default_ttl: Default TTL for tasks in seconds
            max_retries: Maximum retry attempts for failed tasks
            cleanup_interval: Interval for cleanup tasks in seconds
        """
        self._redis_url = redis_url
        self._stream_key = stream_key
        self._task_prefix = task_prefix
        self._queue_prefix = queue_prefix
        self._default_ttl = default_ttl
        self._max_retries = max_retries
        self._cleanup_interval = cleanup_interval
        self._redis = None
        self._cleanup_task = None

        # Priority queue keys
        self._priority_queues = {
            TaskPriority.CRITICAL: f"{queue_prefix}critical",
            TaskPriority.HIGH: f"{queue_prefix}high",
            TaskPriority.NORMAL: f"{queue_prefix}normal",
            TaskPriority.LOW: f"{queue_prefix}low",
        }

    async def initialize(self):
        """Initialize repository and start background tasks."""
        await self._get_redis()
        await self._start_cleanup_task()

    async def shutdown(self):
        """Shutdown repository and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._redis:
            await self._redis.close()

    async def _get_redis(self):
        """Get Redis connection (lazy initialization)."""
        if self._redis is None:
            redis = optional_import("redis")
            if redis is None:
                raise ImportError(
                    "redis[asyncio] is required. Install with: pip install redis[asyncio]"
                )

            self._redis = redis.asyncio.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _get_task_key(self, task_id: str) -> str:
        """Get Redis key for task data."""
        return f"{self._task_prefix}{task_id}"

    def _get_task_metadata_key(self, task_id: str) -> str:
        """Get Redis key for task metadata."""
        return f"{self._task_prefix}meta:{task_id}"

    async def create_task(self, task_id: str, task_data: Dict[str, Any]) -> None:
        """
        Create a new task with priority support.

        Args:
            task_id: Unique task identifier
            task_data: Task data including priority, TTL, etc.
        """
        redis = await self._get_redis()

        # Extract task parameters
        priority = task_data.get("priority", TaskPriority.NORMAL.value)
        ttl = task_data.get("ttl", self._default_ttl)

        # Add timestamps and metadata
        now = datetime.utcnow()
        task_data.update(
            {
                "id": task_id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "status": TaskStatus.PENDING.value,
                "retry_count": 0,
                "priority": priority,
            }
        )

        # Create metadata
        metadata = {
            "task_id": task_id,
            "priority": priority,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
            "ttl": ttl,
        }

        # Transaction to ensure atomicity
        pipe = redis.pipeline()

        # Store task data
        task_key = self._get_task_key(task_id)
        pipe.hset(task_key, mapping=task_data)
        pipe.expire(task_key, ttl)

        # Store metadata
        meta_key = self._get_task_metadata_key(task_id)
        pipe.hset(meta_key, mapping=metadata)
        pipe.expire(meta_key, ttl)

        # Add to stream
        stream_data = {
            "task_id": task_id,
            "priority": priority,
            "created_at": now.isoformat(),
        }
        pipe.xadd(self._stream_key, stream_data)

        # Add to priority queue
        priority_queue = self._get_priority_queue(priority)
        pipe.zadd(priority_queue, {task_id: -time.time()})  # Negative for FIFO

        await pipe.execute()

        logger.info(f"Task created: {task_id} with priority {priority}")

    async def enqueue_task(self, task_id: str) -> None:
        """Move task to processing queue."""
        await self._get_redis()

        task = await self.get_task(task_id)
        if not task:
            raise TaskNotFoundError(f"Task not found: {task_id}")

        # Update status
        await self.update_task(
            task_id,
            {
                "status": TaskStatus.QUEUED.value,
                "queued_at": datetime.utcnow().isoformat(),
            },
        )

        logger.debug(f"Task enqueued: {task_id}")

    async def dequeue_task(self, priorities: Optional[List[TaskPriority]] = None) -> Optional[str]:
        """
        Dequeue next task based on priority.

        Args:
            priorities: List of priorities to check (default: all)

        Returns:
            Task ID if available, None otherwise
        """
        redis = await self._get_redis()

        if priorities is None:
            priorities = list(TaskPriority)

        # Check queues in priority order
        for priority in sorted(priorities, key=lambda p: p.value, reverse=True):
            queue_key = self._priority_queues[priority]

            # Get oldest task from queue (ZPOPMIN)
            result = await redis.zpopmin(queue_key, 1)
            if result:
                task_id = result[0][0]

                # Update task status
                await self.update_task(
                    task_id,
                    {
                        "status": TaskStatus.PROCESSING.value,
                        "started_at": datetime.utcnow().isoformat(),
                    },
                )

                logger.debug(f"Task dequeued: {task_id} from {priority.name} queue")
                return task_id

        return None

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        redis = await self._get_redis()

        task_key = self._get_task_key(task_id)
        task_data = await redis.hgetall(task_key)

        if not task_data:
            return None

        # Convert numeric fields
        for field in ["priority", "retry_count"]:
            if field in task_data:
                task_data[field] = int(task_data[field])

        return task_data

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> None:
        """Update task data."""
        redis = await self._get_redis()

        task_key = self._get_task_key(task_id)

        # Check if task exists
        exists = await redis.exists(task_key)
        if not exists:
            raise TaskNotFoundError(f"Task not found: {task_id}")

        # Add update timestamp
        updates["updated_at"] = datetime.utcnow().isoformat()

        # Handle status transitions
        if "status" in updates:
            old_status = await redis.hget(task_key, "status")
            new_status = updates["status"]

            # Log status transition
            logger.info(f"Task {task_id} status: {old_status} -> {new_status}")

            # Handle completion
            if new_status == TaskStatus.COMPLETED.value:
                updates["completed_at"] = datetime.utcnow().isoformat()

                # Calculate processing time
                started_at = await redis.hget(task_key, "started_at")
                if started_at:
                    start_time = datetime.fromisoformat(started_at)
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    updates["processing_time"] = processing_time

            # Handle failure
            elif new_status == TaskStatus.FAILED.value:
                retry_count = int(await redis.hget(task_key, "retry_count") or 0)

                if retry_count < self._max_retries:
                    # Re-queue for retry
                    updates["retry_count"] = retry_count + 1
                    updates["status"] = TaskStatus.PENDING.value

                    # Re-add to priority queue
                    priority = int(
                        await redis.hget(task_key, "priority") or TaskPriority.NORMAL.value
                    )
                    priority_queue = self._get_priority_queue(priority)
                    await redis.zadd(priority_queue, {task_id: -time.time()})

                    logger.info(
                        f"Task {task_id} scheduled for retry ({retry_count + 1}/{self._max_retries})"
                    )

        # Update fields
        await redis.hset(task_key, mapping=updates)

    async def delete_task(self, task_id: str) -> None:
        """Delete a task."""
        redis = await self._get_redis()

        pipe = redis.pipeline()

        # Delete task data
        task_key = self._get_task_key(task_id)
        pipe.delete(task_key)

        # Delete metadata
        meta_key = self._get_task_metadata_key(task_id)
        pipe.delete(meta_key)

        # Remove from all priority queues
        for queue_key in self._priority_queues.values():
            pipe.zrem(queue_key, task_id)

        await pipe.execute()

        logger.debug(f"Task deleted: {task_id}")

    async def list_tasks(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        redis = await self._get_redis()

        # Get all task keys
        pattern = f"{self._task_prefix}*"
        cursor = 0
        keys = []

        # Scan for keys
        while True:
            cursor, batch = await redis.scan(cursor, match=pattern, count=1000, _type="hash")
            keys.extend(batch)
            if cursor == 0:
                break

        # Filter out metadata keys
        task_keys = [k for k in keys if ":meta:" not in k]

        # Get task data
        tasks = []
        for key in task_keys[offset : offset + limit]:
            task_data = await redis.hgetall(key)
            if task_data:
                # Filter by status if specified
                if status is None or task_data.get("status") == status:
                    # Convert numeric fields
                    for field in ["priority", "retry_count"]:
                        if field in task_data:
                            task_data[field] = int(task_data[field])
                    tasks.append(task_data)

        # Sort by creation time
        tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)

        return tasks

    async def get_task_metrics(self) -> TaskMetrics:
        """Get task metrics for monitoring."""
        redis = await self._get_redis()

        metrics = TaskMetrics()

        # Get all tasks
        all_tasks = await self.list_tasks(limit=10000)
        metrics.total_tasks = len(all_tasks)

        # Count by status
        status_counts = {}
        processing_times = []

        for task in all_tasks:
            status = task.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

            # Collect processing times
            if "processing_time" in task:
                processing_times.append(float(task["processing_time"]))

        metrics.pending_tasks = status_counts.get(TaskStatus.PENDING.value, 0)
        metrics.processing_tasks = status_counts.get(TaskStatus.PROCESSING.value, 0)
        metrics.completed_tasks = status_counts.get(TaskStatus.COMPLETED.value, 0)
        metrics.failed_tasks = status_counts.get(TaskStatus.FAILED.value, 0)

        # Calculate average processing time
        if processing_times:
            metrics.avg_processing_time = sum(processing_times) / len(processing_times)

        # Get queue lengths
        queue_length = 0
        for queue_key in self._priority_queues.values():
            queue_length += await redis.zcard(queue_key)
        metrics.queue_length = queue_length

        return metrics

    async def get_stream_info(self) -> Dict[str, Any]:
        """Get Redis stream information."""
        redis = await self._get_redis()

        info = await redis.xinfo_stream(self._stream_key)
        return {
            "length": info.get("length", 0),
            "first_entry": info.get("first-entry"),
            "last_entry": info.get("last-entry"),
            "consumer_groups": info.get("groups", 0),
        }

    def _get_priority_queue(self, priority: int) -> str:
        """Get priority queue key based on priority value."""
        if priority >= TaskPriority.CRITICAL.value:
            return self._priority_queues[TaskPriority.CRITICAL]
        elif priority >= TaskPriority.HIGH.value:
            return self._priority_queues[TaskPriority.HIGH]
        elif priority >= TaskPriority.LOW.value:
            return self._priority_queues[TaskPriority.NORMAL]
        else:
            return self._priority_queues[TaskPriority.LOW]

    async def _start_cleanup_task(self):
        """Start background cleanup task."""

        async def cleanup():
            while True:
                try:
                    await asyncio.sleep(self._cleanup_interval)
                    await self._cleanup_expired_tasks()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")

        self._cleanup_task = asyncio.create_task(cleanup())

    async def _cleanup_expired_tasks(self):
        """Clean up expired tasks."""
        redis = await self._get_redis()

        # Get all metadata keys
        pattern = f"{self._task_prefix}meta:*"
        cursor = 0

        expired_count = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=100)

            for meta_key in keys:
                metadata = await redis.hgetall(meta_key)
                if metadata and "expires_at" in metadata:
                    expires_at = datetime.fromisoformat(metadata["expires_at"])
                    if expires_at < datetime.utcnow():
                        # Task has expired
                        task_id = metadata.get("task_id")
                        if task_id:
                            task = await self.get_task(task_id)
                            if task and task.get("status") not in [
                                TaskStatus.COMPLETED.value,
                                TaskStatus.FAILED.value,
                            ]:
                                # Mark as expired
                                await self.update_task(
                                    task_id,
                                    {
                                        "status": TaskStatus.EXPIRED.value,
                                        "expired_at": datetime.utcnow().isoformat(),
                                    },
                                )
                                expired_count += 1

            if cursor == 0:
                break

        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired tasks")

    async def recover_stalled_tasks(self, stall_timeout: int = 3600) -> int:
        """
        Recover tasks that have been processing for too long.

        Args:
            stall_timeout: Time in seconds before considering a task stalled

        Returns:
            Number of recovered tasks
        """
        redis = await self._get_redis()

        processing_tasks = await self.list_tasks(status=TaskStatus.PROCESSING.value, limit=1000)

        recovered_count = 0
        stall_threshold = datetime.utcnow() - timedelta(seconds=stall_timeout)

        for task in processing_tasks:
            started_at = task.get("started_at")
            if started_at:
                start_time = datetime.fromisoformat(started_at)
                if start_time < stall_threshold:
                    # Task has been processing too long
                    task_id = task["id"]

                    # Reset to pending
                    await self.update_task(
                        task_id,
                        {
                            "status": TaskStatus.PENDING.value,
                            "stalled_at": datetime.utcnow().isoformat(),
                        },
                    )

                    # Re-add to priority queue
                    priority = task.get("priority", TaskPriority.NORMAL.value)
                    priority_queue = self._get_priority_queue(priority)
                    await redis.zadd(priority_queue, {task_id: -time.time()})

                    recovered_count += 1
                    logger.warning(f"Recovered stalled task: {task_id}")

        return recovered_count
