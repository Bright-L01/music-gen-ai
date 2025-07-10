from datetime import timedelta

"""
Task monitoring endpoints for the Music Gen AI API.

Provides real-time insights into task processing, queue status, and system health.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from music_gen.core.container import get_container
from music_gen.infrastructure.repositories.redis_task_repository_advanced import (
    RedisTaskRepositoryAdvanced,
    TaskMetrics,
    TaskPriority,
    TaskStatus,
)

router = APIRouter(prefix="/monitoring/tasks", tags=["monitoring"])


# Response models
class TaskQueueStatus(BaseModel):
    """Task queue status information."""

    priority: str = Field(..., description="Queue priority level")
    length: int = Field(..., description="Number of tasks in queue")
    oldest_task_age: Optional[float] = Field(None, description="Age of oldest task in seconds")


class TaskSystemHealth(BaseModel):
    """Overall task system health."""

    status: str = Field(..., description="System status (healthy, degraded, unhealthy)")
    metrics: TaskMetrics = Field(..., description="Task metrics")
    queue_status: List[TaskQueueStatus] = Field(..., description="Status of each priority queue")
    stream_info: Dict[str, Any] = Field(..., description="Redis stream information")
    warnings: List[str] = Field(default_factory=list, description="System warnings")
    last_cleanup: Optional[str] = Field(None, description="Last cleanup timestamp")


class TaskDistribution(BaseModel):
    """Task distribution by status and priority."""

    by_status: Dict[str, int] = Field(..., description="Task count by status")
    by_priority: Dict[str, int] = Field(..., description="Task count by priority")
    by_age: Dict[str, int] = Field(..., description="Task count by age buckets")


class TaskPerformance(BaseModel):
    """Task performance metrics."""

    avg_processing_time: float = Field(..., description="Average processing time in seconds")
    p50_processing_time: float = Field(..., description="50th percentile processing time")
    p95_processing_time: float = Field(..., description="95th percentile processing time")
    p99_processing_time: float = Field(..., description="99th percentile processing time")
    success_rate: float = Field(..., description="Task success rate (0-1)")
    retry_rate: float = Field(..., description="Task retry rate (0-1)")
    throughput_per_minute: float = Field(..., description="Tasks completed per minute")


# Dependency injection
def get_task_repository() -> RedisTaskRepositoryAdvanced:
    """Get advanced task repository from DI container."""
    container = get_container()
    repo = container.get(RedisTaskRepositoryAdvanced)
    if not isinstance(repo, RedisTaskRepositoryAdvanced):
        raise HTTPException(
            status_code=503, detail="Advanced task monitoring requires RedisTaskRepositoryAdvanced"
        )
    return repo


@router.get("/health", response_model=TaskSystemHealth)
async def get_task_system_health(repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository)):
    """Get overall task system health and metrics."""

    # Get basic metrics
    metrics = await repo.get_task_metrics()

    # Get stream info
    try:
        stream_info = await repo.get_stream_info()
    except Exception as e:
        stream_info = {"error": str(e)}

    # Check queue status for each priority
    queue_status = []
    warnings = []

    for priority in TaskPriority:
        queue_key = repo._priority_queues[priority]
        redis = await repo._get_redis()

        length = await redis.zcard(queue_key)

        # Get oldest task age
        oldest_task_age = None
        oldest = await redis.zrange(queue_key, 0, 0, withscores=True)
        if oldest:
            # Score is negative timestamp
            task_time = -oldest[0][1]
            oldest_task_age = datetime.utcnow().timestamp() - task_time

        queue_status.append(
            TaskQueueStatus(priority=priority.name, length=length, oldest_task_age=oldest_task_age)
        )

        # Add warnings for old tasks
        if oldest_task_age and oldest_task_age > 3600:  # 1 hour
            warnings.append(f"{priority.name} queue has tasks older than 1 hour")

    # Determine overall health status
    if metrics.failed_tasks > metrics.completed_tasks * 0.5:
        status = "unhealthy"
        warnings.append("High failure rate detected")
    elif metrics.processing_tasks > 100:
        status = "degraded"
        warnings.append("High number of processing tasks")
    elif any(q.length > 1000 for q in queue_status):
        status = "degraded"
        warnings.append("Large queue backlog detected")
    else:
        status = "healthy"

    return TaskSystemHealth(
        status=status,
        metrics=metrics,
        queue_status=queue_status,
        stream_info=stream_info,
        warnings=warnings,
        last_cleanup=None,  # TODO: Track cleanup times
    )


@router.get("/metrics", response_model=TaskMetrics)
async def get_task_metrics(repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository)):
    """Get detailed task metrics."""
    return await repo.get_task_metrics()


@router.get("/distribution", response_model=TaskDistribution)
async def get_task_distribution(repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository)):
    """Get task distribution by various dimensions."""

    # Get all tasks
    all_tasks = await repo.list_tasks(limit=10000)

    # Count by status
    by_status = {}
    for status in TaskStatus:
        by_status[status.value] = sum(1 for t in all_tasks if t.get("status") == status.value)

    # Count by priority
    by_priority = {}
    for priority in TaskPriority:
        by_priority[priority.name] = sum(
            1 for t in all_tasks if t.get("priority") == priority.value
        )

    # Count by age buckets
    now = datetime.utcnow()
    age_buckets = {
        "< 1 min": 0,
        "1-5 min": 0,
        "5-15 min": 0,
        "15-60 min": 0,
        "1-6 hours": 0,
        "6-24 hours": 0,
        "> 24 hours": 0,
    }

    for task in all_tasks:
        created_at = task.get("created_at")
        if created_at:
            try:
                created_time = datetime.fromisoformat(created_at)
                age = (now - created_time).total_seconds()

                if age < 60:
                    age_buckets["< 1 min"] += 1
                elif age < 300:
                    age_buckets["1-5 min"] += 1
                elif age < 900:
                    age_buckets["5-15 min"] += 1
                elif age < 3600:
                    age_buckets["15-60 min"] += 1
                elif age < 21600:
                    age_buckets["1-6 hours"] += 1
                elif age < 86400:
                    age_buckets["6-24 hours"] += 1
                else:
                    age_buckets["> 24 hours"] += 1
            except:
                pass

    return TaskDistribution(by_status=by_status, by_priority=by_priority, by_age=age_buckets)


@router.get("/performance", response_model=TaskPerformance)
async def get_task_performance(
    time_window: int = Query(3600, description="Time window in seconds"),
    repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository),
):
    """Get task performance metrics for a time window."""

    # Get completed tasks within time window
    all_tasks = await repo.list_tasks(limit=10000)
    cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)

    completed_tasks = []
    failed_tasks = 0
    retried_tasks = 0

    for task in all_tasks:
        completed_at = task.get("completed_at")
        if completed_at:
            try:
                completed_time = datetime.fromisoformat(completed_at)
                if completed_time >= cutoff_time:
                    completed_tasks.append(task)
            except:
                pass

        # Count failures and retries
        if task.get("status") == TaskStatus.FAILED.value:
            updated_at = task.get("updated_at")
            if updated_at:
                try:
                    updated_time = datetime.fromisoformat(updated_at)
                    if updated_time >= cutoff_time:
                        failed_tasks += 1
                except:
                    pass

        if task.get("retry_count", 0) > 0:
            retried_tasks += 1

    # Calculate processing times
    processing_times = []
    for task in completed_tasks:
        if "processing_time" in task:
            processing_times.append(float(task["processing_time"]))

    # Calculate percentiles
    if processing_times:
        processing_times.sort()
        n = len(processing_times)

        avg_time = sum(processing_times) / n
        p50_time = processing_times[int(n * 0.5)]
        p95_time = processing_times[int(n * 0.95)]
        p99_time = processing_times[int(n * 0.99)]
    else:
        avg_time = p50_time = p95_time = p99_time = 0.0

    # Calculate rates
    total_processed = len(completed_tasks) + failed_tasks
    success_rate = len(completed_tasks) / total_processed if total_processed > 0 else 0.0
    retry_rate = retried_tasks / len(all_tasks) if all_tasks else 0.0

    # Calculate throughput
    throughput_per_minute = (len(completed_tasks) / time_window) * 60

    return TaskPerformance(
        avg_processing_time=avg_time,
        p50_processing_time=p50_time,
        p95_processing_time=p95_time,
        p99_processing_time=p99_time,
        success_rate=success_rate,
        retry_rate=retry_rate,
        throughput_per_minute=throughput_per_minute,
    )


@router.post("/cleanup")
async def trigger_cleanup(repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository)):
    """Manually trigger task cleanup."""

    # Clean up expired tasks
    await repo._cleanup_expired_tasks()

    # Recover stalled tasks
    recovered = await repo.recover_stalled_tasks()

    return {
        "status": "cleanup completed",
        "recovered_tasks": recovered,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/queue/{priority}")
async def get_queue_details(
    priority: str,
    limit: int = Query(100, le=1000),
    repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository),
):
    """Get detailed information about a specific priority queue."""

    # Validate priority
    try:
        priority_enum = TaskPriority[priority.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")

    # Get queue info
    queue_key = repo._priority_queues[priority_enum]
    redis = await repo._get_redis()

    # Get queue length
    length = await redis.zcard(queue_key)

    # Get task IDs from queue
    task_ids = await redis.zrange(queue_key, 0, limit - 1, withscores=True)

    # Get task details
    tasks = []
    for task_id, score in task_ids:
        task = await repo.get_task(task_id)
        if task:
            # Add queue position
            task["queue_position"] = len(tasks)
            task["queue_time"] = datetime.utcnow().timestamp() - (-score)
            tasks.append(task)

    return {
        "priority": priority,
        "total_length": length,
        "tasks": tasks,
        "oldest_task_age": tasks[0]["queue_time"] if tasks else None,
    }


@router.get("/stalled")
async def get_stalled_tasks(
    stall_timeout: int = Query(3600, description="Stall timeout in seconds"),
    repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository),
):
    """Get tasks that appear to be stalled."""

    processing_tasks = await repo.list_tasks(status=TaskStatus.PROCESSING.value, limit=1000)

    stalled_tasks = []
    stall_threshold = datetime.utcnow() - timedelta(seconds=stall_timeout)

    for task in processing_tasks:
        started_at = task.get("started_at")
        if started_at:
            try:
                start_time = datetime.fromisoformat(started_at)
                if start_time < stall_threshold:
                    task["stall_duration"] = (datetime.utcnow() - start_time).total_seconds()
                    stalled_tasks.append(task)
            except:
                pass

    return {
        "total_stalled": len(stalled_tasks),
        "stall_timeout": stall_timeout,
        "tasks": stalled_tasks,
    }


@router.get("/trends")
async def get_task_trends(
    hours: int = Query(24, description="Number of hours to analyze"),
    repo: RedisTaskRepositoryAdvanced = Depends(get_task_repository),
):
    """Get task processing trends over time."""

    # Get all tasks
    all_tasks = await repo.list_tasks(limit=10000)
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)

    # Create hourly buckets
    hourly_buckets = {}
    for i in range(hours):
        bucket_time = cutoff_time + timedelta(hours=i)
        bucket_key = bucket_time.strftime("%Y-%m-%d %H:00")
        hourly_buckets[bucket_key] = {
            "created": 0,
            "completed": 0,
            "failed": 0,
        }

    # Populate buckets
    for task in all_tasks:
        # Count created tasks
        created_at = task.get("created_at")
        if created_at:
            try:
                created_time = datetime.fromisoformat(created_at)
                if created_time >= cutoff_time:
                    bucket_key = created_time.strftime("%Y-%m-%d %H:00")
                    if bucket_key in hourly_buckets:
                        hourly_buckets[bucket_key]["created"] += 1
            except:
                pass

        # Count completed tasks
        completed_at = task.get("completed_at")
        if completed_at:
            try:
                completed_time = datetime.fromisoformat(completed_at)
                if completed_time >= cutoff_time:
                    bucket_key = completed_time.strftime("%Y-%m-%d %H:00")
                    if bucket_key in hourly_buckets:
                        hourly_buckets[bucket_key]["completed"] += 1
            except:
                pass

        # Count failed tasks
        if task.get("status") == TaskStatus.FAILED.value:
            updated_at = task.get("updated_at")
            if updated_at:
                try:
                    updated_time = datetime.fromisoformat(updated_at)
                    if updated_time >= cutoff_time:
                        bucket_key = updated_time.strftime("%Y-%m-%d %H:00")
                        if bucket_key in hourly_buckets:
                            hourly_buckets[bucket_key]["failed"] += 1
                except:
                    pass

    return {
        "time_range": {
            "start": cutoff_time.isoformat(),
            "end": datetime.utcnow().isoformat(),
            "hours": hours,
        },
        "trends": hourly_buckets,
    }
