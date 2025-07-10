"""
Celery tasks for Music Gen AI background processing.

This module defines all Celery tasks for music generation, processing,
and system monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from celery import Task, current_task, group
from celery.exceptions import SoftTimeLimitExceeded

from music_gen.core.container import get_container
from music_gen.infrastructure.repositories.redis_task_repository_advanced import (
    TaskPriority,
    TaskStatus,
)

from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Temporary directory for audio files
TEMP_DIR = Path("/tmp/musicgen")
TEMP_DIR.mkdir(exist_ok=True)


class MusicGenerationTask(Task):
    """Base class for music generation tasks with retry logic."""

    autoretry_for = (Exception,)
    retry_kwargs = {
        "max_retries": 3,
        "countdown": 60,  # Retry after 60 seconds
    }
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minutes
    retry_jitter = True

    def __init__(self):
        super().__init__()
        self._model_service = None
        self._generation_service = None
        self._task_repository = None

    @property
    def model_service(self):
        """Lazy load model service."""
        if self._model_service is None:
            container = get_container()
            self._model_service = container.get("ModelService")
        return self._model_service

    @property
    def generation_service(self):
        """Lazy load generation service."""
        if self._generation_service is None:
            container = get_container()
            self._generation_service = container.get("GenerationService")
        return self._generation_service

    @property
    def task_repository(self):
        """Lazy load task repository."""
        if self._task_repository is None:
            container = get_container()
            self._task_repository = container.get("TaskRepository")
        return self._task_repository


@celery_app.task(
    bind=True,
    base=MusicGenerationTask,
    name="musicgen.workers.tasks.generate_music_task",
    time_limit=3600,  # 1 hour hard limit
    soft_time_limit=3300,  # 55 minute soft limit
)
def generate_music_task(self, task_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate music from text prompt.

    Args:
        task_id: Unique task identifier
        request_data: Generation request parameters

    Returns:
        Generation result with audio file path
    """
    start_time = time.time()

    try:
        # Update task status
        asyncio.run(
            self.task_repository.update_task(
                task_id,
                {
                    "status": TaskStatus.PROCESSING.value,
                    "worker_id": current_task.request.id,
                    "worker_hostname": current_task.request.hostname,
                    "started_at": datetime.utcnow().isoformat(),
                },
            )
        )

        # Log task start
        logger.info(
            f"Starting music generation task {task_id} on worker {current_task.request.hostname}"
        )

        # Extract parameters
        prompt = request_data.get("prompt", "")
        duration = request_data.get("duration", 10.0)
        temperature = request_data.get("temperature", 1.0)
        top_k = request_data.get("top_k", 50)
        top_p = request_data.get("top_p", 0.9)
        seed = request_data.get("seed")

        # Conditioning parameters
        genre = request_data.get("genre")
        mood = request_data.get("mood")
        tempo = request_data.get("tempo")
        instruments = request_data.get("instruments", [])

        # Create generation request
        from music_gen.core.interfaces.services import GenerationRequest

        gen_request = GenerationRequest(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            conditioning={
                "genre": genre,
                "mood": mood,
                "tempo": tempo,
                "instruments": instruments,
            },
        )

        # Report progress
        self.update_state(state="GENERATING", meta={"progress": 0.1, "status": "Loading model..."})

        # Generate audio
        result = asyncio.run(self.generation_service.generate(gen_request))

        # Report progress
        self.update_state(state="GENERATING", meta={"progress": 0.8, "status": "Saving audio..."})

        # Save audio file
        audio_path = TEMP_DIR / f"{task_id}.wav"
        import scipy.io.wavfile

        scipy.io.wavfile.write(
            str(audio_path),
            rate=result.sample_rate,
            data=(result.audio.cpu().numpy() * 32767).astype(np.int16),
        )

        # Calculate metrics
        processing_time = time.time() - start_time
        file_size = audio_path.stat().st_size

        # Prepare result
        task_result = {
            "task_id": task_id,
            "status": "completed",
            "audio_path": str(audio_path),
            "audio_url": f"/download/{task_id}",
            "duration": result.duration,
            "sample_rate": result.sample_rate,
            "file_size": file_size,
            "processing_time": processing_time,
            "metadata": result.metadata,
            "worker_id": current_task.request.id,
            "completed_at": datetime.utcnow().isoformat(),
        }

        # Update task repository
        asyncio.run(self.task_repository.update_task(task_id, task_result))

        # Log completion
        logger.info(f"Completed music generation task {task_id} in {processing_time:.2f}s")

        return task_result

    except SoftTimeLimitExceeded:
        # Handle soft time limit
        logger.warning(f"Task {task_id} exceeded soft time limit")

        error_data = {
            "status": TaskStatus.FAILED.value,
            "error": "Task exceeded time limit",
            "failed_at": datetime.utcnow().isoformat(),
        }

        asyncio.run(self.task_repository.update_task(task_id, error_data))
        raise

    except Exception as e:
        # Handle other errors
        logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)

        error_data = {
            "status": TaskStatus.FAILED.value,
            "error": str(e),
            "error_type": type(e).__name__,
            "failed_at": datetime.utcnow().isoformat(),
        }

        asyncio.run(self.task_repository.update_task(task_id, error_data))

        # Retry if within limits
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        else:
            # Send to dead letter queue
            send_to_dead_letter_queue(task_id, error_data)
            raise


@celery_app.task(
    bind=True,
    base=MusicGenerationTask,
    name="musicgen.workers.tasks.generate_batch_task",
    time_limit=7200,  # 2 hour limit for batch
)
def generate_batch_task(self, batch_id: str, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate multiple music clips in batch.

    Args:
        batch_id: Batch identifier
        requests: List of generation requests

    Returns:
        Batch result with individual task results
    """
    start_time = time.time()
    results = []

    try:
        logger.info(f"Starting batch generation {batch_id} with {len(requests)} tasks")

        # Create sub-tasks for each request
        subtasks = []
        for i, request in enumerate(requests):
            task_id = f"{batch_id}_{i}"

            # Create task in repository
            asyncio.run(
                self.task_repository.create_task(
                    task_id,
                    {
                        "batch_id": batch_id,
                        "batch_index": i,
                        "request": request,
                        "priority": request.get("priority", TaskPriority.NORMAL.value),
                    },
                )
            )

            # Create Celery signature
            sig = generate_music_task.s(task_id, request)
            subtasks.append(sig)

        # Execute batch as a group
        job = group(subtasks)
        result = job.apply_async()

        # Wait for completion with progress updates
        while not result.ready():
            completed = sum(1 for r in result.results if r.ready())
            progress = completed / len(requests)

            self.update_state(
                state="PROCESSING",
                meta={
                    "progress": progress,
                    "completed": completed,
                    "total": len(requests),
                    "batch_id": batch_id,
                },
            )

            time.sleep(5)  # Check every 5 seconds

        # Collect results
        for i, task_result in enumerate(result.results):
            try:
                results.append(task_result.get())
            except Exception as e:
                results.append(
                    {
                        "task_id": f"{batch_id}_{i}",
                        "status": "failed",
                        "error": str(e),
                    }
                )

        # Calculate batch metrics
        processing_time = time.time() - start_time
        successful = sum(1 for r in results if r.get("status") == "completed")
        failed = len(results) - successful

        batch_result = {
            "batch_id": batch_id,
            "status": "completed",
            "total_tasks": len(requests),
            "successful": successful,
            "failed": failed,
            "processing_time": processing_time,
            "results": results,
            "completed_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Completed batch {batch_id}: {successful}/{len(requests)} successful in {processing_time:.2f}s"
        )

        return batch_result

    except Exception as e:
        logger.error(f"Batch {batch_id} failed: {str(e)}", exc_info=True)

        # Mark all tasks as failed
        for i in range(len(requests)):
            task_id = f"{batch_id}_{i}"
            asyncio.run(
                self.task_repository.update_task(
                    task_id,
                    {
                        "status": TaskStatus.FAILED.value,
                        "error": f"Batch failed: {str(e)}",
                    },
                )
            )

        raise


@celery_app.task(name="musicgen.workers.tasks.process_audio_task")
def process_audio_task(audio_path: str, operations: List[Dict[str, Any]]) -> str:
    """
    Process audio file with various operations.

    Args:
        audio_path: Path to audio file
        operations: List of processing operations

    Returns:
        Path to processed audio file
    """
    try:
        container = get_container()
        audio_service = container.get("AudioProcessingService")

        # Load audio
        audio_data, sample_rate = asyncio.run(audio_service.load_audio(audio_path))

        # Apply operations
        for op in operations:
            op_type = op.get("type")
            params = op.get("params", {})

            if op_type == "normalize":
                audio_data = audio_service.normalize(audio_data, **params)
            elif op_type == "trim_silence":
                audio_data = audio_service.trim_silence(audio_data, **params)
            elif op_type == "fade":
                audio_data = audio_service.apply_fade(audio_data, **params)
            # Add more operations as needed

        # Save processed audio
        output_path = audio_path.replace(".wav", "_processed.wav")
        asyncio.run(audio_service.save_audio(output_path, audio_data, sample_rate))

        return output_path

    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
        raise


# Monitoring and health check tasks


@celery_app.task(name="musicgen.workers.tasks.health_check_task")
def health_check_task() -> Dict[str, Any]:
    """Perform worker health check."""
    try:
        container = get_container()

        # Check Redis connection
        task_repo = container.get("TaskRepository")
        redis_healthy = False
        try:
            asyncio.run(task_repo.get_task("health_check_test"))
            redis_healthy = True
        except:
            pass

        # Check model service
        model_service = container.get("ModelService")
        model_healthy = False
        try:
            models = asyncio.run(model_service.list_models())
            model_healthy = len(models) > 0
        except:
            pass

        # Get worker stats
        from celery import current_app

        stats = current_app.control.inspect().stats()
        active = current_app.control.inspect().active()

        return {
            "status": "healthy" if redis_healthy and model_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "worker_id": current_task.request.id,
            "hostname": current_task.request.hostname,
            "checks": {
                "redis": redis_healthy,
                "models": model_healthy,
            },
            "stats": stats,
            "active_tasks": active,
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@celery_app.task(name="musicgen.workers.tasks.cleanup_stalled_tasks")
def cleanup_stalled_tasks() -> Dict[str, Any]:
    """Clean up stalled tasks and recover them."""
    try:
        container = get_container()
        task_repo = container.get("TaskRepository")

        # Only works with advanced Redis repository
        if hasattr(task_repo, "recover_stalled_tasks"):
            recovered = asyncio.run(task_repo.recover_stalled_tasks())

            logger.info(f"Recovered {recovered} stalled tasks")

            return {
                "recovered_tasks": recovered,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            return {
                "error": "Task recovery not supported by current repository",
                "timestamp": datetime.utcnow().isoformat(),
            }

    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@celery_app.task(name="musicgen.workers.tasks.collect_worker_metrics")
def collect_worker_metrics() -> Dict[str, Any]:
    """Collect and store worker metrics."""
    try:
        from celery import current_app

        # Get inspector
        inspector = current_app.control.inspect()

        # Collect various metrics
        stats = inspector.stats() or {}
        active = inspector.active() or {}
        reserved = inspector.reserved() or {}
        scheduled = inspector.scheduled() or {}

        # Calculate aggregate metrics
        total_workers = len(stats)
        total_active = sum(len(tasks) for tasks in active.values())
        total_reserved = sum(len(tasks) for tasks in reserved.values())
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())

        # Get queue lengths
        with current_app.connection_or_acquire() as conn:
            queue_lengths = {}
            for queue_name in ["critical", "generation-high", "generation", "processing", "batch"]:
                try:
                    queue = conn.default_channel.queue_declare(queue=queue_name, passive=True)
                    queue_lengths[queue_name] = queue.message_count
                except:
                    queue_lengths[queue_name] = 0

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "workers": {
                "total": total_workers,
                "stats": stats,
            },
            "tasks": {
                "active": total_active,
                "reserved": total_reserved,
                "scheduled": total_scheduled,
                "by_worker": {
                    "active": active,
                    "reserved": reserved,
                    "scheduled": scheduled,
                },
            },
            "queues": queue_lengths,
        }

        # Store metrics (could send to monitoring system)
        logger.info(f"Worker metrics: {total_workers} workers, {total_active} active tasks")

        return metrics

    except Exception as e:
        logger.error(f"Failed to collect metrics: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@celery_app.task(name="musicgen.workers.tasks.dead_letter_task")
def dead_letter_task(task_id: str, error_info: Dict[str, Any]) -> None:
    """
    Handle failed tasks in dead letter queue.

    Args:
        task_id: Failed task ID
        error_info: Error information
    """
    try:
        logger.error(f"Task {task_id} sent to dead letter queue: {error_info}")

        # Could implement additional handling:
        # - Send notifications
        # - Store in permanent failure log
        # - Trigger alerts

        # For now, just log
        container = get_container()
        task_repo = container.get("TaskRepository")

        asyncio.run(
            task_repo.update_task(
                task_id,
                {
                    "status": "dead_letter",
                    "dead_letter_info": error_info,
                    "dead_letter_at": datetime.utcnow().isoformat(),
                },
            )
        )

    except Exception as e:
        logger.error(f"Failed to process dead letter task: {str(e)}", exc_info=True)


def send_to_dead_letter_queue(task_id: str, error_data: Dict[str, Any]) -> None:
    """Send failed task to dead letter queue."""
    try:
        dead_letter_task.apply_async(
            args=[task_id, error_data],
            queue="dead-letter",
            routing_key="failed.task",
            priority=0,
        )
    except Exception as e:
        logger.error(f"Failed to send task to dead letter queue: {str(e)}")
