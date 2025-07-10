"""
Celery application configuration for Music Gen AI workers.

This module configures Celery with Redis as the broker and result backend,
with advanced features like task routing, retries, and monitoring.
"""

import os
from datetime import timedelta

from celery import Celery
from kombu import Exchange, Queue

# Get configuration from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Create Celery application
celery_app = Celery("musicgen")

# Task routing configuration
task_routes = {
    "musicgen.workers.tasks.generate_music_task": {
        "queue": "generation",
        "routing_key": "generation.music",
    },
    "musicgen.workers.tasks.generate_batch_task": {
        "queue": "generation",
        "routing_key": "generation.batch",
    },
    "musicgen.workers.tasks.process_audio_task": {
        "queue": "processing",
        "routing_key": "processing.audio",
    },
    "musicgen.workers.tasks.health_check_task": {
        "queue": "monitoring",
        "routing_key": "monitoring.health",
    },
}


# Queue configuration with priorities
class CeleryConfig:
    """Celery configuration class."""

    # Broker settings
    broker_url = CELERY_BROKER_URL
    result_backend = CELERY_RESULT_BACKEND

    # Task settings
    task_serializer = "json"
    accept_content = ["json"]
    result_serializer = "json"
    timezone = "UTC"
    enable_utc = True

    # Task execution settings
    task_acks_late = True  # Tasks acknowledged after completion
    task_reject_on_worker_lost = True  # Reject tasks if worker dies
    task_track_started = True  # Track when tasks start
    task_time_limit = 3600  # 1 hour hard limit
    task_soft_time_limit = 3300  # 55 minute soft limit

    # Result backend settings
    result_expires = 86400  # Results expire after 24 hours
    result_persistent = True  # Persist results

    # Worker settings
    worker_prefetch_multiplier = 1  # Disable prefetching for long tasks
    worker_max_tasks_per_child = 50  # Restart worker after 50 tasks (memory management)
    worker_disable_rate_limits = False
    worker_send_task_events = True  # Enable task events for monitoring

    # Queue definitions with priority support
    task_queues = (
        # Critical priority queue
        Queue(
            "critical", Exchange("musicgen", type="topic"), routing_key="*.critical", priority=10
        ),
        # High priority generation queue
        Queue(
            "generation-high",
            Exchange("musicgen", type="topic"),
            routing_key="generation.high.*",
            priority=7,
        ),
        # Normal generation queue
        Queue(
            "generation", Exchange("musicgen", type="topic"), routing_key="generation.*", priority=5
        ),
        # Audio processing queue
        Queue(
            "processing", Exchange("musicgen", type="topic"), routing_key="processing.*", priority=3
        ),
        # Low priority batch queue
        Queue("batch", Exchange("musicgen", type="topic"), routing_key="batch.*", priority=1),
        # Monitoring queue
        Queue(
            "monitoring", Exchange("musicgen", type="topic"), routing_key="monitoring.*", priority=0
        ),
        # Dead letter queue for failed tasks
        Queue(
            "dead-letter",
            Exchange("musicgen-dlx", type="topic"),
            routing_key="failed.*",
            priority=0,
            queue_arguments={
                "x-message-ttl": 604800000,  # 7 days in milliseconds
            },
        ),
    )

    # Task routing
    task_routes = task_routes

    # Retry configuration
    task_default_retry_delay = 60  # 1 minute
    task_max_retries = 3

    # Task annotations for specific retry policies
    task_annotations = {
        "musicgen.workers.tasks.generate_music_task": {
            "rate_limit": "100/m",  # 100 tasks per minute
            "max_retries": 3,
            "default_retry_delay": 60,
            "retry_backoff": True,
            "retry_backoff_max": 600,  # Max 10 minutes
            "retry_jitter": True,
        },
        "musicgen.workers.tasks.generate_batch_task": {
            "rate_limit": "20/m",  # 20 batch tasks per minute
            "max_retries": 2,
            "default_retry_delay": 120,
        },
    }

    # Beat schedule for periodic tasks
    beat_schedule = {
        "health-check": {
            "task": "musicgen.workers.tasks.health_check_task",
            "schedule": timedelta(minutes=5),
            "options": {
                "queue": "monitoring",
                "priority": 0,
            },
        },
        "cleanup-stalled-tasks": {
            "task": "musicgen.workers.tasks.cleanup_stalled_tasks",
            "schedule": timedelta(minutes=15),
            "options": {
                "queue": "monitoring",
                "priority": 1,
            },
        },
        "collect-metrics": {
            "task": "musicgen.workers.tasks.collect_worker_metrics",
            "schedule": timedelta(minutes=1),
            "options": {
                "queue": "monitoring",
                "priority": 0,
            },
        },
    }

    # Error handling
    task_default_queue = "generation"
    task_default_exchange = "musicgen"
    task_default_routing_key = "generation.default"

    # Monitoring and events
    worker_send_task_events = True
    task_send_sent_event = True

    # Redis-specific optimizations
    broker_transport_options = {
        "visibility_timeout": 3600,  # 1 hour
        "fanout_prefix": True,
        "fanout_patterns": True,
        "priority_steps": list(range(11)),  # 0-10 priority levels
        "sep": ":",
        "queue_order_strategy": "priority",
    }

    # Result backend options
    result_backend_transport_options = {
        "master_name": "mymaster",  # For Redis Sentinel
    }


# Apply configuration
celery_app.config_from_object(CeleryConfig())


# Configure error handling
@celery_app.task(bind=True, name="musicgen.workers.error_handler")
def error_handler(self, uuid):
    """Handle task errors and send to dead letter queue."""
    result = self.app.AsyncResult(uuid)
    exc = result.get(propagate=False)

    # Log error details
    import logging

    logger = logging.getLogger(__name__)
    logger.error(f"Task {uuid} failed with exception: {exc}")

    # Send to dead letter queue
    celery_app.send_task(
        "musicgen.workers.tasks.dead_letter_task",
        args=[uuid, str(exc)],
        queue="dead-letter",
        routing_key="failed.task",
    )


# Task failure handler
@celery_app.task(bind=True)
def on_task_failure(self, exc, task_id, args, kwargs, einfo):
    """Handle task failures."""
    import logging

    logger = logging.getLogger(__name__)

    logger.error(f"Task {task_id} failed: {exc}")

    # Update task repository if available
    try:
        from music_gen.core.container import get_container

        container = get_container()
        task_repo = container.get("TaskRepository")

        import asyncio

        asyncio.run(
            task_repo.update_task(
                task_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "error_traceback": str(einfo),
                },
            )
        )
    except Exception as e:
        logger.error(f"Failed to update task repository: {e}")


# Worker initialization
@celery_app.task
def worker_init(**kwargs):
    """Initialize worker with necessary resources."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Worker initialized")

    # Warm up models if needed
    try:
        from music_gen.core.container import get_container

        container = get_container()
        model_service = container.get("ModelService")

        import asyncio

        asyncio.run(model_service.load_model("musicgen-small"))
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to preload model: {e}")


# Worker shutdown
@celery_app.task
def worker_shutdown(**kwargs):
    """Clean up worker resources."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Worker shutting down")

    # Clean up resources
    try:
        from music_gen.core.container import get_container

        container = get_container()

        # Close repository connections
        task_repo = container.get("TaskRepository")
        if hasattr(task_repo, "shutdown"):
            import asyncio

            asyncio.run(task_repo.shutdown())
    except Exception as e:
        logger.error(f"Error during worker shutdown: {e}")


# Signal handlers for worker lifecycle
from celery.signals import task_failure, task_postrun, task_prerun, worker_ready, worker_shutdown


@worker_ready.connect
def on_worker_ready(sender=None, **kwargs):
    """Handle worker ready signal."""
    worker_init.delay()


@worker_shutdown.connect
def on_worker_shutdown(sender=None, **kwargs):
    """Handle worker shutdown signal."""
    worker_shutdown.delay()


@task_prerun.connect
def on_task_prerun(sender=None, task_id=None, task=None, **kwargs):
    """Handle task pre-run signal."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Task {task_id} starting: {task.name}")


@task_postrun.connect
def on_task_postrun(sender=None, task_id=None, task=None, state=None, **kwargs):
    """Handle task post-run signal."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Task {task_id} completed: {task.name} (state: {state})")


@task_failure.connect
def on_task_failure_signal(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failure signal."""
    error_handler.delay(task_id)


# Export for easy access
__all__ = ["celery_app", "CeleryConfig"]
