"""
Background worker system for Music Gen AI.

This package provides Celery-based workers for handling music generation tasks.
"""

from .celery_app import celery_app
from .tasks import generate_batch_task, generate_music_task

__all__ = ["celery_app", "generate_music_task", "generate_batch_task"]
