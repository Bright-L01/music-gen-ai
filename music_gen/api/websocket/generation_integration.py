"""
Integration between generation tasks and WebSocket updates.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from ...core.task_queue import TaskQueue, TaskStatus
from .manager import WebSocketManager

logger = logging.getLogger(__name__)


class GenerationWebSocketIntegration:
    """
    Integrates music generation with WebSocket for real-time updates.

    Features:
    - Progress updates during generation
    - Status changes notifications
    - Audio chunk streaming
    - Error notifications
    """

    def __init__(
        self,
        ws_manager: WebSocketManager,
        task_queue: TaskQueue,
    ):
        self.ws_manager = ws_manager
        self.task_queue = task_queue
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start monitoring generation tasks."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_tasks())
        logger.info("Generation WebSocket integration started")

    async def stop(self):
        """Stop monitoring generation tasks."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Generation WebSocket integration stopped")

    async def _monitor_tasks(self):
        """Monitor generation tasks and send updates."""
        while self._running:
            try:
                # Get all active tasks
                active_tasks = await self._get_active_tasks()

                # Send updates for each task
                for task in active_tasks:
                    await self._send_task_update(task)

                await asyncio.sleep(1)  # Update interval

            except Exception as e:
                logger.error(f"Task monitor error: {e}")
                await asyncio.sleep(1)

    async def _get_active_tasks(self) -> list:
        """Get all active generation tasks."""
        # TODO: Implement task retrieval from task queue
        return []

    async def _send_task_update(self, task: Dict[str, Any]):
        """Send update for a specific task."""
        user_id = task.get("user_id")
        if not user_id:
            return

        # Prepare update data
        update_data = {
            "id": task["id"],
            "status": task["status"],
            "progress": task.get("progress", 0),
            "stage": task.get("stage", "initializing"),
        }

        # Add additional data based on status
        if task["status"] == TaskStatus.COMPLETED:
            update_data.update(
                {
                    "audioUrl": task.get("result", {}).get("audio_url"),
                    "waveformData": task.get("result", {}).get("waveform_data"),
                    "duration": task.get("result", {}).get("duration"),
                }
            )
        elif task["status"] == TaskStatus.FAILED:
            update_data["error"] = task.get("error", "Generation failed")

        # Send update via WebSocket
        await self.ws_manager.send_generation_update(
            user_id=user_id,
            task_id=task["id"],
            status=task["status"],
            progress=update_data["progress"],
            stage=update_data["stage"],
            data=update_data,
        )

    async def notify_generation_started(
        self,
        user_id: str,
        task_id: str,
        prompt: str,
        params: Dict[str, Any],
    ):
        """Notify user that generation has started."""
        await self.ws_manager.send_generation_update(
            user_id=user_id,
            task_id=task_id,
            status="pending",
            progress=0,
            stage="queued",
            data={
                "prompt": prompt,
                "parameters": params,
                "queued_at": datetime.utcnow().isoformat(),
            },
        )

    async def notify_generation_progress(
        self,
        user_id: str,
        task_id: str,
        progress: float,
        stage: str,
        chunk_data: Optional[bytes] = None,
    ):
        """Send generation progress update."""
        data = {
            "stage_details": self._get_stage_details(stage),
        }

        # If we have audio chunk, send it separately
        if chunk_data:
            # Send to generation-specific room for streaming
            await self.ws_manager.send_to_room(
                f"generation:{task_id}",
                {
                    "type": "audio_chunk",
                    "data": {
                        "task_id": task_id,
                        "chunk": chunk_data.hex(),  # Convert to hex for JSON
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                },
            )

        await self.ws_manager.send_generation_update(
            user_id=user_id,
            task_id=task_id,
            status="processing",
            progress=progress,
            stage=stage,
            data=data,
        )

    async def notify_generation_completed(
        self,
        user_id: str,
        task_id: str,
        audio_url: str,
        duration: float,
        waveform_data: list,
        metadata: Dict[str, Any],
    ):
        """Notify user that generation has completed."""
        await self.ws_manager.send_generation_update(
            user_id=user_id,
            task_id=task_id,
            status="completed",
            progress=1.0,
            stage="completed",
            data={
                "audioUrl": audio_url,
                "duration": duration,
                "waveformData": waveform_data,
                "metadata": metadata,
                "completed_at": datetime.utcnow().isoformat(),
            },
        )

    async def notify_generation_failed(
        self,
        user_id: str,
        task_id: str,
        error: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Notify user that generation has failed."""
        await self.ws_manager.send_generation_update(
            user_id=user_id,
            task_id=task_id,
            status="failed",
            progress=0,
            stage="error",
            data={
                "error": error,
                "details": details or {},
                "failed_at": datetime.utcnow().isoformat(),
            },
        )

    def _get_stage_details(self, stage: str) -> Dict[str, Any]:
        """Get detailed information about generation stage."""
        stage_info = {
            "initializing": {
                "description": "Preparing generation models",
                "estimated_duration": 5,
            },
            "text_encoding": {
                "description": "Processing text prompt",
                "estimated_duration": 2,
            },
            "audio_generation": {
                "description": "Generating audio tokens",
                "estimated_duration": 30,
            },
            "audio_decoding": {
                "description": "Decoding audio from tokens",
                "estimated_duration": 10,
            },
            "post_processing": {
                "description": "Applying final audio processing",
                "estimated_duration": 5,
            },
            "saving": {
                "description": "Saving generated audio",
                "estimated_duration": 2,
            },
        }

        return stage_info.get(
            stage,
            {
                "description": stage,
                "estimated_duration": 0,
            },
        )
