"""
Session management for streaming generation.
"""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from .audio_streamer import AdaptiveStreamer, AudioStreamer
from .generator import StreamingGenerator, create_streaming_generator

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Streaming session states."""

    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    STREAMING = "streaming"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamingRequest:
    """Request to start streaming generation."""

    prompt: str
    duration: Optional[float] = None
    chunk_duration: float = 1.0
    quality_mode: str = "balanced"  # "fast", "balanced", "quality"

    # Generation parameters
    temperature: float = 0.9
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15

    # Conditioning
    genre: Optional[str] = None
    mood: Optional[str] = None
    tempo: Optional[int] = None
    instruments: Optional[List[str]] = None

    # Streaming options
    enable_interruption: bool = True
    adaptive_quality: bool = True
    crossfade_duration: float = 0.1

    def to_generation_params(self) -> Dict[str, Any]:
        """Convert to parameters for model generation."""
        params = {}

        if self.genre:
            genre_vocab = {"jazz": 0, "classical": 1, "rock": 2, "electronic": 3, "ambient": 4}
            genre_id = genre_vocab.get(self.genre.lower(), 0)
            params["genre_ids"] = [genre_id]

        if self.mood:
            mood_vocab = {"happy": 0, "sad": 1, "energetic": 2, "calm": 3, "dramatic": 4}
            mood_id = mood_vocab.get(self.mood.lower(), 0)
            params["mood_ids"] = [mood_id]

        if self.tempo:
            params["tempo"] = [float(self.tempo)]

        return params


@dataclass
class SessionInfo:
    """Information about a streaming session."""

    session_id: str
    state: SessionState
    request: StreamingRequest
    created_at: float
    started_at: Optional[float] = None
    total_duration: float = 0.0
    chunks_generated: int = 0
    last_activity: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "request": asdict(self.request),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "total_duration": self.total_duration,
            "chunks_generated": self.chunks_generated,
            "last_activity": self.last_activity,
            "error_message": self.error_message,
        }


class StreamingSession:
    """Manages a single streaming generation session."""

    def __init__(
        self,
        session_id: str,
        model,
        request: StreamingRequest,
        event_callback: Optional[Callable] = None,
    ):
        self.session_id = session_id
        self.model = model
        self.request = request
        self.event_callback = event_callback

        # Session state
        self.info = SessionInfo(
            session_id=session_id,
            state=SessionState.CREATED,
            request=request,
            created_at=time.time(),
            last_activity=time.time(),
        )

        # Components
        self.generator: Optional[StreamingGenerator] = None
        self.audio_streamer: Optional[AudioStreamer] = None

        # Async coordination
        self.chunk_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.generation_task: Optional[asyncio.Task] = None

        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix=f"session-{session_id}"
        )

        logger.info(f"Created streaming session {session_id}")

    async def prepare(self) -> Dict[str, Any]:
        """Prepare the session for streaming."""
        try:
            self.info.state = SessionState.PREPARING
            self.info.last_activity = time.time()
            await self._emit_event("session_preparing")

            # Create streaming generator
            self.generator = create_streaming_generator(
                model=self.model,
                chunk_duration=self.request.chunk_duration,
                quality_mode=self.request.quality_mode,
                temperature=self.request.temperature,
                top_k=self.request.top_k,
                top_p=self.request.top_p,
                repetition_penalty=self.request.repetition_penalty,
                enable_interruption=self.request.enable_interruption,
                adaptive_quality=self.request.adaptive_quality,
            )

            # Create audio streamer
            if self.request.adaptive_quality:
                self.audio_streamer = AdaptiveStreamer(
                    sample_rate=self.model.audio_tokenizer.sample_rate,
                    crossfade_duration=self.request.crossfade_duration,
                )
            else:
                self.audio_streamer = AudioStreamer(
                    sample_rate=self.model.audio_tokenizer.sample_rate,
                    crossfade_duration=self.request.crossfade_duration,
                )

            # Prepare generation
            generation_params = self.request.to_generation_params()
            prepare_result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.generator.prepare_streaming(
                    texts=[self.request.prompt], **generation_params
                ),
            )

            self.info.state = SessionState.READY
            await self._emit_event("session_ready", prepare_result)

            logger.info(f"Session {self.session_id} prepared successfully")
            return prepare_result

        except Exception as e:
            logger.error(f"Failed to prepare session {self.session_id}: {e}")
            await self._handle_error(e)
            raise

    async def start_streaming(self) -> AsyncIterator[Dict[str, Any]]:
        """Start streaming generation."""
        try:
            if self.info.state != SessionState.READY:
                raise RuntimeError(f"Session not ready for streaming (state: {self.info.state})")

            self.info.state = SessionState.STREAMING
            self.info.started_at = time.time()
            self.info.last_activity = time.time()

            # Start audio streamer
            self.audio_streamer.start_streaming()

            # Start generation task
            self.generation_task = asyncio.create_task(self._generation_worker())

            await self._emit_event("streaming_started")
            logger.info(f"Started streaming for session {self.session_id}")

            # Stream chunks
            async for chunk_data in self._stream_chunks():
                yield chunk_data

        except Exception as e:
            logger.error(f"Streaming error in session {self.session_id}: {e}")
            await self._handle_error(e)
            raise
        finally:
            await self._cleanup_streaming()

    async def _generation_worker(self):
        """Background worker for token generation."""
        try:
            # Run generation in thread pool to avoid blocking
            def run_generation():
                return list(self.generator.start_streaming())

            generation_results = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, run_generation
            )

            # Process generation results
            for result in generation_results:
                if self.stop_event.is_set():
                    break

                await self.chunk_queue.put(result)

                # Update session info
                if result.get("type") == "chunk":
                    self.info.chunks_generated += 1
                    self.info.total_duration = result.get("total_duration", 0.0)
                    self.info.last_activity = time.time()

            # Signal end of generation
            await self.chunk_queue.put({"type": "generation_complete"})

        except Exception as e:
            logger.error(f"Generation worker error in session {self.session_id}: {e}")
            await self.chunk_queue.put({"type": "error", "error": str(e)})

    async def _stream_chunks(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream processed audio chunks."""

        while not self.stop_event.is_set():
            try:
                # Wait for next chunk with timeout
                chunk_data = await asyncio.wait_for(
                    self.chunk_queue.get(), timeout=self.generator.config.max_latency_ms / 1000.0
                )

                if chunk_data.get("type") == "chunk":
                    # Process audio chunk
                    processed_chunk = await self._process_audio_chunk(chunk_data)
                    yield processed_chunk

                elif chunk_data.get("type") == "generation_complete":
                    # End of generation
                    yield {"type": "stream_complete", "session_id": self.session_id}
                    break

                elif chunk_data.get("type") == "error":
                    # Generation error
                    yield chunk_data
                    break

                elif chunk_data.get("type") == "buffer_underrun":
                    # Handle buffer underrun
                    yield {
                        "type": "buffer_underrun",
                        "session_id": self.session_id,
                        "message": "Generation not keeping up with real-time",
                        "stats": chunk_data.get("stats", {}),
                    }

                else:
                    # Forward other chunk types
                    yield chunk_data

            except asyncio.TimeoutError:
                # Timeout waiting for chunk
                yield {
                    "type": "timeout",
                    "session_id": self.session_id,
                    "message": "Timeout waiting for next chunk",
                }
                break

            except Exception as e:
                logger.error(f"Error streaming chunk in session {self.session_id}: {e}")
                yield {
                    "type": "error",
                    "session_id": self.session_id,
                    "error": str(e),
                }
                break

    async def _process_audio_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw audio chunk through audio streamer."""

        try:
            # Extract audio from chunk data
            raw_audio = chunk_data.get("audio")
            if raw_audio is None:
                return {
                    "type": "error",
                    "session_id": self.session_id,
                    "error": "No audio in chunk data",
                }

            # Add to audio streamer
            chunk_id = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.audio_streamer.add_audio_chunk(
                    raw_audio, chunk_data.get("duration", self.request.chunk_duration)
                ),
            )

            # Get processed audio segment
            audio_segment = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self.audio_streamer.get_next_audio_segment
            )

            if audio_segment is None:
                return {
                    "type": "buffering",
                    "session_id": self.session_id,
                    "message": "Audio streamer buffering",
                }

            # Convert to streaming format
            processed_chunk = {
                "type": "audio_chunk",
                "session_id": self.session_id,
                "chunk_id": audio_segment.chunk_id,
                "audio": audio_segment.to_numpy(),
                "sample_rate": audio_segment.sample_rate,
                "duration": audio_segment.duration,
                "timestamp": audio_segment.timestamp,
                "is_final": audio_segment.is_final,
                "original_data": {
                    "generation_time_ms": chunk_data.get("generation_time_ms"),
                    "total_duration": chunk_data.get("total_duration"),
                },
                "buffer_info": self.audio_streamer.get_buffer_info(),
            }

            return processed_chunk

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                "type": "error",
                "session_id": self.session_id,
                "error": f"Audio processing failed: {e}",
            }

    async def pause(self):
        """Pause streaming generation."""
        if self.info.state == SessionState.STREAMING:
            self.info.state = SessionState.PAUSED
            if self.generator:
                self.generator.pause_streaming()
            await self._emit_event("streaming_paused")
            logger.info(f"Paused session {self.session_id}")

    async def resume(self):
        """Resume streaming generation."""
        if self.info.state == SessionState.PAUSED:
            self.info.state = SessionState.STREAMING
            if self.generator:
                self.generator.resume_streaming()
            await self._emit_event("streaming_resumed")
            logger.info(f"Resumed session {self.session_id}")

    async def stop(self):
        """Stop streaming generation."""
        self.info.state = SessionState.STOPPED
        self.stop_event.set()

        if self.generator:
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self.generator.stop_streaming
            )

        if self.audio_streamer:
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self.audio_streamer.stop_streaming
            )

        await self._emit_event("streaming_stopped")
        logger.info(f"Stopped session {self.session_id}")

    async def interrupt_and_modify(self, new_prompt: str, **kwargs) -> bool:
        """Interrupt and modify streaming parameters."""
        if not self.request.enable_interruption:
            return False

        if self.generator:
            success = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, lambda: self.generator.interrupt_and_modify(new_prompt, **kwargs)
            )

            if success:
                await self._emit_event("streaming_modified", {"new_prompt": new_prompt})
                logger.info(f"Modified session {self.session_id} with new prompt: {new_prompt}")

            return success

        return False

    async def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        status = self.info.to_dict()

        if self.generator:
            generator_stats = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self.generator.get_stats
            )
            status["generator_stats"] = generator_stats

        if self.audio_streamer:
            buffer_info = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, self.audio_streamer.get_buffer_info
            )
            status["audio_buffer"] = buffer_info

        return status

    async def _cleanup_streaming(self):
        """Clean up streaming resources."""
        try:
            # Cancel generation task
            if self.generation_task and not self.generation_task.done():
                self.generation_task.cancel()
                try:
                    await self.generation_task
                except asyncio.CancelledError:
                    pass

            # Stop components
            if self.audio_streamer:
                await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, self.audio_streamer.stop_streaming
                )

            logger.info(f"Cleaned up session {self.session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up session {self.session_id}: {e}")

    async def _handle_error(self, error: Exception):
        """Handle session error."""
        self.info.state = SessionState.ERROR
        self.info.error_message = str(error)
        self.info.last_activity = time.time()

        await self._emit_event("session_error", {"error": str(error)})
        await self._cleanup_streaming()

    async def _emit_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Emit session event."""
        if self.event_callback:
            event_data = {
                "session_id": self.session_id,
                "event_type": event_type,
                "timestamp": time.time(),
                "data": data or {},
            }
            try:
                await self.event_callback(event_data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def __del__(self):
        """Cleanup when session is destroyed."""
        try:
            self.thread_pool.shutdown(wait=False)
        except:
            pass


class SessionManager:
    """Manages multiple streaming sessions."""

    def __init__(self, model, max_concurrent_sessions: int = 10):
        self.model = model
        self.max_concurrent_sessions = max_concurrent_sessions

        self.sessions: Dict[str, StreamingSession] = {}
        self.session_lock = asyncio.Lock()

        # Cleanup task
        self.cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()

        logger.info(f"Session manager initialized (max sessions: {max_concurrent_sessions})")

    async def create_session(
        self,
        request: StreamingRequest,
        session_id: Optional[str] = None,
        event_callback: Optional[Callable] = None,
    ) -> str:
        """Create a new streaming session."""

        async with self.session_lock:
            # Check session limit
            if len(self.sessions) >= self.max_concurrent_sessions:
                # Clean up old sessions
                await self._cleanup_inactive_sessions()

                if len(self.sessions) >= self.max_concurrent_sessions:
                    raise RuntimeError(
                        f"Maximum concurrent sessions reached ({self.max_concurrent_sessions})"
                    )

            # Generate session ID
            if session_id is None:
                session_id = str(uuid.uuid4())

            if session_id in self.sessions:
                raise ValueError(f"Session {session_id} already exists")

            # Create session
            session = StreamingSession(session_id, self.model, request, event_callback)
            self.sessions[session_id] = session

            logger.info(f"Created session {session_id} ({len(self.sessions)} total)")
            return session_id

    async def get_session(self, session_id: str) -> Optional[StreamingSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    async def remove_session(self, session_id: str) -> bool:
        """Remove and cleanup session."""
        async with self.session_lock:
            session = self.sessions.pop(session_id, None)
            if session:
                await session.stop()
                logger.info(f"Removed session {session_id}")
                return True
            return False

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with their status."""
        sessions_info = []
        for session in self.sessions.values():
            info = await session.get_status()
            sessions_info.append(info)
        return sessions_info

    async def get_stats(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        active_count = sum(
            1
            for s in self.sessions.values()
            if s.info.state in [SessionState.STREAMING, SessionState.READY]
        )

        total_duration = sum(s.info.total_duration for s in self.sessions.values())
        total_chunks = sum(s.info.chunks_generated for s in self.sessions.values())

        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_count,
            "total_generated_duration": total_duration,
            "total_chunks_generated": total_chunks,
            "max_concurrent": self.max_concurrent_sessions,
        }

    async def _cleanup_inactive_sessions(self):
        """Clean up inactive or expired sessions."""
        current_time = time.time()
        session_timeout = 300  # 5 minutes

        to_remove = []
        for session_id, session in self.sessions.items():
            # Remove sessions that are stopped, errored, or inactive
            if (
                session.info.state in [SessionState.STOPPED, SessionState.ERROR]
                or current_time - session.info.last_activity > session_timeout
            ):
                to_remove.append(session_id)

        for session_id in to_remove:
            await self.remove_session(session_id)

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive sessions")

    def _start_cleanup_task(self):
        """Start background cleanup task."""

        async def cleanup_worker():
            while True:
                try:
                    await asyncio.sleep(60)  # Run every minute
                    await self._cleanup_inactive_sessions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")

        self.cleanup_task = asyncio.create_task(cleanup_worker())

    async def shutdown(self):
        """Shutdown session manager."""
        logger.info("Shutting down session manager")

        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop all sessions
        for session_id in list(self.sessions.keys()):
            await self.remove_session(session_id)

        logger.info("Session manager shutdown complete")
