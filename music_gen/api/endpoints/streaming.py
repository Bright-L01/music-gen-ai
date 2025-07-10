"""
Streaming endpoints for Music Gen AI API.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ...streaming import SessionManager
from ...streaming.generator import StreamingConfig

router = APIRouter()

# Global session manager
session_manager: Optional[SessionManager] = None


class StreamingRequest(BaseModel):
    """Request model for streaming generation."""

    prompt: str = Field(..., description="Text description of the music")
    duration: float = Field(10.0, ge=1.0, le=60.0, description="Duration in seconds")
    chunk_duration: float = Field(1.0, ge=0.5, le=5.0, description="Chunk duration in seconds")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    guidance_scale: float = Field(3.0, ge=1.0, le=5.0, description="Guidance scale")


class StreamingResponse(BaseModel):
    """Response model for streaming session."""

    session_id: str = Field(..., description="Streaming session ID")
    websocket_url: str = Field(..., description="WebSocket URL for streaming")


@router.on_event("startup")
async def startup_streaming():
    """Initialize streaming manager on startup."""
    global session_manager

    # Initialize session manager if not already done
    if session_manager is None:
        from ...core.model_manager import ModelManager

        model_manager = ModelManager()

        # Only initialize if we have models
        if model_manager.has_loaded_models():
            session_manager = SessionManager(max_sessions=10)


@router.post("/session", response_model=StreamingResponse)
async def create_streaming_session(request: StreamingRequest):
    """Create a new streaming session."""

    if session_manager is None:
        raise HTTPException(status_code=503, detail="Streaming service not available")

    # Create session
    session_id = str(uuid.uuid4())

    # Configure streaming
    config = StreamingConfig(
        chunk_duration=request.chunk_duration,
        temperature=request.temperature,
    )

    # Create session
    session = session_manager.create_session(
        session_id=session_id,
        prompt=request.prompt,
        duration=request.duration,
        config=config,
    )

    if session is None:
        raise HTTPException(status_code=503, detail="Could not create streaming session")

    return StreamingResponse(
        session_id=session_id,
        websocket_url=f"/api/v1/stream/ws/{session_id}",
    )


@router.websocket("/ws/{session_id}")
async def streaming_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming audio."""

    if session_manager is None:
        await websocket.close(code=1003, reason="Streaming service not available")
        return

    # Get session
    session = session_manager.get_session(session_id)
    if session is None:
        await websocket.close(code=1003, reason="Session not found")
        return

    await websocket.accept()

    try:
        # Start generation
        await session.start()

        # Stream audio chunks
        async for chunk in session.stream():
            # Send audio chunk as binary data
            await websocket.send_bytes(chunk.tobytes())

            # Send metadata
            await websocket.send_json(
                {
                    "type": "chunk",
                    "chunk_index": session.current_chunk,
                    "total_chunks": session.total_chunks,
                    "progress": session.progress,
                }
            )

        # Send completion message
        await websocket.send_json(
            {
                "type": "complete",
                "duration": session.generated_duration,
                "chunks": session.current_chunk,
            }
        )

    except WebSocketDisconnect:
        # Client disconnected
        await session.stop()
    except Exception as e:
        # Send error message
        await websocket.send_json(
            {
                "type": "error",
                "error": str(e),
            }
        )
        await websocket.close(code=1003, reason=str(e))
    finally:
        # Cleanup session
        session_manager.remove_session(session_id)


@router.delete("/session/{session_id}")
async def stop_streaming_session(session_id: str):
    """Stop a streaming session."""

    if session_manager is None:
        raise HTTPException(status_code=503, detail="Streaming service not available")

    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Stop session
    await session.stop()
    session_manager.remove_session(session_id)

    return {"message": "Session stopped", "session_id": session_id}


@router.get("/sessions")
async def list_active_sessions():
    """List all active streaming sessions."""

    if session_manager is None:
        return {"sessions": [], "count": 0}

    sessions = session_manager.list_sessions()

    return {
        "sessions": [
            {
                "session_id": s["session_id"],
                "prompt": s["prompt"],
                "duration": s["duration"],
                "progress": s["progress"],
                "status": s["status"],
                "created_at": s["created_at"],
            }
            for s in sessions
        ],
        "count": len(sessions),
        "max_sessions": session_manager.max_sessions,
    }
