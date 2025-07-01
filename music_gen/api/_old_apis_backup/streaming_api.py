"""
WebSocket API for real-time streaming generation.
"""

import json
import logging
import time
from typing import Any, Dict

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..streaming import (
    NetworkConditionMonitor,
    SessionManager,
    StreamingProtocol,
    StreamingRequest,
    StreamingSession,
    audio_to_base64,
    validate_streaming_request,
)

logger = logging.getLogger(__name__)


class WebSocketStreamingManager:
    """Manages WebSocket connections for streaming."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_sessions: Dict[str, str] = {}  # connection_id -> session_id
        self.network_monitors: Dict[str, NetworkConditionMonitor] = {}

    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.network_monitors[connection_id] = NetworkConditionMonitor()

        logger.info(f"WebSocket connected: {connection_id}")

        # Send welcome message
        welcome_msg = StreamingProtocol.create_status_message(
            session_id="", status="connected", details={"connection_id": connection_id}
        )
        await self.send_message(connection_id, welcome_msg)

    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        # Stop associated session if any
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            await self.session_manager.remove_session(session_id)
            del self.connection_sessions[connection_id]

        # Remove connection
        self.active_connections.pop(connection_id, None)
        self.network_monitors.pop(connection_id, None)

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to WebSocket connection."""
        websocket = self.active_connections.get(connection_id)
        if not websocket:
            return False

        try:
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False

    async def handle_message(self, connection_id: str, message: Dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = message.get("type")

        try:
            if message_type == "start_streaming":
                await self._handle_start_streaming(connection_id, message)
            elif message_type == "stop_streaming":
                await self._handle_stop_streaming(connection_id, message)
            elif message_type == "pause_streaming":
                await self._handle_pause_streaming(connection_id, message)
            elif message_type == "resume_streaming":
                await self._handle_resume_streaming(connection_id, message)
            elif message_type == "modify_streaming":
                await self._handle_modify_streaming(connection_id, message)
            elif message_type == "get_status":
                await self._handle_get_status(connection_id, message)
            elif message_type == "heartbeat":
                await self._handle_heartbeat(connection_id, message)
            else:
                error_msg = StreamingProtocol.create_error_message(
                    session_id="",
                    error_code="UNKNOWN_MESSAGE_TYPE",
                    error_message=f"Unknown message type: {message_type}",
                )
                await self.send_message(connection_id, error_msg)

        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")
            error_msg = StreamingProtocol.create_error_message(
                session_id="", error_code="MESSAGE_HANDLING_ERROR", error_message=str(e)
            )
            await self.send_message(connection_id, error_msg)

    async def _handle_start_streaming(self, connection_id: str, message: Dict[str, Any]):
        """Handle start streaming request."""

        # Validate request
        request_data = message.get("request", {})
        validation_errors = validate_streaming_request(request_data)

        if validation_errors:
            error_msg = StreamingProtocol.create_error_message(
                session_id="",
                error_code="VALIDATION_ERROR",
                error_message="Request validation failed",
                details={"errors": validation_errors},
            )
            await self.send_message(connection_id, error_msg)
            return

        # Check network conditions for adaptive settings
        network_monitor = self.network_monitors.get(connection_id)
        if network_monitor:
            adaptive_settings = network_monitor.get_adaptive_settings()
            # Apply adaptive settings to request
            for key, value in adaptive_settings.items():
                if key not in request_data:
                    request_data[key] = value

        try:
            # Create streaming request
            streaming_request = StreamingRequest(**request_data)

            # Create session with event callback
            session_id = await self.session_manager.create_session(
                request=streaming_request,
                event_callback=lambda event: self._session_event_callback(connection_id, event),
            )

            # Associate connection with session
            self.connection_sessions[connection_id] = session_id

            # Prepare session
            session = await self.session_manager.get_session(session_id)
            prepare_result = await session.prepare()

            # Send preparation result
            status_msg = StreamingProtocol.create_status_message(
                session_id=session_id, status="prepared", details=prepare_result
            )
            await self.send_message(connection_id, status_msg)

            # Start streaming
            await self._start_session_streaming(connection_id, session)

        except Exception as e:
            logger.error(f"Failed to start streaming for {connection_id}: {e}")
            error_msg = StreamingProtocol.create_error_message(
                session_id="", error_code="STREAMING_START_ERROR", error_message=str(e)
            )
            await self.send_message(connection_id, error_msg)

    async def _start_session_streaming(self, connection_id: str, session: StreamingSession):
        """Start streaming for a session."""

        try:
            async for chunk_data in session.start_streaming():
                if connection_id not in self.active_connections:
                    # Connection closed
                    await session.stop()
                    break

                # Convert chunk to WebSocket message
                if chunk_data.get("type") == "audio_chunk":
                    # Convert audio to base64
                    audio_tensor = chunk_data["audio"]
                    audio_b64 = audio_to_base64(audio_tensor, chunk_data["sample_rate"])

                    # Create streaming message
                    chunk_msg = StreamingProtocol.create_chunk_message(
                        session_id=session.session_id,
                        chunk_id=chunk_data["chunk_id"],
                        audio_data=audio_b64,
                        sample_rate=chunk_data["sample_rate"],
                        duration=chunk_data["duration"],
                        metadata=chunk_data.get("original_data", {}),
                    )

                    # Track network performance
                    start_time = time.time()
                    success = await self.send_message(connection_id, chunk_msg)
                    latency = time.time() - start_time

                    # Update network monitor
                    network_monitor = self.network_monitors.get(connection_id)
                    if network_monitor:
                        chunk_size = len(audio_b64) if audio_b64 else 0
                        network_monitor.record_request(latency, success, chunk_size)

                    if not success:
                        # Failed to send, stop streaming
                        await session.stop()
                        break

                elif chunk_data.get("type") in ["buffer_underrun", "timeout"]:
                    # Send status update
                    status_msg = StreamingProtocol.create_status_message(
                        session_id=session.session_id, status=chunk_data["type"], details=chunk_data
                    )
                    await self.send_message(connection_id, status_msg)

                elif chunk_data.get("type") == "error":
                    # Send error message
                    error_msg = StreamingProtocol.create_error_message(
                        session_id=session.session_id,
                        error_code="GENERATION_ERROR",
                        error_message=chunk_data.get("error", "Unknown error"),
                    )
                    await self.send_message(connection_id, error_msg)
                    break

                elif chunk_data.get("type") == "stream_complete":
                    # Send completion message
                    status_msg = StreamingProtocol.create_status_message(
                        session_id=session.session_id,
                        status="completed",
                        details={"total_duration": session.info.total_duration},
                    )
                    await self.send_message(connection_id, status_msg)
                    break

        except Exception as e:
            logger.error(f"Streaming error for connection {connection_id}: {e}")
            error_msg = StreamingProtocol.create_error_message(
                session_id=session.session_id, error_code="STREAMING_ERROR", error_message=str(e)
            )
            await self.send_message(connection_id, error_msg)

    async def _handle_stop_streaming(self, connection_id: str, message: Dict[str, Any]):
        """Handle stop streaming request."""
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            session = await self.session_manager.get_session(session_id)
            if session:
                await session.stop()

                status_msg = StreamingProtocol.create_status_message(
                    session_id=session_id, status="stopped"
                )
                await self.send_message(connection_id, status_msg)

    async def _handle_pause_streaming(self, connection_id: str, message: Dict[str, Any]):
        """Handle pause streaming request."""
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            session = await self.session_manager.get_session(session_id)
            if session:
                await session.pause()

                status_msg = StreamingProtocol.create_status_message(
                    session_id=session_id, status="paused"
                )
                await self.send_message(connection_id, status_msg)

    async def _handle_resume_streaming(self, connection_id: str, message: Dict[str, Any]):
        """Handle resume streaming request."""
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            session = await self.session_manager.get_session(session_id)
            if session:
                await session.resume()

                status_msg = StreamingProtocol.create_status_message(
                    session_id=session_id, status="resumed"
                )
                await self.send_message(connection_id, status_msg)

    async def _handle_modify_streaming(self, connection_id: str, message: Dict[str, Any]):
        """Handle modify streaming request."""
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            session = await self.session_manager.get_session(session_id)
            if session:
                new_prompt = message.get("new_prompt", "")
                if new_prompt:
                    success = await session.interrupt_and_modify(new_prompt)

                    status_msg = StreamingProtocol.create_status_message(
                        session_id=session_id,
                        status="modified" if success else "modification_failed",
                        details={"new_prompt": new_prompt},
                    )
                    await self.send_message(connection_id, status_msg)

    async def _handle_get_status(self, connection_id: str, message: Dict[str, Any]):
        """Handle get status request."""
        if connection_id in self.connection_sessions:
            session_id = self.connection_sessions[connection_id]
            session = await self.session_manager.get_session(session_id)
            if session:
                status = await session.get_status()

                status_msg = StreamingProtocol.create_status_message(
                    session_id=session_id, status="status_update", details=status
                )
                await self.send_message(connection_id, status_msg)

    async def _handle_heartbeat(self, connection_id: str, message: Dict[str, Any]):
        """Handle heartbeat message."""
        heartbeat_msg = StreamingProtocol.create_heartbeat_message("")
        await self.send_message(connection_id, heartbeat_msg)

    async def _session_event_callback(self, connection_id: str, event: Dict[str, Any]):
        """Callback for session events."""
        status_msg = StreamingProtocol.create_status_message(
            session_id=event.get("session_id", ""),
            status=event.get("event_type", "unknown"),
            details=event.get("data", {}),
        )
        await self.send_message(connection_id, status_msg)


def setup_streaming_routes(app, model, session_manager: SessionManager):
    """Set up streaming WebSocket routes."""

    streaming_manager = WebSocketStreamingManager(session_manager)

    @app.websocket("/ws/stream/{connection_id}")
    async def websocket_streaming_endpoint(websocket: WebSocket, connection_id: str):
        """WebSocket endpoint for real-time streaming."""

        await streaming_manager.connect(websocket, connection_id)

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle message
                await streaming_manager.handle_message(connection_id, message)

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            await streaming_manager.disconnect(connection_id)

    @app.get("/stream/demo")
    async def streaming_demo():
        """Demo page for streaming interface."""
        return HTMLResponse(content=STREAMING_DEMO_HTML)

    @app.get("/stream/sessions")
    async def list_streaming_sessions():
        """List all active streaming sessions."""
        sessions = await session_manager.list_sessions()
        return {"sessions": sessions}

    @app.get("/stream/stats")
    async def get_streaming_stats():
        """Get streaming statistics."""
        stats = await session_manager.get_stats()
        return {"stats": stats}

    @app.delete("/stream/sessions/{session_id}")
    async def stop_streaming_session(session_id: str):
        """Stop a specific streaming session."""
        success = await session_manager.remove_session(session_id)
        if success:
            return {"message": f"Session {session_id} stopped"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")


# Demo HTML for testing streaming
STREAMING_DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>MusicGen Streaming Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .controls { margin: 20px 0; }
        .controls input, .controls select, .controls button {
            margin: 5px; padding: 8px;
        }
        .status {
            background: #f0f0f0;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .error { background: #ffebee; color: #c62828; }
        .success { background: #e8f5e8; color: #2e7d32; }
        .info { background: #e3f2fd; color: #1565c0; }
        #audioPlayer { width: 100%; margin: 10px 0; }
        .audio-chunk {
            margin: 5px 0;
            padding: 5px;
            background: #f5f5f5;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MusicGen Real-time Streaming Demo</h1>

        <div class="controls">
            <h3>Generate Music</h3>
            <input type="text" id="promptInput" placeholder="Enter music description..."
                   value="upbeat jazz piano with drums">
            <br>
            <label>Genre:</label>
            <select id="genreSelect">
                <option value="">Auto</option>
                <option value="jazz">Jazz</option>
                <option value="classical">Classical</option>
                <option value="rock">Rock</option>
                <option value="electronic">Electronic</option>
                <option value="ambient">Ambient</option>
            </select>

            <label>Mood:</label>
            <select id="moodSelect">
                <option value="">Auto</option>
                <option value="happy">Happy</option>
                <option value="sad">Sad</option>
                <option value="energetic">Energetic</option>
                <option value="calm">Calm</option>
                <option value="dramatic">Dramatic</option>
            </select>

            <label>Quality:</label>
            <select id="qualitySelect">
                <option value="balanced">Balanced</option>
                <option value="fast">Fast</option>
                <option value="quality">High Quality</option>
            </select>
            <br>

            <button onclick="startStreaming()">Start Streaming</button>
            <button onclick="pauseStreaming()">Pause</button>
            <button onclick="resumeStreaming()">Resume</button>
            <button onclick="stopStreaming()">Stop</button>
        </div>

        <div class="status" id="status">Ready to stream...</div>

        <audio id="audioPlayer" controls style="display: none;">
            Your browser does not support the audio element.
        </audio>

        <div id="audioChunks"></div>

        <h3>Network & Performance</h3>
        <div id="networkStats" class="status info">
            Latency: -- | Chunks: -- | Buffer: --
        </div>
    </div>

    <script>
        let websocket = null;
        let sessionId = null;
        let audioContext = null;
        let audioBuffer = [];
        let isPlaying = false;
        let chunkCount = 0;

        const connectionId = 'demo_' + Math.random().toString(36).substr(2, 9);

        function updateStatus(message, type = 'info') {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + type;
        }

        function addAudioChunk(chunkData) {
            chunkCount++;
            const chunksDiv = document.getElementById('audioChunks');
            const chunkDiv = document.createElement('div');
            chunkDiv.className = 'audio-chunk';
            chunkDiv.textContent = `Chunk ${chunkData.chunk_id}: ${chunkData.duration.toFixed(2)}s`;
            chunksDiv.appendChild(chunkDiv);

            // Keep only last 10 chunks visible
            while (chunksDiv.children.length > 10) {
                chunksDiv.removeChild(chunksDiv.firstChild);
            }
        }

        function updateNetworkStats(latency, chunks, bufferSize) {
            document.getElementById('networkStats').textContent =
                `Latency: ${latency}ms | Chunks: ${chunks} | Buffer: ${bufferSize}`;
        }

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/stream/${connectionId}`;

            websocket = new WebSocket(wsUrl);

            websocket.onopen = function(event) {
                updateStatus('Connected to streaming server', 'success');
            };

            websocket.onmessage = function(event) {
                const message = JSON.parse(event.data);

                if (message.type === 'audio_chunk') {
                    addAudioChunk(message);
                    // In a real implementation, you'd decode and play the audio
                    updateStatus(`Streaming... (chunk ${message.chunk_id})`, 'success');

                } else if (message.type === 'status_update') {
                    updateStatus(`Status: ${message.status}`, 'info');

                    if (message.status === 'prepared') {
                        sessionId = message.session_id;
                    }

                } else if (message.type === 'error') {
                    updateStatus(`Error: ${message.error_message}`, 'error');

                } else if (message.type === 'heartbeat') {
                    // Handle heartbeat
                }
            };

            websocket.onclose = function(event) {
                updateStatus('Disconnected from server', 'error');
                websocket = null;
                sessionId = null;
            };

            websocket.onerror = function(error) {
                updateStatus('WebSocket error occurred', 'error');
            };
        }

        function startStreaming() {
            if (!websocket) {
                connectWebSocket();
                // Wait a bit for connection
                setTimeout(startStreaming, 1000);
                return;
            }

            const prompt = document.getElementById('promptInput').value;
            const genre = document.getElementById('genreSelect').value;
            const mood = document.getElementById('moodSelect').value;
            const quality = document.getElementById('qualitySelect').value;

            if (!prompt.trim()) {
                updateStatus('Please enter a prompt', 'error');
                return;
            }

            const request = {
                type: 'start_streaming',
                request: {
                    prompt: prompt,
                    chunk_duration: 1.0,
                    quality_mode: quality,
                    temperature: 0.9,
                    enable_interruption: true,
                    adaptive_quality: true
                }
            };

            if (genre) request.request.genre = genre;
            if (mood) request.request.mood = mood;

            chunkCount = 0;
            document.getElementById('audioChunks').innerHTML = '';

            websocket.send(JSON.stringify(request));
            updateStatus('Starting streaming...', 'info');
        }

        function pauseStreaming() {
            if (websocket && sessionId) {
                websocket.send(JSON.stringify({
                    type: 'pause_streaming',
                    session_id: sessionId
                }));
            }
        }

        function resumeStreaming() {
            if (websocket && sessionId) {
                websocket.send(JSON.stringify({
                    type: 'resume_streaming',
                    session_id: sessionId
                }));
            }
        }

        function stopStreaming() {
            if (websocket && sessionId) {
                websocket.send(JSON.stringify({
                    type: 'stop_streaming',
                    session_id: sessionId
                }));
            }
        }

        // Initialize
        window.onload = function() {
            updateStatus('Click "Start Streaming" to begin', 'info');
        };

        // Clean up on page unload
        window.onbeforeunload = function() {
            if (websocket) {
                stopStreaming();
            }
        };
    </script>
</body>
</html>
"""
