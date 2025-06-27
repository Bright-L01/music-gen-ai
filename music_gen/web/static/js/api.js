// API module for MusicGen backend communication

class MusicGenAPI {
    constructor() {
        this.baseURL = window.location.origin;
        this.wsBaseURL = this.baseURL.replace(/^http/, 'ws');
        this.currentTaskId = null;
        this.currentWebSocket = null;
    }

    // Standard generation endpoint
    async generateMusic(params) {
        try {
            const response = await fetch(`${this.baseURL}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params),
            });

            if (!response.ok) {
                throw new Error(`Generation failed: ${response.statusText}`);
            }

            const data = await response.json();
            this.currentTaskId = data.task_id;
            return data;
        } catch (error) {
            console.error('Generation request failed:', error);
            throw error;
        }
    }

    // Poll for generation status
    async getGenerationStatus(taskId) {
        try {
            const response = await fetch(`${this.baseURL}/generate/${taskId}`);
            
            if (!response.ok) {
                throw new Error(`Status check failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Status check failed:', error);
            throw error;
        }
    }

    // Download generated audio
    async downloadAudio(taskId) {
        const url = `${this.baseURL}/download/${taskId}`;
        return url;
    }

    // Cancel current generation
    async cancelGeneration() {
        if (this.currentTaskId) {
            // In a real implementation, we'd have a cancel endpoint
            console.log('Cancelling generation:', this.currentTaskId);
            this.currentTaskId = null;
        }
    }

    // WebSocket streaming connection
    connectStreaming(connectionId, callbacks) {
        const wsURL = `${this.wsBaseURL}/ws/stream/${connectionId}`;
        
        this.currentWebSocket = new WebSocket(wsURL);
        
        this.currentWebSocket.onopen = (event) => {
            console.log('WebSocket connected');
            if (callbacks.onConnect) callbacks.onConnect(event);
        };

        this.currentWebSocket.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                
                switch (message.type) {
                    case 'audio_chunk':
                        if (callbacks.onAudioChunk) callbacks.onAudioChunk(message);
                        break;
                    case 'status_update':
                        if (callbacks.onStatusUpdate) callbacks.onStatusUpdate(message);
                        break;
                    case 'error':
                        if (callbacks.onError) callbacks.onError(message);
                        break;
                    case 'heartbeat':
                        // Respond to heartbeat
                        this.sendStreamingMessage({ type: 'heartbeat' });
                        break;
                    default:
                        console.log('Unknown message type:', message.type);
                }
            } catch (error) {
                console.error('WebSocket message parsing error:', error);
            }
        };

        this.currentWebSocket.onclose = (event) => {
            console.log('WebSocket disconnected');
            if (callbacks.onDisconnect) callbacks.onDisconnect(event);
            this.currentWebSocket = null;
        };

        this.currentWebSocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            if (callbacks.onError) callbacks.onError(error);
        };

        return this.currentWebSocket;
    }

    // Send message through WebSocket
    sendStreamingMessage(message) {
        if (this.currentWebSocket && this.currentWebSocket.readyState === WebSocket.OPEN) {
            this.currentWebSocket.send(JSON.stringify(message));
        } else {
            console.error('WebSocket not connected');
        }
    }

    // Start streaming generation
    startStreaming(params) {
        this.sendStreamingMessage({
            type: 'start_streaming',
            request: params
        });
    }

    // Control streaming
    pauseStreaming() {
        this.sendStreamingMessage({ type: 'pause_streaming' });
    }

    resumeStreaming() {
        this.sendStreamingMessage({ type: 'resume_streaming' });
    }

    stopStreaming() {
        this.sendStreamingMessage({ type: 'stop_streaming' });
    }

    // Disconnect WebSocket
    disconnectStreaming() {
        if (this.currentWebSocket) {
            this.currentWebSocket.close();
            this.currentWebSocket = null;
        }
    }

    // Get available genres
    async getGenres() {
        try {
            const response = await fetch(`${this.baseURL}/genres`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch genres:', error);
            return { genres: [] };
        }
    }

    // Get available moods
    async getMoods() {
        try {
            const response = await fetch(`${this.baseURL}/moods`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch moods:', error);
            return { moods: [] };
        }
    }

    // Evaluate audio
    async evaluateAudio(audioBlob, metrics = ['all']) {
        try {
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'audio.wav');
            
            const response = await fetch(`${this.baseURL}/evaluate`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Evaluation failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Audio evaluation failed:', error);
            throw error;
        }
    }

    // Get streaming sessions
    async getStreamingSessions() {
        try {
            const response = await fetch(`${this.baseURL}/stream/sessions`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch streaming sessions:', error);
            return { sessions: [] };
        }
    }

    // Get streaming stats
    async getStreamingStats() {
        try {
            const response = await fetch(`${this.baseURL}/stream/stats`);
            return await response.json();
        } catch (error) {
            console.error('Failed to fetch streaming stats:', error);
            return { stats: {} };
        }
    }

    // Health check
    async checkHealth() {
        try {
            const response = await fetch(`${this.baseURL}/health`);
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'unhealthy', error: error.message };
        }
    }
}

// Export as global
window.MusicGenAPI = MusicGenAPI;