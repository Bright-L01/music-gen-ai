// Streaming module for real-time music generation

class StreamingManager {
    constructor(api, audioManager, ui) {
        this.api = api;
        this.audioManager = audioManager;
        this.ui = ui;
        
        this.connectionId = null;
        this.sessionId = null;
        this.isStreaming = false;
        this.streamingHandler = null;
        this.visualizer = null;
        
        this.stats = {
            chunks: 0,
            duration: 0,
            latency: [],
            startTime: null
        };
        
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        // Streaming control buttons
        document.getElementById('startStreamBtn').addEventListener('click', () => this.startStreaming());
        document.getElementById('pauseStreamBtn').addEventListener('click', () => this.pauseStreaming());
        document.getElementById('resumeStreamBtn').addEventListener('click', () => this.resumeStreaming());
        document.getElementById('stopStreamBtn').addEventListener('click', () => this.stopStreaming());

        // Enter key on prompt input
        document.getElementById('streamingPrompt').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !this.isStreaming) {
                this.startStreaming();
            }
        });
    }

    async startStreaming() {
        const prompt = document.getElementById('streamingPrompt').value.trim();
        
        if (!prompt) {
            this.ui.showNotification('Please enter a prompt for streaming', 'error');
            return;
        }

        // Initialize
        this.connectionId = 'web_' + Math.random().toString(36).substr(2, 9);
        this.streamingHandler = new StreamingAudioHandler(this.audioManager);
        this.stats = {
            chunks: 0,
            duration: 0,
            latency: [],
            startTime: Date.now()
        };

        // Setup visualizer
        const canvas = document.getElementById('streamingWaveform');
        this.visualizer = new WaveformVisualizer(canvas);
        
        // Update UI
        this.updateStreamingUI('connecting');

        // Connect WebSocket
        this.api.connectStreaming(this.connectionId, {
            onConnect: () => this.handleConnect(),
            onAudioChunk: (chunk) => this.handleAudioChunk(chunk),
            onStatusUpdate: (status) => this.handleStatusUpdate(status),
            onError: (error) => this.handleError(error),
            onDisconnect: () => this.handleDisconnect()
        });

        // Start streaming generation
        const params = {
            prompt: prompt,
            chunk_duration: 1.0,
            quality_mode: 'balanced',
            temperature: 0.9,
            enable_interruption: true,
            adaptive_quality: true,
            crossfade_duration: 0.1
        };

        // Add optional parameters
        const genre = document.getElementById('genreSelect').value;
        const mood = document.getElementById('moodSelect').value;
        const tempo = parseInt(document.getElementById('tempoSlider').value);
        
        if (genre) params.genre = genre;
        if (mood) params.mood = mood;
        if (tempo) params.tempo = tempo;

        setTimeout(() => {
            this.api.startStreaming(params);
        }, 100);
    }

    handleConnect() {
        console.log('Streaming connected');
        this.isStreaming = true;
        this.updateStreamingUI('streaming');
        this.visualizer.startAnimation(this.audioManager);
    }

    async handleAudioChunk(chunk) {
        const receiveTime = Date.now();
        
        // Update stats
        this.stats.chunks++;
        this.stats.duration += chunk.duration;
        
        // Calculate latency
        if (chunk.metadata && chunk.metadata.generation_time_ms) {
            this.stats.latency.push(chunk.metadata.generation_time_ms);
            if (this.stats.latency.length > 10) {
                this.stats.latency.shift();
            }
        }

        // Add to audio queue
        await this.streamingHandler.addChunk(
            chunk.audio_data,
            chunk.sample_rate,
            chunk.duration
        );

        // Update UI
        this.updateStreamingStats();

        // Log performance
        const processingTime = Date.now() - receiveTime;
        console.log(`Chunk ${chunk.chunk_id} processed in ${processingTime}ms`);
    }

    handleStatusUpdate(status) {
        console.log('Streaming status:', status.status);
        
        if (status.status === 'prepared') {
            this.sessionId = status.session_id;
            this.ui.showNotification('Streaming ready', 'success');
        } else if (status.status === 'completed') {
            this.ui.showNotification('Streaming completed', 'success');
            this.stopStreaming();
        } else if (status.status === 'buffer_underrun') {
            this.ui.showNotification('Buffering...', 'info');
        }
    }

    handleError(error) {
        console.error('Streaming error:', error);
        this.ui.showNotification(`Streaming error: ${error.error_message || error}`, 'error');
        this.stopStreaming();
    }

    handleDisconnect() {
        console.log('Streaming disconnected');
        if (this.isStreaming) {
            this.stopStreaming();
        }
    }

    pauseStreaming() {
        if (!this.isStreaming) return;
        
        this.api.pauseStreaming();
        this.streamingHandler.pause();
        this.updateStreamingUI('paused');
        this.ui.showNotification('Streaming paused', 'info');
    }

    resumeStreaming() {
        if (!this.isStreaming) return;
        
        this.api.resumeStreaming();
        this.streamingHandler.resume();
        this.updateStreamingUI('streaming');
        this.ui.showNotification('Streaming resumed', 'info');
    }

    stopStreaming() {
        if (!this.isStreaming) return;
        
        this.isStreaming = false;
        
        // Stop streaming
        this.api.stopStreaming();
        this.api.disconnectStreaming();
        
        // Stop audio
        if (this.streamingHandler) {
            this.streamingHandler.stop();
        }
        
        // Stop visualization
        if (this.visualizer) {
            this.visualizer.stopAnimation();
            this.visualizer.clear();
        }
        
        // Update UI
        this.updateStreamingUI('stopped');
        this.ui.showNotification('Streaming stopped', 'info');
        
        // Show final stats
        this.showFinalStats();
    }

    updateStreamingUI(state) {
        const startBtn = document.getElementById('startStreamBtn');
        const pauseBtn = document.getElementById('pauseStreamBtn');
        const resumeBtn = document.getElementById('resumeStreamBtn');
        const stopBtn = document.getElementById('stopStreamBtn');
        const promptInput = document.getElementById('streamingPrompt');
        
        switch (state) {
            case 'connecting':
                startBtn.disabled = true;
                pauseBtn.disabled = true;
                resumeBtn.disabled = true;
                stopBtn.disabled = false;
                promptInput.disabled = true;
                break;
                
            case 'streaming':
                startBtn.disabled = true;
                pauseBtn.disabled = false;
                resumeBtn.disabled = true;
                stopBtn.disabled = false;
                promptInput.disabled = true;
                break;
                
            case 'paused':
                startBtn.disabled = true;
                pauseBtn.disabled = true;
                resumeBtn.disabled = false;
                stopBtn.disabled = false;
                promptInput.disabled = true;
                break;
                
            case 'stopped':
                startBtn.disabled = false;
                pauseBtn.disabled = true;
                resumeBtn.disabled = true;
                stopBtn.disabled = true;
                promptInput.disabled = false;
                break;
        }
    }

    updateStreamingStats() {
        // Update chunk count
        document.getElementById('chunksGenerated').textContent = this.stats.chunks;
        
        // Update duration
        document.getElementById('streamDuration').textContent = `${this.stats.duration.toFixed(1)}s`;
        
        // Update latency
        if (this.stats.latency.length > 0) {
            const avgLatency = this.stats.latency.reduce((a, b) => a + b, 0) / this.stats.latency.length;
            document.getElementById('streamLatency').textContent = `${avgLatency.toFixed(0)}ms`;
        }
    }

    showFinalStats() {
        const elapsed = (Date.now() - this.stats.startTime) / 1000;
        const realTimeFactor = this.stats.duration / elapsed;
        
        console.log('Streaming stats:', {
            totalChunks: this.stats.chunks,
            totalDuration: this.stats.duration,
            elapsedTime: elapsed,
            realTimeFactor: realTimeFactor,
            averageLatency: this.stats.latency.length > 0 
                ? this.stats.latency.reduce((a, b) => a + b, 0) / this.stats.latency.length 
                : 0
        });
    }

    // Modify streaming prompt in real-time
    async modifyPrompt(newPrompt) {
        if (!this.isStreaming) return;
        
        const message = {
            type: 'modify_streaming',
            new_prompt: newPrompt
        };
        
        this.api.sendStreamingMessage(message);
        this.ui.showNotification('Prompt modified', 'info');
    }

    // Get current streaming status
    async getStatus() {
        if (!this.isStreaming) return null;
        
        this.api.sendStreamingMessage({ type: 'get_status' });
    }
}

// Export as global
window.StreamingManager = StreamingManager;