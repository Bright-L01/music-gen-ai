// Audio handling module for MusicGen

class AudioManager {
    constructor() {
        this.audioContext = null;
        this.audioBuffers = new Map();
        this.currentSource = null;
        this.analyser = null;
        this.initializeContext();
    }

    initializeContext() {
        try {
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioContext = new AudioContext();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 2048;
            this.analyser.connect(this.audioContext.destination);
        } catch (error) {
            console.error('Failed to initialize audio context:', error);
        }
    }

    // Resume audio context (required for some browsers)
    async resumeContext() {
        if (this.audioContext && this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    // Decode base64 audio to AudioBuffer
    async decodeAudioData(base64Audio, sampleRate) {
        try {
            // Convert base64 to ArrayBuffer
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Convert Int16 PCM to Float32
            const int16Array = new Int16Array(bytes.buffer);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }

            // Create AudioBuffer
            const audioBuffer = this.audioContext.createBuffer(1, float32Array.length, sampleRate);
            audioBuffer.getChannelData(0).set(float32Array);

            return audioBuffer;
        } catch (error) {
            console.error('Failed to decode audio data:', error);
            throw error;
        }
    }

    // Play AudioBuffer
    playBuffer(audioBuffer, onEnded) {
        if (this.currentSource) {
            this.currentSource.stop();
        }

        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.analyser);
        
        if (onEnded) {
            source.onended = onEnded;
        }

        source.start(0);
        this.currentSource = source;

        return source;
    }

    // Stop current playback
    stop() {
        if (this.currentSource) {
            this.currentSource.stop();
            this.currentSource = null;
        }
    }

    // Get frequency data for visualization
    getFrequencyData() {
        const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
        this.analyser.getByteFrequencyData(dataArray);
        return dataArray;
    }

    // Get time domain data for waveform
    getWaveformData() {
        const dataArray = new Uint8Array(this.analyser.fftSize);
        this.analyser.getByteTimeDomainData(dataArray);
        return dataArray;
    }

    // Load audio from URL
    async loadAudioFromURL(url) {
        try {
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            return audioBuffer;
        } catch (error) {
            console.error('Failed to load audio from URL:', error);
            throw error;
        }
    }

    // Convert AudioBuffer to WAV blob
    audioBufferToWav(audioBuffer) {
        const numberOfChannels = audioBuffer.numberOfChannels;
        const length = audioBuffer.length * numberOfChannels * 2;
        const buffer = new ArrayBuffer(44 + length);
        const view = new DataView(buffer);

        // WAV header
        const writeString = (offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(0, 'RIFF');
        view.setUint32(4, 36 + length, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true); // fmt chunk size
        view.setUint16(20, 1, true); // PCM format
        view.setUint16(22, numberOfChannels, true);
        view.setUint32(24, audioBuffer.sampleRate, true);
        view.setUint32(28, audioBuffer.sampleRate * numberOfChannels * 2, true);
        view.setUint16(32, numberOfChannels * 2, true);
        view.setUint16(34, 16, true); // 16-bit
        writeString(36, 'data');
        view.setUint32(40, length, true);

        // Convert float samples to 16-bit PCM
        let offset = 44;
        for (let channel = 0; channel < numberOfChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            for (let i = 0; i < channelData.length; i++) {
                const sample = Math.max(-1, Math.min(1, channelData[i]));
                view.setInt16(offset, sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }
}

// Waveform visualization
class WaveformVisualizer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.isAnimating = false;
    }

    // Draw static waveform from AudioBuffer
    drawWaveform(audioBuffer) {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const data = audioBuffer.getChannelData(0);
        
        this.ctx.clearRect(0, 0, width, height);
        this.ctx.strokeStyle = '#9333ea';
        this.ctx.lineWidth = 2;
        
        const sliceWidth = width / data.length;
        let x = 0;
        
        this.ctx.beginPath();
        this.ctx.moveTo(0, height / 2);
        
        for (let i = 0; i < data.length; i += Math.ceil(data.length / width)) {
            const y = (data[i] + 1) * height / 2;
            this.ctx.lineTo(x, y);
            x += sliceWidth;
        }
        
        this.ctx.lineTo(width, height / 2);
        this.ctx.stroke();
    }

    // Animate real-time waveform
    animateWaveform(audioManager) {
        if (!this.isAnimating) return;

        const width = this.canvas.width;
        const height = this.canvas.height;
        const dataArray = audioManager.getWaveformData();
        
        this.ctx.fillStyle = 'rgba(31, 41, 55, 0.1)';
        this.ctx.fillRect(0, 0, width, height);
        
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = '#9333ea';
        this.ctx.beginPath();
        
        const sliceWidth = width / dataArray.length;
        let x = 0;
        
        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * height / 2;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        this.ctx.lineTo(width, height / 2);
        this.ctx.stroke();
        
        requestAnimationFrame(() => this.animateWaveform(audioManager));
    }

    // Start animation
    startAnimation(audioManager) {
        this.isAnimating = true;
        this.animateWaveform(audioManager);
    }

    // Stop animation
    stopAnimation() {
        this.isAnimating = false;
    }

    // Clear canvas
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

// Streaming audio handler
class StreamingAudioHandler {
    constructor(audioManager) {
        this.audioManager = audioManager;
        this.chunkQueue = [];
        this.isPlaying = false;
        this.currentChunkIndex = 0;
        this.totalDuration = 0;
    }

    // Add audio chunk to queue
    async addChunk(base64Audio, sampleRate, duration) {
        try {
            const audioBuffer = await this.audioManager.decodeAudioData(base64Audio, sampleRate);
            this.chunkQueue.push({
                buffer: audioBuffer,
                duration: duration,
                index: this.currentChunkIndex++
            });

            // Start playback if not already playing and we have enough chunks
            if (!this.isPlaying && this.chunkQueue.length >= 2) {
                this.startPlayback();
            }
        } catch (error) {
            console.error('Failed to add audio chunk:', error);
        }
    }

    // Start playing chunks
    startPlayback() {
        if (this.isPlaying || this.chunkQueue.length === 0) return;

        this.isPlaying = true;
        this.playNextChunk();
    }

    // Play next chunk in queue
    playNextChunk() {
        if (this.chunkQueue.length === 0) {
            this.isPlaying = false;
            return;
        }

        const chunk = this.chunkQueue.shift();
        this.totalDuration += chunk.duration;

        this.audioManager.playBuffer(chunk.buffer, () => {
            // Play next chunk when current one ends
            if (this.isPlaying) {
                this.playNextChunk();
            }
        });
    }

    // Stop streaming playback
    stop() {
        this.isPlaying = false;
        this.audioManager.stop();
        this.chunkQueue = [];
        this.currentChunkIndex = 0;
        this.totalDuration = 0;
    }

    // Pause streaming
    pause() {
        this.isPlaying = false;
        this.audioManager.stop();
    }

    // Resume streaming
    resume() {
        if (!this.isPlaying && this.chunkQueue.length > 0) {
            this.startPlayback();
        }
    }

    // Get streaming stats
    getStats() {
        return {
            queueLength: this.chunkQueue.length,
            totalChunks: this.currentChunkIndex,
            totalDuration: this.totalDuration,
            isPlaying: this.isPlaying
        };
    }
}

// Export as globals
window.AudioManager = AudioManager;
window.WaveformVisualizer = WaveformVisualizer;
window.StreamingAudioHandler = StreamingAudioHandler;