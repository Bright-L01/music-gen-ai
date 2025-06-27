// Main application module

class MusicGenApp {
    constructor() {
        this.api = new MusicGenAPI();
        this.audioManager = new AudioManager();
        this.ui = new UIManager();
        this.streamingManager = new StreamingManager(this.api, this.audioManager, this.ui);
        
        this.currentTask = null;
        this.pollingInterval = null;
        
        this.initialize();
    }

    async initialize() {
        // Check API health
        const health = await this.api.checkHealth();
        if (health.status !== 'healthy') {
            this.ui.showNotification('API is not available. Some features may not work.', 'error', 5000);
        }

        // Setup event handlers
        this.setupEventHandlers();

        // Resume audio context on first interaction
        document.addEventListener('click', () => {
            this.audioManager.resumeContext();
        }, { once: true });

        console.log('MusicGen App initialized');
    }

    setupEventHandlers() {
        // Generate button
        document.getElementById('generateBtn').addEventListener('click', () => this.generateMusic());

        // Cancel button
        document.getElementById('cancelBtn').addEventListener('click', () => this.cancelGeneration());

        // Audio player action buttons
        this.setupAudioPlayerActions();

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to generate
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.generateMusic();
            }
            // Escape to cancel
            if (e.key === 'Escape' && this.currentTask) {
                this.cancelGeneration();
            }
        });
    }

    setupAudioPlayerActions() {
        const audioElement = document.getElementById('audioElement');
        
        // Download button
        document.querySelector('#audioPlayer .fa-download').parentElement.addEventListener('click', async () => {
            if (audioElement.src) {
                const link = document.createElement('a');
                link.href = audioElement.src;
                link.download = `musicgen_${Date.now()}.wav`;
                link.click();
                this.ui.showNotification('Download started', 'success');
            }
        });

        // Share button
        document.querySelector('#audioPlayer .fa-share-alt').parentElement.addEventListener('click', async () => {
            if (audioElement.src) {
                try {
                    await navigator.share({
                        title: 'Generated Music',
                        text: 'Check out this AI-generated music!',
                        url: window.location.href
                    });
                } catch (error) {
                    // Copy link to clipboard as fallback
                    await navigator.clipboard.writeText(window.location.href);
                    this.ui.showNotification('Link copied to clipboard', 'success');
                }
            }
        });

        // Save to library button
        document.querySelector('#audioPlayer .fa-save').parentElement.addEventListener('click', () => {
            if (this.currentTask && this.currentTask.audioURL) {
                this.ui.addToLibrary({
                    prompt: this.currentTask.params.prompt,
                    duration: this.currentTask.params.duration,
                    genre: this.currentTask.params.genre,
                    mood: this.currentTask.params.mood,
                    audioURL: this.currentTask.audioURL,
                    metadata: this.currentTask.metadata,
                    timestamp: Date.now()
                });
            }
        });

        // Generate again button
        document.querySelector('#audioPlayer .fa-redo').parentElement.addEventListener('click', () => {
            this.generateMusic();
        });

        // Audio loaded event for waveform
        audioElement.addEventListener('loadeddata', async () => {
            try {
                const audioBuffer = await this.audioManager.loadAudioFromURL(audioElement.src);
                const canvas = document.getElementById('waveform');
                const visualizer = new WaveformVisualizer(canvas);
                visualizer.drawWaveform(audioBuffer);
            } catch (error) {
                console.error('Failed to draw waveform:', error);
            }
        });
    }

    async generateMusic() {
        // Get parameters
        const params = this.ui.getGenerationParameters();

        // Validate
        if (!params.prompt) {
            this.ui.showNotification('Please enter a music description', 'error');
            return;
        }

        // Check if already generating
        if (this.currentTask) {
            this.ui.showNotification('Generation already in progress', 'warning');
            return;
        }

        try {
            // Show progress
            this.ui.showProgress(true);
            this.ui.updateProgress(0, 'Initializing generation...');
            this.ui.hideAudioPlayer();

            // Start generation
            const response = await this.api.generateMusic(params);
            
            this.currentTask = {
                taskId: response.task_id,
                params: params,
                startTime: Date.now()
            };

            this.ui.showNotification('Music generation started', 'success');

            // Start polling for status
            this.startPolling();

        } catch (error) {
            console.error('Generation failed:', error);
            this.ui.showNotification(`Generation failed: ${error.message}`, 'error');
            this.ui.showProgress(false);
            this.currentTask = null;
        }
    }

    startPolling() {
        let progress = 0;
        
        this.pollingInterval = setInterval(async () => {
            try {
                const status = await this.api.getGenerationStatus(this.currentTask.taskId);
                
                if (status.status === 'completed') {
                    // Generation completed
                    this.stopPolling();
                    await this.handleGenerationComplete(status);
                    
                } else if (status.status === 'failed') {
                    // Generation failed
                    this.stopPolling();
                    this.handleGenerationError(status.error || 'Generation failed');
                    
                } else if (status.status === 'processing') {
                    // Update progress
                    progress = Math.min(progress + 10, 90);
                    const elapsed = (Date.now() - this.currentTask.startTime) / 1000;
                    this.ui.updateProgress(progress, `Generating... (${elapsed.toFixed(1)}s)`);
                }
                
            } catch (error) {
                console.error('Polling error:', error);
                // Continue polling unless too many errors
            }
        }, 1000);
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    async handleGenerationComplete(status) {
        try {
            // Get audio URL
            const audioURL = await this.api.downloadAudio(status.task_id);
            
            // Update progress
            this.ui.updateProgress(100, 'Generation complete!');
            
            // Store audio URL
            this.currentTask.audioURL = audioURL;
            this.currentTask.metadata = status.metadata;
            
            // Show audio player
            this.ui.showAudioPlayer(audioURL, status.metadata);
            
            // Hide progress after delay
            setTimeout(() => {
                this.ui.showProgress(false);
                this.currentTask = null;
            }, 1000);
            
            this.ui.showNotification('Music generation completed successfully!', 'success');
            
            // Evaluate audio quality (optional)
            this.evaluateGeneratedAudio(audioURL);
            
        } catch (error) {
            console.error('Failed to handle completion:', error);
            this.handleGenerationError(error.message);
        }
    }

    handleGenerationError(error) {
        this.ui.showProgress(false);
        this.ui.showNotification(`Generation failed: ${error}`, 'error', 5000);
        this.currentTask = null;
    }

    async cancelGeneration() {
        if (!this.currentTask) return;
        
        this.stopPolling();
        await this.api.cancelGeneration();
        
        this.ui.showProgress(false);
        this.ui.showNotification('Generation cancelled', 'info');
        this.currentTask = null;
    }

    async evaluateGeneratedAudio(audioURL) {
        try {
            // Fetch audio as blob
            const response = await fetch(audioURL);
            const audioBlob = await response.blob();
            
            // Evaluate
            const evaluation = await this.api.evaluateAudio(audioBlob);
            
            console.log('Audio evaluation:', evaluation);
            
            // Could display quality metrics in UI
            if (evaluation.quality_score) {
                console.log(`Quality score: ${evaluation.quality_score.toFixed(2)}`);
            }
            
        } catch (error) {
            console.error('Audio evaluation failed:', error);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MusicGenApp();
});

// Service worker for offline support (optional)
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/js/sw.js').catch(err => {
        console.log('Service worker registration failed:', err);
    });
}