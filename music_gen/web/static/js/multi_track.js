// Multi-track Studio JavaScript

class MultiTrackStudio {
    constructor() {
        this.tracks = [];
        this.selectedTrack = null;
        this.isPlaying = false;
        this.currentTaskId = null;
        
        this.initializeEventListeners();
        this.loadInstruments();
    }

    initializeEventListeners() {
        // Track controls
        document.getElementById('addTrackBtn').addEventListener('click', () => this.showInstrumentPanel());
        
        // Instrument selection
        document.querySelectorAll('.instrument-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const instrument = e.currentTarget.dataset.instrument;
                this.addTrack(instrument);
                this.hideInstrumentPanel();
            });
        });

        // Transport controls
        document.getElementById('playBtn').addEventListener('click', () => this.togglePlayback());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopPlayback());
        document.getElementById('generateAllBtn').addEventListener('click', () => this.generateAllTracks());

        // Master volume
        document.getElementById('masterVolume').addEventListener('input', (e) => {
            document.getElementById('masterVolumeDisplay').textContent = e.target.value + '%';
        });

        // Track details panel
        document.getElementById('trackVolume').addEventListener('input', () => this.updateTrackSettings());
        document.getElementById('trackPan').addEventListener('input', () => this.updateTrackSettings());
        document.getElementById('trackReverb').addEventListener('input', () => this.updateTrackSettings());
        document.getElementById('trackDelay').addEventListener('input', () => this.updateTrackSettings());
        document.getElementById('trackStartTime').addEventListener('input', () => this.updateTrackSettings());
        document.getElementById('trackName').addEventListener('input', () => this.updateTrackSettings());
        
        document.getElementById('deleteTrackBtn').addEventListener('click', () => this.deleteSelectedTrack());

        // Export
        document.querySelector('a[href="#export"]').addEventListener('click', (e) => {
            e.preventDefault();
            this.showExportModal();
        });

        document.getElementById('confirmExport').addEventListener('click', () => this.exportProject());
        document.getElementById('cancelExport').addEventListener('click', () => this.hideExportModal());
    }

    async loadInstruments() {
        try {
            const response = await fetch('/instruments');
            const data = await response.json();
            // Could dynamically create instrument buttons based on available instruments
            console.log('Available instruments:', data.instruments);
        } catch (error) {
            console.error('Failed to load instruments:', error);
        }
    }

    showInstrumentPanel() {
        document.getElementById('instrumentPanel').classList.remove('hidden');
    }

    hideInstrumentPanel() {
        document.getElementById('instrumentPanel').classList.add('hidden');
    }

    addTrack(instrument) {
        const track = {
            id: Date.now(),
            instrument: instrument,
            name: this.getInstrumentName(instrument),
            volume: 70,
            pan: 0,
            reverb: 20,
            delay: 0,
            startTime: 0,
            muted: false,
            solo: false,
            audioBuffer: null,
            waveformData: null
        };

        this.tracks.push(track);
        this.renderTrack(track);
        this.selectTrack(track);
    }

    getInstrumentName(instrument) {
        const names = {
            piano: 'Piano',
            guitar: 'Guitar',
            drums: 'Drums',
            bass: 'Bass',
            violin: 'Violin',
            saxophone: 'Saxophone',
            synthesizer: 'Synthesizer',
            flute: 'Flute'
        };
        return names[instrument] || instrument;
    }

    getInstrumentIcon(instrument) {
        const icons = {
            piano: 'fa-piano-keyboard',
            guitar: 'fa-guitar',
            drums: 'fa-drum',
            bass: 'fa-guitar',
            violin: 'fa-violin',
            saxophone: 'fa-saxophone', 
            synthesizer: 'fa-keyboard',
            flute: 'fa-wind'
        };
        return icons[instrument] || 'fa-music';
    }

    getInstrumentColor(instrument) {
        const colors = {
            piano: 'bg-blue-600',
            guitar: 'bg-orange-600',
            drums: 'bg-red-600',
            bass: 'bg-purple-600',
            violin: 'bg-green-600',
            saxophone: 'bg-yellow-600',
            synthesizer: 'bg-pink-600',
            flute: 'bg-indigo-600'
        };
        return colors[instrument] || 'bg-gray-600';
    }

    renderTrack(track) {
        const trackElement = document.createElement('div');
        trackElement.className = 'track-lane flex items-center border-b border-gray-700 h-20';
        trackElement.dataset.trackId = track.id;
        
        trackElement.innerHTML = `
            <!-- Track Header -->
            <div class="w-64 p-4 flex items-center space-x-3 bg-gray-800">
                <div class="instrument-icon ${this.getInstrumentColor(track.instrument)}">
                    <i class="fas ${this.getInstrumentIcon(track.instrument)}"></i>
                </div>
                <div class="flex-1">
                    <div class="font-medium">${track.name}</div>
                    <div class="text-xs text-gray-400">${track.instrument}</div>
                </div>
                <div class="flex space-x-1">
                    <button class="mute-btn p-1 text-xs ${track.muted ? 'text-red-500' : 'text-gray-400'} hover:text-white">
                        <i class="fas fa-volume-mute"></i>
                    </button>
                    <button class="solo-btn p-1 text-xs ${track.solo ? 'text-yellow-500' : 'text-gray-400'} hover:text-white">
                        S
                    </button>
                </div>
            </div>
            
            <!-- Waveform Area -->
            <div class="flex-1 p-2">
                <div class="waveform-container" id="waveform-${track.id}">
                    <canvas class="w-full h-full"></canvas>
                    <div class="absolute inset-0 flex items-center justify-center text-gray-500">
                        <i class="fas fa-waveform-lines mr-2"></i>
                        <span>Click "Generate" to create audio</span>
                    </div>
                </div>
            </div>
        `;

        // Add event listeners
        trackElement.addEventListener('click', () => this.selectTrack(track));
        
        const muteBtn = trackElement.querySelector('.mute-btn');
        muteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleMute(track);
        });

        const soloBtn = trackElement.querySelector('.solo-btn');
        soloBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.toggleSolo(track);
        });

        document.getElementById('tracksContainer').appendChild(trackElement);
    }

    selectTrack(track) {
        this.selectedTrack = track;
        
        // Update UI
        document.querySelectorAll('.track-lane').forEach(el => {
            el.classList.remove('ring-2', 'ring-purple-500');
        });
        
        const trackElement = document.querySelector(`[data-track-id="${track.id}"]`);
        if (trackElement) {
            trackElement.classList.add('ring-2', 'ring-purple-500');
        }

        // Show track details panel
        this.showTrackDetails(track);
    }

    showTrackDetails(track) {
        const panel = document.getElementById('trackDetailsPanel');
        panel.classList.remove('hidden');

        document.getElementById('trackName').value = track.name;
        document.getElementById('trackVolume').value = track.volume;
        document.getElementById('trackPan').value = track.pan;
        document.getElementById('trackReverb').value = track.reverb;
        document.getElementById('trackDelay').value = track.delay;
        document.getElementById('trackStartTime').value = track.startTime;
    }

    updateTrackSettings() {
        if (!this.selectedTrack) return;

        this.selectedTrack.name = document.getElementById('trackName').value;
        this.selectedTrack.volume = parseInt(document.getElementById('trackVolume').value);
        this.selectedTrack.pan = parseInt(document.getElementById('trackPan').value);
        this.selectedTrack.reverb = parseInt(document.getElementById('trackReverb').value);
        this.selectedTrack.delay = parseInt(document.getElementById('trackDelay').value);
        this.selectedTrack.startTime = parseFloat(document.getElementById('trackStartTime').value);

        // Update track display
        const trackElement = document.querySelector(`[data-track-id="${this.selectedTrack.id}"]`);
        if (trackElement) {
            trackElement.querySelector('.font-medium').textContent = this.selectedTrack.name;
        }
    }

    toggleMute(track) {
        track.muted = !track.muted;
        const trackElement = document.querySelector(`[data-track-id="${track.id}"]`);
        const muteBtn = trackElement.querySelector('.mute-btn');
        muteBtn.classList.toggle('text-red-500', track.muted);
        muteBtn.classList.toggle('text-gray-400', !track.muted);
    }

    toggleSolo(track) {
        track.solo = !track.solo;
        const trackElement = document.querySelector(`[data-track-id="${track.id}"]`);
        const soloBtn = trackElement.querySelector('.solo-btn');
        soloBtn.classList.toggle('text-yellow-500', track.solo);
        soloBtn.classList.toggle('text-gray-400', !track.solo);
    }

    deleteSelectedTrack() {
        if (!this.selectedTrack) return;

        const trackElement = document.querySelector(`[data-track-id="${this.selectedTrack.id}"]`);
        if (trackElement) {
            trackElement.remove();
        }

        this.tracks = this.tracks.filter(t => t.id !== this.selectedTrack.id);
        this.selectedTrack = null;
        document.getElementById('trackDetailsPanel').classList.add('hidden');
    }

    async generateAllTracks() {
        if (this.tracks.length === 0) {
            alert('Please add at least one track');
            return;
        }

        const btn = document.getElementById('generateAllBtn');
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Generating...';

        try {
            const requestData = {
                prompt: document.getElementById('masterPrompt').value || 'Create music',
                duration: parseFloat(document.getElementById('duration').value),
                tracks: this.tracks.map(track => ({
                    instrument: track.instrument,
                    volume: track.volume / 100,
                    pan: track.pan / 100,
                    reverb: track.reverb / 100,
                    start_time: track.startTime
                })),
                auto_mix: true,
                export_stems: true
            };

            const response = await fetch('/generate/multi-instrument', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();
            this.currentTaskId = result.task_id;

            // Poll for completion
            this.pollGenerationStatus();

        } catch (error) {
            console.error('Generation failed:', error);
            alert('Failed to generate tracks');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-magic mr-2"></i>Generate All Tracks';
        }
    }

    async pollGenerationStatus() {
        const checkStatus = async () => {
            try {
                const response = await fetch(`/generate/${this.currentTaskId}`);
                const result = await response.json();

                if (result.status === 'completed') {
                    // Load generated audio
                    this.loadGeneratedAudio(result);
                } else if (result.status === 'failed') {
                    alert('Generation failed: ' + result.error);
                } else {
                    // Continue polling
                    setTimeout(checkStatus, 1000);
                }
            } catch (error) {
                console.error('Status check failed:', error);
            }
        };

        checkStatus();
    }

    async loadGeneratedAudio(result) {
        // Load mixed audio
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        try {
            const response = await fetch(result.audio_url);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Store audio buffer
            this.mixedAudioBuffer = audioBuffer;
            
            // Update waveforms
            this.tracks.forEach(track => {
                this.drawWaveform(track);
            });

            alert('Generation complete!');
        } catch (error) {
            console.error('Failed to load audio:', error);
        }
    }

    drawWaveform(track) {
        const container = document.getElementById(`waveform-${track.id}`);
        const canvas = container.querySelector('canvas');
        const ctx = canvas.getContext('2d');
        
        // Clear placeholder
        container.querySelector('.absolute').style.display = 'none';
        
        // Set canvas size
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        
        // Draw simple waveform visualization
        ctx.fillStyle = '#8b5cf6';
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 2;
        
        // Generate random waveform for demo
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        
        for (let x = 0; x < canvas.width; x += 2) {
            const y = canvas.height / 2 + (Math.random() - 0.5) * canvas.height * 0.6;
            ctx.lineTo(x, y);
        }
        
        ctx.stroke();
    }

    togglePlayback() {
        const playBtn = document.getElementById('playBtn');
        
        if (this.isPlaying) {
            this.pausePlayback();
            playBtn.innerHTML = '<i class="fas fa-play text-xl"></i>';
        } else {
            this.startPlayback();
            playBtn.innerHTML = '<i class="fas fa-pause text-xl"></i>';
        }
        
        this.isPlaying = !this.isPlaying;
    }

    startPlayback() {
        if (!this.mixedAudioBuffer) {
            alert('Please generate tracks first');
            return;
        }
        
        // Playback implementation would go here
        console.log('Starting playback...');
    }

    pausePlayback() {
        console.log('Pausing playback...');
    }

    stopPlayback() {
        this.isPlaying = false;
        document.getElementById('playBtn').innerHTML = '<i class="fas fa-play text-xl"></i>';
        console.log('Stopping playback...');
    }

    showExportModal() {
        document.getElementById('exportModal').classList.remove('hidden');
    }

    hideExportModal() {
        document.getElementById('exportModal').classList.add('hidden');
    }

    async exportProject() {
        const exportMixed = document.getElementById('exportMixed').checked;
        const exportStems = document.getElementById('exportStems').checked;
        const exportMidi = document.getElementById('exportMidi').checked;

        if (!exportMixed && !exportStems && !exportMidi) {
            alert('Please select at least one export option');
            return;
        }

        // Export implementation would go here
        console.log('Exporting project...', { exportMixed, exportStems, exportMidi });
        
        this.hideExportModal();
        alert('Export started. Files will be downloaded when ready.');
    }
}

// Initialize the studio
const studio = new MultiTrackStudio();