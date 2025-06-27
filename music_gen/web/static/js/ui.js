// UI utilities and interactions

class UIManager {
    constructor() {
        this.notifications = [];
        this.library = [];
        this.setupEventHandlers();
        this.initializeUI();
    }

    setupEventHandlers() {
        // Prompt suggestions
        document.querySelectorAll('.prompt-suggestion').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const suggestions = {
                    'Jazz Piano Trio': 'Smooth jazz piano trio with walking bass, brushed drums, and melodic piano improvisation',
                    'Epic Orchestral': 'Epic orchestral music with powerful strings, brass fanfares, and dramatic timpani rolls',
                    'Chill Lofi Hip-Hop': 'Chill lofi hip-hop beat with vinyl crackle, jazzy chords, and mellow bass',
                    'Electronic Dance': 'Energetic electronic dance music with pulsing bass, synthesizers, and driving beat',
                    'Acoustic Folk': 'Warm acoustic folk music with fingerpicked guitar, gentle vocals, and harmonica'
                };
                
                const prompt = suggestions[e.target.textContent.trim()];
                if (prompt) {
                    document.getElementById('promptInput').value = prompt;
                    this.showNotification('Prompt suggestion applied', 'info');
                }
            });
        });

        // Quality mode buttons
        document.querySelectorAll('.quality-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.quality-btn').forEach(b => b.classList.remove('selected', 'bg-purple-600'));
                btn.classList.add('selected', 'bg-purple-600');
            });
        });

        // Sliders
        this.setupSlider('tempoSlider', 'tempoValue', ' BPM');
        this.setupSlider('durationSlider', 'durationValue', 's');
        this.setupSlider('numBeamsSlider', 'numBeamsValue', ' beams');

        // Beam search toggle
        document.getElementById('useBeamSearch').addEventListener('change', (e) => {
            const options = document.getElementById('beamSearchOptions');
            if (e.target.checked) {
                options.classList.remove('hidden');
            } else {
                options.classList.add('hidden');
            }
        });

        // Navigation smooth scrolling
        document.querySelectorAll('nav a[href^="#"]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(e.target.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    }

    setupSlider(sliderId, valueId, suffix = '') {
        const slider = document.getElementById(sliderId);
        const valueDisplay = document.getElementById(valueId);
        
        if (slider && valueDisplay) {
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = e.target.value + suffix;
            });
        }
    }

    initializeUI() {
        // Set initial quality mode
        document.querySelector('.quality-btn[data-quality="balanced"]').click();
        
        // Initialize canvas sizes
        this.resizeCanvases();
        window.addEventListener('resize', () => this.resizeCanvases());
        
        // Load library from localStorage
        this.loadLibrary();
    }

    resizeCanvases() {
        const canvases = ['waveform', 'streamingWaveform'];
        canvases.forEach(id => {
            const canvas = document.getElementById(id);
            if (canvas) {
                const rect = canvas.getBoundingClientRect();
                canvas.width = rect.width * window.devicePixelRatio;
                canvas.height = rect.height * window.devicePixelRatio;
                const ctx = canvas.getContext('2d');
                ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
            }
        });
    }

    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="flex items-center">
                <i class="fas fa-${this.getNotificationIcon(type)} mr-2"></i>
                <span>${message}</span>
                <button class="ml-4" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        this.notifications.push(notification);
        
        // Auto remove after duration
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                notification.remove();
                this.notifications = this.notifications.filter(n => n !== notification);
            }, 300);
        }, duration);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            info: 'info-circle',
            warning: 'exclamation-triangle'
        };
        return icons[type] || 'info-circle';
    }

    showProgress(show = true) {
        const progressDiv = document.getElementById('generationProgress');
        if (show) {
            progressDiv.classList.remove('hidden');
        } else {
            progressDiv.classList.add('hidden');
        }
    }

    updateProgress(percent, text) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        
        if (progressBar) {
            progressBar.style.width = `${percent}%`;
        }
        if (progressText) {
            progressText.textContent = text;
        }
    }

    showAudioPlayer(audioURL, metadata) {
        const playerDiv = document.getElementById('audioPlayer');
        const audioElement = document.getElementById('audioElement');
        
        // Show player
        playerDiv.classList.remove('hidden');
        
        // Set audio source
        audioElement.src = audioURL;
        
        // Display metadata
        this.displayMetadata(metadata);
        
        // Scroll to player
        playerDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    hideAudioPlayer() {
        const playerDiv = document.getElementById('audioPlayer');
        playerDiv.classList.add('hidden');
    }

    displayMetadata(metadata) {
        const metadataDiv = document.getElementById('metadata');
        metadataDiv.innerHTML = '';
        
        const fields = [
            { label: 'Prompt', value: metadata.prompt },
            { label: 'Duration', value: `${metadata.duration}s` },
            { label: 'Quality Mode', value: metadata.generation_params?.method || 'Unknown' },
            { label: 'Temperature', value: metadata.generation_params?.temperature },
            { label: 'Genre', value: metadata.conditioning?.genre || 'Auto' },
            { label: 'Mood', value: metadata.conditioning?.mood || 'Auto' }
        ];
        
        fields.forEach(field => {
            if (field.value) {
                const div = document.createElement('div');
                div.innerHTML = `
                    <span class="font-medium">${field.label}:</span>
                    <span>${field.value}</span>
                `;
                metadataDiv.appendChild(div);
            }
        });
    }

    addToLibrary(item) {
        // Add to library array
        this.library.unshift(item);
        
        // Save to localStorage
        this.saveLibrary();
        
        // Update UI
        this.renderLibrary();
        
        this.showNotification('Added to library', 'success');
    }

    renderLibrary() {
        const libraryGrid = document.getElementById('libraryGrid');
        libraryGrid.innerHTML = '';
        
        if (this.library.length === 0) {
            libraryGrid.innerHTML = `
                <div class="col-span-full text-center py-12 text-gray-400">
                    <i class="fas fa-music text-4xl mb-4"></i>
                    <p>Your library is empty. Generate some music to get started!</p>
                </div>
            `;
            return;
        }
        
        this.library.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'library-card bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700';
            card.innerHTML = `
                <div class="mb-3">
                    <h3 class="font-semibold text-lg mb-1 truncate">${item.prompt}</h3>
                    <p class="text-sm text-gray-400">
                        ${new Date(item.timestamp).toLocaleDateString()} • 
                        ${item.duration}s • 
                        ${item.genre || 'No genre'}
                    </p>
                </div>
                <div class="flex justify-between items-center">
                    <button class="play-btn px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm" data-index="${index}">
                        <i class="fas fa-play mr-1"></i> Play
                    </button>
                    <button class="delete-btn text-red-400 hover:text-red-300" data-index="${index}">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;
            
            // Add event listeners
            card.querySelector('.play-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                this.playLibraryItem(index);
            });
            
            card.querySelector('.delete-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteLibraryItem(index);
            });
            
            libraryGrid.appendChild(card);
        });
    }

    playLibraryItem(index) {
        const item = this.library[index];
        if (item && item.audioURL) {
            this.showAudioPlayer(item.audioURL, item.metadata);
            this.showNotification('Playing from library', 'info');
        }
    }

    deleteLibraryItem(index) {
        if (confirm('Are you sure you want to delete this item?')) {
            this.library.splice(index, 1);
            this.saveLibrary();
            this.renderLibrary();
            this.showNotification('Item deleted from library', 'info');
        }
    }

    saveLibrary() {
        try {
            localStorage.setItem('musicgen_library', JSON.stringify(this.library));
        } catch (error) {
            console.error('Failed to save library:', error);
        }
    }

    loadLibrary() {
        try {
            const saved = localStorage.getItem('musicgen_library');
            if (saved) {
                this.library = JSON.parse(saved);
                this.renderLibrary();
            }
        } catch (error) {
            console.error('Failed to load library:', error);
        }
    }

    getGenerationParameters() {
        return {
            prompt: document.getElementById('promptInput').value.trim(),
            duration: parseFloat(document.getElementById('durationSlider').value),
            temperature: parseFloat(document.getElementById('temperatureInput').value),
            top_k: parseInt(document.getElementById('topKInput').value),
            top_p: parseFloat(document.getElementById('topPInput').value),
            do_sample: true,
            repetition_penalty: 1.1,
            num_beams: document.getElementById('useBeamSearch').checked 
                ? parseInt(document.getElementById('numBeamsSlider').value) 
                : 1,
            genre: document.getElementById('genreSelect').value || undefined,
            mood: document.getElementById('moodSelect').value || undefined,
            tempo: parseInt(document.getElementById('tempoSlider').value) || undefined
        };
    }

    resetGenerationForm() {
        document.getElementById('promptInput').value = '';
        document.getElementById('genreSelect').value = '';
        document.getElementById('moodSelect').value = '';
        document.getElementById('tempoSlider').value = 120;
        document.getElementById('durationSlider').value = 30;
        document.getElementById('useBeamSearch').checked = false;
        document.getElementById('beamSearchOptions').classList.add('hidden');
    }
}

// Export as global
window.UIManager = UIManager;