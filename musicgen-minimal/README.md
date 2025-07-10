# MusicGen-Minimal v1.2.0

Text-to-music generation that actually works. No bullshit, just music.

## What This Does
- 🎵 Generates music from text descriptions
- 🎤 VocalGen - Add realistic singing vocals to your music (NEW in v1.2.0!)
- 🎯 Prompt Engineering Assistant - craft better prompts, get 30-50% better results
- 🎼 Melody-guided generation - provide audio, get styled variations
- 🚀 Extended generation beyond 30 seconds 
- 📦 Batch processing from CSV files
- 🎧 MP3 output support
- 📊 Real-time progress bars
- ✅ Uses Facebook's MusicGen models (including MusicGen-Melody)
- 💻 Works on CPU (slow) or GPU (fast)

## What This Doesn't Do
- No web interface (yet)
- No real-time generation
- No MIDI export (audio files only)

## Installation

⚠️ **Reality Check**: This package downloads 2.7GB of dependencies (PyTorch, etc.) on first install.

```bash
# Basic installation
pip install musicgen-minimal

# With MP3 support  
pip install 'musicgen-minimal[audio]'
```

**First install includes:**
- PyTorch (2.1GB) - ML framework
- Transformers (400MB) - HuggingFace models  
- Other dependencies (300MB)
- **Total**: ~2.7GB download, ~7GB disk space

**Optional MP3 support:**
- Install pydub: `pip install pydub`
- Install ffmpeg: https://ffmpeg.org/download.html
- Without these, falls back to WAV output

**First use downloads model:**
- Small model: 300MB (recommended)
- Medium model: 1.5GB (better quality)
- Large model: 3.2GB (best quality)

## Usage

### Prompt Engineering (NEW!)
```bash
# Build better prompts interactively
musicgen prompt

# Validate and improve any prompt
musicgen validate "epic masterpiece music"
# Output: ⚠️ 'epic' is too vague... 
# Improved: orchestral music with strings, dramatic mood

# Get genre-specific examples
musicgen prompt --genre jazz --example
```

### Basic Generation
```bash
# Generate 10 seconds of music (auto-validates prompt)
musicgen generate "upbeat jazz piano" --duration 10

# Generate 2-minute track (uses extended generation)
musicgen generate "epic orchestral symphony" --duration 120

# Generate with MP3 output (requires pydub)
musicgen generate "lofi hip hop" --output beats.mp3

# Disable prompt validation
musicgen generate "my prompt" --no-validate
```

### Melody-Guided Generation (NEW!)
```bash
# Generate music that follows a melody
musicgen melody "jazz style" melody.wav --output jazzy.wav

# Transform a piano piece into orchestral epic
musicgen melody "epic orchestral" piano.mp3 --output epic.wav

# Auto-detect duration from melody file
musicgen melody "lofi hip hop" beat.wav

# Style transfer - keep melody, change genre
musicgen melody "8-bit chiptune" classical.wav --output retro.mp3
```

### VocalGen - Music with Vocals (NEW!)

⚠️ **Experimental Feature**: Requires additional dependencies for best results.

```bash
# Basic vocal generation (uses built-in TTS)
musicgen vocal "upbeat pop song" "La la la, dancing in the moonlight" --output song.mp3

# Full song with custom lyrics
musicgen vocal "jazz ballad" "Blue skies, gentle breeze, memories of you" \
  --duration 60 --vocal-style jazz

# Different vocal styles
musicgen vocal "rock anthem" "We will rise up" --vocal-style rock --mix-style vocal_forward

# Enhance existing instrumental
musicgen remix instrumental.wav "Add these lyrics to the track" --output with_vocals.wav
```

**Installation for VocalGen:**
```bash
# Required for basic vocals
pip install TTS

# Optional for better quality (recommended)
pip install demucs  # Stem separation for mixing
pip install pedalboard  # Audio effects and mastering
pip install noisereduce  # Artifact reduction
```

**Vocal Styles:**
- `pop` - Clear, modern vocals
- `rock` - Powerful, gritty vocals  
- `jazz` - Smooth, expressive vocals
- `electronic` - Processed, spacey vocals
- `auto` - Automatically detect from music style

**Mix Styles:**
- `balanced` - Equal prominence (default)
- `vocal_forward` - Vocals more prominent
- `instrumental_forward` - Music more prominent

### Batch Processing
```bash
# Create sample CSV template
musicgen create-sample-csv

# Process multiple tracks from CSV
musicgen batch playlist.csv

# Parallel processing with 4 workers
musicgen batch jobs.csv --workers 4
```

### Python API
```python
from musicgen import quick_generate, quick_generate_with_melody, PromptEngineer
from musicgen import quick_generate_with_vocals, VocalGenPipeline

# Improve your prompts
engineer = PromptEngineer()
prompt = "epic music"
improved = engineer.improve_prompt(prompt)  # "orchestral music, moderate tempo"

# Generate 60-second MP3
quick_generate(
    improved,
    "trailer.mp3", 
    duration=60,
    format="mp3",
    bitrate="320k"
)

# Melody-guided generation
quick_generate_with_melody(
    "orchestral arrangement",
    "simple_melody.wav",
    "orchestral_version.mp3"
)

# Generate with vocals (NEW!)
quick_generate_with_vocals(
    "pop song",
    "song_with_vocals.mp3",
    lyrics="Dancing in the starlight, feeling so alive",
    duration=45
)

# Advanced vocal control
pipeline = VocalGenPipeline()
audio, sr = pipeline.generate_with_vocals(
    prompt="electronic dance music",
    lyrics="Feel the beat, move your feet",
    vocal_style="electronic",
    mix_style="instrumental_forward",
    enhance_level="heavy"
)
```

## Performance (Brutal Honesty)

**CPU Performance (Most Users)**
- **Model loading**: ~30 seconds
- **Generation speed**: 0.1x realtime (10x slower)
- **30-second song**: ~5 minutes
- **2-minute song**: ~20 minutes  
- **Memory usage**: ~4GB RAM

**GPU Performance (CUDA)**
- **Model loading**: ~5 seconds
- **Generation speed**: 2-5x realtime
- **30-second song**: ~10 seconds
- **2-minute song**: ~40 seconds
- **Memory usage**: ~4GB VRAM

⚠️ **WARNING**: Long generations on CPU will test your patience!

## Why So Simple?

Because it actually works. The original project had 343 files and didn't run.
This has 4 files and ships music.

## What's New in v1.1.0

### Extended Generation
- Generate music longer than 30 seconds (up to unlimited duration)
- Automatic segment blending with crossfade
- Progress tracking for multi-segment generation

### Batch Processing 
- Process multiple tracks from CSV files
- Parallel generation with multiprocessing
- Mixed format support (WAV and MP3 in same batch)

### MP3 Output
- 75-90% smaller files than WAV
- Configurable bitrate (128k-320k)
- Graceful fallback if dependencies missing

### Progress Bars
- Real-time progress for all operations
- ETA calculation for long generations
- Optional --no-progress for automation

## Contributing

Keep it simple. If your PR adds more than 50 lines, it's too complex.