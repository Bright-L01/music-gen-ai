# MusicGen-Minimal Changelog

## [1.2.0] - 2025-07-10

### 🎤 Added - VocalGen: AI Music with Vocals (EXPERIMENTAL)

#### Core Features
- **AI Vocal Synthesis**: Generate complete songs with AI-synthesized singing vocals
- **Two-Stage Pipeline**: MusicGen creates instrumental → TTS adds vocals → Intelligent mixing
- **Multiple Vocal Styles**: Pop, rock, jazz, electronic, or auto-detect from prompt
- **Structured Lyrics**: Support for verses, choruses, and bridges with automatic parsing
- **Intelligent Mixing**: Style-aware mixing parameters for optimal vocal/instrumental balance
- **Audio Enhancement**: Optional stem separation and mastering with Demucs and Pedalboard

#### CLI Integration
- **New command**: `musicgen vocal "prompt" "lyrics" --output song.mp3`
- **Vocal style control**: `--vocal-style` (auto, pop, rock, jazz, electronic)
- **Mix control**: `--mix-style` (balanced, vocal_forward, instrumental_forward)
- **Enhancement levels**: `--enhance` (none, light, moderate, heavy)
- **All standard options**: Duration, model size, format, bitrate, device selection

#### Python API
- **Quick function**: `quick_generate_with_vocals()` for simple vocal generation
- **Full pipeline**: `VocalGenPipeline` class for advanced control
- **Remix capability**: Add vocals to existing instrumental tracks
- **Progress callbacks**: Real-time progress tracking for vocal generation
- **Flexible mixing**: Programmatic control over all mixing parameters

#### Audio Processing
- **Lyrics Processing**: `LyricsProcessor` for parsing and timing alignment
- **Vocal Synthesis**: `VocalSynthesizer` with TTS integration and singing effects
- **Audio Enhancement**: `AudioEnhancer` with stem separation and mastering chains
- **Smart Mixing**: `AudioMixer` with style-aware parameters and effects

### 🔧 Changed

#### Test Suite
- **Improved test robustness**: Tests now handle missing optional dependencies gracefully
- **Better mocking**: Fixed issues with module-level imports in tests
- **Float precision**: Fixed audio normalization tests with proper tolerance
- **Coverage improvements**: Added comprehensive tests for all VocalGen components

### 📚 Documentation

#### New Documentation
- **VocalGen Guide**: Comprehensive guide at `docs/VOCALGEN_GUIDE.md`
- **Usage examples**: Multiple examples for different music styles
- **Troubleshooting**: Common issues and solutions
- **Best practices**: Tips for writing lyrics and getting better results

### 🧪 Testing

#### New Test Suite
- **VocalGen tests**: `test_vocalgen.py` with 22 comprehensive tests
- **Component tests**: Individual tests for each VocalGen component
- **Integration tests**: Full pipeline testing with mocked dependencies
- **Real-world validation**: Scripts for testing with actual TTS models

### 📦 Dependencies

#### Optional Dependencies for VocalGen
- **TTS**: Text-to-speech synthesis (required for vocals)
- **demucs**: Stem separation for better mixing (recommended)
- **pedalboard**: Audio effects and mastering (recommended)
- **librosa**: Advanced audio processing (optional)
- **noisereduce**: Artifact reduction (optional)

### 💡 Usage Examples

#### Basic Vocal Generation
```bash
# Simple pop song
musicgen vocal "upbeat pop music" "La la la, dancing in the sunshine" --output pop_song.mp3

# Jazz ballad with style
musicgen vocal "smooth jazz" "Blue moon, shining above" --vocal-style jazz --duration 60

# Rock anthem with vocal-forward mix
musicgen vocal "rock anthem" "We will rise" --vocal-style rock --mix-style vocal_forward
```

#### Python API
```python
from musicgen import quick_generate_with_vocals, VocalGenPipeline

# Quick generation
quick_generate_with_vocals(
    "happy pop song",
    "happy_song.mp3",
    lyrics="Sunshine day, come out and play",
    duration=30
)

# Advanced control
pipeline = VocalGenPipeline()
audio, sr = pipeline.generate_with_vocals(
    prompt="epic orchestral",
    lyrics="Heroes rise from ashes",
    vocal_style="rock",
    enhance_level="heavy"
)
```

### ⚠️ Limitations

- **Experimental feature**: Vocal quality depends on TTS model capabilities
- **Simple melodies**: Vocals follow basic pitch patterns, not complex melodies
- **Timing sync**: Lyrics may not perfectly align with beat
- **Language support**: Best results with English lyrics
- **Performance**: Vocal generation adds significant processing time

### 🚀 Installation

```bash
# Basic vocals (required)
pip install TTS

# Full quality (recommended)
pip install TTS demucs pedalboard librosa noisereduce
```

## [1.1.2] - 2025-07-09

### 🎯 Fixed

#### User Experience Improvements
- **Fixed misleading "Success!" message**: Now shows "⚠️ Partial Success" when MP3 conversion fails
- **Clear MP3 failure warnings**: Prominent instructions on how to install ffmpeg
- **Better error visibility**: ffmpeg errors are now clearly displayed with installation instructions

### 🔧 Changed

#### Error Messages
- MP3 conversion errors now show platform-specific ffmpeg installation commands
- convert_to_mp3 now returns (path, success) tuple for better error handling
- CLI shows yellow warning when MP3 was requested but WAV was delivered

#### User Feedback
- "Partial Success" status when format conversion fails
- Step-by-step instructions to enable MP3 support
- Clear distinction between successful generation and failed format conversion

## [1.1.1] - 2025-07-09

### 🐛 Fixed

#### Critical MP3 Bug Fix
- **Fixed MP3 save failure**: save_audio now handles MP3 filenames gracefully
- **Root cause**: soundfile doesn't support MP3 format, was causing "Invalid combination of format, subtype and endian" error
- **Solution**: save_audio now converts MP3 filenames to WAV automatically with warning

### 🔧 Changed

#### Error Handling
- Improved error messages when MP3 dependencies are missing
- Better filename handling in save_audio method
- More robust format detection

## [1.1.0] - 2025-07-09

### 🚀 Added

#### Extended Generation
- **Break the 30-second limit**: Generate audio of any duration using segment-based generation with crossfading
- **Automatic detection**: CLI automatically uses extended generation for durations > 30 seconds
- **Progress tracking**: Real-time progress updates for multi-segment generation
- **Crossfade blending**: Smooth transitions between segments (2-second overlap by default)

#### Batch Processing
- **CSV-based workflows**: Process multiple music generation tasks from CSV files
- **Multiprocessing support**: Parallel generation using up to 4 workers
- **Mixed format support**: WAV and MP3 outputs in the same batch
- **Progress tracking**: Real-time batch processing progress with ETA
- **Results logging**: Comprehensive JSON results with success rates and performance metrics
- **Template generation**: `musicgen create-sample-csv` for easy batch setup

#### MP3 Output Support
- **Compact file sizes**: 75-90% smaller than WAV files
- **Quality control**: Configurable bitrates (128k, 192k, 256k, 320k)
- **Auto-detection**: Automatic format detection from file extensions
- **Graceful fallback**: Falls back to WAV if pydub/ffmpeg not available
- **Batch compatible**: Full MP3 support in batch processing

#### Progress Tracking
- **Real-time progress**: Visual progress bars for all generation operations
- **Time estimates**: ETA calculation for long operations
- **Detailed feedback**: Step-by-step progress messages
- **Optional disable**: `--no-progress` flag for headless operation
- **Batch integration**: Progress tracking in batch processing

### 🔧 Changed

#### CLI Enhancements
- **New options**: `--format`, `--bitrate`, `--no-progress` flags
- **Auto-detection**: Automatic format detection from output filename
- **Enhanced output**: Shows format, bitrate, and compression information
- **Better validation**: Comprehensive parameter validation with helpful messages

#### Python API Improvements
- **Enhanced `quick_generate()`**: Added format and bitrate parameters
- **Progress callbacks**: Added progress callback support to all generation methods
- **Return values**: `quick_generate()` now returns actual output file path
- **Backward compatible**: All existing code continues to work unchanged

#### Performance Optimizations
- **Memory management**: Optimized memory usage for long generations
- **Worker scaling**: Intelligent worker count selection for batch processing
- **Progress overhead**: Minimal performance impact from progress tracking

### 🐛 Fixed

#### Error Handling
- **Graceful degradation**: Better handling of missing dependencies
- **Clear error messages**: Helpful guidance for common issues
- **Recovery mechanisms**: Robust error recovery in batch processing
- **Validation improvements**: Better parameter validation and fallbacks

#### Generation Quality
- **Crossfade artifacts**: Improved audio blending between segments
- **Duration accuracy**: Better duration matching for extended generation
- **Format consistency**: Reliable format detection and conversion

### 📚 Documentation

#### New Documentation
- **Feature guides**: Comprehensive documentation for all new features
- **Usage examples**: Real-world examples for all functionality
- **API reference**: Complete API documentation with examples
- **Performance notes**: Honest performance expectations and optimization tips

#### Updated Documentation
- **README**: Updated with new features and examples
- **CLI help**: Enhanced help text with all new options
- **Error messages**: Clearer error messages with actionable guidance

### 🏗️ Internal Changes

#### Code Organization
- **New modules**: `batch.py` for batch processing functionality
- **Progress classes**: `GenerationProgress` for progress tracking
- **Type hints**: Complete type annotations for better IDE support
- **Test coverage**: Comprehensive test suites for all new features

#### Dependencies
- **Optional dependencies**: `pydub` as optional dependency for MP3 support
- **Pandas integration**: For robust CSV processing in batch mode
- **Rich enhancements**: Enhanced progress bars and UI elements

### 🧪 Testing

#### New Test Suites
- **Extended generation tests**: `test_extended_generation.py`
- **Batch processing tests**: `test_batch_processing.py`
- **MP3 support tests**: `test_mp3_support.py`
- **Integration tests**: `test_all_features.py`

#### Test Coverage
- **Feature integration**: Tests for all features working together
- **Error conditions**: Tests for error handling and recovery
- **Performance tests**: Basic performance and memory usage tests
- **CLI integration**: Tests for all CLI functionality

### 💡 Usage Examples

#### Extended Generation
```bash
# Generate 2-minute orchestral piece
musicgen generate "epic orchestral symphony" --duration 120

# With progress tracking
musicgen generate "ambient soundscape" --duration 90 --format mp3
```

#### Batch Processing
```bash
# Create sample CSV
musicgen create-sample-csv

# Process batch with mixed formats
musicgen batch sample_batch.csv --workers 2

# Custom output directory
musicgen batch jobs.csv --output-dir my_music --format mp3
```

#### MP3 Output
```bash
# High-quality MP3
musicgen generate "jazz piano" --format mp3 --bitrate 320k

# Auto-detect from filename
musicgen generate "rock guitar" -o song.mp3

# Batch with MP3 output
musicgen batch jobs.csv --format mp3 --bitrate 192k
```

#### Progress Control
```bash
# Default progress bars
musicgen generate "classical music" --duration 60

# Disable progress (headless mode)
musicgen generate "ambient music" --no-progress
```

### 🔄 Migration Guide

#### From 1.0.0 to 1.1.0

**Existing code continues to work unchanged**, but you can now take advantage of new features:

```python
# Old way (still works)
from musicgen import quick_generate
quick_generate("music", "output.wav", duration=30)

# New way (enhanced)
from musicgen import quick_generate
output_path = quick_generate(
    "music", 
    "output.mp3", 
    duration=60,      # Extended generation
    format="mp3",     # MP3 output
    bitrate="320k"    # High quality
)
```

#### New Dependencies

For full MP3 support, install:
```bash
pip install 'musicgen-minimal[audio]'
# or
pip install pydub
```

### 🎯 Performance Notes

#### Generation Speed
- **CPU**: ~0.1-0.2x realtime (slow but functional)
- **GPU**: ~5-10x faster with CUDA support
- **Extended generation**: Proportionally longer (2x duration = 2x time)
- **Batch processing**: Linear scaling with worker count

#### Memory Usage
- **Basic generation**: ~4GB RAM
- **Extended generation**: ~4GB RAM (constant)
- **Batch processing**: ~4GB per worker
- **MP3 conversion**: Minimal additional memory

#### File Sizes
- **WAV**: ~10MB per minute
- **MP3 192k**: ~1.5MB per minute (85% smaller)
- **MP3 320k**: ~2.5MB per minute (75% smaller)

### 🏆 Success Metrics

#### Feature Delivery
- ✅ **Extended Generation**: Generate unlimited duration audio
- ✅ **Batch Processing**: Process 100+ files in one command
- ✅ **MP3 Support**: 75-90% file size reduction
- ✅ **Progress Tracking**: Real-time feedback for all operations

#### User Experience
- ✅ **Zero breaking changes**: All existing code works
- ✅ **Intuitive CLI**: Auto-detection reduces complexity
- ✅ **Clear documentation**: Comprehensive guides and examples
- ✅ **Reliable operation**: Robust error handling and recovery

## [1.0.0] - 2024-01-XX

### 🚀 Initial Release

- **Text-to-music generation**: Core functionality using Facebook's MusicGen
- **Simple CLI**: Basic command-line interface
- **Python API**: `quick_generate()` function for easy integration
- **WAV output**: High-quality audio output
- **Model support**: Small, medium, and large model variants
- **Basic validation**: Parameter validation and error handling

---

**Full Changelog**: https://github.com/musicgen-minimal/musicgen-minimal/releases