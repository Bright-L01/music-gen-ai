# VocalGen - AI Music with Vocals Guide

## Overview

VocalGen adds vocal synthesis capabilities to MusicGen, allowing you to create complete songs with AI-generated singing. This is an experimental feature that combines instrumental generation with text-to-speech technology.

## How It Works

VocalGen uses a two-stage pipeline:

1. **Instrumental Generation**: MusicGen creates the background music based on your prompt
2. **Vocal Synthesis**: A TTS model generates singing from your lyrics
3. **Intelligent Mixing**: Audio tracks are mixed with style-aware parameters
4. **Enhancement** (optional): Stem separation and audio effects improve quality

## Installation

### Basic Installation

```bash
# Install musicgen-minimal first
pip install musicgen-minimal

# Required for vocals
pip install TTS
```

### Full Installation (Recommended)

```bash
# All optional dependencies for best quality
pip install TTS           # Vocal synthesis (required)
pip install demucs        # Stem separation for better mixing
pip install pedalboard    # Audio effects and mastering
pip install librosa       # Advanced audio processing
pip install noisereduce   # Artifact reduction
```

## Usage

### Command Line

#### Basic Usage

```bash
# Simple vocal generation
musicgen vocal "upbeat pop song" "La la la, dancing in the moonlight" --output song.mp3

# Longer song with custom duration
musicgen vocal "jazz ballad" "Blue skies above, gentle breeze" --duration 60

# Different vocal styles
musicgen vocal "rock anthem" "We will rise up, never give up" --vocal-style rock
```

#### Advanced Options

```bash
# Full control over generation
musicgen vocal "electronic dance music" "Feel the beat, move your feet" \
  --output edm_track.mp3 \
  --duration 45 \
  --vocal-style electronic \
  --mix-style instrumental_forward \
  --enhance heavy \
  --model medium \
  --device cuda
```

#### Options Explained

- `--vocal-style`: How the vocals sound (auto, pop, rock, jazz, electronic)
- `--mix-style`: Balance between vocals and instrumental
  - `balanced`: Equal prominence (default)
  - `vocal_forward`: Vocals more prominent
  - `instrumental_forward`: Music more prominent
- `--enhance`: Audio enhancement level (none, light, moderate, heavy)
- `--model`: MusicGen model size (small, medium, large)

### Python API

#### Basic Usage

```python
from musicgen import quick_generate_with_vocals

# Generate a song with vocals
quick_generate_with_vocals(
    prompt="happy pop music",
    output_file="happy_song.mp3",
    lyrics="Sunshine day, come out and play",
    duration=30
)
```

#### Advanced Usage

```python
from musicgen import VocalGenPipeline

# Create pipeline with custom settings
pipeline = VocalGenPipeline(
    model_name="facebook/musicgen-medium",
    enhance_output=True
)

# Generate with full control
audio, sample_rate = pipeline.generate_with_vocals(
    prompt="epic orchestral music",
    lyrics="""[Verse 1]
    Rising from the ashes
    We stand tall and proud
    
    [Chorus]
    Heroes of the light
    Fighting through the night""",
    duration=60,
    vocal_style="rock",
    mix_style="balanced",
    enhance_level="moderate",
    progress_callback=lambda percent, msg: print(f"{percent}% - {msg}")
)

# Save the result
pipeline.music_generator.save_audio(audio, sample_rate, "epic_song.wav")
```

#### Add Vocals to Existing Music

```python
# Remix an existing instrumental with vocals
pipeline.remix_with_vocals(
    instrumental_path="my_beat.wav",
    lyrics="Add these words to my beat",
    vocal_style="pop",
    output_path="beat_with_vocals.wav"
)
```

## Writing Lyrics

### Format

VocalGen supports structured lyrics with sections:

```
[Verse 1]
First line of verse
Second line of verse

[Chorus]
Catchy chorus line 1
Catchy chorus line 2

[Verse 2]
Another verse here

[Bridge]
Bridge section
```

### Tips for Better Results

1. **Keep it Simple**: Simple, repetitive lyrics work best
2. **Use Natural Phrasing**: Write lyrics that flow naturally when spoken
3. **Avoid Complex Words**: Stick to common vocabulary
4. **Include Repetition**: Choruses and repeated phrases sound more musical
5. **Match the Style**: Write lyrics appropriate for your music style

### Example Lyrics by Style

**Pop**:
```
[Verse]
Walking down the street today
Sunshine brightens up my way

[Chorus]
La la la, feeling free
This is where I want to be
```

**Rock**:
```
[Verse]
Thunder rolling through the night
We're gonna win this fight

[Chorus]
Stand up, stand strong
We've been waiting for so long
```

**Jazz**:
```
[Verse]
Blue notes floating in the air
Moonlight dancing everywhere

[Chorus]
Swing with me, through the night
Everything's gonna be alright
```

## Vocal Styles

### Available Styles

- **auto**: Automatically detects from music prompt
- **pop**: Clear, modern vocals with moderate expression
- **rock**: Powerful, slightly gritty vocals
- **jazz**: Smooth, expressive vocals with vibrato
- **electronic**: Processed, ethereal vocals

### Style Detection

If you use `--vocal-style auto`, VocalGen detects the style from your music prompt:

- "jazz", "swing", "bebop" → jazz style
- "rock", "metal", "punk" → rock style
- "electronic", "techno", "edm" → electronic style
- "classical", "orchestra" → classical style
- Default → pop style

## Troubleshooting

### No Vocals Generated

**Problem**: Music generates but no vocals are added.

**Solution**: Install TTS:
```bash
pip install TTS
```

### Poor Vocal Quality

**Problem**: Vocals sound robotic or unclear.

**Solutions**:
1. Install enhancement dependencies:
   ```bash
   pip install demucs pedalboard
   ```
2. Use `--enhance moderate` or `--enhance heavy`
3. Try different vocal styles
4. Simplify your lyrics

### Vocals Too Quiet/Loud

**Problem**: Vocal balance is off.

**Solutions**:
- Too quiet: Use `--mix-style vocal_forward`
- Too loud: Use `--mix-style instrumental_forward`
- Fine-tune in Python: `mix_style="balanced"` with custom `vocal_level`

### Generation Too Slow

**Problem**: Takes too long to generate.

**Solutions**:
1. Use GPU: `--device cuda`
2. Use smaller model: `--model small`
3. Disable enhancement: `--enhance none`
4. Reduce duration

### Out of Memory

**Problem**: CUDA out of memory errors.

**Solutions**:
1. Use CPU: `--device cpu`
2. Use smaller model: `--model small`
3. Reduce duration
4. Disable enhancement

## Limitations

### Current Limitations

1. **Vocal Quality**: TTS-based vocals don't match human singing quality
2. **Melody**: Vocals follow simple melody patterns, not complex melodies
3. **Timing**: Lyrics may not perfectly sync with beat
4. **Languages**: Best results with English lyrics
5. **Duration**: Quality degrades for very long generations

### Not Supported

- Real-time generation
- Specific voice selection
- MIDI export
- Explicit tempo/key control
- Multi-voice harmonies

## Best Practices

### For Best Results

1. **Start Small**: Test with 15-30 second clips first
2. **Match Style**: Ensure lyrics match the music style
3. **Use Enhancement**: Enable audio enhancement for better quality
4. **Experiment**: Try different vocal and mix styles
5. **Post-Process**: Consider external audio editing for final polish

### Workflow Tips

1. Generate instrumental first to test the music
2. Add simple lyrics and test vocal generation
3. Refine lyrics based on results
4. Adjust mix parameters
5. Apply enhancement for final version

## Examples

### Complete Pop Song

```bash
musicgen vocal "upbeat pop music with synth and drums" \
  "[Verse 1]
  Waking up to a brand new day
  All my worries fade away
  
  [Chorus]
  Dance dance dance, feel the beat
  Move your body, move your feet
  Dance dance dance, all night long
  This is our favorite song
  
  [Verse 2]
  Lights are flashing everywhere
  Music pumping in the air" \
  --output pop_hit.mp3 \
  --duration 60 \
  --vocal-style pop \
  --mix-style balanced \
  --enhance moderate
```

### Jazz Ballad

```python
from musicgen import VocalGenPipeline

pipeline = VocalGenPipeline(model_name="facebook/musicgen-medium")

jazz_lyrics = """[Verse]
Strolling through the city lights
Jazz notes fill these summer nights
Saxophone plays sweet and low
That's the only sound I know

[Chorus]
Blue moon shining up above
This is what they call jazz love
Syncopated heartbeat true
All I need is me and you"""

audio, sr = pipeline.generate_with_vocals(
    prompt="smooth jazz ballad with saxophone and piano",
    lyrics=jazz_lyrics,
    duration=90,
    vocal_style="jazz",
    mix_style="balanced",
    enhance_level="moderate"
)

pipeline.music_generator.save_audio(audio, sr, "jazz_ballad.wav")
```

## Technical Details

### Architecture

```
Text Prompt → MusicGen → Instrumental Audio
                              ↓
Lyrics → TTS → Vocal Audio → Mix → Enhancement → Final Output
```

### Audio Processing Pipeline

1. **Instrumental Generation**: 32kHz mono from MusicGen
2. **Vocal Synthesis**: TTS output resampled to match
3. **Mixing**: Intelligent mixing with style parameters
4. **Enhancement** (optional):
   - Stem separation with Demucs
   - Effects with Pedalboard
   - Noise reduction
   - Mastering chain

### Performance

- CPU: ~10x slower than realtime
- GPU: ~2-5x slower than realtime
- Memory: 4-8GB depending on model
- Disk: ~100MB per minute of audio

## Future Improvements

Planned enhancements for VocalGen:

1. **Better Vocal Models**: Integration with singing-specific models
2. **Melody Control**: MIDI-based melody specification
3. **Voice Selection**: Multiple voice options
4. **Real-time Generation**: Streaming output
5. **Multi-language**: Support for non-English lyrics
6. **Harmonies**: Multi-voice arrangements

## Contributing

VocalGen is experimental and we welcome contributions! Areas for improvement:

- Better TTS model integration
- Improved lyrics-to-melody mapping
- Advanced mixing algorithms
- Performance optimizations
- Additional vocal styles

See the main project README for contribution guidelines.

## Support

- GitHub Issues: Report bugs and request features
- Discussions: Share your creations and get help
- Examples: Check the `examples/` folder for more use cases

Remember: VocalGen is experimental. Results will vary, but it's a starting point for AI-generated music with vocals!