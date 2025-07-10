# VocalGen Documentation - AI-Synthesized Vocals for MusicGen

## Overview

VocalGen is an experimental feature that adds AI-synthesized vocals to MusicGen-generated instrumental tracks. It uses text-to-speech (TTS) technology combined with audio processing to create singing vocals that blend with the instrumental music.

## Installation

### Required Dependencies
```bash
pip install TTS  # Coqui TTS for vocal synthesis
```

### Optional Dependencies (for enhanced quality)
```bash
pip install demucs       # For stem separation
pip install pedalboard   # For audio effects
pip install librosa      # For audio processing
pip install noisereduce  # For noise reduction
```

## Usage

### Command Line Interface

Basic usage:
```bash
musicgen vocal "upbeat pop song" "La la la, this is my song" --duration 30
```

With options:
```bash
musicgen vocal "jazz ballad" "Verse lyrics here..." \
  --output song.mp3 \
  --duration 45 \
  --vocal-style jazz \
  --mix-style vocal_forward \
  --enhance moderate
```

#### CLI Parameters
- `prompt`: Music style description (required)
- `lyrics`: Song lyrics to sing (required)
- `--output`: Output filename (default: output_with_vocals.wav)
- `--duration`: Duration in seconds (default: 30.0)
- `--model`: Model size - small, medium, large (default: small)
- `--vocal-style`: Vocal style - auto, pop, rock, jazz, electronic (default: auto)
- `--mix-style`: Mix style - balanced, vocal_forward, instrumental_forward (default: balanced)
- `--enhance`: Enhancement level - light, moderate, heavy, none (default: moderate)
- `--format`: Output format - wav, mp3 (default: auto-detect)

### Python API

Simple usage:
```python
from musicgen import quick_generate_with_vocals

output = quick_generate_with_vocals(
    prompt="upbeat pop song",
    output_file="song_with_vocals.mp3",
    lyrics="La la la, this is my song"
)
```

Advanced usage with VocalGenPipeline:
```python
from musicgen.vocalgen import VocalGenPipeline

# Initialize pipeline
pipeline = VocalGenPipeline(
    model_name="facebook/musicgen-small",
    device="cuda",  # or "cpu"
    enhance_output=True
)

# Generate with custom parameters
audio, sample_rate = pipeline.generate_with_vocals(
    prompt="smooth jazz",
    lyrics="""
    [Verse 1]
    Walking down the street tonight
    Stars are shining oh so bright
    
    [Chorus]
    This is our song, our melody
    Dancing free, just you and me
    """,
    duration=60.0,
    vocal_style="jazz",
    mix_style="vocal_forward",
    enhance_level="moderate",
    progress_callback=lambda percent, msg: print(f"{percent}% - {msg}")
)

# Save the output
pipeline.music_generator.save_audio(audio, sample_rate, "output.wav")
```

## Architecture

### Pipeline Components

1. **VocalGenPipeline** - Main orchestrator
   - Coordinates instrumental generation, vocal synthesis, and mixing
   - Handles progress tracking and parameter management

2. **VocalSynthesizer** - Vocal generation
   - Uses Coqui TTS for text-to-speech
   - Applies singing effects (pitch modulation, vibrato, formant shifting)
   - Supports multiple vocal styles

3. **AudioEnhancer** - Quality improvement
   - Stem separation with Demucs
   - Audio effects with Pedalboard
   - Noise reduction and artifact removal

4. **AudioMixer** - Intelligent mixing
   - Style-aware mixing parameters
   - Automatic gain compensation
   - Optional sidechain compression

5. **LyricsProcessor** - Lyrics handling
   - Parses structured lyrics ([Verse], [Chorus], etc.)
   - Aligns lyrics to musical timing
   - Handles tempo-based timing

### Generation Process

1. **Instrumental Generation**: MusicGen creates the backing track
2. **Style Detection**: Analyzes prompt for musical style
3. **Lyrics Processing**: Parses and structures the lyrics
4. **Vocal Synthesis**: Generates vocals with TTS + effects
5. **Mixing**: Blends vocals and instrumental intelligently
6. **Enhancement**: (Optional) Applies mastering effects

## Vocal Styles

### Pop
- Clear, upfront vocals
- Moderate reverb
- Balanced compression
- Bright EQ curve

### Rock
- Powerful, forward vocals
- Less reverb, more presence
- Heavy compression
- Mid-range emphasis

### Jazz
- Smooth, warm vocals
- More reverb for space
- Gentle compression
- Rolled-off highs

### Electronic
- Processed, blended vocals
- Creative effects
- Moderate compression
- Can be more experimental

### Auto (Default)
- Detects style from prompt
- Applies appropriate settings

## Mix Styles

### Balanced
- Equal focus on vocals and instrumental
- Natural blending
- Default mixing approach

### Vocal Forward
- Vocals prominently featured
- Instrumental acts as backing
- Good for lyric-focused songs

### Instrumental Forward
- Vocals blend into the mix
- Instrumental takes focus
- Good for ambient/atmospheric tracks

## Enhancement Levels

### None
- No post-processing
- Raw output from synthesis

### Light
- Basic normalization
- Minimal processing
- Preserves original character

### Moderate (Default)
- Noise reduction
- Balanced mastering
- Good for most use cases

### Heavy
- Full stem separation
- Individual stem processing
- Maximum quality (slower)

## Lyrics Format

VocalGen supports structured lyrics with section markers:

```
[Verse 1]
First verse lyrics here
Multiple lines supported

[Chorus]
Chorus lyrics
Can repeat sections

[Verse 2]
Second verse

[Bridge]
Bridge section

[Outro]
Ending lyrics
```

Without markers, lyrics are treated as continuous verses.

## Quality Tips

### For Best Results:
1. **Keep lyrics simple** - Complex lyrics may be harder to understand
2. **Match style to prompt** - Use jazz vocals for jazz music
3. **Shorter duration** - Quality is better for 30-60 second clips
4. **Use enhancement** - The moderate setting improves most outputs
5. **Experiment with mix** - Try different mix styles for your song

### Common Issues:
- **Robotic vocals**: Try different vocal styles or increase enhancement
- **Timing issues**: Simplify lyrics or adjust tempo
- **Muddy mix**: Use vocal_forward mix style
- **Artifacts**: Enable enhancement or use heavy enhancement level

## Technical Details

### Dependencies Status
The pipeline checks for optional dependencies and gracefully degrades:
- Without Demucs: No stem separation, simpler mixing
- Without Pedalboard: Limited audio effects
- Without librosa: Basic audio processing only
- Without noisereduce: No artifact reduction

### Performance
- GPU recommended for faster generation
- Enhancement adds 20-50% to generation time
- Heavy enhancement can take 2-3x longer
- Typical generation: 5-10x faster than realtime on GPU

### Limitations
- Vocal quality depends on TTS model
- Complex lyrics may have pronunciation issues
- Long durations (>60s) may have consistency issues
- Best suited for simple, melodic vocals

## Examples

### Pop Song
```bash
musicgen vocal "catchy pop song with upbeat tempo" \
  "Hey hey hey, we're dancing all night / \
   Feel the rhythm, feel the light / \
   La la la, this is our song / \
   Come on everybody, sing along" \
  --vocal-style pop \
  --mix-style vocal_forward \
  --duration 30
```

### Jazz Ballad
```bash
musicgen vocal "smooth jazz ballad" \
  "[Verse] / \
   Blue moon shining up above / \
   Whispering words of love / \
   [Chorus] / \
   In the still of the night / \
   Everything feels so right" \
  --vocal-style jazz \
  --enhance heavy \
  --duration 45
```

### Electronic Track
```bash
musicgen vocal "electronic dance music" \
  "Move your body / Feel the beat / \
   Electronic symphony / \
   Lost in the heat" \
  --vocal-style electronic \
  --mix-style instrumental_forward \
  --enhance moderate
```

## Future Improvements

Planned enhancements:
- Fine-tuned singing models
- Multi-voice harmonies
- Better pitch tracking
- Real-time generation
- MIDI alignment
- Custom voice models

## Troubleshooting

### No audio output
- Check if TTS is properly installed
- Verify CUDA/CPU availability
- Try shorter duration or simpler lyrics

### Poor quality vocals
- Increase enhancement level
- Try different vocal style
- Simplify lyrics
- Adjust mix style

### Installation issues
- Install TTS first: `pip install TTS`
- For Mac/Linux: May need `apt-get install espeak` or `brew install espeak`
- Check PyTorch installation for GPU support

## Contributing

VocalGen is experimental and welcomes contributions:
- Report issues on GitHub
- Submit PRs for improvements
- Share your generated songs
- Suggest new features

## License

Same as MusicGen-Minimal - MIT License