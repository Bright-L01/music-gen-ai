# API Reference

## Overview

The Music Gen AI API provides programmatic access to music generation capabilities through both Python and REST interfaces.

## Python API

### Basic Usage

```python
from music_gen import MusicGenerator

# Initialize generator
generator = MusicGenerator(model="facebook/musicgen-medium")

# Generate music
audio = generator.generate(
    prompt="Calm piano melody",
    duration=30.0,
    temperature=0.8
)

# Save audio
audio.save("output.wav")
```

### Core Classes

#### `MusicGenerator`

Main class for music generation.

**Parameters:**
- `model` (str): Model name or path
- `device` (str): Device to use ("cuda", "cpu", "auto")
- `dtype` (torch.dtype): Model precision
- `cache_dir` (str): Cache directory for models

**Methods:**

##### `generate(prompt, duration, temperature, top_k, top_p, seed)`
Generate audio from text prompt.

**Returns:** `AudioOutput` object

##### `generate_batch(prompts, **kwargs)`
Generate multiple audio clips in batch.

**Returns:** List of `AudioOutput` objects

#### `MultiInstrumentGenerator`

Generate multi-track music with separate instruments.

**Methods:**

##### `generate_multi_track(prompts, duration, mix_settings)`
Generate multiple instrument tracks.

**Parameters:**
- `prompts` (dict): Mapping of instrument names to prompts
- `duration` (float): Duration in seconds
- `mix_settings` (dict): Mixing parameters

**Returns:** Dictionary of instrument tracks

#### `AudioMixer`

Professional audio mixing capabilities.

**Methods:**

##### `mix_tracks(tracks, levels, panning, effects)`
Mix multiple audio tracks.

##### `apply_effects(audio, effects_chain)`
Apply audio effects chain.

##### `master(audio, target_lufs, headroom_db)`
Master audio to professional standards.

## REST API

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
```bash
curl -H "Authorization: Bearer YOUR_API_KEY"
```

### Endpoints

#### `POST /generate`
Generate music from text prompt.

**Request:**
```json
{
  "prompt": "Upbeat jazz piano",
  "duration": 30.0,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.9
}
```

**Response:**
```json
{
  "task_id": "uuid",
  "status": "processing",
  "eta": 45
}
```

#### `GET /generate/{task_id}`
Check generation status.

**Response:**
```json
{
  "task_id": "uuid",
  "status": "completed",
  "audio_url": "https://...",
  "duration": 30.0,
  "metadata": {}
}
```

#### `POST /generate/batch`
Batch generation endpoint.

**Request:**
```json
{
  "requests": [
    {"prompt": "Piano melody", "duration": 15},
    {"prompt": "Guitar solo", "duration": 20}
  ]
}
```

#### `POST /multi-instrument`
Generate multi-track music.

**Request:**
```json
{
  "instruments": {
    "piano": "Jazz piano comping",
    "bass": "Walking bass line",
    "drums": "Swing pattern"
  },
  "duration": 60.0,
  "mix_preset": "jazz_club"
}
```

#### `GET /models`
List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "musicgen-small",
      "size": "300M",
      "quality": "good"
    }
  ]
}
```

#### `GET /health`
Health check endpoint.

### WebSocket API

#### `/ws/stream`
Real-time audio streaming.

**Message format:**
```json
{
  "type": "generate",
  "prompt": "Ambient music",
  "stream": true
}
```

**Response stream:**
```json
{
  "type": "audio_chunk",
  "data": "base64_encoded_audio",
  "timestamp": 1234567890
}
```

### Error Responses

```json
{
  "error": {
    "code": "INVALID_PROMPT",
    "message": "Prompt cannot be empty",
    "details": {}
  }
}
```

### Rate Limiting

- 100 requests per minute per API key
- 10 concurrent generations per account
- 1GB monthly bandwidth limit (free tier)

### SDK Examples

#### Python
```python
from music_gen import APIClient

client = APIClient(api_key="YOUR_KEY")
result = client.generate("Jazz piano", duration=30)
result.save("jazz.wav")
```

#### JavaScript
```javascript
const MusicGenAPI = require('musicgen-api');

const client = new MusicGenAPI({apiKey: 'YOUR_KEY'});
const result = await client.generate({
  prompt: 'Jazz piano',
  duration: 30
});
```

#### cURL
```bash
curl -X POST https://api.musicgen.ai/generate \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Jazz piano", "duration": 30}'
```