"""Multi-instrument API endpoints for MusicGen."""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..audio.mixing import MixingConfig, MixingEngine, TrackConfig
from ..audio.separation import DemucsSeparator, HybridSeparator, SpleeterSeparator
from ..export.midi import MIDIConverter, MIDIExportConfig
from ..models.multi_instrument import (
    MultiInstrumentMusicGen,
    MultiTrackGenerator,
    TrackGenerationConfig,
)
from ..utils.audio import load_audio_file, save_audio_file


# Pydantic models for multi-instrument API
class InstrumentTrackRequest(BaseModel):
    """Configuration for a single instrument track."""

    instrument: str = Field(..., description="Instrument name")
    volume: float = Field(0.7, ge=0.0, le=1.0, description="Track volume")
    pan: float = Field(0.0, ge=-1.0, le=1.0, description="Stereo pan (-1=left, 1=right)")
    reverb: float = Field(0.2, ge=0.0, le=1.0, description="Reverb amount")
    start_time: float = Field(0.0, ge=0.0, description="Start time in seconds")
    duration: Optional[float] = Field(None, description="Track duration (None=full length)")
    midi_program: Optional[int] = Field(None, description="MIDI program number override")


class MultiInstrumentGenerationRequest(BaseModel):
    """Request for multi-instrument generation."""

    prompt: str = Field(..., description="Text description")
    tracks: List[InstrumentTrackRequest] = Field(..., description="Instrument tracks")
    duration: float = Field(30.0, ge=1.0, le=120.0, description="Total duration")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Sampling temperature")
    use_beam_search: bool = Field(False, description="Use beam search")
    auto_mix: bool = Field(True, description="Enable automatic mixing")
    export_stems: bool = Field(False, description="Export individual stems")
    export_midi: bool = Field(False, description="Export MIDI file")
    seed: Optional[int] = Field(None, description="Random seed")


class TrackSeparationRequest(BaseModel):
    """Request for track separation."""

    method: str = Field("hybrid", description="Separation method: demucs, spleeter, hybrid")
    targets: Optional[List[str]] = Field(None, description="Target stems to extract")
    enhance_quality: bool = Field(True, description="Apply quality enhancement")


class MixingRequest(BaseModel):
    """Request for audio mixing."""

    tracks: List[Dict[str, Any]] = Field(..., description="Track configurations")
    master_volume: float = Field(0.8, ge=0.0, le=1.0, description="Master volume")
    apply_mastering: bool = Field(True, description="Apply mastering chain")


def setup_multi_instrument_routes(app: FastAPI, model: MultiInstrumentMusicGen, temp_dir: Path):
    """Setup multi-instrument API routes."""

    # Initialize components
    mixing_engine = MixingEngine(MixingConfig())
    midi_converter = MIDIConverter(MIDIExportConfig())
    separators = {
        "demucs": DemucsSeparator(),
        "spleeter": SpleeterSeparator(),
        "hybrid": HybridSeparator(),
    }

    # Load separation models
    for separator in separators.values():
        separator.load_model()

    @app.get("/instruments")
    async def list_instruments():
        """List all available instruments."""
        return {
            "instruments": model.multi_config.get_instrument_names(),
            "categories": {
                "keyboards": ["piano", "electric_piano", "harpsichord", "organ", "synthesizer"],
                "strings": ["violin", "viola", "cello", "double_bass", "harp"],
                "guitars": ["acoustic_guitar", "electric_guitar", "bass_guitar"],
                "brass": ["trumpet", "trombone", "french_horn", "tuba"],
                "woodwinds": ["flute", "clarinet", "saxophone", "oboe"],
                "percussion": ["drums", "timpani", "xylophone", "vibraphone"],
                "voice": ["soprano", "alto", "tenor", "bass_voice", "choir"],
                "synth": ["synth_pad", "synth_lead"],
            },
        }

    @app.get("/instruments/{instrument_name}")
    async def get_instrument_info(instrument_name: str):
        """Get detailed information about an instrument."""
        config = model.multi_config.get_instrument_config(instrument_name)
        if not config:
            raise HTTPException(status_code=404, detail="Instrument not found")

        return {
            "name": config.name,
            "midi_program": config.midi_program,
            "frequency_range": config.frequency_range,
            "octave_range": config.typical_octave_range,
            "properties": {
                "polyphonic": config.polyphonic,
                "percussion": config.percussion,
                "sustained": config.sustained,
            },
            "defaults": {"volume": config.default_volume, "pan": config.default_pan},
        }

    @app.post("/generate/multi-instrument")
    async def generate_multi_instrument(
        request: MultiInstrumentGenerationRequest, background_tasks: BackgroundTasks
    ):
        """Generate multi-instrument music."""
        task_id = str(uuid.uuid4())

        # Start background task
        background_tasks.add_task(
            generate_multi_instrument_task,
            task_id,
            request,
            model,
            mixing_engine,
            midi_converter,
            temp_dir,
        )

        return {"task_id": task_id, "status": "processing"}

    @app.post("/separate-tracks")
    async def separate_tracks(
        audio_file: UploadFile = File(...),
        request: TrackSeparationRequest = TrackSeparationRequest(),
    ):
        """Separate audio into instrument tracks."""
        # Save uploaded file
        temp_path = temp_dir / f"sep_{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # Load audio
        audio, sr = load_audio_file(str(temp_path))

        # Perform separation
        separator = separators.get(request.method, separators["hybrid"])
        result = separator.separate(audio, sr, request.targets)

        # Save separated tracks
        output_paths = {}
        for stem_name, stem_audio in result.stems.items():
            stem_path = temp_dir / f"stem_{stem_name}_{uuid.uuid4()}.wav"
            save_audio_file(stem_audio, str(stem_path), result.sample_rate)
            output_paths[stem_name] = str(stem_path)

        # Cleanup
        temp_path.unlink()

        return {
            "stems": output_paths,
            "confidence_scores": result.confidence_scores,
            "processing_time": result.processing_time,
        }

    @app.post("/mix-tracks")
    async def mix_tracks(request: MixingRequest):
        """Mix multiple audio tracks."""
        # Load tracks
        tracks = {}
        track_configs = {}

        for track_data in request.tracks:
            track_name = track_data["name"]
            audio_path = track_data["audio_path"]

            # Load audio
            audio, sr = load_audio_file(audio_path)
            tracks[track_name] = audio

            # Create track config
            track_configs[track_name] = TrackConfig(
                name=track_name,
                volume=track_data.get("volume", 0.7),
                pan=track_data.get("pan", 0.0),
                reverb_send=track_data.get("reverb", 0.0),
                delay_send=track_data.get("delay", 0.0),
                eq_low_gain=track_data.get("eq_low", 0.0),
                eq_mid_gain=track_data.get("eq_mid", 0.0),
                eq_high_gain=track_data.get("eq_high", 0.0),
            )

        # Mix tracks
        mixed_audio = mixing_engine.mix(tracks, track_configs)

        # Save mixed audio
        output_path = temp_dir / f"mixed_{uuid.uuid4()}.wav"
        save_audio_file(mixed_audio, str(output_path), mixing_engine.config.sample_rate)

        return {
            "mixed_audio_path": str(output_path),
            "duration": mixed_audio.shape[-1] / mixing_engine.config.sample_rate,
            "metering": mixing_engine.get_metering(mixed_audio),
        }

    @app.post("/export-midi")
    async def export_midi(audio_file: UploadFile = File(...), instrument: str = "piano"):
        """Convert audio to MIDI."""
        # Save uploaded file
        temp_path = temp_dir / f"midi_{uuid.uuid4()}.wav"
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # Load audio
        audio, sr = load_audio_file(str(temp_path))

        # Convert to MIDI
        midi = midi_converter.convert_single_track(audio, instrument)

        # Save MIDI file
        midi_path = temp_dir / f"output_{uuid.uuid4()}.mid"
        midi.write(str(midi_path))

        # Cleanup
        temp_path.unlink()

        return FileResponse(
            midi_path, media_type="audio/midi", filename=f"converted_{instrument}.mid"
        )


async def generate_multi_instrument_task(
    task_id: str,
    request: MultiInstrumentGenerationRequest,
    model: MultiInstrumentMusicGen,
    mixing_engine: MixingEngine,
    midi_converter: MIDIConverter,
    temp_dir: Path,
):
    """Background task for multi-instrument generation."""
    try:
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)

        # Create track configurations
        track_configs = []
        for track in request.tracks:
            config = TrackGenerationConfig(
                instrument=track.instrument,
                volume=track.volume,
                pan=track.pan,
                reverb=track.reverb,
                start_time=track.start_time,
                duration=track.duration,
            )
            track_configs.append(config)

        # Generate multi-track audio
        generator = MultiTrackGenerator(model, model.multi_config)
        result = generator.generate(
            prompt=request.prompt,
            track_configs=track_configs,
            duration=request.duration,
            temperature=request.temperature,
            use_beam_search=request.use_beam_search,
        )

        # Save mixed audio
        output_path = temp_dir / f"{task_id}_mixed.wav"
        save_audio_file(result.mixed_audio, str(output_path), result.sample_rate)

        # Save stems if requested
        stem_paths = {}
        if request.export_stems:
            for instrument, audio in result.audio_tracks.items():
                stem_path = temp_dir / f"{task_id}_stem_{instrument}.wav"
                save_audio_file(audio, str(stem_path), result.sample_rate)
                stem_paths[instrument] = str(stem_path)

        # Export MIDI if requested
        midi_path = None
        if request.export_midi:
            midi_path = temp_dir / f"{task_id}.mid"
            midi_converter.export_to_file(result.audio_tracks, str(midi_path))

        # Store result
        # In production, use database or cache
        print(f"Task {task_id} completed successfully")

    except Exception as e:
        print(f"Task {task_id} failed: {e}")
