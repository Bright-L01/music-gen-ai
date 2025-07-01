"""MIDI converter for audio to MIDI export."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pretty_midi
import torch

from music_gen.export.midi.quantizer import MIDIQuantizer
from music_gen.export.midi.transcriber import AudioTranscriber


@dataclass
class MIDIExportConfig:
    """Configuration for MIDI export."""

    # Transcription settings
    hop_length: int = 512
    fmin: float = 27.5  # A0
    fmax: float = 4186.0  # C8

    # Quantization settings
    quantize: bool = True
    quantize_strength: float = 0.8  # 0-1, how much to snap to grid
    min_note_duration: float = 0.0625  # 1/16 note at 120 BPM

    # MIDI settings
    tempo: float = 120.0
    time_signature: Tuple[int, int] = (4, 4)
    velocity_curve: str = "linear"  # "linear", "exponential", "logarithmic"

    # Multi-track settings
    separate_tracks: bool = True
    program_change_per_track: bool = True


class MIDIConverter:
    """Convert audio to MIDI format with multi-track support."""

    def __init__(self, config: MIDIExportConfig, sample_rate: int = 44100):
        self.config = config
        self.sample_rate = sample_rate

        # Initialize components
        self.transcriber = AudioTranscriber(
            sample_rate=sample_rate,
            hop_length=config.hop_length,
            fmin=config.fmin,
            fmax=config.fmax,
        )

        self.quantizer = MIDIQuantizer(
            tempo=config.tempo,
            time_signature=config.time_signature,
            strength=config.quantize_strength,
        )

    def convert(
        self,
        audio_tracks: Dict[str, torch.Tensor],
        instrument_configs: Optional[Dict[str, Dict]] = None,
    ) -> pretty_midi.PrettyMIDI:
        """Convert audio tracks to MIDI.

        Args:
            audio_tracks: Dictionary of instrument_name -> audio tensor
            instrument_configs: Optional instrument-specific settings

        Returns:
            PrettyMIDI object with all tracks
        """
        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=self.config.tempo)

        # Process each track
        for track_idx, (instrument_name, audio) in enumerate(audio_tracks.items()):
            # Get instrument config
            inst_config = (instrument_configs or {}).get(instrument_name, {})

            # Transcribe audio to notes
            notes = self.transcriber.transcribe(audio, instrument_type=instrument_name)

            # Quantize if enabled
            if self.config.quantize and notes:
                notes = self.quantizer.quantize_notes(notes)

            # Filter short notes
            notes = [n for n in notes if n.duration >= self.config.min_note_duration]

            # Create MIDI instrument
            program = self._get_midi_program(instrument_name, inst_config)
            instrument = pretty_midi.Instrument(
                program=program, is_drum=(instrument_name.lower() == "drums"), name=instrument_name
            )

            # Add notes to instrument
            for note_data in notes:
                note = pretty_midi.Note(
                    velocity=self._calculate_velocity(note_data.amplitude),
                    pitch=note_data.pitch,
                    start=note_data.start,
                    end=note_data.start + note_data.duration,
                )
                instrument.notes.append(note)

            # Add control changes if specified
            if "control_changes" in inst_config:
                for cc_data in inst_config["control_changes"]:
                    cc = pretty_midi.ControlChange(
                        number=cc_data["number"],
                        value=cc_data["value"],
                        time=cc_data.get("time", 0.0),
                    )
                    instrument.control_changes.append(cc)

            # Add instrument to MIDI
            midi.instruments.append(instrument)

        # Add tempo changes if specified
        if hasattr(self.config, "tempo_changes"):
            for tempo_change in self.config.tempo_changes:
                midi.tempo_changes.append(
                    pretty_midi.TempoChange(tempo_change["tempo"], tempo_change["time"])
                )

        # Add time signature
        midi.time_signature_changes.append(
            pretty_midi.TimeSignature(
                self.config.time_signature[0], self.config.time_signature[1], 0.0
            )
        )

        return midi

    def _get_midi_program(self, instrument_name: str, config: Dict) -> int:
        """Get MIDI program number for instrument."""
        # Check if program is specified in config
        if "midi_program" in config:
            return config["midi_program"]

        # Default mapping
        program_map = {
            # Piano
            "piano": 0,
            "electric_piano": 4,
            "harpsichord": 6,
            "organ": 19,
            # Chromatic Percussion
            "vibraphone": 11,
            "xylophone": 13,
            # Guitar
            "acoustic_guitar": 24,
            "electric_guitar": 26,
            "bass_guitar": 33,
            # Strings
            "violin": 40,
            "viola": 41,
            "cello": 42,
            "double_bass": 43,
            "harp": 46,
            # Brass
            "trumpet": 56,
            "trombone": 57,
            "tuba": 58,
            "french_horn": 60,
            # Reed
            "saxophone": 66,
            "oboe": 68,
            "clarinet": 71,
            "flute": 73,
            # Synth
            "synthesizer": 38,
            "synth_pad": 88,
            "synth_lead": 80,
            # Voice
            "soprano": 52,
            "alto": 53,
            "tenor": 54,
            "bass_voice": 55,
            "choir": 52,
            # Drums (special)
            "drums": 0,  # Channel 10
        }

        return program_map.get(instrument_name.lower(), 0)

    def _calculate_velocity(self, amplitude: float) -> int:
        """Convert amplitude to MIDI velocity."""
        # Normalize amplitude to 0-1 range
        amplitude = np.clip(amplitude, 0.0, 1.0)

        if self.config.velocity_curve == "exponential":
            # Exponential curve for more dynamic range
            velocity = int(127 * (amplitude**2))
        elif self.config.velocity_curve == "logarithmic":
            # Logarithmic curve for compressed dynamics
            velocity = int(127 * np.log1p(amplitude * 9) / np.log(10))
        else:  # linear
            velocity = int(127 * amplitude)

        return np.clip(velocity, 1, 127)

    def export_to_file(
        self,
        audio_tracks: Dict[str, torch.Tensor],
        output_path: str,
        instrument_configs: Optional[Dict[str, Dict]] = None,
    ):
        """Export audio tracks to MIDI file.

        Args:
            audio_tracks: Dictionary of audio tracks
            output_path: Path to save MIDI file
            instrument_configs: Optional instrument settings
        """
        midi = self.convert(audio_tracks, instrument_configs)
        midi.write(output_path)

    def convert_single_track(
        self, audio: torch.Tensor, instrument_name: str = "piano"
    ) -> pretty_midi.PrettyMIDI:
        """Convert single audio track to MIDI."""
        return self.convert(
            {instrument_name: audio},
            {instrument_name: {"midi_program": self._get_midi_program(instrument_name, {})}},
        )
