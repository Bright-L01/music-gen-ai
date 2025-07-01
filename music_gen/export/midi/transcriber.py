"""Audio to MIDI transcription using pitch detection."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np
import torch


@dataclass
class Note:
    """Represents a musical note."""

    pitch: int  # MIDI pitch (0-127)
    start: float  # Start time in seconds
    duration: float  # Duration in seconds
    amplitude: float  # Amplitude/velocity (0-1)
    confidence: float  # Detection confidence (0-1)


class AudioTranscriber:
    """Transcribe audio to MIDI notes."""

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 512,
        fmin: float = 27.5,
        fmax: float = 4186.0,
        threshold: float = 0.1,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold

    def transcribe(self, audio: torch.Tensor, instrument_type: Optional[str] = None) -> List[Note]:
        """Transcribe audio to notes.

        Args:
            audio: Audio tensor [channels, samples]
            instrument_type: Type of instrument for optimized detection

        Returns:
            List of detected notes
        """
        # Convert to mono if stereo
        if audio.dim() > 1 and audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        elif audio.dim() > 1:
            audio = audio.squeeze(0)

        audio_np = audio.cpu().numpy()

        # Use different methods based on instrument type
        if instrument_type and instrument_type.lower() == "drums":
            return self._transcribe_drums(audio_np)
        else:
            return self._transcribe_pitched(audio_np)

    def _transcribe_pitched(self, audio: np.ndarray) -> List[Note]:
        """Transcribe pitched instruments."""
        # Pitch detection using librosa
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            threshold=self.threshold,
        )

        # Extract notes from pitch contour
        notes = []
        time_frames = librosa.frames_to_time(
            np.arange(pitches.shape[1]), sr=self.sample_rate, hop_length=self.hop_length
        )

        current_note = None

        for t, time in enumerate(time_frames):
            # Get pitch at this frame
            index = magnitudes[:, t].argmax()
            pitch_hz = pitches[index, t]
            magnitude = magnitudes[index, t]

            if pitch_hz > 0 and magnitude > self.threshold:
                # Convert to MIDI pitch
                midi_pitch = librosa.hz_to_midi(pitch_hz)
                midi_pitch = int(np.round(midi_pitch))

                if current_note is None:
                    # Start new note
                    current_note = {
                        "pitch": midi_pitch,
                        "start": time,
                        "amplitude": magnitude,
                        "frames": 1,
                    }
                elif abs(current_note["pitch"] - midi_pitch) <= 1:
                    # Continue current note
                    current_note["amplitude"] = max(current_note["amplitude"], magnitude)
                    current_note["frames"] += 1
                else:
                    # End current note and start new one
                    duration = time - current_note["start"]
                    notes.append(
                        Note(
                            pitch=current_note["pitch"],
                            start=current_note["start"],
                            duration=duration,
                            amplitude=current_note["amplitude"],
                            confidence=0.8,
                        )
                    )

                    current_note = {
                        "pitch": midi_pitch,
                        "start": time,
                        "amplitude": magnitude,
                        "frames": 1,
                    }
            else:
                # No pitch detected
                if current_note is not None:
                    # End current note
                    duration = time - current_note["start"]
                    notes.append(
                        Note(
                            pitch=current_note["pitch"],
                            start=current_note["start"],
                            duration=duration,
                            amplitude=current_note["amplitude"],
                            confidence=0.8,
                        )
                    )
                    current_note = None

        # Add final note if exists
        if current_note is not None:
            duration = time_frames[-1] - current_note["start"]
            notes.append(
                Note(
                    pitch=current_note["pitch"],
                    start=current_note["start"],
                    duration=duration,
                    amplitude=current_note["amplitude"],
                    confidence=0.8,
                )
            )

        return notes

    def _transcribe_drums(self, audio: np.ndarray) -> List[Note]:
        """Transcribe drum hits."""
        # Onset detection for drums
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length, backtrack=True
        )

        onset_times = librosa.frames_to_time(
            onset_frames, sr=self.sample_rate, hop_length=self.hop_length
        )

        notes = []

        # Analyze each onset
        for i, onset_time in enumerate(onset_times):
            # Get audio segment around onset
            start_sample = int(onset_time * self.sample_rate)
            end_sample = min(start_sample + self.sample_rate // 10, len(audio))
            segment = audio[start_sample:end_sample]

            if len(segment) > 0:
                # Classify drum type based on spectral features
                drum_type, confidence = self._classify_drum(segment)

                # Standard GM drum mapping
                drum_map = {
                    "kick": 36,  # C1
                    "snare": 38,  # D1
                    "hihat": 42,  # F#1
                    "crash": 49,  # C#2
                    "tom": 45,  # A1
                }

                midi_pitch = drum_map.get(drum_type, 38)

                # Calculate amplitude
                amplitude = np.abs(segment).max()

                notes.append(
                    Note(
                        pitch=midi_pitch,
                        start=onset_time,
                        duration=0.1,  # Short duration for drums
                        amplitude=amplitude,
                        confidence=confidence,
                    )
                )

        return notes

    def _classify_drum(self, segment: np.ndarray) -> Tuple[str, float]:
        """Classify drum type from audio segment."""
        # Simple spectral centroid-based classification
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)[0, 0]

        # Basic classification based on frequency
        if spectral_centroid < 200:
            return "kick", 0.7
        elif spectral_centroid < 500:
            return "tom", 0.6
        elif spectral_centroid < 2000:
            return "snare", 0.7
        elif spectral_centroid < 6000:
            return "hihat", 0.8
        else:
            return "crash", 0.6
