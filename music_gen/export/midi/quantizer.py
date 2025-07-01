"""MIDI quantization for note timing correction."""

from typing import List, Tuple

import numpy as np

from music_gen.export.midi.transcriber import Note


class MIDIQuantizer:
    """Quantize MIDI notes to musical grid."""

    def __init__(
        self, tempo: float = 120.0, time_signature: Tuple[int, int] = (4, 4), strength: float = 0.8
    ):
        self.tempo = tempo
        self.time_signature = time_signature
        self.strength = np.clip(strength, 0.0, 1.0)

        # Calculate grid positions
        self.beat_duration = 60.0 / tempo
        self.measure_duration = self.beat_duration * time_signature[0]

        # Common subdivisions
        self.subdivisions = [1, 2, 3, 4, 6, 8, 12, 16]  # Whole to 16th notes

    def quantize_notes(self, notes: List[Note]) -> List[Note]:
        """Quantize note timings to grid."""
        quantized_notes = []

        for note in notes:
            # Quantize start time
            quantized_start = self._quantize_time(note.start)

            # Quantize duration
            quantized_end = self._quantize_time(note.start + note.duration)
            quantized_duration = max(quantized_end - quantized_start, self.beat_duration / 16)

            # Apply strength (blend between original and quantized)
            final_start = note.start + (quantized_start - note.start) * self.strength
            final_duration = note.duration + (quantized_duration - note.duration) * self.strength

            quantized_notes.append(
                Note(
                    pitch=note.pitch,
                    start=final_start,
                    duration=final_duration,
                    amplitude=note.amplitude,
                    confidence=note.confidence,
                )
            )

        return quantized_notes

    def _quantize_time(self, time: float) -> float:
        """Quantize time to nearest grid position."""
        # Find nearest beat
        beat_position = time / self.beat_duration

        # Try different subdivisions
        best_position = round(beat_position)
        best_distance = abs(beat_position - best_position)

        for subdivision in self.subdivisions:
            grid_position = round(beat_position * subdivision) / subdivision
            distance = abs(beat_position - grid_position)

            if distance < best_distance:
                best_distance = distance
                best_position = grid_position

        return best_position * self.beat_duration
