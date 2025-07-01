"""Automation system for mixing parameters."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class InterpolationType(Enum):
    """Interpolation types for automation curves."""

    LINEAR = "linear"
    CUBIC = "cubic"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    STEP = "step"


@dataclass
class AutomationPoint:
    """A single automation point."""

    time: float  # Time in seconds
    value: float  # Parameter value
    curve_type: InterpolationType = InterpolationType.LINEAR
    tension: float = 0.0  # For cubic interpolation


class AutomationLane:
    """Automation lane for a single parameter."""

    def __init__(
        self,
        parameter_name: str,
        default_value: float = 0.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        self.parameter_name = parameter_name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.points: List[AutomationPoint] = []

    def add_point(
        self, time: float, value: float, curve_type: InterpolationType = InterpolationType.LINEAR
    ):
        """Add an automation point."""
        # Clamp value to range
        value = np.clip(value, self.min_value, self.max_value)

        # Remove any existing point at the same time
        self.points = [p for p in self.points if abs(p.time - time) > 1e-6]

        # Add new point
        point = AutomationPoint(time, value, curve_type)
        self.points.append(point)

        # Sort points by time
        self.points.sort(key=lambda p: p.time)

    def remove_point(self, time: float, tolerance: float = 0.01):
        """Remove automation point near the given time."""
        self.points = [p for p in self.points if abs(p.time - time) > tolerance]

    def clear(self):
        """Clear all automation points."""
        self.points = []

    def get_value(self, time: float) -> float:
        """Get interpolated value at given time."""
        if not self.points:
            return self.default_value

        # Before first point
        if time <= self.points[0].time:
            return self.points[0].value

        # After last point
        if time >= self.points[-1].time:
            return self.points[-1].value

        # Find surrounding points
        for i in range(len(self.points) - 1):
            if self.points[i].time <= time <= self.points[i + 1].time:
                return self._interpolate(self.points[i], self.points[i + 1], time)

        return self.default_value

    def _interpolate(self, p1: AutomationPoint, p2: AutomationPoint, time: float) -> float:
        """Interpolate between two points."""
        # Calculate normalized position
        t = (time - p1.time) / (p2.time - p1.time)

        if p1.curve_type == InterpolationType.LINEAR:
            return p1.value + (p2.value - p1.value) * t

        elif p1.curve_type == InterpolationType.EXPONENTIAL:
            # Exponential curve (for volume fades)
            if p1.value <= 0:
                return p2.value * t
            ratio = p2.value / p1.value
            return p1.value * (ratio**t)

        elif p1.curve_type == InterpolationType.LOGARITHMIC:
            # Logarithmic curve
            return p1.value + (p2.value - p1.value) * np.log1p(t * 9) / np.log(10)

        elif p1.curve_type == InterpolationType.CUBIC:
            # Cubic bezier interpolation
            t2 = t * t
            t3 = t2 * t

            # Hermite basis functions
            h1 = 2 * t3 - 3 * t2 + 1
            h2 = -2 * t3 + 3 * t2
            h3 = t3 - 2 * t2 + t
            h4 = t3 - t2

            # Calculate tangents based on tension
            tension = p1.tension
            m1 = tension * (p2.value - p1.value)
            m2 = tension * (p2.value - p1.value)

            return h1 * p1.value + h2 * p2.value + h3 * m1 + h4 * m2

        elif p1.curve_type == InterpolationType.STEP:
            # Step (no interpolation)
            return p1.value

        return p1.value

    def get_values(self, num_samples: int, sample_rate: int, start_time: float = 0.0) -> np.ndarray:
        """Get array of values for given number of samples."""
        duration = num_samples / sample_rate
        times = np.linspace(start_time, start_time + duration, num_samples)

        # Vectorized version for efficiency
        values = np.zeros(num_samples)

        if not self.points:
            values.fill(self.default_value)
            return values

        # Process each segment
        point_times = [p.time for p in self.points]
        point_values = [p.value for p in self.points]

        # Before first point
        mask = times <= point_times[0]
        values[mask] = point_values[0]

        # Between points
        for i in range(len(self.points) - 1):
            mask = (times > point_times[i]) & (times <= point_times[i + 1])
            if mask.any():
                segment_times = times[mask]
                segment_values = np.zeros(len(segment_times))

                for j, t in enumerate(segment_times):
                    segment_values[j] = self._interpolate(self.points[i], self.points[i + 1], t)

                values[mask] = segment_values

        # After last point
        mask = times > point_times[-1]
        values[mask] = point_values[-1]

        return values

    def scale_time(self, factor: float):
        """Scale all automation times by a factor."""
        for point in self.points:
            point.time *= factor

    def shift_time(self, offset: float):
        """Shift all automation times by an offset."""
        for point in self.points:
            point.time += offset

    def scale_values(self, factor: float):
        """Scale all automation values by a factor."""
        for point in self.points:
            point.value = np.clip(point.value * factor, self.min_value, self.max_value)

    def copy(self) -> "AutomationLane":
        """Create a copy of this automation lane."""
        new_lane = AutomationLane(
            self.parameter_name, self.default_value, self.min_value, self.max_value
        )

        for point in self.points:
            new_lane.points.append(
                AutomationPoint(point.time, point.value, point.curve_type, point.tension)
            )

        return new_lane


class AutomationClip:
    """A clip containing multiple automation lanes."""

    def __init__(self, name: str = "Automation Clip"):
        self.name = name
        self.lanes: Dict[str, AutomationLane] = {}
        self.start_time: float = 0.0
        self.duration: float = 0.0

    def add_lane(self, lane: AutomationLane):
        """Add an automation lane to the clip."""
        self.lanes[lane.parameter_name] = lane
        self._update_duration()

    def remove_lane(self, parameter_name: str):
        """Remove an automation lane."""
        if parameter_name in self.lanes:
            del self.lanes[parameter_name]

    def get_lane(self, parameter_name: str) -> Optional[AutomationLane]:
        """Get automation lane by parameter name."""
        return self.lanes.get(parameter_name)

    def _update_duration(self):
        """Update clip duration based on automation points."""
        max_time = 0.0

        for lane in self.lanes.values():
            if lane.points:
                max_time = max(max_time, lane.points[-1].time)

        self.duration = max_time

    def get_values(
        self, parameter_name: str, num_samples: int, sample_rate: int, start_time: float = 0.0
    ) -> Optional[np.ndarray]:
        """Get automation values for a parameter."""
        lane = self.lanes.get(parameter_name)
        if lane:
            return lane.get_values(num_samples, sample_rate, start_time)
        return None


class AutomationPreset:
    """Preset automation patterns."""

    @staticmethod
    def fade_in(
        duration: float, curve: InterpolationType = InterpolationType.EXPONENTIAL
    ) -> AutomationLane:
        """Create a fade-in automation."""
        lane = AutomationLane("volume", 0.0, 0.0, 1.0)
        lane.add_point(0.0, 0.0, curve)
        lane.add_point(duration, 1.0, curve)
        return lane

    @staticmethod
    def fade_out(
        duration: float, curve: InterpolationType = InterpolationType.EXPONENTIAL
    ) -> AutomationLane:
        """Create a fade-out automation."""
        lane = AutomationLane("volume", 1.0, 0.0, 1.0)
        lane.add_point(0.0, 1.0, curve)
        lane.add_point(duration, 0.0, curve)
        return lane

    @staticmethod
    def tremolo(rate: float, depth: float, duration: float) -> AutomationLane:
        """Create tremolo automation."""
        lane = AutomationLane("volume", 1.0, 0.0, 1.0)

        # Create sine wave automation
        num_points = int(duration * rate * 4)  # 4 points per cycle
        for i in range(num_points + 1):
            time = i * duration / num_points
            phase = 2 * np.pi * rate * time
            value = 1.0 - depth * 0.5 * (1 - np.cos(phase))
            lane.add_point(time, value, InterpolationType.CUBIC)

        return lane

    @staticmethod
    def auto_pan(rate: float, width: float, duration: float) -> AutomationLane:
        """Create auto-pan automation."""
        lane = AutomationLane("pan", 0.0, -1.0, 1.0)

        # Create sine wave panning
        num_points = int(duration * rate * 4)
        for i in range(num_points + 1):
            time = i * duration / num_points
            phase = 2 * np.pi * rate * time
            value = width * np.sin(phase)
            lane.add_point(time, value, InterpolationType.CUBIC)

        return lane

    @staticmethod
    def filter_sweep(
        start_freq: float,
        end_freq: float,
        duration: float,
        curve: InterpolationType = InterpolationType.EXPONENTIAL,
    ) -> AutomationLane:
        """Create filter frequency sweep automation."""
        lane = AutomationLane("filter_freq", start_freq, 20.0, 20000.0)
        lane.add_point(0.0, start_freq, curve)
        lane.add_point(duration, end_freq, curve)
        return lane
