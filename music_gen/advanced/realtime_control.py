"""
Real-Time Parameter Adjustment System

Implements real-time control and parameter adjustment for music generation,
allowing dynamic modification of generation parameters during the creation process.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.musicgen import MusicGenForConditionalGeneration
from .conditioning import AdvancedConditioningSystem
from .music_theory import MusicTheoryEngine


class ParameterType(Enum):
    """Types of adjustable parameters."""

    TEMPO = "tempo"
    KEY = "key"
    INTENSITY = "intensity"
    COMPLEXITY = "complexity"
    ARRANGEMENT_DENSITY = "arrangement_density"
    GENRE = "genre"
    MOOD = "mood"
    STYLE = "style"
    REVERB = "reverb"
    STEREO_WIDTH = "stereo_width"
    HARMONIC_TENSION = "harmonic_tension"
    RHYTHMIC_VARIATION = "rhythmic_variation"


class ControlMode(Enum):
    """Control interaction modes."""

    IMMEDIATE = "immediate"
    SMOOTH = "smooth"
    QUANTIZED = "quantized"
    GESTURE_BASED = "gesture_based"


@dataclass
class ParameterControl:
    """Configuration for a controllable parameter."""

    param_type: ParameterType
    min_value: float
    max_value: float
    default_value: float
    step_size: float = 0.01
    control_mode: ControlMode = ControlMode.SMOOTH
    smoothing_factor: float = 0.1
    quantize_steps: Optional[List[float]] = None
    description: str = ""

    def __post_init__(self):
        """Validate parameter configuration."""
        if self.min_value >= self.max_value:
            raise ValueError("min_value must be less than max_value")
        if not (self.min_value <= self.default_value <= self.max_value):
            raise ValueError("default_value must be within min/max range")


@dataclass
class ControlEvent:
    """Real-time control event."""

    parameter: ParameterType
    value: float
    timestamp: float
    event_id: Optional[str] = None
    source: str = "user"  # "user", "automation", "gesture", etc.


@dataclass
class GenerationState:
    """Current state of generation parameters."""

    parameters: Dict[ParameterType, float]
    timestamp: float
    generation_position: float  # 0.0 to 1.0
    active_section: Optional[str] = None

    def copy(self) -> "GenerationState":
        """Create a copy of the current state."""
        return GenerationState(
            parameters=self.parameters.copy(),
            timestamp=self.timestamp,
            generation_position=self.generation_position,
            active_section=self.active_section,
        )


class ParameterSmoother:
    """Smooths parameter changes to avoid artifacts."""

    def __init__(self, smoothing_factor: float = 0.1):
        self.smoothing_factor = smoothing_factor
        self.current_values: Dict[ParameterType, float] = {}
        self.target_values: Dict[ParameterType, float] = {}

    def set_target(self, parameter: ParameterType, target_value: float):
        """Set a target value for smoothing."""
        self.target_values[parameter] = target_value

        if parameter not in self.current_values:
            self.current_values[parameter] = target_value

    def update(self) -> Dict[ParameterType, float]:
        """Update current values towards targets."""
        for param, target in self.target_values.items():
            if param in self.current_values:
                current = self.current_values[param]
                # Exponential smoothing
                new_value = current + (target - current) * self.smoothing_factor
                self.current_values[param] = new_value
            else:
                self.current_values[param] = target

        return self.current_values.copy()

    def get_current_value(self, parameter: ParameterType) -> Optional[float]:
        """Get current smoothed value."""
        return self.current_values.get(parameter)


class GestureRecognizer:
    """Recognizes control gestures and maps them to parameters."""

    def __init__(self):
        self.gesture_patterns = self._load_gesture_patterns()
        self.gesture_history: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
        self.max_history_length = 100

    def _load_gesture_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined gesture patterns."""

        return {
            "intensity_sweep": {
                "pattern": "horizontal_line",
                "parameter": ParameterType.INTENSITY,
                "mapping": "linear",
                "description": "Horizontal movement controls intensity",
            },
            "tempo_control": {
                "pattern": "vertical_line",
                "parameter": ParameterType.TEMPO,
                "mapping": "exponential",
                "description": "Vertical movement controls tempo",
            },
            "complexity_spiral": {
                "pattern": "circular",
                "parameter": ParameterType.COMPLEXITY,
                "mapping": "circular_radius",
                "description": "Circular gestures control complexity",
            },
            "mood_wipe": {
                "pattern": "diagonal_swipe",
                "parameter": ParameterType.MOOD,
                "mapping": "categorical",
                "description": "Diagonal swipes change mood",
            },
        }

    def add_gesture_point(self, x: float, y: float, timestamp: float):
        """Add a point to the gesture history."""
        self.gesture_history.append((x, y, timestamp))

        # Limit history length
        if len(self.gesture_history) > self.max_history_length:
            self.gesture_history.pop(0)

    def recognize_gesture(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Recognize gesture pattern and return parameter changes."""

        if len(self.gesture_history) < 5:  # Need minimum points
            return None

        # Simple gesture recognition (placeholder)
        recent_points = self.gesture_history[-10:]

        # Calculate gesture characteristics
        x_coords = [p[0] for p in recent_points]
        y_coords = [p[1] for p in recent_points]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # Determine gesture type
        if x_range > y_range * 2:
            # Horizontal gesture
            gesture_type = "intensity_sweep"
            value = (max(x_coords) + min(x_coords)) / 2  # Average x position
        elif y_range > x_range * 2:
            # Vertical gesture
            gesture_type = "tempo_control"
            value = (max(y_coords) + min(y_coords)) / 2  # Average y position
        else:
            # Mixed gesture - check for circular pattern
            gesture_type = "complexity_spiral"
            # Calculate distance from center
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            distances = [
                ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                for x, y in zip(x_coords, y_coords)
            ]
            value = sum(distances) / len(distances)

        pattern = self.gesture_patterns.get(gesture_type)
        if pattern:
            return gesture_type, {
                "parameter": pattern["parameter"],
                "value": value,
                "mapping": pattern["mapping"],
            }

        return None

    def clear_history(self):
        """Clear gesture history."""
        self.gesture_history.clear()


class AutomationCurve:
    """Automation curve for parameter control."""

    def __init__(
        self,
        parameter: ParameterType,
        curve_points: List[Tuple[float, float]],  # (time, value) pairs
        curve_type: str = "linear",
    ):
        self.parameter = parameter
        self.curve_points = sorted(curve_points)  # Sort by time
        self.curve_type = curve_type

    def get_value_at_time(self, time: float) -> float:
        """Get interpolated value at given time."""

        if not self.curve_points:
            return 0.0

        if time <= self.curve_points[0][0]:
            return self.curve_points[0][1]

        if time >= self.curve_points[-1][0]:
            return self.curve_points[-1][1]

        # Find surrounding points
        for i in range(len(self.curve_points) - 1):
            t1, v1 = self.curve_points[i]
            t2, v2 = self.curve_points[i + 1]

            if t1 <= time <= t2:
                # Interpolate
                if self.curve_type == "linear":
                    alpha = (time - t1) / (t2 - t1)
                    return v1 + (v2 - v1) * alpha
                elif self.curve_type == "smooth":
                    alpha = (time - t1) / (t2 - t1)
                    # Smooth step interpolation
                    alpha = alpha * alpha * (3 - 2 * alpha)
                    return v1 + (v2 - v1) * alpha
                elif self.curve_type == "exponential":
                    alpha = (time - t1) / (t2 - t1)
                    alpha = alpha * alpha
                    return v1 + (v2 - v1) * alpha

        return 0.0


class RealTimeController:
    """Real-time parameter controller for music generation."""

    def __init__(
        self,
        model: MusicGenForConditionalGeneration,
        conditioning_system: AdvancedConditioningSystem,
        music_theory: MusicTheoryEngine,
        update_rate: float = 30.0,  # Updates per second
    ):
        self.model = model
        self.conditioning_system = conditioning_system
        self.music_theory = music_theory
        self.update_rate = update_rate

        self.logger = logging.getLogger(__name__)

        # Initialize parameter controls
        self.parameter_controls = self._initialize_parameter_controls()

        # State management
        self.current_state = GenerationState(
            parameters={
                param: control.default_value for param, control in self.parameter_controls.items()
            },
            timestamp=time.time(),
            generation_position=0.0,
        )

        # Control components
        self.parameter_smoother = ParameterSmoother()
        self.gesture_recognizer = GestureRecognizer()

        # Event handling
        self.control_queue = Queue()
        self.event_callbacks: Dict[ParameterType, List[Callable]] = {}

        # Automation
        self.automation_curves: Dict[ParameterType, AutomationCurve] = {}
        self.automation_active = False
        self.automation_start_time = 0.0

        # Threading
        self.control_thread = None
        self.running = False

    def _initialize_parameter_controls(self) -> Dict[ParameterType, ParameterControl]:
        """Initialize parameter control configurations."""

        controls = {
            ParameterType.TEMPO: ParameterControl(
                param_type=ParameterType.TEMPO,
                min_value=60.0,
                max_value=180.0,
                default_value=120.0,
                step_size=1.0,
                control_mode=ControlMode.SMOOTH,
                description="Musical tempo in BPM",
            ),
            ParameterType.INTENSITY: ParameterControl(
                param_type=ParameterType.INTENSITY,
                min_value=0.0,
                max_value=1.0,
                default_value=0.5,
                step_size=0.01,
                control_mode=ControlMode.SMOOTH,
                description="Musical intensity and energy",
            ),
            ParameterType.COMPLEXITY: ParameterControl(
                param_type=ParameterType.COMPLEXITY,
                min_value=0.0,
                max_value=1.0,
                default_value=0.5,
                step_size=0.01,
                control_mode=ControlMode.SMOOTH,
                description="Harmonic and rhythmic complexity",
            ),
            ParameterType.ARRANGEMENT_DENSITY: ParameterControl(
                param_type=ParameterType.ARRANGEMENT_DENSITY,
                min_value=0.0,
                max_value=1.0,
                default_value=0.5,
                step_size=0.01,
                control_mode=ControlMode.SMOOTH,
                description="Instrument arrangement density",
            ),
            ParameterType.REVERB: ParameterControl(
                param_type=ParameterType.REVERB,
                min_value=0.0,
                max_value=1.0,
                default_value=0.3,
                step_size=0.01,
                control_mode=ControlMode.SMOOTH,
                description="Reverb amount",
            ),
            ParameterType.STEREO_WIDTH: ParameterControl(
                param_type=ParameterType.STEREO_WIDTH,
                min_value=0.0,
                max_value=2.0,
                default_value=1.0,
                step_size=0.01,
                control_mode=ControlMode.SMOOTH,
                description="Stereo width enhancement",
            ),
            ParameterType.HARMONIC_TENSION: ParameterControl(
                param_type=ParameterType.HARMONIC_TENSION,
                min_value=0.0,
                max_value=1.0,
                default_value=0.3,
                step_size=0.01,
                control_mode=ControlMode.SMOOTH,
                description="Harmonic tension and dissonance",
            ),
            ParameterType.RHYTHMIC_VARIATION: ParameterControl(
                param_type=ParameterType.RHYTHMIC_VARIATION,
                min_value=0.0,
                max_value=1.0,
                default_value=0.4,
                step_size=0.01,
                control_mode=ControlMode.SMOOTH,
                description="Rhythmic variation and syncopation",
            ),
        }

        return controls

    def start_control_thread(self):
        """Start the real-time control thread."""
        if self.running:
            return

        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        self.logger.info("Real-time control thread started")

    def stop_control_thread(self):
        """Stop the real-time control thread."""
        self.running = False

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)

        self.logger.info("Real-time control thread stopped")

    def _control_loop(self):
        """Main control loop running in separate thread."""

        update_interval = 1.0 / self.update_rate

        while self.running:
            start_time = time.time()

            try:
                # Process control events
                self._process_control_events()

                # Update automation
                if self.automation_active:
                    self._update_automation()

                # Update parameter smoothing
                smoothed_params = self.parameter_smoother.update()

                # Update current state
                self.current_state.parameters.update(smoothed_params)
                self.current_state.timestamp = time.time()

                # Trigger callbacks for changed parameters
                self._trigger_parameter_callbacks()

            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")

            # Maintain update rate
            elapsed = time.time() - start_time
            sleep_time = max(0, update_interval - elapsed)
            time.sleep(sleep_time)

    def _process_control_events(self):
        """Process events from the control queue."""

        try:
            while True:
                event = self.control_queue.get_nowait()
                self._handle_control_event(event)
        except Empty:
            pass  # No more events in queue

    def _handle_control_event(self, event: ControlEvent):
        """Handle a single control event."""

        control = self.parameter_controls.get(event.parameter)
        if not control:
            self.logger.warning(f"Unknown parameter: {event.parameter}")
            return

        # Clamp value to valid range
        clamped_value = max(control.min_value, min(control.max_value, event.value))

        # Apply control mode
        if control.control_mode == ControlMode.IMMEDIATE:
            self.current_state.parameters[event.parameter] = clamped_value
        elif control.control_mode == ControlMode.SMOOTH:
            self.parameter_smoother.set_target(event.parameter, clamped_value)
        elif control.control_mode == ControlMode.QUANTIZED:
            if control.quantize_steps:
                # Find nearest quantize step
                quantized_value = min(control.quantize_steps, key=lambda x: abs(x - clamped_value))
                self.parameter_smoother.set_target(event.parameter, quantized_value)
            else:
                self.parameter_smoother.set_target(event.parameter, clamped_value)

        self.logger.debug(f"Parameter {event.parameter.value} set to {clamped_value}")

    def _update_automation(self):
        """Update parameters based on automation curves."""

        current_time = time.time() - self.automation_start_time

        for parameter, curve in self.automation_curves.items():
            automated_value = curve.get_value_at_time(current_time)
            self.parameter_smoother.set_target(parameter, automated_value)

    def _trigger_parameter_callbacks(self):
        """Trigger callbacks for parameter changes."""

        for parameter, callbacks in self.event_callbacks.items():
            if parameter in self.current_state.parameters:
                value = self.current_state.parameters[parameter]
                for callback in callbacks:
                    try:
                        callback(parameter, value)
                    except Exception as e:
                        self.logger.error(f"Error in parameter callback: {e}")

    def set_parameter(self, parameter: ParameterType, value: float, source: str = "user"):
        """Set a parameter value."""

        event = ControlEvent(parameter=parameter, value=value, timestamp=time.time(), source=source)

        self.control_queue.put(event)

    def get_parameter(self, parameter: ParameterType) -> Optional[float]:
        """Get current parameter value."""
        return self.current_state.parameters.get(parameter)

    def get_all_parameters(self) -> Dict[ParameterType, float]:
        """Get all current parameter values."""
        return self.current_state.parameters.copy()

    def add_parameter_callback(self, parameter: ParameterType, callback: Callable):
        """Add callback for parameter changes."""

        if parameter not in self.event_callbacks:
            self.event_callbacks[parameter] = []

        self.event_callbacks[parameter].append(callback)

    def remove_parameter_callback(self, parameter: ParameterType, callback: Callable):
        """Remove parameter callback."""

        if parameter in self.event_callbacks:
            try:
                self.event_callbacks[parameter].remove(callback)
            except ValueError:
                pass

    def add_gesture_point(self, x: float, y: float):
        """Add gesture control point."""

        self.gesture_recognizer.add_gesture_point(x, y, time.time())

        # Check for gesture recognition
        gesture_result = self.gesture_recognizer.recognize_gesture()
        if gesture_result:
            gesture_type, gesture_data = gesture_result

            parameter = gesture_data["parameter"]
            value = gesture_data["value"]

            # Map gesture value to parameter range
            control = self.parameter_controls.get(parameter)
            if control:
                # Normalize gesture value (assuming 0-1 range)
                mapped_value = control.min_value + value * (control.max_value - control.min_value)
                self.set_parameter(parameter, mapped_value, source="gesture")

    def set_automation_curve(self, parameter: ParameterType, curve: AutomationCurve):
        """Set automation curve for a parameter."""
        self.automation_curves[parameter] = curve

    def start_automation(self):
        """Start automation playback."""
        self.automation_active = True
        self.automation_start_time = time.time()
        self.logger.info("Automation started")

    def stop_automation(self):
        """Stop automation playback."""
        self.automation_active = False
        self.logger.info("Automation stopped")

    def clear_automation(self):
        """Clear all automation curves."""
        self.automation_curves.clear()
        self.automation_active = False
        self.logger.info("Automation cleared")

    def create_parameter_preset(self, name: str) -> Dict[str, Any]:
        """Create a parameter preset from current values."""

        preset = {
            "name": name,
            "parameters": self.current_state.parameters.copy(),
            "timestamp": time.time(),
        }

        return preset

    def load_parameter_preset(self, preset: Dict[str, Any]):
        """Load parameter values from preset."""

        if "parameters" not in preset:
            raise ValueError("Invalid preset format")

        for parameter, value in preset["parameters"].items():
            if isinstance(parameter, str):
                # Convert string to enum
                try:
                    param_enum = ParameterType(parameter)
                    self.set_parameter(param_enum, value, source="preset")
                except ValueError:
                    self.logger.warning(f"Unknown parameter in preset: {parameter}")
            else:
                self.set_parameter(parameter, value, source="preset")

        self.logger.info(f"Loaded preset: {preset.get('name', 'Unknown')}")

    def get_conditioning_from_state(self) -> Dict[str, torch.Tensor]:
        """Generate conditioning based on current parameter state."""

        params = self.current_state.parameters

        # Create musical parameters dict
        musical_params = {
            "tempo": params.get(ParameterType.TEMPO, 120.0),
            "intensity": params.get(ParameterType.INTENSITY, 0.5),
            "complexity": params.get(ParameterType.COMPLEXITY, 0.5),
            "arrangement_density": params.get(ParameterType.ARRANGEMENT_DENSITY, 0.5),
        }

        # Generate conditioning
        conditioning = self.conditioning_system.create_comprehensive_conditioning(
            text_prompt="", musical_params=musical_params  # Would be provided separately
        )

        return conditioning


class RealTimeGenerationManager:
    """Manager for real-time generation with parameter control."""

    def __init__(self, controller: RealTimeController, model: MusicGenForConditionalGeneration):
        self.controller = controller
        self.model = model

        self.logger = logging.getLogger(__name__)

        # Generation state
        self.is_generating = False
        self.generation_task = None

        # Audio streaming
        self.audio_buffer = Queue(maxsize=10)
        self.buffer_duration = 2.0  # seconds

    async def start_real_time_generation(
        self, base_prompt: str, duration: float = 60.0, chunk_size: float = 2.0
    ):
        """Start real-time generation with parameter control."""

        if self.is_generating:
            self.logger.warning("Generation already in progress")
            return

        self.is_generating = True
        self.controller.start_control_thread()

        try:
            # Generate in chunks
            num_chunks = int(duration / chunk_size)

            for chunk_idx in range(num_chunks):
                if not self.is_generating:
                    break

                self.logger.debug(f"Generating chunk {chunk_idx + 1}/{num_chunks}")

                # Update generation position
                position = chunk_idx / num_chunks
                self.controller.current_state.generation_position = position

                # Get current conditioning
                conditioning = self.controller.get_conditioning_from_state()

                # Generate audio chunk
                with torch.no_grad():
                    chunk_audio = await self._generate_chunk(base_prompt, conditioning, chunk_size)

                # Add to buffer
                try:
                    self.audio_buffer.put(chunk_audio, timeout=1.0)
                except:
                    self.logger.warning("Audio buffer full, dropping chunk")

                # Brief pause to allow parameter updates
                await asyncio.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in real-time generation: {e}")

        finally:
            self.is_generating = False
            self.controller.stop_control_thread()

    async def _generate_chunk(
        self, prompt: str, conditioning: Dict[str, torch.Tensor], duration: float
    ) -> torch.Tensor:
        """Generate a single audio chunk."""

        # Create enhanced prompt with current parameters
        params = self.controller.get_all_parameters()

        enhanced_prompt = prompt
        if params.get(ParameterType.INTENSITY, 0.5) > 0.7:
            enhanced_prompt += ", energetic, intense"
        elif params.get(ParameterType.INTENSITY, 0.5) < 0.3:
            enhanced_prompt += ", calm, gentle"

        # Generate audio (placeholder implementation)
        # In practice, this would use the actual model generation
        sample_rate = 32000
        samples = int(duration * sample_rate)

        # Create simple test audio (sine wave)
        t = torch.linspace(0, duration, samples)
        frequency = params.get(ParameterType.TEMPO, 120.0) * 2  # Simple mapping
        audio = torch.sin(2 * np.pi * frequency * t) * 0.1

        return audio.unsqueeze(0)  # Add batch dimension

    def stop_generation(self):
        """Stop real-time generation."""
        self.is_generating = False
        self.logger.info("Real-time generation stopped")

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[torch.Tensor]:
        """Get next audio chunk from buffer."""
        try:
            return self.audio_buffer.get(timeout=timeout)
        except Empty:
            return None

    def clear_audio_buffer(self):
        """Clear the audio buffer."""
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except Empty:
                break


class RealTimeControlAPI:
    """API interface for real-time control."""

    def __init__(self, controller: RealTimeController):
        self.controller = controller
        self.logger = logging.getLogger(__name__)

    def get_parameter_info(self) -> Dict[str, Any]:
        """Get information about all controllable parameters."""

        info = {}
        for param_type, control in self.controller.parameter_controls.items():
            info[param_type.value] = {
                "min_value": control.min_value,
                "max_value": control.max_value,
                "default_value": control.default_value,
                "step_size": control.step_size,
                "control_mode": control.control_mode.value,
                "description": control.description,
            }

        return info

    def set_parameter_value(self, parameter_name: str, value: float) -> bool:
        """Set parameter value via API."""

        try:
            param_type = ParameterType(parameter_name)
            self.controller.set_parameter(param_type, value, source="api")
            return True
        except ValueError:
            self.logger.error(f"Unknown parameter: {parameter_name}")
            return False

    def get_parameter_value(self, parameter_name: str) -> Optional[float]:
        """Get parameter value via API."""

        try:
            param_type = ParameterType(parameter_name)
            return self.controller.get_parameter(param_type)
        except ValueError:
            self.logger.error(f"Unknown parameter: {parameter_name}")
            return None

    def get_all_parameter_values(self) -> Dict[str, float]:
        """Get all parameter values via API."""

        values = {}
        for param_type, value in self.controller.get_all_parameters().items():
            values[param_type.value] = value

        return values

    def load_preset(self, preset_data: Dict[str, Any]) -> bool:
        """Load parameter preset via API."""

        try:
            self.controller.load_parameter_preset(preset_data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load preset: {e}")
            return False

    def create_preset(self, name: str) -> Dict[str, Any]:
        """Create preset from current parameters."""
        return self.controller.create_parameter_preset(name)

    def add_automation_point(self, parameter_name: str, time: float, value: float) -> bool:
        """Add automation point via API."""

        try:
            param_type = ParameterType(parameter_name)

            # Get or create automation curve
            if param_type not in self.controller.automation_curves:
                self.controller.automation_curves[param_type] = AutomationCurve(
                    parameter=param_type, curve_points=[]
                )

            curve = self.controller.automation_curves[param_type]
            curve.curve_points.append((time, value))
            curve.curve_points.sort()  # Keep sorted by time

            return True
        except ValueError:
            self.logger.error(f"Unknown parameter: {parameter_name}")
            return False

    def start_automation(self) -> bool:
        """Start automation playback."""
        self.controller.start_automation()
        return True

    def stop_automation(self) -> bool:
        """Stop automation playback."""
        self.controller.stop_automation()
        return True

    def clear_automation(self) -> bool:
        """Clear all automation."""
        self.controller.clear_automation()
        return True
