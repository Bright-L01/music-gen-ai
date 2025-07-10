"""Custom exceptions for Music Gen AI."""


class MusicGenException(Exception):
    """Base exception for Music Gen AI."""


class ModelNotFoundError(MusicGenException):
    """Raised when a model is not found."""


class ModelLoadError(MusicGenException):
    """Raised when a model fails to load."""


class ModelSaveError(MusicGenException):
    """Raised when a model fails to save."""


class DatasetError(MusicGenException):
    """Raised when there's an error with dataset operations."""


class GenerationError(MusicGenException):
    """Raised when music generation fails."""


class ConfigurationError(MusicGenException):
    """Raised when there's a configuration error."""


class AudioProcessingError(MusicGenException):
    """Raised when audio processing fails."""


class InsufficientResourcesError(MusicGenException):
    """Raised when system resources are insufficient."""


class ResourceExhaustionError(MusicGenException):
    """Raised when resources are exhausted during operation."""


class AudioNotFoundError(MusicGenException):
    """Raised when audio file is not found."""


class AudioSaveError(MusicGenException):
    """Raised when audio fails to save."""


class StreamingError(MusicGenException):
    """Raised when streaming operations fail."""


class TaskNotFoundError(MusicGenException):
    """Raised when a task is not found."""


class MetadataNotFoundError(MusicGenException):
    """Raised when metadata is not found."""


class TrainingError(MusicGenException):
    """Raised when training fails."""


class AuthenticationError(MusicGenException):
    """Raised when authentication fails."""


class AuthorizationError(MusicGenException):
    """Raised when authorization fails."""


class SessionExpiredError(AuthenticationError):
    """Raised when session has expired."""


class InvalidTokenError(AuthenticationError):
    """Raised when token is invalid."""


class RateLimitExceededError(MusicGenException):
    """Raised when rate limit is exceeded."""


class ValidationError(MusicGenException):
    """Raised when validation fails."""


class DatabaseError(MusicGenException):
    """Raised when database operations fail."""


class CacheError(MusicGenException):
    """Raised when cache operations fail."""


class WebSocketError(MusicGenException):
    """Raised when WebSocket operations fail."""


class CSRFError(MusicGenException):
    """Raised when CSRF validation fails."""
