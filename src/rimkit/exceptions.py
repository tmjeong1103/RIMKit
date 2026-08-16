"""Public exception hierarchy."""


class RIMKitError(Exception):
    """Base class for expected RIMKit failures."""


CoReError = RIMKitError


class ConfigurationError(CoReError):
    """Raised when a run or robot configuration is invalid."""


class MotionValidationError(CoReError):
    """Raised when an input motion violates the SOMA contract."""


class ModelVerificationError(CoReError):
    """Raised when a packaged robot model violates its contract."""


class ArtifactError(CoReError):
    """Raised when a review artifact cannot be created safely."""


class PipelineNotAvailableError(CoReError):
    """Raised while an algorithm stage is not yet available in the port."""
