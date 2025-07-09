class ParameterValidationError(ValueError):
    """Raised when generated parameters fail validation."""

class MissingInputError(KeyError):
    """Raised when a required memory key is absent."""

class ToolExecutionError(RuntimeError):
    """Raised when executing a tool fails."""

