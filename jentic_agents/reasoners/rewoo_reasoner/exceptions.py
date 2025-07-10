class MissingInputError(KeyError):
    """Raised when a required memory key is absent."""

class ToolExecutionError(RuntimeError):
    """Raised when executing a tool fails."""

class ReasoningStepError(RuntimeError):
    """Raised when a reasoning-only step cannot produce a valid result."""
