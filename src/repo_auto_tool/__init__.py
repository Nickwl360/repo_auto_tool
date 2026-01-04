"""
repo-improver: Continuously improve a codebase toward a goal using Claude Code CLI.
"""

from .config import ImproverConfig
from .exceptions import (
    ClaudeInterfaceError,
    ClaudeNotFoundError,
    ClaudeResponseError,
    ClaudeTimeoutError,
    ConfigurationError,
    GitError,
    GitNotInitializedError,
    GitOperationError,
    InvalidConfigValueError,
    InvalidPathError,
    LintFailureError,
    RepoAutoToolError,
    StateCorruptedError,
    StateError,
    StateLoadError,
    StateSaveError,
    SyntaxError_,
    TestFailureError,
    ValidationError,
)
from .improver import RepoImprover
from .state import ImprovementState

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "RepoImprover",
    "ImproverConfig",
    "ImprovementState",
    # Base exception
    "RepoAutoToolError",
    # Configuration errors
    "ConfigurationError",
    "InvalidPathError",
    "InvalidConfigValueError",
    # State errors
    "StateError",
    "StateLoadError",
    "StateSaveError",
    "StateCorruptedError",
    # Validation errors
    "ValidationError",
    "TestFailureError",
    "LintFailureError",
    "SyntaxError_",
    # Claude interface errors
    "ClaudeInterfaceError",
    "ClaudeNotFoundError",
    "ClaudeTimeoutError",
    "ClaudeResponseError",
    # Git errors
    "GitError",
    "GitNotInitializedError",
    "GitOperationError",
]
