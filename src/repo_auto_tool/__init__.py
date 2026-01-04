"""
repo-improver: Continuously improve a codebase toward a goal using Claude Code CLI.
"""

from .agents import (
    Agent,
    AgentMode,
    AgentResult,
    GoalDecomposerAgent,
    PreAnalysisAgent,
    ReviewerAgent,
    create_agent,
)
from .config import ImproverConfig, find_venv_path, get_venv_command
from .convergence import (
    ChangeMetrics,
    ChangeTracker,
    ConvergenceConfig,
    ConvergenceDetector,
    ConvergenceState,
)
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
from .git_helper import GitHelper, GitStatus
from .improver import RepoImprover
from .logging import JSONFormatter, setup_logging
from .safety import (
    DangerousCommand,
    DangerousCommandDetector,
    SafetyManager,
    SecretMatch,
    SecretsRedactor,
)
from .state import ImprovementState, StatusType, truncate_text

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "RepoImprover",
    "ImproverConfig",
    "ImprovementState",
    "StatusType",
    # Git utilities
    "GitHelper",
    "GitStatus",
    # Utilities
    "find_venv_path",
    "get_venv_command",
    "truncate_text",
    # Agents
    "Agent",
    "AgentMode",
    "AgentResult",
    "PreAnalysisAgent",
    "GoalDecomposerAgent",
    "ReviewerAgent",
    "create_agent",
    # Safety
    "SafetyManager",
    "SecretsRedactor",
    "DangerousCommandDetector",
    "SecretMatch",
    "DangerousCommand",
    # Convergence
    "ConvergenceDetector",
    "ConvergenceConfig",
    "ConvergenceState",
    "ChangeTracker",
    "ChangeMetrics",
    # Logging
    "setup_logging",
    "JSONFormatter",
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
