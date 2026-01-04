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
from .claude_interface import ClaudeCodeInterface, ClaudeResponse, TokenUsage
from .config import ImproverConfig, find_venv_path, get_venv_command
from .context_manager import ContextManager, ContextSummary
from .convergence import (
    ChangeMetrics,
    ChangeTracker,
    ConvergenceAction,
    ConvergenceConfig,
    ConvergenceDetector,
    ConvergenceState,
)
from .exceptions import (
    BudgetError,
    ClaudeInterfaceError,
    ClaudeNotFoundError,
    ClaudeResponseError,
    ClaudeTimeoutError,
    ConfigurationError,
    CostLimitExceededError,
    GitError,
    GitNotInitializedError,
    GitOperationError,
    InvalidConfigValueError,
    InvalidPathError,
    LintFailureError,
    PromptParseError,
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
from .logging import (
    VALID_LOG_LEVELS,
    ConsoleFormatter,
    JSONFormatter,
    LogLevel,
    get_logger,
    setup_logging,
)
from .model_selector import (
    MODEL_HAIKU,
    MODEL_OPUS,
    MODEL_SONNET,
    ModelChoice,
    ModelSelector,
    TaskComplexity,
)
from .prompt_adapter import (
    AdaptiveGuidance,
    FailureRecord,
    PromptAdapter,
)
from .prompt_parser import (
    ParsedPrompt,
    PromptParser,
    parse_prompt_file,
    parse_prompt_string,
)
from .safety import (
    DangerousCommand,
    DangerousCommandDetector,
    SafetyManager,
    SecretMatch,
    SecretsRedactor,
)
from .session_history import (
    ErrorPattern,
    SessionHistory,
    SessionSummary,
)
from .session_metrics import (
    EfficiencyMetrics,
    FailureAnalysis,
    SessionMetrics,
    SuccessMetrics,
    TimeMetrics,
)
from .state import ImprovementState, IterationRecord, StatusType, truncate_text
from .validators import (
    CommandValidator,
    LintValidator,
    SyntaxValidator,
    TestValidator,
    TypeCheckValidator,
    ValidationPipeline,
    ValidationResult,
    Validator,
)

__version__ = "0.1.0"
__all__ = [
    # Core classes
    "RepoImprover",
    "ImproverConfig",
    "ImprovementState",
    "IterationRecord",
    "StatusType",
    # Context management
    "ContextManager",
    "ContextSummary",
    # Claude interface
    "ClaudeCodeInterface",
    "ClaudeResponse",
    "TokenUsage",
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
    "ConvergenceAction",
    "ChangeTracker",
    "ChangeMetrics",
    # Model selection
    "ModelSelector",
    "ModelChoice",
    "TaskComplexity",
    "MODEL_HAIKU",
    "MODEL_SONNET",
    "MODEL_OPUS",
    # Prompt adaptation
    "PromptAdapter",
    "AdaptiveGuidance",
    "FailureRecord",
    # Session history
    "SessionHistory",
    "SessionSummary",
    "ErrorPattern",
    # Session metrics
    "SessionMetrics",
    "SuccessMetrics",
    "EfficiencyMetrics",
    "TimeMetrics",
    "FailureAnalysis",
    # Logging
    "setup_logging",
    "get_logger",
    "JSONFormatter",
    "ConsoleFormatter",
    "LogLevel",
    "VALID_LOG_LEVELS",
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
    # Budget errors
    "BudgetError",
    "CostLimitExceededError",
    # Prompt parsing errors
    "PromptParseError",
    # Prompt parsing
    "ParsedPrompt",
    "PromptParser",
    "parse_prompt_file",
    "parse_prompt_string",
    # Validators
    "Validator",
    "CommandValidator",
    "TestValidator",
    "LintValidator",
    "TypeCheckValidator",
    "SyntaxValidator",
    "ValidationPipeline",
    "ValidationResult",
]
