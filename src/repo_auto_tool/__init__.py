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
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    CircuitStats,
    create_claude_circuit_breaker,
    create_file_io_circuit_breaker,
    create_git_circuit_breaker,
    get_circuit_breaker,
)
from .circuit_breaker import (
    registry as circuit_registry,
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
from .goal_analyzer import (
    GoalAnalyzer,
    GoalAssessment,
    GoalComplexity,
    GoalRisk,
    ProgressEstimate,
    ProgressTracker,
    analyze_goal,
)
from .improver import RepoImprover
from .interrupt_handler import (
    InterruptAction,
    InterruptContext,
    InterruptHandler,
    InterruptResult,
    InterruptState,
    create_interrupt_handler,
)
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
from .smart_defaults import (
    DetectedTools,
    ProjectProfile,
    SmartDefaults,
    detect_smart_defaults,
    get_validation_commands,
)
from .state import ImprovementState, IterationRecord, StatusType, truncate_text
from .tui import (
    TUI,
    KeyAction,
    PanelType,
    TUIConfig,
    TUIError,
    TUILogHandler,
    TUINotAvailableError,
    TUIState,
    create_plain_text_tui,
    create_tui,
    setup_tui_logging,
)
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
from .watch_mode import (
    FileState,
    FileWatcher,
    WatchConfig,
    WatchMode,
    WatchModeAborted,
    WatchModeError,
    run_watch_mode,
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
    # Interrupt handling
    "InterruptAction",
    "InterruptContext",
    "InterruptHandler",
    "InterruptResult",
    "InterruptState",
    "create_interrupt_handler",
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
    # Smart defaults
    "SmartDefaults",
    "DetectedTools",
    "ProjectProfile",
    "detect_smart_defaults",
    "get_validation_commands",
    # Goal analysis
    "GoalAnalyzer",
    "GoalAssessment",
    "GoalComplexity",
    "GoalRisk",
    "ProgressEstimate",
    "ProgressTracker",
    "analyze_goal",
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
    # Watch mode
    "WatchMode",
    "WatchConfig",
    "WatchModeError",
    "WatchModeAborted",
    "FileWatcher",
    "FileState",
    "run_watch_mode",
    # TUI
    "TUI",
    "TUIConfig",
    "TUIState",
    "TUIError",
    "TUINotAvailableError",
    "TUILogHandler",
    "KeyAction",
    "PanelType",
    "create_tui",
    "create_plain_text_tui",
    "setup_tui_logging",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "CircuitState",
    "CircuitStats",
    "circuit_registry",
    "get_circuit_breaker",
    "create_claude_circuit_breaker",
    "create_git_circuit_breaker",
    "create_file_io_circuit_breaker",
]
