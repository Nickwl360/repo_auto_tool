"""Configuration for the repo improver."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .exceptions import InvalidConfigValueError, InvalidPathError


@dataclass
class ImproverConfig:
    """Configuration for a repo improvement session."""
    
    # Target repository
    repo_path: Path
    
    # The goal to achieve (natural language description)
    goal: str
    
    # Improvement loop settings
    max_iterations: int = 20
    max_consecutive_failures: int = 3
    
    # Claude Code settings
    allowed_tools: list[str] = field(default_factory=lambda: [
        "Bash(*)", "Read(*)", "Edit(*)", "Write(*)", "Glob(*)", "Grep(*)"
    ])
    model: str | None = None  # Use default if None
    
    # Validation settings
    run_tests: bool = True
    test_command: str = "pytest"
    run_linter: bool = True
    lint_command: str = "ruff check ."
    custom_validators: list[str] = field(default_factory=list)  # Shell commands
    
    # Git safety
    use_git: bool = True
    commit_each_iteration: bool = True
    branch_name: str = "repo-improver/auto"
    
    # State persistence
    state_file: Path | None = None  # Auto-set if None
    
    # Output settings
    verbose: bool = True
    log_file: Path | None = None
    output_format: Literal["text", "json"] = "text"
    
    def __post_init__(self) -> None:
        """Initialize and validate configuration.

        Raises:
            InvalidPathError: If repo_path is invalid.
            InvalidConfigValueError: If any configuration value is invalid.
        """
        self.repo_path = Path(self.repo_path).resolve()
        if self.state_file is None:
            self.state_file = self.repo_path / ".repo-improver-state.json"
        if self.log_file is None:
            self.log_file = self.repo_path / ".repo-improver.log"

        self._validate()

    def _validate(self) -> None:
        """Validate all configuration values.

        Raises:
            InvalidPathError: If repo_path is invalid.
            InvalidConfigValueError: If any configuration value is invalid.
        """
        self._validate_repo_path()
        self._validate_iteration_limits()
        self._validate_goal()

    def _validate_repo_path(self) -> None:
        """Validate that repo_path exists and is a directory."""
        if not self.repo_path.exists():
            raise InvalidPathError(self.repo_path, "Path does not exist")
        if not self.repo_path.is_dir():
            raise InvalidPathError(self.repo_path, "Path is not a directory")

    def _validate_iteration_limits(self) -> None:
        """Validate iteration limit configuration values."""
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise InvalidConfigValueError(
                "max_iterations",
                self.max_iterations,
                "Must be a positive integer"
            )
        if not isinstance(self.max_consecutive_failures, int) or self.max_consecutive_failures < 1:
            raise InvalidConfigValueError(
                "max_consecutive_failures",
                self.max_consecutive_failures,
                "Must be a positive integer"
            )
        if self.max_consecutive_failures > self.max_iterations:
            raise InvalidConfigValueError(
                "max_consecutive_failures",
                self.max_consecutive_failures,
                f"Cannot exceed max_iterations ({self.max_iterations})"
            )

    def _validate_goal(self) -> None:
        """Validate the improvement goal."""
        if not self.goal or not self.goal.strip():
            raise InvalidConfigValueError(
                "goal",
                self.goal,
                "Goal cannot be empty"
            )
