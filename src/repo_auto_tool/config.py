"""Configuration for the repo improver."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .exceptions import InvalidConfigValueError, InvalidPathError
from .logging import VALID_LOG_LEVELS


def find_venv_path(repo_path: Path) -> Path | None:
    """Find Python virtual environment in or above the repo.

    Checks common venv locations:
    - ./venv, ./.venv, ./env (in repo)
    - Active VIRTUAL_ENV environment variable

    Args:
        repo_path: The repository path to search from.

    Returns:
        Path to the venv directory, or None if not found.
    """
    # Check if we're already in a venv
    if venv := os.environ.get("VIRTUAL_ENV"):
        return Path(venv)

    # Check common venv directory names in repo
    for name in ["venv", ".venv", "env"]:
        venv_path = repo_path / name
        if venv_path.is_dir():
            # Verify it's a valid venv by checking for scripts/bin
            if (venv_path / "Scripts").is_dir() or (venv_path / "bin").is_dir():
                return venv_path
    return None


def get_venv_command(repo_path: Path, command: str) -> str:
    """Get command with venv prefix if applicable.

    For commands like 'pytest' or 'ruff', returns the full path
    if a venv is found, otherwise returns the original command.

    Args:
        repo_path: The repository path.
        command: The command to prefix (e.g., 'pytest', 'ruff check .').

    Returns:
        The command, possibly prefixed with venv path.
    """
    venv = find_venv_path(repo_path)
    if not venv:
        return command

    # Determine scripts directory based on OS
    if sys.platform == "win32":
        scripts_dir = venv / "Scripts"
    else:
        scripts_dir = venv / "bin"

    if not scripts_dir.is_dir():
        return command

    # Extract the base command (first word)
    parts = command.split(None, 1)
    if not parts:
        return command

    base_cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    # Check if command exists in venv scripts
    for ext in ["", ".exe", ".bat", ".cmd"]:
        cmd_path = scripts_dir / f"{base_cmd}{ext}"
        if cmd_path.is_file():
            # Use relative path from repo for portability
            try:
                rel_path = cmd_path.relative_to(repo_path)
                prefix = f"./{rel_path}".replace("\\", "/")
            except ValueError:
                # Not relative, use absolute
                prefix = str(cmd_path)
            return f"{prefix} {args}".strip() if args else prefix

    return command


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
    run_tests: bool = True  # Smart detection skips if no test files found
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
    state_dir: Path | None = None  # Directory for state files

    # Safety settings
    redact_secrets: bool = True
    detect_dangerous_commands: bool = True

    # Goal type for different improvement strategies
    goal_type: Literal["open-ended", "bounded", "exploratory"] = "open-ended"

    # Convergence detection settings
    convergence_threshold: float = 0.1
    plateau_window: int = 3
    checkpoint_interval: int = 5
    early_stop_enabled: bool = True

    # Output settings
    verbose: bool = True
    log_file: Path | None = None
    log_level: str = "INFO"
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
        self._validate_log_level()

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

    def _validate_log_level(self) -> None:
        """Validate the log level setting."""
        level_upper = self.log_level.upper()
        if level_upper not in VALID_LOG_LEVELS:
            raise InvalidConfigValueError(
                "log_level",
                self.log_level,
                f"Must be one of: {', '.join(sorted(VALID_LOG_LEVELS))}"
            )
        # Normalize to uppercase
        self.log_level = level_upper
