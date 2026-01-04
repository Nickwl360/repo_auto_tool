"""Custom exceptions for repo-auto-tool.

This module defines a hierarchy of exceptions for structured error handling
throughout the application. All exceptions inherit from RepoAutoToolError.

Exception Hierarchy:
    RepoAutoToolError (base)
    +-- ConfigurationError
    |   +-- InvalidPathError
    |   +-- InvalidConfigValueError
    +-- StateError
    |   +-- StateLoadError
    |   +-- StateSaveError
    |   +-- StateCorruptedError
    +-- ValidationError
    |   +-- TestFailureError
    |   +-- LintFailureError
    |   +-- SyntaxError_
    +-- ClaudeInterfaceError
    |   +-- ClaudeNotFoundError
    |   +-- ClaudeTimeoutError
    |   +-- ClaudeResponseError
    +-- GitError
    |   +-- GitNotInitializedError
    |   +-- GitOperationError
"""

from pathlib import Path
from typing import Any


class RepoAutoToolError(Exception):
    """Base exception for all repo-auto-tool errors.

    All custom exceptions in this package inherit from this class,
    allowing for broad exception catching when needed.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# Configuration Errors

class ConfigurationError(RepoAutoToolError):
    """Base exception for configuration-related errors."""
    pass


class InvalidPathError(ConfigurationError):
    """Raised when a required path is invalid or inaccessible."""

    def __init__(self, path: Path | str, reason: str):
        self.path = Path(path) if isinstance(path, str) else path
        self.reason = reason
        super().__init__(
            f"Invalid path '{self.path}': {reason}",
            details={"path": str(self.path), "reason": reason}
        )


class InvalidConfigValueError(ConfigurationError):
    """Raised when a configuration value is invalid."""

    def __init__(self, field: str, value: Any, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid value for '{field}': {reason}",
            details={"field": field, "value": value, "reason": reason}
        )


# State Errors

class StateError(RepoAutoToolError):
    """Base exception for state management errors."""
    pass


class StateLoadError(StateError):
    """Raised when state cannot be loaded from file."""

    def __init__(self, path: Path | str, reason: str):
        self.path = Path(path) if isinstance(path, str) else path
        self.reason = reason
        super().__init__(
            f"Failed to load state from '{self.path}': {reason}",
            details={"path": str(self.path), "reason": reason}
        )


class StateSaveError(StateError):
    """Raised when state cannot be saved to file."""

    def __init__(self, path: Path | str, reason: str):
        self.path = Path(path) if isinstance(path, str) else path
        self.reason = reason
        super().__init__(
            f"Failed to save state to '{self.path}': {reason}",
            details={"path": str(self.path), "reason": reason}
        )


class StateCorruptedError(StateError):
    """Raised when state file exists but contains invalid data."""

    def __init__(self, path: Path | str, reason: str):
        self.path = Path(path) if isinstance(path, str) else path
        self.reason = reason
        super().__init__(
            f"Corrupted state file '{self.path}': {reason}",
            details={"path": str(self.path), "reason": reason}
        )


# Validation Errors

class ValidationError(RepoAutoToolError):
    """Base exception for validation-related errors."""
    pass


class TestFailureError(ValidationError):
    """Raised when tests fail during validation."""

    def __init__(self, command: str, output: str):
        self.command = command
        self.output = output
        super().__init__(
            f"Tests failed: {command}",
            details={"command": command, "output": output[:500]}
        )


class LintFailureError(ValidationError):
    """Raised when linting fails during validation."""

    def __init__(self, command: str, output: str):
        self.command = command
        self.output = output
        super().__init__(
            f"Linting failed: {command}",
            details={"command": command, "output": output[:500]}
        )


class SyntaxError_(ValidationError):
    """Raised when syntax errors are detected.

    Named with underscore to avoid shadowing builtin SyntaxError.
    """

    def __init__(self, file_path: Path | str, errors: list[str]):
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self.errors = errors
        super().__init__(
            f"Syntax errors in '{self.file_path}'",
            details={"file": str(self.file_path), "errors": errors}
        )


# Claude Interface Errors

class ClaudeInterfaceError(RepoAutoToolError):
    """Base exception for Claude Code CLI integration errors."""
    pass


class ClaudeNotFoundError(ClaudeInterfaceError):
    """Raised when Claude Code CLI is not installed or not in PATH."""

    def __init__(self):
        super().__init__(
            "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
            details={"install_command": "npm install -g @anthropic-ai/claude-code"}
        )


class ClaudeTimeoutError(ClaudeInterfaceError):
    """Raised when Claude Code CLI times out."""

    def __init__(self, timeout_seconds: int, prompt: str):
        self.timeout_seconds = timeout_seconds
        self.prompt = prompt
        super().__init__(
            f"Claude Code timed out after {timeout_seconds} seconds",
            details={"timeout_seconds": timeout_seconds, "prompt_preview": prompt[:100]}
        )


class ClaudeResponseError(ClaudeInterfaceError):
    """Raised when Claude Code returns an invalid or unparseable response."""

    def __init__(self, reason: str, raw_output: str | None = None):
        self.reason = reason
        self.raw_output = raw_output
        details: dict[str, Any] = {"reason": reason}
        if raw_output:
            details["output_preview"] = raw_output[:200]
        super().__init__(
            f"Invalid Claude response: {reason}",
            details=details
        )


# Git Errors

class GitError(RepoAutoToolError):
    """Base exception for Git-related errors."""
    pass


class GitNotInitializedError(GitError):
    """Raised when git operations are attempted on a non-git directory."""

    def __init__(self, path: Path | str):
        self.path = Path(path) if isinstance(path, str) else path
        super().__init__(
            f"'{self.path}' is not a git repository",
            details={"path": str(self.path)}
        )


class GitOperationError(GitError):
    """Raised when a git operation fails."""

    def __init__(self, operation: str, reason: str, output: str | None = None):
        self.operation = operation
        self.reason = reason
        self.output = output
        details: dict[str, Any] = {"operation": operation, "reason": reason}
        if output:
            details["output"] = output[:500]
        super().__init__(
            f"Git {operation} failed: {reason}",
            details=details
        )
