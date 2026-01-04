"""State management for improvement sessions."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .exceptions import StateCorruptedError, StateLoadError, StateSaveError

logger = logging.getLogger(__name__)

# Required fields for state validation
_REQUIRED_STATE_FIELDS = {"goal", "repo_path", "status"}
_REQUIRED_ITERATION_FIELDS = {
    "iteration", "timestamp", "prompt", "result", "success", "validation_passed"
}
_VALID_STATUSES: set[str] = {"running", "completed", "failed", "paused", "converged"}

# Context efficiency settings
MAX_RESULT_LENGTH = 2000  # Max chars to store per iteration result
MAX_ERROR_LENGTH = 1000   # Max chars to store per error
MAX_STORED_ITERATIONS = 50  # Keep only recent iterations in state file


def truncate_text(text: str, max_length: int, suffix: str = "...[truncated]") -> str:
    """Truncate text to max_length, adding suffix if truncated.

    Args:
        text: The text to truncate.
        max_length: Maximum length including suffix.
        suffix: String to append when truncated.

    Returns:
        Original text if short enough, otherwise truncated with suffix.
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


@dataclass
class IterationRecord:
    """Record of a single improvement iteration.

    Attributes:
        iteration: The iteration number (1-indexed).
        timestamp: ISO format timestamp of when this iteration ran.
        prompt: The prompt sent to Claude.
        result: The result/output from Claude.
        success: Whether Claude call succeeded.
        validation_passed: Whether validation (lint/tests) passed.
        git_commit: Git commit hash if changes were committed.
        error: Error message if iteration failed.
        token_usage: Token usage statistics for this iteration.
    """
    iteration: int
    timestamp: str
    prompt: str
    result: str
    success: bool
    validation_passed: bool
    git_commit: str | None = None
    error: str | None = None
    token_usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, handling token_usage specially."""
        data = asdict(self)
        # token_usage is already a dict or None, no special handling needed
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IterationRecord:
        """Create an IterationRecord from a dictionary.

        Args:
            data: Dictionary with iteration data.

        Returns:
            IterationRecord instance.

        Raises:
            StateCorruptedError: If required fields are missing or have invalid types.
        """
        # Validate required fields
        missing = _REQUIRED_ITERATION_FIELDS - set(data.keys())
        if missing:
            raise StateCorruptedError(
                path="<unknown>",
                reason=f"Missing required iteration fields: {missing}"
            )

        # Validate field types
        if not isinstance(data.get("iteration"), int):
            raise StateCorruptedError(
                path="<unknown>",
                reason=f"'iteration' must be an integer, got {type(data.get('iteration'))}"
            )
        if not isinstance(data.get("success"), bool):
            raise StateCorruptedError(
                path="<unknown>",
                reason=f"'success' must be a boolean, got {type(data.get('success'))}"
            )
        if not isinstance(data.get("validation_passed"), bool):
            val_type = type(data.get("validation_passed"))
            raise StateCorruptedError(
                path="<unknown>",
                reason=f"'validation_passed' must be a boolean, got {val_type}"
            )

        # Filter to only expected fields to prevent unexpected kwargs
        expected_fields = _REQUIRED_ITERATION_FIELDS | {"git_commit", "error", "token_usage"}
        filtered_data = {k: v for k, v in data.items() if k in expected_fields}
        return cls(**filtered_data)


# Valid status values for improvement sessions
StatusType = Literal["running", "completed", "failed", "paused", "converged"]


@dataclass
class ImprovementState:
    """
    Persistent state for an improvement session.

    Tracks progress, allows resumption, and provides history for context.
    Includes cumulative token usage statistics for cost monitoring.
    """
    goal: str
    repo_path: str
    status: StatusType = "running"
    current_iteration: int = 0
    total_iterations: int = 0
    consecutive_failures: int = 0
    iterations: list[IterationRecord] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    summary: str | None = None
    # Cumulative token usage across all iterations
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this session (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def get_token_summary(self) -> dict[str, int]:
        """Get a summary of token usage for this session."""
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
            "cache_creation_tokens": self.total_cache_creation_tokens,
        }
    
    def record_iteration(
        self,
        prompt: str,
        result: str,
        success: bool,
        validation_passed: bool,
        git_commit: str | None = None,
        error: str | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        """Record the results of an iteration.

        Automatically truncates long results and errors to prevent
        unbounded state file growth and reduce context overhead.

        Args:
            prompt: The prompt sent to Claude.
            result: The result/output from Claude.
            success: Whether the Claude call succeeded.
            validation_passed: Whether validation passed.
            git_commit: Git commit hash if changes were committed.
            error: Error message if iteration failed.
            token_usage: Token usage dict with input_tokens, output_tokens, etc.
        """
        self.current_iteration += 1
        self.total_iterations += 1

        # Truncate long outputs for efficiency
        truncated_result = truncate_text(result, MAX_RESULT_LENGTH)
        truncated_error = truncate_text(error, MAX_ERROR_LENGTH) if error else None

        if len(result) > MAX_RESULT_LENGTH:
            logger.debug(f"Truncated result from {len(result)} to {MAX_RESULT_LENGTH} chars")

        # Accumulate token usage if provided
        if token_usage:
            self.total_input_tokens += token_usage.get("input_tokens", 0)
            self.total_output_tokens += token_usage.get("output_tokens", 0)
            self.total_cache_read_tokens += token_usage.get("cache_read_tokens", 0)
            self.total_cache_creation_tokens += token_usage.get("cache_creation_tokens", 0)
            logger.debug(
                f"Iteration {self.current_iteration} tokens: "
                f"{token_usage.get('input_tokens', 0)} in, "
                f"{token_usage.get('output_tokens', 0)} out. "
                f"Session total: {self.total_tokens}"
            )

        record = IterationRecord(
            iteration=self.current_iteration,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            result=truncated_result,
            success=success,
            validation_passed=validation_passed,
            git_commit=git_commit,
            error=truncated_error,
            token_usage=token_usage,
        )
        self.iterations.append(record)

        if success and validation_passed:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get context from recent iterations for Claude."""
        if not self.iterations:
            return "No previous iterations."
        
        recent = self.iterations[-n:]
        context_parts = [f"Goal: {self.goal}", "", "Recent progress:"]
        
        for record in recent:
            status = "[OK]" if record.success and record.validation_passed else "[FAIL]"
            context_parts.append(
                f"  [{status}] Iteration {record.iteration}: {record.result[:200]}..."
            )
        
        return "\n".join(context_parts)
    
    def mark_complete(self, summary: str) -> None:
        """Mark the improvement session as complete."""
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        self.summary = summary
    
    def mark_failed(self, reason: str) -> None:
        """Mark the improvement session as failed."""
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()
        self.summary = f"Failed: {reason}"
    
    def save(self, path: Path) -> None:
        """Save state to JSON file.

        Stores only the most recent iterations (up to MAX_STORED_ITERATIONS)
        to prevent unbounded state file growth while preserving total count.

        Args:
            path: Path to save the state file.

        Raises:
            StateSaveError: If the file cannot be written.
        """
        # Keep only recent iterations for storage efficiency
        stored_iterations = self.iterations[-MAX_STORED_ITERATIONS:]
        if len(self.iterations) > MAX_STORED_ITERATIONS:
            trimmed = len(self.iterations) - MAX_STORED_ITERATIONS
            logger.debug(f"Trimming {trimmed} old iterations from state file")

        data = {
            "goal": self.goal,
            "repo_path": self.repo_path,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "consecutive_failures": self.consecutive_failures,
            "iterations": [it.to_dict() for it in stored_iterations],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
            # Token usage statistics
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
        }

        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
            logger.debug(f"State saved to {path}")
        except PermissionError as e:
            raise StateSaveError(path, f"Permission denied: {e}") from e
        except OSError as e:
            raise StateSaveError(path, f"OS error: {e}") from e
    
    @classmethod
    def load(cls, path: Path) -> ImprovementState:
        """Load state from JSON file.

        Args:
            path: Path to the state file.

        Returns:
            ImprovementState instance.

        Raises:
            StateLoadError: If the file cannot be read.
            StateCorruptedError: If the file contains invalid data.
        """
        # Try to read the file
        try:
            content = path.read_text()
        except FileNotFoundError as e:
            raise StateLoadError(path, "File not found") from e
        except PermissionError as e:
            raise StateLoadError(path, f"Permission denied: {e}") from e
        except OSError as e:
            raise StateLoadError(path, f"OS error: {e}") from e

        # Try to parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise StateCorruptedError(path, f"Invalid JSON: {e}") from e

        # Validate it's a dictionary
        if not isinstance(data, dict):
            raise StateCorruptedError(path, f"Expected object, got {type(data).__name__}")

        # Validate required fields
        missing = _REQUIRED_STATE_FIELDS - set(data.keys())
        if missing:
            raise StateCorruptedError(path, f"Missing required fields: {missing}")

        # Validate status value
        status = data.get("status")
        if status not in _VALID_STATUSES:
            raise StateCorruptedError(
                path, f"Invalid status '{status}', must be one of {_VALID_STATUSES}"
            )

        # Parse iterations with validation
        iterations_data = data.pop("iterations", [])
        if not isinstance(iterations_data, list):
            iter_type = type(iterations_data).__name__
            raise StateCorruptedError(
                path, f"'iterations' must be a list, got {iter_type}"
            )

        try:
            iterations = [IterationRecord.from_dict(it) for it in iterations_data]
        except StateCorruptedError as e:
            # Re-raise with correct path
            raise StateCorruptedError(path, e.reason) from e

        # Filter to expected fields (including token usage stats)
        expected_fields = {
            "goal", "repo_path", "status", "current_iteration", "total_iterations",
            "consecutive_failures", "started_at", "completed_at", "summary",
            "total_input_tokens", "total_output_tokens",
            "total_cache_read_tokens", "total_cache_creation_tokens",
        }
        filtered_data = {k: v for k, v in data.items() if k in expected_fields}

        state = cls(**filtered_data)
        state.iterations = iterations
        return state
    
    @classmethod
    def load_or_create(cls, path: Path, goal: str, repo_path: str) -> ImprovementState:
        """Load existing state or create new one.

        Args:
            path: Path to the state file.
            goal: The improvement goal (used for new state).
            repo_path: Path to the repository (used for new state).

        Returns:
            ImprovementState instance (loaded or newly created).

        Raises:
            StateLoadError: If file exists but cannot be read.
            StateCorruptedError: If file exists but contains invalid data.
        """
        if path.exists():
            logger.info(f"Resuming from existing state: {path}")
            return cls.load(path)
        return cls(goal=goal, repo_path=repo_path)
