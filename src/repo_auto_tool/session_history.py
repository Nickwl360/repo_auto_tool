"""Cross-session learning and history persistence.

This module provides persistent storage of learnings across multiple sessions,
allowing the tool to improve over time by remembering:
1. What patterns typically cause failures
2. What fixes worked for specific error types
3. Repository-specific insights (e.g., "this repo requires specific lint config")

The history is stored in a JSON file (.repo-improver-history.json) and loaded
at the start of each session to bootstrap the PromptAdapter with prior knowledge.

Example:
    # At end of session, save learnings
    history = SessionHistory.load(repo_path)
    history.record_session(state)
    history.save()

    # At start of new session, load learnings
    history = SessionHistory.load(repo_path)
    adapter = history.get_prompt_adapter()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# History file name
HISTORY_FILE = ".repo-improver-history.json"

# Limits to prevent unbounded growth
MAX_STORED_PATTERNS = 50
MAX_SESSIONS_TRACKED = 20
MAX_FIXES_PER_ERROR = 10


@dataclass
class ErrorPattern:
    """A pattern of errors encountered across sessions.

    Attributes:
        error_type: Category of the error (syntax_error, import_error, etc.)
        occurrence_count: Total times this error has occurred.
        last_seen: ISO timestamp of when this error was last seen.
        example_messages: Sample error messages for context.
        successful_fixes: What has worked to fix this error type.
    """
    error_type: str
    occurrence_count: int = 0
    last_seen: str = ""
    example_messages: list[str] = field(default_factory=list)
    successful_fixes: list[str] = field(default_factory=list)

    def record_occurrence(self, message: str | None = None) -> None:
        """Record an occurrence of this error pattern.

        Args:
            message: Optional error message for context.
        """
        self.occurrence_count += 1
        self.last_seen = datetime.now().isoformat()

        if message and len(message) > 10:
            # Keep only unique, non-duplicate messages
            truncated = message[:200]
            if truncated not in self.example_messages:
                self.example_messages.append(truncated)
                # Keep only recent examples
                if len(self.example_messages) > 5:
                    self.example_messages = self.example_messages[-5:]

    def record_fix(self, fix_description: str) -> None:
        """Record a successful fix for this error type.

        Args:
            fix_description: Brief description of what fixed the error.
        """
        if not fix_description or len(fix_description) < 5:
            return

        truncated = fix_description[:150]
        if truncated not in self.successful_fixes:
            self.successful_fixes.append(truncated)
            # Keep only recent successful fixes
            if len(self.successful_fixes) > MAX_FIXES_PER_ERROR:
                self.successful_fixes = self.successful_fixes[-MAX_FIXES_PER_ERROR:]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "occurrence_count": self.occurrence_count,
            "last_seen": self.last_seen,
            "example_messages": self.example_messages,
            "successful_fixes": self.successful_fixes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorPattern:
        """Create from dictionary."""
        return cls(
            error_type=data.get("error_type", "unknown"),
            occurrence_count=data.get("occurrence_count", 0),
            last_seen=data.get("last_seen", ""),
            example_messages=data.get("example_messages", []),
            successful_fixes=data.get("successful_fixes", []),
        )


@dataclass
class SessionSummary:
    """Summary of a completed session for history tracking.

    Attributes:
        session_id: Unique identifier (usually started_at timestamp).
        goal: The improvement goal.
        started_at: ISO timestamp of session start.
        completed_at: ISO timestamp of session end.
        status: Final status (completed, failed, paused, etc.)
        total_iterations: Number of iterations run.
        success_count: Number of successful iterations.
        failure_count: Number of failed iterations.
        error_types_encountered: List of error types seen.
        key_learnings: Important insights from the session.
    """
    session_id: str
    goal: str
    started_at: str
    completed_at: str
    status: str
    total_iterations: int = 0
    success_count: int = 0
    failure_count: int = 0
    error_types_encountered: list[str] = field(default_factory=list)
    key_learnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "goal": self.goal,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "total_iterations": self.total_iterations,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "error_types_encountered": self.error_types_encountered,
            "key_learnings": self.key_learnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSummary:
        """Create from dictionary."""
        return cls(
            session_id=data.get("session_id", ""),
            goal=data.get("goal", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            status=data.get("status", "unknown"),
            total_iterations=data.get("total_iterations", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            error_types_encountered=data.get("error_types_encountered", []),
            key_learnings=data.get("key_learnings", []),
        )


@dataclass
class SessionHistory:
    """Persistent cross-session learning history.

    Tracks error patterns, successful fixes, and session outcomes
    to help the tool improve over time.

    Attributes:
        repo_path: Path to the repository this history belongs to.
        error_patterns: Map of error type to pattern data.
        past_sessions: Summaries of past sessions.
        global_insights: Repo-specific insights learned over time.
        created_at: When this history was first created.
        updated_at: When this history was last updated.
    """
    repo_path: str
    error_patterns: dict[str, ErrorPattern] = field(default_factory=dict)
    past_sessions: list[SessionSummary] = field(default_factory=list)
    global_insights: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""

    def _history_file_path(self) -> Path:
        """Get the path to the history file."""
        return Path(self.repo_path) / HISTORY_FILE

    def record_error(
        self,
        error_type: str,
        message: str | None = None,
    ) -> None:
        """Record an error occurrence.

        Args:
            error_type: Category of the error.
            message: Optional error message.
        """
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = ErrorPattern(error_type=error_type)

        self.error_patterns[error_type].record_occurrence(message)

    def record_fix(self, error_type: str, fix_description: str) -> None:
        """Record a successful fix for an error type.

        Args:
            error_type: The error type that was fixed.
            fix_description: Brief description of the fix.
        """
        if error_type not in self.error_patterns:
            self.error_patterns[error_type] = ErrorPattern(error_type=error_type)

        self.error_patterns[error_type].record_fix(fix_description)
        logger.debug(f"Recorded fix for {error_type}: {fix_description[:50]}...")

    def add_insight(self, insight: str) -> None:
        """Add a global insight about the repository.

        Args:
            insight: The insight to add.
        """
        if insight and insight not in self.global_insights:
            self.global_insights.append(insight)
            # Keep only recent insights
            if len(self.global_insights) > 20:
                self.global_insights = self.global_insights[-20:]

    def record_session(
        self,
        state: Any,  # ImprovementState, using Any to avoid circular import
    ) -> None:
        """Record a completed session to history.

        Extracts learnings from the session state and adds them to
        the persistent history.

        Args:
            state: The ImprovementState from the completed session.
        """
        # Collect error types and analyze patterns
        error_types_seen: set[str] = set()
        key_learnings: list[str] = []

        from .prompt_adapter import PromptAdapter
        adapter = PromptAdapter()

        # Process iterations to extract patterns
        for i, iteration in enumerate(state.iterations):
            if not iteration.validation_passed and iteration.error:
                # Record the error
                error_type = adapter.analyze_error_message(iteration.error)
                error_types_seen.add(error_type)
                self.record_error(error_type, iteration.error)

                # Check if next iteration fixed it
                if i + 1 < len(state.iterations):
                    next_it = state.iterations[i + 1]
                    if next_it.validation_passed and next_it.success:
                        # Extract what fixed it
                        fix_summary = self._extract_fix_summary(next_it.result)
                        if fix_summary:
                            self.record_fix(error_type, fix_summary)
                            key_learnings.append(
                                f"Fixed {error_type} by: {fix_summary[:80]}"
                            )

        # Create session summary
        summary = SessionSummary(
            session_id=state.started_at,
            goal=state.goal,
            started_at=state.started_at,
            completed_at=state.completed_at or datetime.now().isoformat(),
            status=state.status,
            total_iterations=state.total_iterations,
            success_count=sum(
                1 for it in state.iterations
                if it.success and it.validation_passed
            ),
            failure_count=sum(
                1 for it in state.iterations
                if not it.validation_passed
            ),
            error_types_encountered=list(error_types_seen),
            key_learnings=key_learnings[:5],  # Keep top 5
        )

        # Add to history (keep only recent sessions)
        self.past_sessions.append(summary)
        if len(self.past_sessions) > MAX_SESSIONS_TRACKED:
            self.past_sessions = self.past_sessions[-MAX_SESSIONS_TRACKED:]

        self.updated_at = datetime.now().isoformat()
        logger.info(
            f"Recorded session to history: {summary.total_iterations} iterations, "
            f"{len(error_types_seen)} error types, {len(key_learnings)} learnings"
        )

    def _extract_fix_summary(self, result: str) -> str:
        """Extract a brief summary of what fixed an error.

        Args:
            result: The result text from a successful iteration.

        Returns:
            Brief summary of the fix, or empty string if none found.
        """
        if not result:
            return ""

        # Try to find a meaningful first line
        lines = result.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            # Skip empty lines and common prefixes
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            if line in ["GOAL_COMPLETE:", "BLOCKED:"]:
                continue
            # Return first meaningful line
            if len(line) > 10:
                return line[:150]

        return result[:100] if len(result) > 10 else ""

    def get_guidance_for_prompt_adapter(self) -> list[tuple[str, str, int]]:
        """Get guidance tuples for enhancing prompts.

        Returns:
            List of (error_type, guidance_text, threshold) tuples
            based on historical patterns and successful fixes.
        """
        guidance: list[tuple[str, str, int]] = []

        for pattern in self.error_patterns.values():
            if pattern.occurrence_count < 2:
                continue  # Not enough data

            # Build guidance from successful fixes
            if pattern.successful_fixes:
                fixes_text = "\n".join(
                    f"  - {fix}" for fix in pattern.successful_fixes[-3:]
                )
                guidance_text = (
                    f"HISTORICAL LEARNING ({pattern.error_type}): "
                    f"This error has occurred {pattern.occurrence_count} times. "
                    f"What has worked before:\n{fixes_text}"
                )
                guidance.append((pattern.error_type, guidance_text, 1))

        return guidance

    def get_context_for_prompt(self, max_length: int = 500) -> str:
        """Get historical context for inclusion in prompts.

        Returns a brief summary of relevant historical learnings
        that can be included in prompts.

        Args:
            max_length: Maximum length of the context string.

        Returns:
            Context string with historical insights.
        """
        parts: list[str] = []

        # Add top error patterns with fixes
        sorted_patterns = sorted(
            self.error_patterns.values(),
            key=lambda p: p.occurrence_count,
            reverse=True,
        )

        patterns_with_fixes = [
            p for p in sorted_patterns if p.successful_fixes
        ][:3]

        if patterns_with_fixes:
            parts.append("Historical learnings from past sessions:")
            for pattern in patterns_with_fixes:
                if pattern.successful_fixes:
                    parts.append(
                        f"  - {pattern.error_type} (seen {pattern.occurrence_count}x): "
                        f"fix by {pattern.successful_fixes[-1][:60]}"
                    )

        # Add global insights
        if self.global_insights:
            parts.append("\nRepository insights:")
            for insight in self.global_insights[-3:]:
                parts.append(f"  - {insight}")

        result = "\n".join(parts)
        if len(result) > max_length:
            result = result[:max_length - 15] + "\n...[truncated]"

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the history.

        Returns:
            Dictionary with history statistics.
        """
        total_occurrences = sum(
            p.occurrence_count for p in self.error_patterns.values()
        )
        total_fixes = sum(
            len(p.successful_fixes) for p in self.error_patterns.values()
        )

        return {
            "sessions_tracked": len(self.past_sessions),
            "error_patterns": len(self.error_patterns),
            "total_error_occurrences": total_occurrences,
            "total_fixes_recorded": total_fixes,
            "global_insights": len(self.global_insights),
            "top_errors": sorted(
                [(p.error_type, p.occurrence_count)
                 for p in self.error_patterns.values()],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }

    def save(self) -> None:
        """Save history to JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        self.updated_at = datetime.now().isoformat()

        # Trim error patterns if too many
        if len(self.error_patterns) > MAX_STORED_PATTERNS:
            # Keep patterns with most occurrences
            sorted_patterns = sorted(
                self.error_patterns.items(),
                key=lambda x: x[1].occurrence_count,
                reverse=True,
            )[:MAX_STORED_PATTERNS]
            self.error_patterns = dict(sorted_patterns)

        data = {
            "repo_path": self.repo_path,
            "error_patterns": {
                k: v.to_dict() for k, v in self.error_patterns.items()
            },
            "past_sessions": [s.to_dict() for s in self.past_sessions],
            "global_insights": self.global_insights,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        path = self._history_file_path()
        try:
            path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Session history saved to {path}")
        except OSError as e:
            logger.error(f"Failed to save session history: {e}")
            raise

    @classmethod
    def load(cls, repo_path: str | Path) -> SessionHistory:
        """Load history from JSON file, or create new if doesn't exist.

        Args:
            repo_path: Path to the repository.

        Returns:
            SessionHistory instance (loaded or newly created).
        """
        repo_path = str(Path(repo_path).resolve())
        history_path = Path(repo_path) / HISTORY_FILE

        if not history_path.exists():
            logger.debug(f"No history file found, creating new history for {repo_path}")
            return cls(repo_path=repo_path)

        try:
            content = history_path.read_text()
            data = json.loads(content)

            history = cls(
                repo_path=repo_path,
                global_insights=data.get("global_insights", []),
                created_at=data.get("created_at", datetime.now().isoformat()),
                updated_at=data.get("updated_at", ""),
            )

            # Load error patterns
            for key, pattern_data in data.get("error_patterns", {}).items():
                history.error_patterns[key] = ErrorPattern.from_dict(pattern_data)

            # Load past sessions
            for session_data in data.get("past_sessions", []):
                history.past_sessions.append(SessionSummary.from_dict(session_data))

            stats = history.get_stats()
            logger.info(
                f"Loaded session history: {stats['sessions_tracked']} sessions, "
                f"{stats['error_patterns']} patterns, "
                f"{stats['total_fixes_recorded']} fixes"
            )

            return history

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load history file (corrupted?): {e}")
            # Create new history if file is corrupted
            return cls(repo_path=repo_path)
        except OSError as e:
            logger.warning(f"Failed to read history file: {e}")
            return cls(repo_path=repo_path)
