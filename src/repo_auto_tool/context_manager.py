"""Smart context management for efficient LLM interactions.

This module provides intelligent context summarization that:
1. Prioritizes recent and relevant iteration history
2. Extracts key patterns from failures and successes
3. Compresses verbose output while preserving essential information
4. Tracks what files were modified for continuity
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import IterationRecord

logger = logging.getLogger(__name__)


@dataclass
class ContextSummary:
    """Summarized context for providing to Claude.

    Attributes:
        goal: The improvement goal.
        iteration_count: Total number of iterations so far.
        success_count: Number of successful iterations.
        failure_count: Number of failed iterations.
        recent_successes: Brief descriptions of recent successes.
        recent_failures: Brief descriptions of recent failures with causes.
        files_modified: Set of files that have been modified.
        recurring_issues: Patterns that keep appearing in failures.
        key_learnings: Insights extracted from the iteration history.
    """

    goal: str
    iteration_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    recent_successes: list[str] = field(default_factory=list)
    recent_failures: list[str] = field(default_factory=list)
    files_modified: set[str] = field(default_factory=set)
    recurring_issues: list[str] = field(default_factory=list)
    key_learnings: list[str] = field(default_factory=list)

    def format(self, max_length: int = 2000) -> str:
        """Format the context summary for inclusion in prompts.

        Args:
            max_length: Maximum length of the formatted output.

        Returns:
            Formatted context string.
        """
        parts = [f"Goal: {self.goal}"]

        # Progress summary
        if self.iteration_count > 0:
            success_rate = (
                (self.success_count / self.iteration_count) * 100
                if self.iteration_count > 0
                else 0
            )
            parts.append(
                f"\nProgress: {self.iteration_count} iterations, "
                f"{self.success_count} successful ({success_rate:.0f}% success rate)"
            )

        # Recent successes (brief)
        if self.recent_successes:
            parts.append("\nRecent progress:")
            for success in self.recent_successes[-3:]:
                parts.append(f"  [OK] {success}")

        # Recent failures with causes (more detail for learning)
        if self.recent_failures:
            parts.append("\nRecent issues to avoid:")
            for failure in self.recent_failures[-3:]:
                parts.append(f"  [FAIL] {failure}")

        # Recurring issues (important for not repeating mistakes)
        if self.recurring_issues:
            parts.append("\nRecurring patterns to watch:")
            for issue in self.recurring_issues[:3]:
                parts.append(f"  - {issue}")

        # Key learnings
        if self.key_learnings:
            parts.append("\nKey learnings from this session:")
            for learning in self.key_learnings[:3]:
                parts.append(f"  - {learning}")

        # Files modified (for continuity)
        if self.files_modified:
            sorted_files = sorted(self.files_modified)[:10]
            parts.append(f"\nFiles modified so far: {', '.join(sorted_files)}")
            if len(self.files_modified) > 10:
                parts.append(f"  ... and {len(self.files_modified) - 10} more")

        result = "\n".join(parts)

        # Truncate if too long
        if len(result) > max_length:
            result = result[: max_length - 20] + "\n...[truncated]"

        return result


class ContextManager:
    """Manages context generation for LLM prompts.

    Provides intelligent summarization of iteration history
    to give Claude useful context without excessive token usage.
    """

    # Common error patterns to track
    ERROR_PATTERNS = {
        "import": re.compile(r"(?:ImportError|ModuleNotFoundError|No module named)", re.I),
        "syntax": re.compile(r"(?:SyntaxError|IndentationError|TabError)", re.I),
        "type": re.compile(r"(?:TypeError|AttributeError|NameError)", re.I),
        "test": re.compile(r"(?:AssertionError|test.*failed|FAILED)", re.I),
        "lint": re.compile(r"(?:ruff|pylint|flake8|E\d{3}|W\d{3}|F\d{3})", re.I),
    }

    # Patterns for extracting file paths from output
    FILE_PATTERN = re.compile(
        r"(?:^|[\s\"'])([a-zA-Z0-9_./\\-]+\.(?:py|js|ts|json|yaml|yml|md|txt|toml))",
        re.MULTILINE,
    )

    def __init__(self, max_context_tokens: int = 1500):
        """Initialize the context manager.

        Args:
            max_context_tokens: Approximate max tokens for context.
                               (Assumes ~4 chars per token on average)
        """
        self.max_context_chars = max_context_tokens * 4

    def summarize(
        self, goal: str, iterations: list[IterationRecord]
    ) -> ContextSummary:
        """Generate a smart summary of iteration history.

        Args:
            goal: The improvement goal.
            iterations: List of iteration records.

        Returns:
            ContextSummary with extracted insights.
        """
        summary = ContextSummary(goal=goal)
        summary.iteration_count = len(iterations)

        if not iterations:
            return summary

        # Categorize iterations
        successes = []
        failures = []
        error_counts: dict[str, int] = {}

        for it in iterations:
            if it.success and it.validation_passed:
                successes.append(it)
                summary.success_count += 1
            else:
                failures.append(it)
                summary.failure_count += 1

                # Track error patterns
                error_text = it.error or it.result
                for error_type, pattern in self.ERROR_PATTERNS.items():
                    if pattern.search(error_text):
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1

            # Extract files modified
            files = self._extract_files(it.result)
            summary.files_modified.update(files)

        # Extract recent successes (brief summaries)
        for it in successes[-5:]:
            brief = self._extract_brief_summary(it.result)
            summary.recent_successes.append(brief)

        # Extract recent failures with causes
        for it in failures[-5:]:
            cause = self._extract_failure_cause(it)
            summary.recent_failures.append(cause)

        # Identify recurring issues
        for error_type, count in error_counts.items():
            if count >= 2:
                summary.recurring_issues.append(
                    f"{error_type} errors occurred {count} times"
                )

        # Extract key learnings from successful recoveries
        summary.key_learnings = self._extract_learnings(iterations)

        return summary

    def get_context(
        self, goal: str, iterations: list[IterationRecord]
    ) -> str:
        """Get formatted context string for prompts.

        Args:
            goal: The improvement goal.
            iterations: List of iteration records.

        Returns:
            Formatted context string.
        """
        summary = self.summarize(goal, iterations)
        return summary.format(max_length=self.max_context_chars)

    def _extract_brief_summary(self, result: str) -> str:
        """Extract a brief summary from an iteration result.

        Args:
            result: The full result text.

        Returns:
            Brief summary (first meaningful line or truncated).
        """
        if not result:
            return "(no output)"

        # Clean up the result
        result = result.strip()

        # Try to find a summary line
        lines = result.split("\n")
        for line in lines[:5]:
            line = line.strip()
            # Skip empty lines and common prefixes
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            # Skip lines that are just status markers
            if line in ["GOAL_COMPLETE:", "BLOCKED:"]:
                continue
            # Return first meaningful line, truncated
            if len(line) > 100:
                return line[:97] + "..."
            return line

        # Fallback to truncated result
        if len(result) > 100:
            return result[:97] + "..."
        return result

    def _extract_failure_cause(self, iteration: IterationRecord) -> str:
        """Extract the cause of a failure from an iteration.

        Args:
            iteration: The failed iteration record.

        Returns:
            Brief description of what failed and why.
        """
        if iteration.error:
            # Error field has the validation failure info
            error = iteration.error
            if len(error) > 150:
                # Try to extract the key part
                lines = error.split("\n")
                for line in lines[:3]:
                    line = line.strip()
                    if line and not line.startswith("-"):
                        if len(line) > 147:
                            return line[:147] + "..."
                        return line
                return error[:147] + "..."
            return error

        # No explicit error, check result
        result = iteration.result
        for error_type, pattern in self.ERROR_PATTERNS.items():
            match = pattern.search(result)
            if match:
                # Extract context around the error
                start = max(0, match.start() - 20)
                end = min(len(result), match.end() + 80)
                context = result[start:end].replace("\n", " ")
                return f"{error_type}: {context.strip()}"

        return "Unknown failure"

    def _extract_files(self, text: str) -> set[str]:
        """Extract file paths mentioned in text.

        Args:
            text: Text that may contain file paths.

        Returns:
            Set of file paths found.
        """
        if not text:
            return set()

        files = set()
        for match in self.FILE_PATTERN.finditer(text):
            path = match.group(1)
            # Skip common false positives
            if path.startswith("http") or path.startswith("//"):
                continue
            # Skip very short paths (likely false positives)
            if len(path) < 5:
                continue
            files.add(path)

        return files

    def _extract_learnings(
        self, iterations: list[IterationRecord]
    ) -> list[str]:
        """Extract key learnings from iteration history.

        Looks for patterns like:
        - Successful recovery after failures (what worked)
        - Consistent success patterns

        Args:
            iterations: Full iteration history.

        Returns:
            List of learning insights.
        """
        learnings = []

        # Look for successful recoveries after failures
        for i in range(1, len(iterations)):
            prev = iterations[i - 1]
            curr = iterations[i]

            # Successful recovery pattern
            if (
                not prev.validation_passed
                and curr.success
                and curr.validation_passed
                and prev.error
            ):
                # What error was fixed?
                for error_type, pattern in self.ERROR_PATTERNS.items():
                    if pattern.search(prev.error):
                        # Extract what fixed it (first line of successful result)
                        fix = self._extract_brief_summary(curr.result)
                        learnings.append(
                            f"Fixed {error_type} error by: {fix[:80]}"
                        )
                        break

        # Deduplicate while preserving order
        seen = set()
        unique_learnings = []
        for learning in learnings:
            # Normalize for comparison
            normalized = learning.lower()[:50]
            if normalized not in seen:
                seen.add(normalized)
                unique_learnings.append(learning)

        return unique_learnings[:5]  # Limit to top 5
