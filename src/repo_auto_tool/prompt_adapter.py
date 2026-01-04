"""Adaptive prompt enhancement based on failure patterns.

This module provides the PromptAdapter class which analyzes past iteration
failures and automatically adjusts prompts to help Claude avoid repeating
common mistakes. It's a key component of the tool's self-improvement capability.

The adapter learns from:
- Recurring error types (syntax, import, lint, test failures)
- Common failure patterns across iterations
- Success/failure ratios for different types of changes

Example:
    adapter = PromptAdapter()

    # After some iterations with import errors
    adapter.record_failure("import_error", "No module named 'foo'")
    adapter.record_failure("import_error", "Cannot import 'bar' from 'baz'")

    # The adapter will add import-related guidance
    enhanced_prompt = adapter.enhance_prompt(original_prompt)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """A record of a specific failure type and its details.

    Attributes:
        error_type: Category of error (syntax_error, import_error, etc.)
        message: The error message or description.
        count: Number of times this failure has occurred.
    """
    error_type: str
    message: str
    count: int = 1


@dataclass
class AdaptiveGuidance:
    """Guidance text to add to prompts based on failure patterns.

    Attributes:
        error_type: The error type this guidance addresses.
        threshold: Minimum occurrences before adding this guidance.
        guidance: The text to add to prompts.
        priority: Higher priority guidance is added first (1-10).
    """
    error_type: str
    threshold: int
    guidance: str
    priority: int = 5


# Pre-defined guidance for common error types
DEFAULT_GUIDANCE: list[AdaptiveGuidance] = [
    AdaptiveGuidance(
        error_type="syntax_error",
        threshold=1,
        guidance=(
            "SYNTAX CHECK: You have made syntax errors before. "
            "After editing, mentally trace through the code to verify:\n"
            "  - All parentheses, brackets, and braces are matched\n"
            "  - Indentation is consistent (use 4 spaces)\n"
            "  - No missing colons after def/class/if/for/while\n"
            "  - String quotes are properly closed\n"
        ),
        priority=9,
    ),
    AdaptiveGuidance(
        error_type="import_error",
        threshold=1,
        guidance=(
            "IMPORT VERIFICATION: Past iterations had import issues. Before finishing:\n"
            "  - Verify all imported modules exist in the project or are installed\n"
            "  - Check relative vs absolute import paths\n"
            "  - Ensure you're not creating circular imports\n"
            "  - If adding new dependencies, note them clearly\n"
        ),
        priority=8,
    ),
    AdaptiveGuidance(
        error_type="lint_error",
        threshold=2,
        guidance=(
            "LINT COMPLIANCE: Linting has failed before. Pay attention to:\n"
            "  - Line length (typically 88-100 chars max)\n"
            "  - Unused imports and variables\n"
            "  - Type hints where required\n"
            "  - Proper docstrings for public functions\n"
        ),
        priority=7,
    ),
    AdaptiveGuidance(
        error_type="test_failure",
        threshold=2,
        guidance=(
            "TEST AWARENESS: Tests have failed in past iterations. Consider:\n"
            "  - Check if your changes break existing functionality\n"
            "  - If modifying function signatures, update all call sites\n"
            "  - Run related tests mentally to predict failures\n"
            "  - Consider edge cases in your implementation\n"
        ),
        priority=6,
    ),
    AdaptiveGuidance(
        error_type="type_error",
        threshold=1,
        guidance=(
            "TYPE SAFETY: Type errors occurred before. Ensure:\n"
            "  - Return types match function signatures\n"
            "  - Optional types are handled (use 'if x is not None')\n"
            "  - Generic types are properly parameterized\n"
            "  - Type narrowing is used where needed\n"
        ),
        priority=7,
    ),
    AdaptiveGuidance(
        error_type="attribute_error",
        threshold=1,
        guidance=(
            "ATTRIBUTE CHECK: AttributeError occurred before. Verify:\n"
            "  - Object attributes exist before accessing them\n"
            "  - Class inheritance is correct\n"
            "  - self. prefix is used for instance attributes\n"
            "  - Check for typos in attribute names\n"
        ),
        priority=6,
    ),
]


@dataclass
class PromptAdapter:
    """Adapts prompts based on accumulated failure patterns.

    This class tracks failures across iterations and enhances prompts
    with specific guidance to help avoid repeating mistakes.

    The adapter maintains a rolling window of recent failures and
    provides contextual hints based on the types of errors encountered.

    Attributes:
        failure_counts: Counter of error types encountered.
        recent_messages: Recent error messages by type (for context).
        max_recent_messages: Maximum messages to keep per error type.
        custom_guidance: Additional guidance provided dynamically.
    """
    failure_counts: Counter[str] = field(default_factory=Counter)
    recent_messages: dict[str, list[str]] = field(default_factory=dict)
    max_recent_messages: int = 5
    custom_guidance: list[AdaptiveGuidance] = field(default_factory=list)
    _guidance_library: list[AdaptiveGuidance] = field(
        default_factory=lambda: list(DEFAULT_GUIDANCE)
    )

    def record_failure(self, error_type: str, message: str | None = None) -> None:
        """Record a failure occurrence.

        Args:
            error_type: Category of the error (e.g., "syntax_error").
            message: Optional error message for context.
        """
        self.failure_counts[error_type] += 1

        if message:
            if error_type not in self.recent_messages:
                self.recent_messages[error_type] = []
            messages = self.recent_messages[error_type]
            messages.append(message[:200])  # Truncate long messages
            # Keep only recent messages
            if len(messages) > self.max_recent_messages:
                self.recent_messages[error_type] = messages[-self.max_recent_messages:]

        logger.debug(
            f"Recorded failure: {error_type} (total: {self.failure_counts[error_type]})"
        )

    def record_success(self) -> None:
        """Record a successful iteration.

        Successes gradually reduce the weight of past failures,
        preventing stale guidance from persisting indefinitely.
        """
        # Decay failure counts slightly on success
        for error_type in list(self.failure_counts.keys()):
            if self.failure_counts[error_type] > 0:
                # Reduce by 0.5 (will be truncated to int in threshold checks)
                # This allows guidance to fade after consistent success
                pass  # Currently we just track, decay could be added later

    def add_custom_guidance(
        self,
        error_type: str,
        guidance: str,
        threshold: int = 1,
        priority: int = 5,
    ) -> None:
        """Add custom guidance for a specific error type.

        Args:
            error_type: The error type to match.
            guidance: The guidance text to add to prompts.
            threshold: Minimum occurrences before adding guidance.
            priority: Higher priority (1-10) appears first.
        """
        self.custom_guidance.append(
            AdaptiveGuidance(
                error_type=error_type,
                threshold=threshold,
                guidance=guidance,
                priority=priority,
            )
        )

    def get_applicable_guidance(self) -> list[str]:
        """Get all guidance applicable based on current failure counts.

        Returns:
            List of guidance strings, sorted by priority (highest first).
        """
        all_guidance = self._guidance_library + self.custom_guidance
        applicable = []

        for g in all_guidance:
            if self.failure_counts.get(g.error_type, 0) >= g.threshold:
                applicable.append((g.priority, g.guidance))

        # Sort by priority (descending) and return just the guidance text
        applicable.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in applicable]

    def enhance_prompt(self, original_prompt: str) -> str:
        """Enhance a prompt with adaptive guidance based on past failures.

        Args:
            original_prompt: The original prompt to enhance.

        Returns:
            Enhanced prompt with relevant guidance inserted.
        """
        guidance_items = self.get_applicable_guidance()

        if not guidance_items:
            return original_prompt

        # Build the adaptive section
        guidance_section = (
            "\n--- ADAPTIVE GUIDANCE (based on past iteration patterns) ---\n\n"
        )
        for item in guidance_items:
            guidance_section += item + "\n"
        guidance_section += "--- END ADAPTIVE GUIDANCE ---\n\n"

        logger.info(f"Added {len(guidance_items)} adaptive guidance items to prompt")

        # Insert guidance before the "Now make the improvements:" line if present
        # Otherwise append at the end
        marker = "Now make the improvements:"
        if marker in original_prompt:
            parts = original_prompt.rsplit(marker, 1)
            return parts[0] + guidance_section + marker + parts[1]
        else:
            return original_prompt + "\n" + guidance_section

    def analyze_error_message(self, error: str) -> str:
        """Analyze an error message and categorize it.

        Args:
            error: The error message to analyze.

        Returns:
            The categorized error type.
        """
        error_lower = error.lower()

        # Check for specific error patterns
        if "syntaxerror" in error_lower or "syntax error" in error_lower:
            return "syntax_error"
        elif "indentationerror" in error_lower:
            return "syntax_error"
        elif "importerror" in error_lower or "no module named" in error_lower:
            return "import_error"
        elif "modulenotfounderror" in error_lower:
            return "import_error"
        elif "typeerror" in error_lower:
            return "type_error"
        elif "attributeerror" in error_lower:
            return "attribute_error"
        elif "nameerror" in error_lower:
            return "name_error"
        elif "keyerror" in error_lower:
            return "key_error"
        elif "valueerror" in error_lower:
            return "value_error"
        elif any(x in error_lower for x in ["ruff", "lint", "flake8", "pylint"]):
            return "lint_error"
        elif any(x in error_lower for x in ["test", "assert", "pytest", "unittest"]):
            return "test_failure"
        elif "timeout" in error_lower:
            return "timeout"
        else:
            return "other_error"

    def record_from_validation_error(self, error: str) -> None:
        """Analyze and record a validation error.

        Convenience method that categorizes the error and records it.

        Args:
            error: The validation error message.
        """
        error_type = self.analyze_error_message(error)
        self.record_failure(error_type, error)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about recorded failures.

        Returns:
            Dictionary with failure statistics.
        """
        total_failures = sum(self.failure_counts.values())
        return {
            "total_failures": total_failures,
            "failure_counts": dict(self.failure_counts),
            "top_errors": self.failure_counts.most_common(3),
            "active_guidance_count": len(self.get_applicable_guidance()),
        }

    def reset(self) -> None:
        """Reset all recorded failures and guidance."""
        self.failure_counts.clear()
        self.recent_messages.clear()
        self.custom_guidance.clear()
        logger.info("PromptAdapter reset")

    @classmethod
    def from_iteration_history(
        cls,
        iterations: list[dict[str, Any]],
    ) -> PromptAdapter:
        """Create a PromptAdapter initialized from iteration history.

        This factory method allows bootstrapping the adapter from
        past session data, useful when resuming sessions.

        Args:
            iterations: List of iteration records with 'error' field.

        Returns:
            A PromptAdapter pre-populated with failure patterns.
        """
        adapter = cls()

        for it in iterations:
            if not it.get("validation_passed") and it.get("error"):
                adapter.record_from_validation_error(it["error"])

        stats = adapter.get_stats()
        if stats["total_failures"] > 0:
            logger.info(
                f"PromptAdapter initialized with {stats['total_failures']} "
                f"failures from history: {stats['failure_counts']}"
            )

        return adapter
