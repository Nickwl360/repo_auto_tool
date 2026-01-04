"""Smart model selection for cost-efficient Claude API usage.

This module provides automatic model selection based on task complexity,
using cheaper models (Haiku) for simple tasks and more capable models
(Sonnet/Opus) for complex coding tasks.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Complexity level of a task."""

    SIMPLE = "simple"  # Analysis, small changes, questions
    MODERATE = "moderate"  # Standard improvements, single-file edits
    COMPLEX = "complex"  # Architecture changes, multi-file refactors


# Model constants
MODEL_HAIKU = "claude-haiku-3-5-20241022"
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_OPUS = "claude-opus-4-20250514"

# Pricing per million tokens (approximate as of late 2024)
MODEL_PRICING: dict[str, dict[str, float]] = {
    MODEL_HAIKU: {"input": 0.25, "output": 1.25},
    MODEL_SONNET: {"input": 3.0, "output": 15.0},
    MODEL_OPUS: {"input": 15.0, "output": 75.0},
}


@dataclass
class ModelChoice:
    """Result of model selection with reasoning."""

    model: str
    complexity: TaskComplexity
    reason: str


class ModelSelector:
    """Automatically selects the appropriate Claude model based on task complexity.

    This helps optimize costs by using cheaper models for simple tasks
    and reserving expensive models for complex operations.

    Usage:
        selector = ModelSelector()
        choice = selector.select_model(prompt, context)
        # Use choice.model for the API call
    """

    # Keywords indicating simple analysis tasks
    SIMPLE_KEYWORDS = frozenset([
        "analyze",
        "analysis",
        "report",
        "explain",
        "describe",
        "list",
        "check",
        "verify",
        "count",
        "find",
        "search",
        "status",
        "summary",
        "overview",
        "what is",
        "how many",
    ])

    # Keywords indicating complex coding tasks
    COMPLEX_KEYWORDS = frozenset([
        "refactor",
        "redesign",
        "architect",
        "restructure",
        "rewrite",
        "implement new",
        "add feature",
        "migrate",
        "convert",
        "optimize performance",
        "security",
        "complex",
        "multi-file",
        "overhaul",
    ])

    # Keywords indicating moderate coding tasks
    MODERATE_KEYWORDS = frozenset([
        "fix",
        "improve",
        "update",
        "modify",
        "change",
        "edit",
        "add",
        "remove",
        "delete",
        "type hint",
        "docstring",
        "comment",
        "format",
        "lint",
        "test",
    ])

    def __init__(
        self,
        simple_model: str = MODEL_HAIKU,
        moderate_model: str = MODEL_SONNET,
        complex_model: str = MODEL_SONNET,
        override_model: str | None = None,
    ):
        """Initialize the model selector.

        Args:
            simple_model: Model to use for simple tasks (default: Haiku).
            moderate_model: Model to use for moderate tasks (default: Sonnet).
            complex_model: Model to use for complex tasks (default: Sonnet).
            override_model: If set, always use this model regardless of complexity.
        """
        self.simple_model = simple_model
        self.moderate_model = moderate_model
        self.complex_model = complex_model
        self.override_model = override_model

    def analyze_complexity(
        self,
        prompt: str,
        context: str | None = None,
    ) -> TaskComplexity:
        """Analyze the complexity of a task based on prompt and context.

        Args:
            prompt: The main instruction/prompt.
            context: Optional additional context.

        Returns:
            TaskComplexity indicating the task's complexity level.
        """
        text = (prompt + " " + (context or "")).lower()

        # Check for explicit complexity markers
        if self._has_complexity_markers(text, "complex"):
            return TaskComplexity.COMPLEX

        if self._has_complexity_markers(text, "simple"):
            return TaskComplexity.SIMPLE

        # Count keyword matches
        simple_score = self._count_keyword_matches(text, self.SIMPLE_KEYWORDS)
        moderate_score = self._count_keyword_matches(text, self.MODERATE_KEYWORDS)
        complex_score = self._count_keyword_matches(text, self.COMPLEX_KEYWORDS)

        # Check for multi-file indicators
        multi_file_patterns = [
            r"multiple files",
            r"across.*files",
            r"all files",
            r"entire codebase",
            r"project-wide",
        ]
        for pattern in multi_file_patterns:
            if re.search(pattern, text):
                complex_score += 2

        # Check for analysis-only indicators
        analysis_patterns = [
            r"analysis only",
            r"do not edit",
            r"don't edit",
            r"just analyze",
            r"only analyze",
            r"no changes",
        ]
        for pattern in analysis_patterns:
            if re.search(pattern, text):
                simple_score += 3

        # Determine complexity based on scores
        if simple_score > complex_score and simple_score > moderate_score:
            return TaskComplexity.SIMPLE
        elif complex_score > moderate_score:
            return TaskComplexity.COMPLEX
        elif moderate_score > 0:
            return TaskComplexity.MODERATE
        else:
            # Default to moderate for unknown tasks
            return TaskComplexity.MODERATE

    def _has_complexity_markers(self, text: str, level: str) -> bool:
        """Check for explicit complexity markers in text."""
        markers = {
            "complex": [
                "complex task",
                "difficult",
                "challenging",
                "major change",
                "significant refactor",
            ],
            "simple": [
                "simple task",
                "quick",
                "minor",
                "small change",
                "trivial",
            ],
        }
        for marker in markers.get(level, []):
            if marker in text:
                return True
        return False

    def _count_keyword_matches(
        self,
        text: str,
        keywords: frozenset[str],
    ) -> int:
        """Count how many keywords from the set appear in the text."""
        count = 0
        for keyword in keywords:
            if keyword in text:
                count += 1
        return count

    def select_model(
        self,
        prompt: str,
        context: str | None = None,
        task_type: Literal["analyze", "improve", "fix"] = "improve",
    ) -> ModelChoice:
        """Select the appropriate model for a task.

        Args:
            prompt: The main instruction/prompt.
            context: Optional additional context.
            task_type: Type of task (analyze, improve, fix).

        Returns:
            ModelChoice with selected model and reasoning.
        """
        # If override is set, always use it
        if self.override_model:
            return ModelChoice(
                model=self.override_model,
                complexity=TaskComplexity.MODERATE,
                reason="User-specified model override",
            )

        # Analysis tasks are typically simple
        if task_type == "analyze":
            return ModelChoice(
                model=self.simple_model,
                complexity=TaskComplexity.SIMPLE,
                reason="Analysis task - using efficient model",
            )

        # Fix tasks are typically moderate
        if task_type == "fix":
            return ModelChoice(
                model=self.moderate_model,
                complexity=TaskComplexity.MODERATE,
                reason="Fix task - using standard model",
            )

        # Analyze prompt complexity for improve tasks
        complexity = self.analyze_complexity(prompt, context)

        if complexity == TaskComplexity.SIMPLE:
            return ModelChoice(
                model=self.simple_model,
                complexity=complexity,
                reason="Simple task detected - using efficient model",
            )
        elif complexity == TaskComplexity.COMPLEX:
            return ModelChoice(
                model=self.complex_model,
                complexity=complexity,
                reason="Complex task detected - using capable model",
            )
        else:
            return ModelChoice(
                model=self.moderate_model,
                complexity=complexity,
                reason="Standard task - using balanced model",
            )

    def get_model_pricing(self, model: str) -> dict[str, float]:
        """Get pricing for a model.

        Args:
            model: Model identifier.

        Returns:
            Dict with 'input' and 'output' costs per million tokens.
        """
        # Normalize model name for lookup
        model_lower = model.lower()

        if "haiku" in model_lower:
            return MODEL_PRICING[MODEL_HAIKU]
        elif "opus" in model_lower:
            return MODEL_PRICING[MODEL_OPUS]
        else:
            # Default to Sonnet pricing
            return MODEL_PRICING[MODEL_SONNET]
