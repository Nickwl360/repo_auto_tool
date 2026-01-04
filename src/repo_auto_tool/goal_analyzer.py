"""Goal analysis and complexity estimation for intelligent automation.

This module provides goal assessment capabilities to help users understand:
- How complex their goal is (scope, risk, estimated effort)
- Whether the goal is too vague and needs refinement
- Potential risks and warnings before execution
- Suggestions for better-scoped alternative goals
- Progress estimation during execution

All operations are defensively coded with robust error handling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GoalComplexity(Enum):
    """Complexity level for a goal."""

    TRIVIAL = "trivial"  # Single file, simple change (e.g., fix typo)
    SIMPLE = "simple"  # Few files, straightforward change
    MODERATE = "moderate"  # Multiple files, some coordination needed
    COMPLEX = "complex"  # Many files, significant changes
    MAJOR = "major"  # Architecture-level changes, high risk


class GoalRisk(Enum):
    """Risk level for a goal."""

    LOW = "low"  # Safe, easily reversible
    MEDIUM = "medium"  # Some risk, but manageable
    HIGH = "high"  # Significant risk, needs careful review
    CRITICAL = "critical"  # Could break things badly


@dataclass
class GoalAssessment:
    """Complete assessment of a goal.

    Attributes:
        goal: The original goal text
        complexity: Estimated complexity level
        risk: Estimated risk level
        estimated_iterations: Estimated number of iterations needed
        is_vague: Whether the goal is too vague
        warnings: List of warnings about the goal
        suggestions: Suggestions for improving the goal
        scope_indicators: What scope indicators were detected
        confidence: Confidence in the assessment (0.0 to 1.0)
    """

    goal: str
    complexity: GoalComplexity = GoalComplexity.MODERATE
    risk: GoalRisk = GoalRisk.MEDIUM
    estimated_iterations: int = 5
    is_vague: bool = False
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    scope_indicators: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "goal": self.goal,
            "complexity": self.complexity.value,
            "risk": self.risk.value,
            "estimated_iterations": self.estimated_iterations,
            "is_vague": self.is_vague,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "scope_indicators": self.scope_indicators,
            "confidence": self.confidence,
        }

    def get_summary(self) -> str:
        """Get a human-readable summary of the assessment.

        Returns:
            Formatted string summary.
        """
        lines = [
            "Goal Assessment:",
            f"  Complexity: {self.complexity.value.upper()}",
            f"  Risk Level: {self.risk.value.upper()}",
            f"  Est. Iterations: {self.estimated_iterations}",
            f"  Confidence: {self.confidence:.0%}",
        ]

        if self.is_vague:
            lines.append("  [!] Goal is VAGUE - consider refining it")

        if self.warnings:
            lines.append("\n  Warnings:")
            for warning in self.warnings[:5]:
                lines.append(f"    - {warning}")

        if self.suggestions:
            lines.append("\n  Suggestions:")
            for suggestion in self.suggestions[:3]:
                lines.append(f"    - {suggestion}")

        return "\n".join(lines)


class GoalAnalyzer:
    """Analyzes goals to estimate complexity, risk, and provide guidance.

    This analyzer examines goal text to determine:
    - Scope of changes (single file vs. codebase-wide)
    - Type of work (fix, refactor, new feature, etc.)
    - Potential risks and warnings
    - Whether the goal needs refinement

    All analysis is done locally without LLM calls for speed.
    """

    # Vague/ambiguous goal patterns
    VAGUE_PATTERNS: list[tuple[str, str]] = [
        (r"^improve\s+", "What specifically should be improved?"),
        (r"^make\s+(?:it\s+)?better", "What does 'better' mean specifically?"),
        (r"^fix\s+(?:it|things|stuff)", "What specifically needs to be fixed?"),
        (r"^clean\s*up", "What areas need cleaning up?"),
        (r"^optimize", "What specific optimization is needed?"),
        (r"^refactor$", "What specific refactoring is needed?"),
        (r"^update", "What specific updates are needed?"),
        (r"^change\s+", "What specific changes are needed?"),
    ]

    # Scope indicators - patterns that suggest scope
    SCOPE_PATTERNS: dict[str, list[str]] = {
        "single_file": [
            r"in\s+\w+\.\w+",  # "in file.py"
            r"(?:file|module)\s+\w+",
            r"the\s+\w+\s+function",
            r"the\s+\w+\s+class",
        ],
        "multi_file": [
            r"all\s+(?:files|modules)",
            r"across\s+(?:the\s+)?(?:project|codebase)",
            r"throughout",
            r"every(?:where)?",
            r"project-?wide",
        ],
        "architecture": [
            r"architect(?:ure)?",
            r"restructur",
            r"rewrite",
            r"redesign",
            r"overhaul",
            r"migrate",
        ],
    }

    # Risk indicators
    RISK_PATTERNS: dict[str, list[str]] = {
        "high_risk": [
            r"database",
            r"migration",
            r"security",
            r"auth(?:entication|orization)?",
            r"payment",
            r"production",
            r"deploy",
            r"secret",
            r"credential",
            r"password",
        ],
        "breaking_change": [
            r"breaking\s+change",
            r"backwards?\s+compat",
            r"deprecate",
            r"remove\s+(?:all|support)",
            r"delete\s+(?:all|files)",
        ],
        "external_deps": [
            r"api\s+(?:change|update|migration)",
            r"third\s*party",
            r"external\s+service",
            r"integrat",
        ],
    }

    # Work type patterns (used for iteration estimation)
    WORK_TYPE_PATTERNS: dict[str, tuple[list[str], int]] = {
        "fix_bug": ([r"fix\s+bug", r"bug\s*fix", r"resolve\s+issue"], 2),
        "fix_typo": ([r"typo", r"spelling", r"grammar"], 1),
        "add_tests": ([r"add\s+tests?", r"test\s+coverage", r"unit\s+tests?"], 4),
        "add_types": ([r"type\s+hints?", r"typing", r"add\s+types"], 5),
        "add_docs": ([r"document", r"docstring", r"readme", r"comment"], 3),
        "refactor": ([r"refactor", r"clean\s*up", r"reorganiz"], 6),
        "new_feature": ([r"implement", r"add\s+feature", r"create\s+new", r"build"], 8),
        "major_change": ([r"rewrite", r"overhaul", r"redesign", r"migrate"], 15),
    }

    def __init__(self, repo_path: Path | None = None):
        """Initialize the goal analyzer.

        Args:
            repo_path: Optional path to the repository for context-aware analysis.
        """
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self._codebase_size: int | None = None

    def analyze(self, goal: str) -> GoalAssessment:
        """Analyze a goal and return a complete assessment.

        Args:
            goal: The goal text to analyze.

        Returns:
            GoalAssessment with complexity, risk, and recommendations.
        """
        if not goal or not isinstance(goal, str):
            return GoalAssessment(
                goal=goal or "",
                complexity=GoalComplexity.MODERATE,
                risk=GoalRisk.MEDIUM,
                is_vague=True,
                warnings=["Empty or invalid goal provided"],
                confidence=0.0,
            )

        goal = goal.strip()
        goal_lower = goal.lower()

        assessment = GoalAssessment(goal=goal)

        try:
            # Check for vagueness
            self._assess_vagueness(goal_lower, assessment)

            # Determine scope
            self._assess_scope(goal_lower, assessment)

            # Assess risk
            self._assess_risk(goal_lower, assessment)

            # Estimate iterations
            self._estimate_iterations(goal_lower, assessment)

            # Determine complexity from scope and work type
            self._determine_complexity(assessment)

            # Generate suggestions
            self._generate_suggestions(goal_lower, assessment)

            # Calculate confidence
            self._calculate_confidence(assessment)

        except Exception as e:
            logger.error(f"Error analyzing goal: {e}")
            assessment.warnings.append(f"Analysis error: {e}")
            assessment.confidence = 0.3

        return assessment

    def _assess_vagueness(self, goal_lower: str, assessment: GoalAssessment) -> None:
        """Check if the goal is too vague.

        Args:
            goal_lower: Lowercase goal text.
            assessment: Assessment to update.
        """
        for pattern, suggestion in self.VAGUE_PATTERNS:
            try:
                if re.search(pattern, goal_lower):
                    assessment.is_vague = True
                    assessment.suggestions.append(suggestion)
            except re.error:
                continue

        # Check for very short goals
        word_count = len(goal_lower.split())
        if word_count <= 3:
            assessment.is_vague = True
            assessment.warnings.append(
                "Goal is very short - more detail would help"
            )

        # Check for lack of specificity
        if not any(char.isdigit() for char in goal_lower):
            has_specific_target = any(
                kw in goal_lower
                for kw in ["file", "class", "function", "module", "test", "error"]
            )
            if not has_specific_target and word_count < 10:
                assessment.suggestions.append(
                    "Consider specifying target files, functions, or areas"
                )

    def _assess_scope(self, goal_lower: str, assessment: GoalAssessment) -> None:
        """Determine the scope of the goal.

        Args:
            goal_lower: Lowercase goal text.
            assessment: Assessment to update.
        """
        scope_indicators = assessment.scope_indicators

        for scope_type, patterns in self.SCOPE_PATTERNS.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, goal_lower):
                        scope_indicators[scope_type] = True
                        break
                except re.error:
                    continue

        # Infer scope from keywords if not explicitly detected
        if not scope_indicators:
            if any(kw in goal_lower for kw in ["all", "every", "entire", "whole"]):
                scope_indicators["multi_file"] = True
            elif any(kw in goal_lower for kw in ["the", "this", "one", "single"]):
                scope_indicators["single_file"] = True
            else:
                scope_indicators["unknown"] = True

    def _assess_risk(self, goal_lower: str, assessment: GoalAssessment) -> None:
        """Assess the risk level of the goal.

        Args:
            goal_lower: Lowercase goal text.
            assessment: Assessment to update.
        """
        risk_score = 0

        for risk_type, patterns in self.RISK_PATTERNS.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, goal_lower):
                        if risk_type == "high_risk":
                            risk_score += 3
                            assessment.warnings.append(
                                f"High-risk area detected: {pattern}"
                            )
                        elif risk_type == "breaking_change":
                            risk_score += 4
                            assessment.warnings.append(
                                "Potential breaking change - review carefully"
                            )
                        elif risk_type == "external_deps":
                            risk_score += 2
                            assessment.warnings.append(
                                "External dependency changes may have side effects"
                            )
                        break
                except re.error:
                    continue

        # Check for architecture-level changes
        if assessment.scope_indicators.get("architecture"):
            risk_score += 3
            assessment.warnings.append(
                "Architecture changes are high risk - consider smaller increments"
            )

        # Map score to risk level
        if risk_score >= 6:
            assessment.risk = GoalRisk.CRITICAL
        elif risk_score >= 4:
            assessment.risk = GoalRisk.HIGH
        elif risk_score >= 2:
            assessment.risk = GoalRisk.MEDIUM
        else:
            assessment.risk = GoalRisk.LOW

    def _estimate_iterations(
        self, goal_lower: str, assessment: GoalAssessment
    ) -> None:
        """Estimate the number of iterations needed.

        Args:
            goal_lower: Lowercase goal text.
            assessment: Assessment to update.
        """
        base_iterations = 3  # Default
        work_type_detected = False

        for work_type, (patterns, iterations) in self.WORK_TYPE_PATTERNS.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, goal_lower):
                        base_iterations = iterations
                        assessment.scope_indicators["work_type"] = work_type
                        work_type_detected = True
                        break
                except re.error:
                    continue
            if work_type_detected:
                break

        # Adjust for scope
        if assessment.scope_indicators.get("multi_file"):
            base_iterations = int(base_iterations * 1.5)
        if assessment.scope_indicators.get("architecture"):
            base_iterations = int(base_iterations * 2.0)

        # Clamp to reasonable range
        assessment.estimated_iterations = max(1, min(base_iterations, 50))

    def _determine_complexity(self, assessment: GoalAssessment) -> None:
        """Determine overall complexity from scope and other factors.

        Args:
            assessment: Assessment to update.
        """
        indicators = assessment.scope_indicators
        iterations = assessment.estimated_iterations

        # Architecture changes are always major
        if indicators.get("architecture"):
            assessment.complexity = GoalComplexity.MAJOR
            return

        # Use iterations as a complexity proxy
        if iterations <= 1:
            assessment.complexity = GoalComplexity.TRIVIAL
        elif iterations <= 3:
            assessment.complexity = GoalComplexity.SIMPLE
        elif iterations <= 6:
            assessment.complexity = GoalComplexity.MODERATE
        elif iterations <= 12:
            assessment.complexity = GoalComplexity.COMPLEX
        else:
            assessment.complexity = GoalComplexity.MAJOR

        # Adjust for multi-file scope
        if indicators.get("multi_file") and assessment.complexity.value in (
            "trivial",
            "simple",
        ):
            assessment.complexity = GoalComplexity.MODERATE

    def _generate_suggestions(
        self, goal_lower: str, assessment: GoalAssessment
    ) -> None:
        """Generate suggestions for improving the goal.

        Args:
            goal_lower: Lowercase goal text.
            assessment: Assessment to update.
        """
        # Suggest breaking down complex goals
        if assessment.complexity in (GoalComplexity.COMPLEX, GoalComplexity.MAJOR):
            assessment.suggestions.append(
                "Consider breaking this into smaller, incremental goals"
            )

        # Suggest being more specific for vague goals
        if assessment.is_vague and not assessment.suggestions:
            assessment.suggestions.append(
                "Add specific file names, function names, or measurable outcomes"
            )

        # Suggest validation for high-risk changes
        if assessment.risk in (GoalRisk.HIGH, GoalRisk.CRITICAL):
            assessment.suggestions.append(
                "Ensure tests exist before making changes to catch regressions"
            )

        # Suggest using plan mode for complex changes
        if assessment.estimated_iterations > 8:
            assessment.suggestions.append(
                "Use --plan mode to review the approach before execution"
            )

        # Suggest checkpoints for long runs
        if assessment.estimated_iterations > 10:
            assessment.suggestions.append(
                "Use --checkpoint-interval to review progress periodically"
            )

    def _calculate_confidence(self, assessment: GoalAssessment) -> None:
        """Calculate confidence in the assessment.

        Args:
            assessment: Assessment to update.
        """
        confidence = 0.7  # Base confidence

        # Reduce confidence for vague goals
        if assessment.is_vague:
            confidence -= 0.2

        # Reduce confidence for unknown scope
        if assessment.scope_indicators.get("unknown"):
            confidence -= 0.1

        # Reduce confidence for very complex goals
        if assessment.complexity in (GoalComplexity.COMPLEX, GoalComplexity.MAJOR):
            confidence -= 0.1

        # Increase confidence if we detected specific work type
        if assessment.scope_indicators.get("work_type"):
            confidence += 0.1

        # Clamp to valid range
        assessment.confidence = max(0.1, min(confidence, 0.95))


@dataclass
class ProgressEstimate:
    """Estimate of progress toward a goal.

    Attributes:
        current_iteration: Current iteration number
        total_estimated: Estimated total iterations
        percent_complete: Estimated percentage complete (0-100)
        remaining_iterations: Estimated remaining iterations
        trend: Whether progress is on track ("on_track", "slower", "faster")
        confidence: Confidence in the estimate (0.0 to 1.0)
    """

    current_iteration: int
    total_estimated: int
    percent_complete: float
    remaining_iterations: int
    trend: str = "on_track"
    confidence: float = 0.5

    def get_summary(self) -> str:
        """Get a human-readable progress summary.

        Returns:
            Formatted progress string.
        """
        bar_width = 20
        filled = int(bar_width * self.percent_complete / 100)
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"

        trend_icon = {"on_track": "->", "slower": ">>", "faster": "<<"}.get(
            self.trend, "->"
        )

        return (
            f"Progress: {bar} {self.percent_complete:.0f}% "
            f"(iter {self.current_iteration}/{self.total_estimated}) {trend_icon}"
        )


class ProgressTracker:
    """Tracks and estimates progress toward a goal.

    Uses iteration history and success patterns to estimate
    how far along the improvement process is.
    """

    def __init__(self, initial_estimate: int = 5):
        """Initialize the progress tracker.

        Args:
            initial_estimate: Initial estimate of total iterations.
        """
        self.initial_estimate = max(1, initial_estimate)
        self.current_estimate = self.initial_estimate
        self.iterations_completed = 0
        self.successful_iterations = 0
        self.consecutive_successes = 0
        self.consecutive_failures = 0

    def record_iteration(self, success: bool, made_progress: bool = True) -> None:
        """Record the outcome of an iteration.

        Args:
            success: Whether the iteration succeeded (validation passed).
            made_progress: Whether meaningful progress was made.
        """
        self.iterations_completed += 1

        if success:
            self.successful_iterations += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0

        # Adjust estimate based on progress
        self._adjust_estimate(success, made_progress)

    def _adjust_estimate(self, success: bool, made_progress: bool) -> None:
        """Adjust the total estimate based on iteration outcome.

        Args:
            success: Whether the iteration succeeded.
            made_progress: Whether meaningful progress was made.
        """
        # If we're struggling (multiple consecutive failures), increase estimate
        if self.consecutive_failures >= 3:
            self.current_estimate = int(self.current_estimate * 1.2)

        # If we're making fast progress, decrease estimate
        if self.consecutive_successes >= 3 and made_progress:
            self.current_estimate = max(
                self.iterations_completed,
                int(self.current_estimate * 0.9),
            )

        # Never estimate less than current iteration
        self.current_estimate = max(self.current_estimate, self.iterations_completed)

    def get_estimate(self) -> ProgressEstimate:
        """Get the current progress estimate.

        Returns:
            ProgressEstimate with current status.
        """
        # Calculate percent complete
        percent = min(
            99.0,  # Never show 100% until actually done
            (self.iterations_completed / max(1, self.current_estimate)) * 100,
        )

        # Determine trend
        if self.consecutive_failures >= 2:
            trend = "slower"
        elif self.consecutive_successes >= 2:
            trend = "faster"
        else:
            trend = "on_track"

        # Calculate confidence
        confidence = 0.5
        if self.iterations_completed >= 3:
            success_rate = self.successful_iterations / self.iterations_completed
            confidence = min(0.9, 0.4 + success_rate * 0.5)

        remaining = max(0, self.current_estimate - self.iterations_completed)

        return ProgressEstimate(
            current_iteration=self.iterations_completed,
            total_estimated=self.current_estimate,
            percent_complete=percent,
            remaining_iterations=remaining,
            trend=trend,
            confidence=confidence,
        )

    def is_likely_complete(self) -> bool:
        """Check if the goal is likely complete.

        Returns:
            True if we estimate the goal is complete.
        """
        estimate = self.get_estimate()
        return (
            estimate.percent_complete >= 95.0
            and self.consecutive_successes >= 2
        )


def analyze_goal(goal: str, repo_path: Path | None = None) -> GoalAssessment:
    """Convenience function to analyze a goal.

    Args:
        goal: The goal text to analyze.
        repo_path: Optional repository path for context.

    Returns:
        GoalAssessment with the analysis results.
    """
    try:
        analyzer = GoalAnalyzer(repo_path)
        return analyzer.analyze(goal)
    except Exception as e:
        logger.error(f"Failed to analyze goal: {e}")
        return GoalAssessment(
            goal=goal or "",
            is_vague=True,
            warnings=[f"Analysis failed: {e}"],
            confidence=0.0,
        )
