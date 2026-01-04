"""Session metrics tracking and analysis.

Provides rich analytics for improvement sessions including:
- Success/failure rates and patterns
- Token efficiency metrics
- Time-based statistics
- Cost analysis
- Historical comparisons

This module helps the tool understand its own performance patterns
and identify areas for self-improvement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import ImprovementState, IterationRecord

logger = logging.getLogger(__name__)


@dataclass
class EfficiencyMetrics:
    """Metrics related to token and cost efficiency.

    Attributes:
        avg_tokens_per_iteration: Average total tokens used per iteration.
        avg_input_tokens: Average input tokens per iteration.
        avg_output_tokens: Average output tokens per iteration.
        tokens_per_success: Tokens spent per successful iteration.
        cost_per_success: Estimated cost per successful iteration.
        cache_hit_rate: Percentage of tokens served from cache (0-100).
    """

    avg_tokens_per_iteration: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    tokens_per_success: float = 0.0
    cost_per_success: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class SuccessMetrics:
    """Metrics related to iteration success rates.

    Attributes:
        total_iterations: Total number of iterations run.
        successful_iterations: Iterations with validation passing.
        failed_iterations: Iterations with validation failing.
        success_rate: Percentage of successful iterations (0-100).
        consecutive_successes: Current streak of successes.
        max_consecutive_successes: Longest success streak.
        consecutive_failures: Current streak of failures.
        max_consecutive_failures: Longest failure streak.
    """

    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    success_rate: float = 0.0
    consecutive_successes: int = 0
    max_consecutive_successes: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0


@dataclass
class TimeMetrics:
    """Time-related session metrics.

    Attributes:
        started_at: When the session started (ISO format).
        ended_at: When the session ended (ISO format), or None if running.
        elapsed_seconds: Total elapsed time in seconds.
        avg_seconds_per_iteration: Average time per iteration (estimated).
    """

    started_at: str | None = None
    ended_at: str | None = None
    elapsed_seconds: float = 0.0
    avg_seconds_per_iteration: float = 0.0


@dataclass
class FailureAnalysis:
    """Analysis of failure patterns.

    Attributes:
        error_categories: Count of failures by category.
        most_common_error: The most frequent error type.
        recovery_rate: Percentage of failures followed by success (0-100).
    """

    error_categories: dict[str, int] = field(default_factory=dict)
    most_common_error: str | None = None
    recovery_rate: float = 0.0


@dataclass
class SessionMetrics:
    """Comprehensive session metrics and analytics.

    Aggregates all metrics types into a single analysis object
    that can be used for reporting, self-diagnosis, and optimization.

    Attributes:
        efficiency: Token and cost efficiency metrics.
        success: Success rate and streak metrics.
        time: Time-related metrics.
        failures: Failure pattern analysis.
        estimated_cost: Total estimated cost in USD.
        goal_progress: Qualitative assessment of progress toward goal.
    """

    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    success: SuccessMetrics = field(default_factory=SuccessMetrics)
    time: TimeMetrics = field(default_factory=TimeMetrics)
    failures: FailureAnalysis = field(default_factory=FailureAnalysis)
    estimated_cost: float = 0.0
    goal_progress: str = "unknown"

    @classmethod
    def from_state(
        cls,
        state: ImprovementState,
        model: str | None = None,
    ) -> SessionMetrics:
        """Create metrics from an improvement state.

        Args:
            state: The improvement state to analyze.
            model: The model used (for cost estimation).

        Returns:
            SessionMetrics with computed values.
        """
        metrics = cls()

        # Calculate success metrics
        metrics.success = _calculate_success_metrics(state.iterations)

        # Calculate efficiency metrics
        metrics.efficiency = _calculate_efficiency_metrics(
            state=state,
            successful_count=metrics.success.successful_iterations,
            model=model,
        )

        # Calculate time metrics
        metrics.time = _calculate_time_metrics(state)

        # Analyze failure patterns
        metrics.failures = _analyze_failures(state.iterations)

        # Estimate total cost
        metrics.estimated_cost = _estimate_session_cost(state, model)

        # Assess goal progress
        metrics.goal_progress = _assess_goal_progress(state)

        return metrics

    def format_report(self) -> str:
        """Format metrics as a human-readable report.

        Returns:
            Multi-line string with formatted metrics.
        """
        lines = [
            "",
            "SESSION METRICS REPORT",
            "=" * 50,
            "",
            "SUCCESS METRICS:",
            f"  Total iterations:      {self.success.total_iterations}",
            f"  Successful:            {self.success.successful_iterations}",
            f"  Failed:                {self.success.failed_iterations}",
            f"  Success rate:          {self.success.success_rate:.1f}%",
            f"  Max success streak:    {self.success.max_consecutive_successes}",
            f"  Max failure streak:    {self.success.max_consecutive_failures}",
            "",
            "EFFICIENCY METRICS:",
            f"  Avg tokens/iteration:  {self.efficiency.avg_tokens_per_iteration:,.0f}",
            f"  Avg input tokens:      {self.efficiency.avg_input_tokens:,.0f}",
            f"  Avg output tokens:     {self.efficiency.avg_output_tokens:,.0f}",
            f"  Tokens per success:    {self.efficiency.tokens_per_success:,.0f}",
            f"  Cache hit rate:        {self.efficiency.cache_hit_rate:.1f}%",
        ]

        if self.estimated_cost > 0:
            lines.extend([
                "",
                "COST ANALYSIS:",
                f"  Estimated total cost:  ${self.estimated_cost:.4f}",
                f"  Cost per success:      ${self.efficiency.cost_per_success:.4f}",
            ])

        if self.time.elapsed_seconds > 0:
            elapsed_min = self.time.elapsed_seconds / 60
            lines.extend([
                "",
                "TIME METRICS:",
                f"  Elapsed time:          {elapsed_min:.1f} minutes",
                f"  Avg time/iteration:    {self.time.avg_seconds_per_iteration:.1f} seconds",
            ])

        if self.failures.error_categories:
            lines.extend([
                "",
                "FAILURE ANALYSIS:",
                f"  Most common error:     {self.failures.most_common_error or 'N/A'}",
                f"  Recovery rate:         {self.failures.recovery_rate:.1f}%",
                "  Error breakdown:",
            ])
            for category, count in sorted(
                self.failures.error_categories.items(),
                key=lambda x: -x[1],
            ):
                lines.append(f"    - {category}: {count}")

        lines.extend([
            "",
            f"GOAL PROGRESS: {self.goal_progress}",
            "=" * 50,
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary for JSON serialization.

        Returns:
            Dictionary with all metrics.
        """
        return {
            "success": {
                "total_iterations": self.success.total_iterations,
                "successful_iterations": self.success.successful_iterations,
                "failed_iterations": self.success.failed_iterations,
                "success_rate": self.success.success_rate,
                "max_consecutive_successes": self.success.max_consecutive_successes,
                "max_consecutive_failures": self.success.max_consecutive_failures,
            },
            "efficiency": {
                "avg_tokens_per_iteration": self.efficiency.avg_tokens_per_iteration,
                "avg_input_tokens": self.efficiency.avg_input_tokens,
                "avg_output_tokens": self.efficiency.avg_output_tokens,
                "tokens_per_success": self.efficiency.tokens_per_success,
                "cost_per_success": self.efficiency.cost_per_success,
                "cache_hit_rate": self.efficiency.cache_hit_rate,
            },
            "time": {
                "started_at": self.time.started_at,
                "ended_at": self.time.ended_at,
                "elapsed_seconds": self.time.elapsed_seconds,
                "avg_seconds_per_iteration": self.time.avg_seconds_per_iteration,
            },
            "failures": {
                "error_categories": self.failures.error_categories,
                "most_common_error": self.failures.most_common_error,
                "recovery_rate": self.failures.recovery_rate,
            },
            "estimated_cost": self.estimated_cost,
            "goal_progress": self.goal_progress,
        }


def _calculate_success_metrics(iterations: list[IterationRecord]) -> SuccessMetrics:
    """Calculate success-related metrics from iteration history.

    Args:
        iterations: List of iteration records.

    Returns:
        SuccessMetrics with computed values.
    """
    metrics = SuccessMetrics()
    metrics.total_iterations = len(iterations)

    if not iterations:
        return metrics

    # Count successes and failures
    for it in iterations:
        if it.success and it.validation_passed:
            metrics.successful_iterations += 1
        else:
            metrics.failed_iterations += 1

    # Calculate success rate
    if metrics.total_iterations > 0:
        metrics.success_rate = (
            metrics.successful_iterations / metrics.total_iterations
        ) * 100

    # Calculate streaks
    current_streak = 0
    max_success_streak = 0
    max_failure_streak = 0
    last_was_success = None

    for it in iterations:
        is_success = it.success and it.validation_passed
        if is_success:
            if last_was_success is True:
                current_streak += 1
            else:
                # Check if previous failure streak was max
                if last_was_success is False:
                    max_failure_streak = max(max_failure_streak, current_streak)
                current_streak = 1
            max_success_streak = max(max_success_streak, current_streak)
        else:
            if last_was_success is False:
                current_streak += 1
            else:
                # Check if previous success streak was max
                if last_was_success is True:
                    max_success_streak = max(max_success_streak, current_streak)
                current_streak = 1
            max_failure_streak = max(max_failure_streak, current_streak)
        last_was_success = is_success

    # Final streak check
    if last_was_success is True:
        max_success_streak = max(max_success_streak, current_streak)
        metrics.consecutive_successes = current_streak
    elif last_was_success is False:
        max_failure_streak = max(max_failure_streak, current_streak)
        metrics.consecutive_failures = current_streak

    metrics.max_consecutive_successes = max_success_streak
    metrics.max_consecutive_failures = max_failure_streak

    return metrics


def _calculate_efficiency_metrics(
    state: ImprovementState,
    successful_count: int,
    model: str | None = None,
) -> EfficiencyMetrics:
    """Calculate efficiency metrics from state.

    Args:
        state: The improvement state.
        successful_count: Number of successful iterations.
        model: Model used for cost calculation.

    Returns:
        EfficiencyMetrics with computed values.
    """
    metrics = EfficiencyMetrics()
    total_iterations = len(state.iterations)

    if total_iterations == 0:
        return metrics

    total_tokens = state.total_input_tokens + state.total_output_tokens

    # Average tokens per iteration
    metrics.avg_tokens_per_iteration = total_tokens / total_iterations
    metrics.avg_input_tokens = state.total_input_tokens / total_iterations
    metrics.avg_output_tokens = state.total_output_tokens / total_iterations

    # Tokens per success (efficiency of finding solutions)
    if successful_count > 0:
        metrics.tokens_per_success = total_tokens / successful_count
        cost = _estimate_session_cost(state, model)
        metrics.cost_per_success = cost / successful_count

    # Cache efficiency
    total_input = state.total_input_tokens + state.total_cache_read_tokens
    if total_input > 0:
        metrics.cache_hit_rate = (state.total_cache_read_tokens / total_input) * 100

    return metrics


def _calculate_time_metrics(state: ImprovementState) -> TimeMetrics:
    """Calculate time-related metrics.

    Args:
        state: The improvement state.

    Returns:
        TimeMetrics with computed values.
    """
    metrics = TimeMetrics()
    metrics.started_at = state.started_at
    metrics.ended_at = state.completed_at

    try:
        start = datetime.fromisoformat(state.started_at)
        end = (
            datetime.fromisoformat(state.completed_at)
            if state.completed_at
            else datetime.now()
        )
        metrics.elapsed_seconds = (end - start).total_seconds()

        if len(state.iterations) > 0:
            metrics.avg_seconds_per_iteration = (
                metrics.elapsed_seconds / len(state.iterations)
            )
    except (ValueError, TypeError):
        # If timestamps are malformed, leave as defaults
        logger.debug("Could not parse session timestamps for time metrics")

    return metrics


def _analyze_failures(iterations: list[IterationRecord]) -> FailureAnalysis:
    """Analyze failure patterns in iterations.

    Args:
        iterations: List of iteration records.

    Returns:
        FailureAnalysis with computed values.
    """
    analysis = FailureAnalysis()

    # Count error categories
    for it in iterations:
        if it.error:
            category = _categorize_error(it.error)
            analysis.error_categories[category] = (
                analysis.error_categories.get(category, 0) + 1
            )

    # Find most common error
    if analysis.error_categories:
        analysis.most_common_error = max(
            analysis.error_categories.keys(),
            key=lambda k: analysis.error_categories[k],
        )

    # Calculate recovery rate (failures followed by success)
    failures_followed_by_success = 0
    total_failures = 0

    for i, it in enumerate(iterations):
        if not (it.success and it.validation_passed):
            total_failures += 1
            # Check if next iteration was successful
            if i + 1 < len(iterations):
                next_it = iterations[i + 1]
                if next_it.success and next_it.validation_passed:
                    failures_followed_by_success += 1

    if total_failures > 0:
        analysis.recovery_rate = (failures_followed_by_success / total_failures) * 100

    return analysis


def _categorize_error(error: str) -> str:
    """Categorize an error message into a type.

    Args:
        error: The error message.

    Returns:
        Error category string.
    """
    error_lower = error.lower()

    if "syntax" in error_lower:
        return "syntax_error"
    if "import" in error_lower or "module" in error_lower:
        return "import_error"
    if "type" in error_lower and ("error" in error_lower or "hint" in error_lower):
        return "type_error"
    if "test" in error_lower or "assert" in error_lower:
        return "test_failure"
    if "lint" in error_lower or "ruff" in error_lower or "flake" in error_lower:
        return "lint_error"
    if "timeout" in error_lower:
        return "timeout"
    if "permission" in error_lower or "access" in error_lower:
        return "permission_error"
    return "other"


def _estimate_session_cost(
    state: ImprovementState,
    model: str | None = None,
) -> float:
    """Estimate total session cost.

    Args:
        state: The improvement state.
        model: The model used.

    Returns:
        Estimated cost in USD.
    """
    # Pricing per million tokens (approximate)
    model_lower = (model or "sonnet").lower()

    if "opus" in model_lower:
        input_cost = 15.0
        output_cost = 75.0
    elif "haiku" in model_lower:
        input_cost = 0.25
        output_cost = 1.25
    else:  # Sonnet
        input_cost = 3.0
        output_cost = 15.0

    # Cache reads are 10% of input cost
    cache_cost = input_cost * 0.1

    total = (
        (state.total_input_tokens / 1_000_000) * input_cost
        + (state.total_output_tokens / 1_000_000) * output_cost
        + (state.total_cache_read_tokens / 1_000_000) * cache_cost
    )

    return total


def _assess_goal_progress(state: ImprovementState) -> str:
    """Assess qualitative progress toward goal.

    Args:
        state: The improvement state.

    Returns:
        Progress assessment string.
    """
    if state.status == "completed":
        return "COMPLETED"
    if state.status == "failed":
        return "FAILED"
    if state.status == "converged":
        return "CONVERGED (likely complete)"

    # Assess based on iteration patterns
    if not state.iterations:
        return "NOT STARTED"

    # Check recent success rate
    recent = state.iterations[-5:] if len(state.iterations) >= 5 else state.iterations
    recent_successes = sum(
        1 for it in recent if it.success and it.validation_passed
    )
    recent_rate = recent_successes / len(recent)

    if recent_rate >= 0.8:
        return "GOOD PROGRESS"
    if recent_rate >= 0.5:
        return "MODERATE PROGRESS"
    if recent_rate >= 0.2:
        return "STRUGGLING"
    return "BLOCKED"
