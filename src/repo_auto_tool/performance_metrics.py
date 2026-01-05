"""Performance metrics tracking for improvement sessions.

Tracks and analyzes iteration performance to identify bottlenecks
and optimization opportunities.

Metrics tracked:
- Time per iteration
- Token efficiency (output tokens per successful change)
- Error recovery time
- Success rate trends
- Validation performance

Example:
    tracker = PerformanceTracker()

    # Track an iteration
    with tracker.track_iteration(iteration_num=1):
        # ... do work ...
        pass

    tracker.record_validation_time(duration=5.2)
    tracker.record_tokens(input=1000, output=500)
    tracker.record_success(True)

    # Get insights
    metrics = tracker.get_metrics()
    print(f"Avg iteration time: {metrics['avg_iteration_time']:.2f}s")
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class IterationMetrics:
    """Metrics for a single iteration.

    Attributes:
        iteration: The iteration number.
        start_time: When the iteration started.
        end_time: When the iteration ended.
        duration: Total iteration time in seconds.
        validation_duration: Time spent on validation in seconds.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        success: Whether the iteration succeeded.
        validation_passed: Whether validation passed.
        error_type: Type of error if failed.
    """
    iteration: int
    start_time: datetime
    end_time: datetime | None = None
    duration: float = 0.0
    validation_duration: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = False
    validation_passed: bool = False
    error_type: str | None = None

    def finalize(self, end_time: datetime | None = None) -> None:
        """Finalize metrics by calculating duration.

        Args:
            end_time: End time (defaults to now).
        """
        self.end_time = end_time or datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()


@dataclass
class PerformanceInsights:
    """Insights derived from performance metrics.

    Attributes:
        total_iterations: Total number of iterations.
        successful_iterations: Number of successful iterations.
        success_rate: Success rate (0.0-1.0).
        avg_iteration_time: Average time per iteration (seconds).
        avg_validation_time: Average validation time (seconds).
        avg_tokens_per_iteration: Average tokens per iteration.
        token_efficiency: Output tokens per successful iteration.
        fastest_iteration: Fastest iteration time (seconds).
        slowest_iteration: Slowest iteration time (seconds).
        error_recovery_rate: Rate of recovery after errors (0.0-1.0).
        validation_overhead: Validation time as percentage of total.
    """
    total_iterations: int = 0
    successful_iterations: int = 0
    success_rate: float = 0.0
    avg_iteration_time: float = 0.0
    avg_validation_time: float = 0.0
    avg_tokens_per_iteration: float = 0.0
    token_efficiency: float = 0.0
    fastest_iteration: float = 0.0
    slowest_iteration: float = 0.0
    error_recovery_rate: float = 0.0
    validation_overhead: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_iterations": self.total_iterations,
            "successful_iterations": self.successful_iterations,
            "success_rate": round(self.success_rate, 3),
            "avg_iteration_time": round(self.avg_iteration_time, 2),
            "avg_validation_time": round(self.avg_validation_time, 2),
            "avg_tokens_per_iteration": round(self.avg_tokens_per_iteration, 1),
            "token_efficiency": round(self.token_efficiency, 1),
            "fastest_iteration": round(self.fastest_iteration, 2),
            "slowest_iteration": round(self.slowest_iteration, 2),
            "error_recovery_rate": round(self.error_recovery_rate, 3),
            "validation_overhead": round(self.validation_overhead * 100, 1),
        }


class PerformanceTracker:
    """Tracks and analyzes iteration performance.

    Attributes:
        metrics_history: List of IterationMetrics for all iterations.
        current_metrics: Metrics for the current iteration (if any).
    """

    def __init__(self):
        self.metrics_history: list[IterationMetrics] = []
        self.current_metrics: IterationMetrics | None = None

    @contextmanager
    def track_iteration(self, iteration_num: int) -> Iterator[IterationMetrics]:
        """Context manager to track an iteration.

        Args:
            iteration_num: The iteration number.

        Yields:
            IterationMetrics for this iteration.
        """
        self.current_metrics = IterationMetrics(
            iteration=iteration_num,
            start_time=datetime.now(),
        )

        try:
            yield self.current_metrics
        finally:
            self.current_metrics.finalize()
            self.metrics_history.append(self.current_metrics)
            self.current_metrics = None

    def record_validation_time(self, duration: float) -> None:
        """Record validation time for current iteration.

        Args:
            duration: Validation duration in seconds.
        """
        if self.current_metrics:
            self.current_metrics.validation_duration = duration

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage for current iteration.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        if self.current_metrics:
            self.current_metrics.input_tokens = input_tokens
            self.current_metrics.output_tokens = output_tokens

    def record_success(self, success: bool, validation_passed: bool = True) -> None:
        """Record success/failure for current iteration.

        Args:
            success: Whether the iteration succeeded.
            validation_passed: Whether validation passed.
        """
        if self.current_metrics:
            self.current_metrics.success = success
            self.current_metrics.validation_passed = validation_passed

    def record_error(self, error_type: str) -> None:
        """Record error type for current iteration.

        Args:
            error_type: Type of error.
        """
        if self.current_metrics:
            self.current_metrics.error_type = error_type

    def get_metrics(self) -> PerformanceInsights:
        """Calculate performance insights from tracked metrics.

        Returns:
            PerformanceInsights with analyzed metrics.
        """
        if not self.metrics_history:
            return PerformanceInsights()

        insights = PerformanceInsights()

        insights.total_iterations = len(self.metrics_history)
        insights.successful_iterations = sum(
            1 for m in self.metrics_history
            if m.success and m.validation_passed
        )

        if insights.total_iterations > 0:
            insights.success_rate = (
                insights.successful_iterations / insights.total_iterations
            )

        # Time metrics
        durations = [m.duration for m in self.metrics_history if m.duration > 0]
        if durations:
            insights.avg_iteration_time = sum(durations) / len(durations)
            insights.fastest_iteration = min(durations)
            insights.slowest_iteration = max(durations)

        validation_durations = [
            m.validation_duration for m in self.metrics_history
            if m.validation_duration > 0
        ]
        if validation_durations:
            insights.avg_validation_time = (
                sum(validation_durations) / len(validation_durations)
            )

        # Validation overhead
        if insights.avg_iteration_time > 0:
            insights.validation_overhead = (
                insights.avg_validation_time / insights.avg_iteration_time
            )

        # Token metrics
        total_tokens = sum(
            m.input_tokens + m.output_tokens for m in self.metrics_history
        )
        if insights.total_iterations > 0:
            insights.avg_tokens_per_iteration = (
                total_tokens / insights.total_iterations
            )

        successful_output_tokens = sum(
            m.output_tokens for m in self.metrics_history
            if m.success and m.validation_passed
        )
        if insights.successful_iterations > 0:
            insights.token_efficiency = (
                successful_output_tokens / insights.successful_iterations
            )

        # Error recovery rate
        # Count cases where an error was followed by success
        recoveries = 0
        errors = 0
        for i, m in enumerate(self.metrics_history[:-1]):
            if not m.validation_passed:
                errors += 1
                next_m = self.metrics_history[i + 1]
                if next_m.validation_passed:
                    recoveries += 1

        if errors > 0:
            insights.error_recovery_rate = recoveries / errors

        return insights

    def get_trend(self, window_size: int = 5) -> dict[str, float]:
        """Get recent performance trend.

        Args:
            window_size: Number of recent iterations to analyze.

        Returns:
            Dictionary with trend metrics.
        """
        if len(self.metrics_history) < window_size:
            window_size = len(self.metrics_history)

        if window_size == 0:
            return {
                "recent_success_rate": 0.0,
                "recent_avg_time": 0.0,
                "improving": False,
            }

        recent = self.metrics_history[-window_size:]

        recent_successes = sum(
            1 for m in recent if m.success and m.validation_passed
        )
        recent_success_rate = recent_successes / len(recent)

        recent_times = [m.duration for m in recent if m.duration > 0]
        recent_avg_time = sum(recent_times) / len(recent_times) if recent_times else 0.0

        # Compare to overall metrics
        overall = self.get_metrics()
        improving = recent_success_rate > overall.success_rate

        return {
            "recent_success_rate": round(recent_success_rate, 3),
            "recent_avg_time": round(recent_avg_time, 2),
            "improving": improving,
            "window_size": window_size,
        }

    def identify_bottlenecks(self) -> list[str]:
        """Identify performance bottlenecks.

        Returns:
            List of bottleneck descriptions.
        """
        bottlenecks = []
        insights = self.get_metrics()

        # Validation overhead
        if insights.validation_overhead > 0.5:
            bottlenecks.append(
                f"Validation takes {insights.validation_overhead:.0%} of iteration time - "
                "consider using smart/differential validation"
            )

        # Slow iterations
        if insights.slowest_iteration > 2 * insights.avg_iteration_time:
            bottlenecks.append(
                f"Some iterations are {insights.slowest_iteration / insights.avg_iteration_time:.1f}x "
                "slower than average - investigate what's different"
            )

        # Low success rate
        if insights.success_rate < 0.5:
            bottlenecks.append(
                f"Low success rate ({insights.success_rate:.0%}) - "
                "consider using prompt learner or adjusting approach"
            )

        # Poor error recovery
        if insights.error_recovery_rate < 0.5 and insights.total_iterations > 5:
            bottlenecks.append(
                f"Low error recovery rate ({insights.error_recovery_rate:.0%}) - "
                "errors are not being fixed effectively"
            )

        # Token inefficiency
        if insights.token_efficiency > 1000:
            bottlenecks.append(
                f"High token usage per success ({insights.token_efficiency:.0f} tokens) - "
                "prompts may be too verbose or unfocused"
            )

        return bottlenecks

    def get_summary(self) -> str:
        """Get a human-readable performance summary.

        Returns:
            Formatted summary string.
        """
        insights = self.get_metrics()
        trend = self.get_trend()
        bottlenecks = self.identify_bottlenecks()

        lines = [
            "Performance Summary:",
            f"  Iterations: {insights.total_iterations} total, "
            f"{insights.successful_iterations} successful ({insights.success_rate:.0%})",
            f"  Timing: {insights.avg_iteration_time:.1f}s avg "
            f"(fastest: {insights.fastest_iteration:.1f}s, "
            f"slowest: {insights.slowest_iteration:.1f}s)",
            f"  Validation: {insights.avg_validation_time:.1f}s avg "
            f"({insights.validation_overhead:.0%} of total time)",
            f"  Tokens: {insights.avg_tokens_per_iteration:.0f} avg per iteration, "
            f"{insights.token_efficiency:.0f} per success",
            f"  Recovery: {insights.error_recovery_rate:.0%} of errors recovered",
            "",
            f"Recent Trend ({trend['window_size']} iterations):",
            f"  Success rate: {trend['recent_success_rate']:.0%} "
            f"({'↑ improving' if trend['improving'] else '↓ declining'})",
            f"  Avg time: {trend['recent_avg_time']:.1f}s",
        ]

        if bottlenecks:
            lines.append("")
            lines.append("Potential Bottlenecks:")
            for bottleneck in bottlenecks:
                lines.append(f"  - {bottleneck}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics_history.clear()
        self.current_metrics = None
