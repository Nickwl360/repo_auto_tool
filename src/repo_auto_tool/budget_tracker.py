"""Token budget tracking with real-time percentage monitoring.

This module provides comprehensive budget tracking with:
- Real-time percentage display
- Cost projections
- Warnings before budget exhaustion
- Model downgrade suggestions

Example:
    tracker = BudgetTracker(max_cost=10.00)

    # Track usage
    status = tracker.track_usage(token_usage)
    print(f"Budget: {status.percentage_used:.0%} used")

    # Check if should downgrade
    if tracker.should_downgrade_model():
        print("Consider using Haiku to conserve budget")

    # Get display string
    print(tracker.get_display_string())
    # Output: "Budget: 35% used ($3.50/$10.00) - ~15 iterations remaining"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Model costs per million tokens (as of Jan 2026)
# Source: https://www.anthropic.com/pricing
MODEL_COSTS = {
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {
        "input": 1.00,  # $1 per million input tokens
        "output": 5.00,  # $5 per million output tokens
    },
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
    },
    # Claude Opus 4.5
    "claude-opus-4-5-20251101": {
        "input": 15.00,
        "output": 75.00,
    },
    # Fallback for unknown models
    "default": {
        "input": 3.00,
        "output": 15.00,
    },
}


class BudgetStatus(Enum):
    """Budget status levels."""
    SAFE = "safe"  # < 50% used
    WARNING = "warning"  # 50-80% used
    CRITICAL = "critical"  # > 80% used
    EXCEEDED = "exceeded"  # > 100% used


@dataclass
class TokenUsage:
    """Token usage for a single operation.

    Attributes:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cache_read_tokens: Number of cached tokens read.
        cache_creation_tokens: Number of tokens written to cache.
        model: Model used (for cost calculation).
    """
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    model: str = "claude-3-5-sonnet-20241022"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenUsage:
        """Create from dictionary.

        Args:
            data: Dictionary with token usage data.

        Returns:
            TokenUsage instance.
        """
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            cache_creation_tokens=data.get("cache_creation_tokens", 0),
            model=data.get("model", "claude-3-5-sonnet-20241022"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "model": self.model,
        }

    def calculate_cost(self, model_costs: dict[str, dict[str, float]] | None = None) -> float:
        """Calculate cost in USD.

        Args:
            model_costs: Optional custom model costs.

        Returns:
            Cost in USD.
        """
        costs = model_costs or MODEL_COSTS
        model_cost = costs.get(self.model, costs["default"])

        input_cost = (self.input_tokens / 1_000_000) * model_cost["input"]
        output_cost = (self.output_tokens / 1_000_000) * model_cost["output"]

        # Cache reads are typically 90% discount
        cache_read_cost = (self.cache_read_tokens / 1_000_000) * model_cost["input"] * 0.1

        # Cache creation same as input
        cache_creation_cost = (self.cache_creation_tokens / 1_000_000) * model_cost["input"]

        return input_cost + output_cost + cache_read_cost + cache_creation_cost


@dataclass
class BudgetInfo:
    """Current budget status information.

    Attributes:
        total_spent: Total cost spent so far (USD).
        max_cost: Maximum budget (USD).
        percentage_used: Percentage of budget used (0.0-1.0+).
        status: Current budget status level.
        projected_iterations: Estimated remaining iterations.
        cost_per_iteration: Average cost per iteration.
        iterations_completed: Number of iterations completed.
        should_downgrade: Whether to suggest model downgrade.
    """
    total_spent: float
    max_cost: float
    percentage_used: float
    status: BudgetStatus
    projected_iterations: int
    cost_per_iteration: float
    iterations_completed: int
    should_downgrade: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_spent": round(self.total_spent, 4),
            "max_cost": self.max_cost,
            "percentage_used": round(self.percentage_used, 4),
            "status": self.status.value,
            "projected_iterations": self.projected_iterations,
            "cost_per_iteration": round(self.cost_per_iteration, 4),
            "iterations_completed": self.iterations_completed,
            "should_downgrade": self.should_downgrade,
        }


class BudgetTracker:
    """Tracks token usage and budget consumption.

    Attributes:
        max_cost: Maximum budget in USD.
        total_spent: Total cost spent so far.
        usage_history: List of TokenUsage records.
        model_costs: Cost per million tokens by model.
        downgrade_threshold: Percentage at which to suggest downgrade.
        warning_threshold: Percentage for warning status.
        critical_threshold: Percentage for critical status.
    """

    def __init__(
        self,
        max_cost: float,
        model_costs: dict[str, dict[str, float]] | None = None,
        downgrade_threshold: float = 0.70,
        warning_threshold: float = 0.50,
        critical_threshold: float = 0.80,
    ):
        self.max_cost = max_cost
        self.total_spent = 0.0
        self.usage_history: list[TokenUsage] = []
        self.model_costs = model_costs or MODEL_COSTS
        self.downgrade_threshold = downgrade_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def track_usage(self, usage: TokenUsage | dict[str, Any]) -> BudgetInfo:
        """Track token usage and return current status.

        Args:
            usage: TokenUsage or dictionary with usage data.

        Returns:
            BudgetInfo with current budget status.
        """
        # Convert dict to TokenUsage if needed
        if isinstance(usage, dict):
            usage = TokenUsage.from_dict(usage)

        # Calculate cost for this usage
        cost = usage.calculate_cost(self.model_costs)
        self.total_spent += cost

        # Add to history
        self.usage_history.append(usage)

        logger.debug(
            f"Tracked usage: {usage.input_tokens} in, {usage.output_tokens} out, "
            f"${cost:.4f} (total: ${self.total_spent:.4f})"
        )

        return self.get_status()

    def get_status(self) -> BudgetInfo:
        """Get current budget status.

        Returns:
            BudgetInfo with current state.
        """
        percentage = self.get_percentage_used()

        # Determine status
        if percentage >= 1.0:
            status = BudgetStatus.EXCEEDED
        elif percentage >= self.critical_threshold:
            status = BudgetStatus.CRITICAL
        elif percentage >= self.warning_threshold:
            status = BudgetStatus.WARNING
        else:
            status = BudgetStatus.SAFE

        # Calculate cost per iteration
        iterations = len(self.usage_history)
        cost_per_iter = self.total_spent / iterations if iterations > 0 else 0.0

        # Project remaining iterations
        remaining_budget = self.max_cost - self.total_spent
        projected = int(remaining_budget / cost_per_iter) if cost_per_iter > 0 else 0

        # Should downgrade?
        should_downgrade = percentage >= self.downgrade_threshold and status != BudgetStatus.EXCEEDED

        return BudgetInfo(
            total_spent=self.total_spent,
            max_cost=self.max_cost,
            percentage_used=percentage,
            status=status,
            projected_iterations=max(0, projected),
            cost_per_iteration=cost_per_iter,
            iterations_completed=iterations,
            should_downgrade=should_downgrade,
        )

    def get_percentage_used(self) -> float:
        """Get percentage of budget used (0.0-1.0+).

        Returns:
            Percentage as a float (can exceed 1.0).
        """
        if self.max_cost <= 0:
            return 0.0
        return self.total_spent / self.max_cost

    def project_remaining_iterations(self) -> int:
        """Estimate how many iterations remain before budget exhaustion.

        Returns:
            Estimated remaining iterations.
        """
        status = self.get_status()
        return status.projected_iterations

    def should_downgrade_model(self) -> bool:
        """Check if should suggest downgrading to cheaper model.

        Returns:
            True if should downgrade.
        """
        status = self.get_status()
        return status.should_downgrade

    def get_display_string(self, color: bool = False) -> str:
        """Get formatted display string for budget status.

        Args:
            color: Whether to include ANSI color codes.

        Returns:
            Formatted string like "Budget: 35% used ($3.50/$10.00) - ~15 iterations remaining"
        """
        status = self.get_status()

        # Format percentage
        pct_str = f"{status.percentage_used * 100:.0f}%"

        # Format costs
        spent_str = f"${status.total_spent:.2f}"
        max_str = f"${status.max_cost:.2f}"

        # Format iterations
        if status.projected_iterations > 0:
            iter_str = f"~{status.projected_iterations} iterations remaining"
        else:
            iter_str = "budget exhausted"

        base_str = f"Budget: {pct_str} used ({spent_str}/{max_str}) - {iter_str}"

        if not color:
            return base_str

        # Add color codes based on status
        if status.status == BudgetStatus.SAFE:
            return f"\033[92m{base_str}\033[0m"  # Green
        elif status.status == BudgetStatus.WARNING:
            return f"\033[93m{base_str}\033[0m"  # Yellow
        elif status.status == BudgetStatus.CRITICAL:
            return f"\033[91m{base_str}\033[0m"  # Red
        else:  # EXCEEDED
            return f"\033[91m\033[1m{base_str}\033[0m"  # Bold red

    def get_cost_breakdown(self) -> dict[str, Any]:
        """Get detailed cost breakdown by model.

        Returns:
            Dictionary with cost breakdown.
        """
        by_model: dict[str, dict[str, Any]] = {}

        for usage in self.usage_history:
            model = usage.model
            if model not in by_model:
                by_model[model] = {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }

            by_model[model]["count"] += 1
            by_model[model]["input_tokens"] += usage.input_tokens
            by_model[model]["output_tokens"] += usage.output_tokens
            by_model[model]["cost"] += usage.calculate_cost(self.model_costs)

        return {
            "total_cost": self.total_spent,
            "by_model": by_model,
            "total_iterations": len(self.usage_history),
        }

    def suggest_budget_adjustment(self) -> str | None:
        """Suggest budget adjustment if needed.

        Returns:
            Suggestion string or None.
        """
        status = self.get_status()

        if status.status == BudgetStatus.EXCEEDED:
            return (
                f"Budget exceeded! Spent ${status.total_spent:.2f} of ${status.max_cost:.2f}. "
                "Consider increasing budget with --max-cost"
            )

        if status.status == BudgetStatus.CRITICAL:
            if status.projected_iterations < 3:
                return (
                    f"Budget critically low ({status.percentage_used:.0%}). "
                    f"Only ~{status.projected_iterations} iterations remaining. "
                    "Consider increasing budget or using cheaper models."
                )

        if status.should_downgrade:
            return (
                f"Budget at {status.percentage_used:.0%}. "
                "Consider using 'haiku' model to conserve budget."
            )

        return None

    def reset(self) -> None:
        """Reset budget tracker (for testing)."""
        self.total_spent = 0.0
        self.usage_history.clear()
