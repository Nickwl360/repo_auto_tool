"""Convergence detection for improvement loops."""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ChangeMetrics:
    """Metrics about changes made in an iteration."""

    files_changed: int = 0
    lines_added: int = 0
    lines_removed: int = 0

    @property
    def total_lines_changed(self) -> int:
        """Total lines modified (added + removed)."""
        return self.lines_added + self.lines_removed

    @property
    def is_empty(self) -> bool:
        """Check if no changes were made."""
        return self.files_changed == 0 and self.total_lines_changed == 0


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection."""

    # Minimum iterations before checking for convergence
    min_iterations: int = 3

    # Threshold: if average change rate drops below this, consider converged
    convergence_threshold: float = 0.1

    # Number of recent iterations to consider for plateau detection
    plateau_window: int = 3

    # Checkpoint every N iterations
    checkpoint_interval: int = 5

    # Enable early stopping when converged
    early_stop_enabled: bool = True


@dataclass
class ConvergenceState:
    """Tracks convergence state over multiple iterations."""

    # History of change metrics per iteration
    history: list[ChangeMetrics] = field(default_factory=list)

    # Current iteration number
    iteration: int = 0

    # Whether convergence was detected
    converged: bool = False

    # Reason for convergence (if any)
    convergence_reason: str = ""

    def add_metrics(self, metrics: ChangeMetrics) -> None:
        """Add metrics for a completed iteration.

        Args:
            metrics: The change metrics for the iteration.
        """
        self.history.append(metrics)
        self.iteration += 1

    def get_recent_metrics(self, window: int) -> list[ChangeMetrics]:
        """Get metrics for the most recent iterations.

        Args:
            window: Number of recent iterations to return.

        Returns:
            List of recent ChangeMetrics (may be fewer than window).
        """
        return self.history[-window:] if self.history else []

    def average_change_rate(self, window: int) -> float:
        """Calculate average total lines changed over recent iterations.

        Args:
            window: Number of recent iterations to average.

        Returns:
            Average lines changed per iteration.
        """
        recent = self.get_recent_metrics(window)
        if not recent:
            return 0.0
        total = sum(m.total_lines_changed for m in recent)
        return total / len(recent)


# Recommendation type for what action to take
ConvergenceAction = Literal["continue", "checkpoint", "stop"]


class ChangeTracker:
    """Tracks changes using git diff statistics."""

    def __init__(self, repo_path: Path) -> None:
        """Initialize the change tracker.

        Args:
            repo_path: Path to the git repository.
        """
        self.repo_path = repo_path

    def get_diff_stats(self, ref: str = "HEAD~1") -> ChangeMetrics:
        """Get diff statistics compared to a reference.

        Args:
            ref: Git reference to compare against (default: previous commit).

        Returns:
            ChangeMetrics with files changed, lines added/removed.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--stat", "--numstat", ref],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return ChangeMetrics()

            return self._parse_numstat(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ChangeMetrics()

    def get_staged_stats(self) -> ChangeMetrics:
        """Get statistics for staged changes.

        Returns:
            ChangeMetrics for currently staged changes.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--numstat"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return ChangeMetrics()

            return self._parse_numstat(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ChangeMetrics()

    def get_unstaged_stats(self) -> ChangeMetrics:
        """Get statistics for unstaged changes.

        Returns:
            ChangeMetrics for current unstaged changes.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--numstat"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return ChangeMetrics()

            return self._parse_numstat(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return ChangeMetrics()

    def _parse_numstat(self, output: str) -> ChangeMetrics:
        """Parse git diff --numstat output.

        Args:
            output: Raw output from git diff --numstat.

        Returns:
            Parsed ChangeMetrics.
        """
        files_changed = 0
        lines_added = 0
        lines_removed = 0

        for line in output.strip().split("\n"):
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                try:
                    # numstat format: added<TAB>removed<TAB>filename
                    # Binary files show "-" for counts
                    added = parts[0]
                    removed = parts[1]

                    if added != "-":
                        lines_added += int(added)
                    if removed != "-":
                        lines_removed += int(removed)
                    files_changed += 1
                except ValueError:
                    # Skip lines that don't parse correctly
                    pass

        return ChangeMetrics(
            files_changed=files_changed,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )


class ConvergenceDetector:
    """Detects when improvement loops have converged."""

    def __init__(self, config: ConvergenceConfig | None = None) -> None:
        """Initialize the convergence detector.

        Args:
            config: Configuration for detection thresholds.
        """
        self.config = config or ConvergenceConfig()

    def analyze(self, state: ConvergenceState) -> tuple[ConvergenceAction, str]:
        """Analyze convergence state and recommend an action.

        Args:
            state: Current convergence state with history.

        Returns:
            Tuple of (recommended action, reason string).
        """
        # Not enough history yet
        if state.iteration < self.config.min_iterations:
            return "continue", "Minimum iterations not reached"

        # Check if we should checkpoint
        if (
            self.config.checkpoint_interval > 0
            and state.iteration % self.config.checkpoint_interval == 0
        ):
            # Still check for convergence first
            if self._is_converged(state):
                return "stop", self._convergence_reason(state)
            return "checkpoint", f"Checkpoint at iteration {state.iteration}"

        # Check for convergence
        if self._is_converged(state):
            if self.config.early_stop_enabled:
                return "stop", self._convergence_reason(state)
            reason = self._convergence_reason(state)
            return "checkpoint", f"Converged but early stop disabled: {reason}"

        return "continue", "Changes still being made"

    def _is_converged(self, state: ConvergenceState) -> bool:
        """Check if the improvement loop has converged.

        Args:
            state: Current convergence state.

        Returns:
            True if converged based on configured thresholds.
        """
        # Check for plateau (very low change rate)
        avg_rate = state.average_change_rate(self.config.plateau_window)
        if avg_rate <= self.config.convergence_threshold:
            return True

        # Check for multiple empty iterations
        recent = state.get_recent_metrics(self.config.plateau_window)
        if len(recent) >= self.config.plateau_window:
            empty_count = sum(1 for m in recent if m.is_empty)
            if empty_count >= self.config.plateau_window:
                return True

        return False

    def _convergence_reason(self, state: ConvergenceState) -> str:
        """Generate a human-readable convergence reason.

        Args:
            state: Current convergence state.

        Returns:
            Description of why convergence was detected.
        """
        avg_rate = state.average_change_rate(self.config.plateau_window)

        if avg_rate <= self.config.convergence_threshold:
            return (
                f"Change rate ({avg_rate:.2f} lines/iteration) "
                f"below threshold ({self.config.convergence_threshold})"
            )

        recent = state.get_recent_metrics(self.config.plateau_window)
        empty_count = sum(1 for m in recent if m.is_empty)
        if empty_count >= self.config.plateau_window:
            return f"No changes in last {self.config.plateau_window} iterations"

        return "Convergence detected"

    def should_stop(self, state: ConvergenceState) -> bool:
        """Simple check if the loop should stop.

        Args:
            state: Current convergence state.

        Returns:
            True if early stop is enabled and convergence detected.
        """
        action, _ = self.analyze(state)
        return action == "stop"

    def should_checkpoint(self, state: ConvergenceState) -> bool:
        """Check if a checkpoint should be created.

        Args:
            state: Current convergence state.

        Returns:
            True if a checkpoint is recommended.
        """
        action, _ = self.analyze(state)
        return action in ("checkpoint", "stop")
