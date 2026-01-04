"""Session reporting and progress tracking.

Provides comprehensive reports at session end and real-time progress tracking.
"""

import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import ImprovementState, IterationRecord

logger = logging.getLogger(__name__)


class ChangeCategory(Enum):
    """Categories of changes made during a session."""
    FEATURE = "feature"      # New functionality
    FIX = "fix"              # Bug fixes
    REFACTOR = "refactor"    # Code restructuring
    TEST = "test"            # Test additions/changes
    DOCS = "docs"            # Documentation
    STYLE = "style"          # Formatting/linting fixes
    OTHER = "other"          # Uncategorized


@dataclass
class IterationSummary:
    """Summary of a single iteration."""
    iteration: int
    timestamp: str
    success: bool
    validation_passed: bool
    description: str
    files_changed: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    category: ChangeCategory = ChangeCategory.OTHER
    error: str | None = None


@dataclass
class DiffSummary:
    """Grouped summary of all changes."""
    features: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    refactors: list[str] = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)

    total_files_changed: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0


@dataclass
class ProgressEstimate:
    """Estimate of progress toward goal completion."""
    percent_complete: float  # 0-100
    iterations_remaining: int  # Estimated
    confidence: str  # "low", "medium", "high"
    reasoning: str


class SessionReporter:
    """Generate comprehensive session reports and track progress."""

    def __init__(self, repo_path: Path, goal: str):
        """Initialize the session reporter.

        Args:
            repo_path: Path to the repository.
            goal: The improvement goal.
        """
        self.repo_path = repo_path
        self.goal = goal
        self._iteration_summaries: list[IterationSummary] = []

    def record_iteration(
        self,
        iteration: "IterationRecord",
        files_changed: list[str] | None = None,
        lines_added: int = 0,
        lines_removed: int = 0,
    ) -> IterationSummary:
        """Record an iteration for reporting.

        Args:
            iteration: The iteration record from state.
            files_changed: List of files modified in this iteration.
            lines_added: Lines added in this iteration.
            lines_removed: Lines removed in this iteration.

        Returns:
            The created IterationSummary.
        """
        # Categorize the change based on content
        category = self._categorize_change(iteration.result)

        # Extract a short description from the result
        description = self._extract_description(iteration.result)

        summary = IterationSummary(
            iteration=iteration.iteration,
            timestamp=iteration.timestamp,
            success=iteration.success,
            validation_passed=iteration.validation_passed,
            description=description,
            files_changed=files_changed or [],
            lines_added=lines_added,
            lines_removed=lines_removed,
            category=category,
            error=iteration.error,
        )

        self._iteration_summaries.append(summary)
        return summary

    def _categorize_change(self, result: str) -> ChangeCategory:
        """Categorize a change based on its description.

        Args:
            result: The Claude response describing the change.

        Returns:
            The appropriate ChangeCategory.
        """
        result_lower = result.lower()

        # Check for keywords indicating category
        if any(kw in result_lower for kw in ["test", "pytest", "unittest", "spec"]):
            return ChangeCategory.TEST
        elif any(kw in result_lower for kw in ["fix", "bug", "error", "issue", "patch"]):
            return ChangeCategory.FIX
        elif any(kw in result_lower for kw in ["refactor", "restructure", "reorganize", "clean"]):
            return ChangeCategory.REFACTOR
        elif any(kw in result_lower for kw in ["doc", "readme", "comment", "docstring"]):
            return ChangeCategory.DOCS
        elif any(kw in result_lower for kw in ["style", "format", "lint", "ruff"]):
            return ChangeCategory.STYLE
        elif any(kw in result_lower for kw in ["add", "implement", "create", "feature", "new"]):
            return ChangeCategory.FEATURE
        else:
            return ChangeCategory.OTHER

    def _extract_description(self, result: str, max_length: int = 100) -> str:
        """Extract a short description from Claude's response.

        Args:
            result: The full Claude response.
            max_length: Maximum length of description.

        Returns:
            A concise description of the change.
        """
        # Skip GOAL_COMPLETE or BLOCKED prefixes
        text = result
        for prefix in ["GOAL_COMPLETE:", "BLOCKED:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Take first sentence or line
        for delimiter in [".", "\n", ";"]:
            if delimiter in text:
                text = text.split(delimiter)[0]
                break

        text = text.strip()

        if len(text) > max_length:
            text = text[:max_length - 3] + "..."

        return text

    def get_diff_summary(self) -> DiffSummary:
        """Get grouped summary of all changes.

        Returns:
            A DiffSummary with changes grouped by category.
        """
        summary = DiffSummary()

        for it_summary in self._iteration_summaries:
            if not it_summary.success or not it_summary.validation_passed:
                continue

            # Add to appropriate category
            if it_summary.category == ChangeCategory.FEATURE:
                summary.features.append(it_summary.description)
            elif it_summary.category == ChangeCategory.FIX:
                summary.fixes.append(it_summary.description)
            elif it_summary.category == ChangeCategory.REFACTOR:
                summary.refactors.append(it_summary.description)
            elif it_summary.category == ChangeCategory.TEST:
                summary.tests.append(it_summary.description)
            elif it_summary.category == ChangeCategory.DOCS:
                summary.docs.append(it_summary.description)
            else:
                summary.other.append(it_summary.description)

            # Aggregate metrics
            summary.total_files_changed += len(it_summary.files_changed)
            summary.total_lines_added += it_summary.lines_added
            summary.total_lines_removed += it_summary.lines_removed

        return summary

    def estimate_progress(self, state: "ImprovementState") -> ProgressEstimate:
        """Estimate progress toward goal completion.

        Uses heuristics based on:
        - Number of successful iterations
        - Recent validation success rate
        - Whether goal-completion markers have been seen

        Args:
            state: Current improvement state.

        Returns:
            A ProgressEstimate with percentage and reasoning.
        """
        if state.status == "completed":
            return ProgressEstimate(
                percent_complete=100.0,
                iterations_remaining=0,
                confidence="high",
                reasoning="Goal was marked as complete by Claude",
            )

        if state.status == "failed":
            return ProgressEstimate(
                percent_complete=0.0,
                iterations_remaining=-1,
                confidence="high",
                reasoning="Session failed",
            )

        # Calculate success rate
        total_iterations = len(state.iterations)
        if total_iterations == 0:
            return ProgressEstimate(
                percent_complete=0.0,
                iterations_remaining=-1,
                confidence="low",
                reasoning="No iterations completed yet",
            )

        successful = sum(1 for it in state.iterations if it.validation_passed)
        success_rate = successful / total_iterations

        # Check recent iterations for trends
        recent = state.iterations[-5:] if len(state.iterations) >= 5 else state.iterations
        recent_success_rate = sum(1 for it in recent if it.validation_passed) / len(recent)

        # Estimate percent based on iterations and success rate
        # Assume average goal takes ~10 successful iterations
        estimated_total_iterations = 10
        percent = min(100.0, (successful / estimated_total_iterations) * 100)

        # Adjust confidence based on data points
        if total_iterations < 3:
            confidence = "low"
        elif total_iterations < 7:
            confidence = "medium"
        else:
            confidence = "high"

        # Estimate remaining iterations
        if success_rate > 0:
            remaining_successes_needed = max(0, estimated_total_iterations - successful)
            iterations_remaining = int(remaining_successes_needed / success_rate)
        else:
            iterations_remaining = -1  # Unknown

        # Build reasoning
        if recent_success_rate >= 0.8:
            trend = "Recent progress is strong"
        elif recent_success_rate >= 0.5:
            trend = "Progress is steady"
        elif recent_success_rate >= 0.2:
            trend = "Encountering some difficulties"
        else:
            trend = "Currently struggling with validation"

        reasoning = (
            f"{successful}/{total_iterations} iterations successful ({success_rate:.0%}). {trend}."
        )

        return ProgressEstimate(
            percent_complete=round(percent, 1),
            iterations_remaining=iterations_remaining,
            confidence=confidence,
            reasoning=reasoning,
        )

    def get_progress_bar(self, state: "ImprovementState", width: int = 40) -> str:
        """Get a visual progress bar string.

        Args:
            state: Current improvement state.
            width: Width of the progress bar in characters.

        Returns:
            A string like "[████████░░░░░░░░░░░░] 40%"
        """
        estimate = self.estimate_progress(state)
        percent = estimate.percent_complete

        filled = int((percent / 100) * width)
        empty = width - filled

        bar = "█" * filled + "░" * empty
        return f"[{bar}] {percent:.0f}%"

    def generate_iteration_log(self, verbose: bool = False) -> str:
        """Generate detailed iteration-by-iteration log.

        Args:
            verbose: If True, include full details. If False, summary only.

        Returns:
            Formatted log string.
        """
        if not self._iteration_summaries:
            return "No iterations recorded."

        lines = ["## Iteration Log\n"]

        for summary in self._iteration_summaries:
            status = "✓" if summary.validation_passed else "✗"
            category_emoji = {
                ChangeCategory.FEATURE: "✨",
                ChangeCategory.FIX: "🔧",
                ChangeCategory.REFACTOR: "♻️",
                ChangeCategory.TEST: "🧪",
                ChangeCategory.DOCS: "📝",
                ChangeCategory.STYLE: "💅",
                ChangeCategory.OTHER: "📦",
            }.get(summary.category, "📦")

            lines.append(
                f"### Iteration {summary.iteration} {status} {category_emoji}"
            )
            lines.append(f"- **Time**: {summary.timestamp}")
            lines.append(f"- **Status**: {'Passed' if summary.validation_passed else 'Failed'}")
            lines.append(f"- **Description**: {summary.description}")

            if verbose:
                if summary.files_changed:
                    lines.append(f"- **Files**: {', '.join(summary.files_changed)}")
                if summary.lines_added or summary.lines_removed:
                    lines.append(
                        f"- **Changes**: +{summary.lines_added} / -{summary.lines_removed} lines"
                    )
                if summary.error:
                    lines.append(f"- **Error**: {summary.error}")

            lines.append("")

        return "\n".join(lines)

    def generate_report(
        self,
        state: "ImprovementState",
        include_git_diff: bool = True,
    ) -> str:
        """Generate comprehensive end-of-session report.

        Args:
            state: The final improvement state.
            include_git_diff: Whether to include git diff output.

        Returns:
            Markdown-formatted report string.
        """
        report_lines = [
            "# Session Report",
            f"\n**Generated**: {datetime.now().isoformat()}",
            f"\n**Repository**: {self.repo_path}",
            f"\n**Goal**: {self.goal}",
            f"\n**Status**: {state.status.upper()}",
            "",
        ]

        # Progress
        estimate = self.estimate_progress(state)
        report_lines.extend([
            "## Progress",
            f"- **Completion**: {estimate.percent_complete:.0f}%",
            f"- **Confidence**: {estimate.confidence}",
            f"- **Assessment**: {estimate.reasoning}",
            "",
        ])

        # Metrics
        successful = sum(1 for it in state.iterations if it.validation_passed)
        failed = len(state.iterations) - successful

        report_lines.extend([
            "## Metrics",
            f"- **Total Iterations**: {len(state.iterations)}",
            f"- **Successful**: {successful}",
            f"- **Failed**: {failed}",
            f"- **Success Rate**: {(successful / max(1, len(state.iterations))):.0%}",
            f"- **Started**: {state.started_at}",
            f"- **Completed**: {state.completed_at or 'In progress'}",
            "",
        ])

        # Token usage
        token_summary = state.get_token_summary()
        report_lines.extend([
            "## Token Usage",
            f"- **Input Tokens**: {token_summary['input_tokens']:,}",
            f"- **Output Tokens**: {token_summary['output_tokens']:,}",
            f"- **Total Tokens**: {token_summary['total_tokens']:,}",
            f"- **Cache Read**: {token_summary.get('cache_read_tokens', 0):,}",
            "",
        ])

        # Diff summary
        diff_summary = self.get_diff_summary()
        report_lines.extend([
            "## Changes Summary",
            f"- **Files Changed**: {diff_summary.total_files_changed}",
            f"- **Lines Added**: +{diff_summary.total_lines_added}",
            f"- **Lines Removed**: -{diff_summary.total_lines_removed}",
            "",
        ])

        # Categorized changes
        if any([diff_summary.features, diff_summary.fixes, diff_summary.refactors,
                diff_summary.tests, diff_summary.docs, diff_summary.other]):
            report_lines.append("### Changes by Category")

            if diff_summary.features:
                report_lines.append("\n**Features:**")
                for item in diff_summary.features:
                    report_lines.append(f"- {item}")

            if diff_summary.fixes:
                report_lines.append("\n**Fixes:**")
                for item in diff_summary.fixes:
                    report_lines.append(f"- {item}")

            if diff_summary.refactors:
                report_lines.append("\n**Refactors:**")
                for item in diff_summary.refactors:
                    report_lines.append(f"- {item}")

            if diff_summary.tests:
                report_lines.append("\n**Tests:**")
                for item in diff_summary.tests:
                    report_lines.append(f"- {item}")

            if diff_summary.docs:
                report_lines.append("\n**Documentation:**")
                for item in diff_summary.docs:
                    report_lines.append(f"- {item}")

            if diff_summary.other:
                report_lines.append("\n**Other:**")
                for item in diff_summary.other:
                    report_lines.append(f"- {item}")

            report_lines.append("")

        # Iteration log
        report_lines.append(self.generate_iteration_log(verbose=False))

        # Git diff (if requested and available)
        if include_git_diff:
            diff_output = self._get_git_diff()
            if diff_output:
                report_lines.extend([
                    "## Git Diff",
                    "```diff",
                    diff_output[:5000],  # Limit diff output
                    "```",
                    "",
                ])

        # Final notes
        if state.summary:
            report_lines.extend([
                "## Summary",
                state.summary,
                "",
            ])

        return "\n".join(report_lines)

    def _get_git_diff(self) -> str | None:
        """Get git diff from repository.

        Returns:
            Git diff output string, or None if unavailable.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD~1", "--stat"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.debug(f"Could not get git diff: {e}")
        return None

    def save_report(
        self,
        state: "ImprovementState",
        output_path: Path | None = None,
    ) -> Path:
        """Save report to a file.

        Args:
            state: The improvement state.
            output_path: Where to save. Defaults to repo_path/.session-report.md

        Returns:
            Path to the saved report.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.repo_path / f".session-report-{timestamp}.md"

        report = self.generate_report(state)
        output_path.write_text(report, encoding="utf-8")

        logger.info(f"Session report saved to {output_path}")
        return output_path
