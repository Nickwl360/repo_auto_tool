"""Tests for session reporting and progress tracking."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from repo_auto_tool.session_report import (
    ChangeCategory,
    DiffSummary,
    IterationSummary,
    ProgressEstimate,
    SessionReporter,
)


@pytest.fixture
def mock_iteration():
    """Create a mock iteration record."""
    def _create(
        iteration: int = 1,
        success: bool = True,
        validation_passed: bool = True,
        result: str = "Added new feature",
        error: str | None = None,
    ):
        record = MagicMock()
        record.iteration = iteration
        record.timestamp = datetime.now().isoformat()
        record.success = success
        record.validation_passed = validation_passed
        record.result = result
        record.error = error
        return record
    return _create


@pytest.fixture
def mock_state():
    """Create a mock improvement state."""
    def _create(
        status: str = "running",
        iterations: list | None = None,
    ):
        state = MagicMock()
        state.status = status
        state.iterations = iterations or []
        state.started_at = datetime.now().isoformat()
        state.completed_at = None
        state.summary = None
        state.get_token_summary.return_value = {
            "input_tokens": 10000,
            "output_tokens": 5000,
            "total_tokens": 15000,
            "cache_read_tokens": 1000,
        }
        return state
    return _create


class TestChangeCategory:
    """Tests for change categorization."""

    def test_categorize_feature(self, temp_dir: Path):
        """Test that feature keywords are detected."""
        reporter = SessionReporter(temp_dir, "Add feature")
        category = reporter._categorize_change("Added new authentication feature")
        assert category == ChangeCategory.FEATURE

    def test_categorize_fix(self, temp_dir: Path):
        """Test that fix keywords are detected."""
        reporter = SessionReporter(temp_dir, "Fix bug")
        category = reporter._categorize_change("Fixed the login bug")
        assert category == ChangeCategory.FIX

    def test_categorize_test(self, temp_dir: Path):
        """Test that test keywords are detected."""
        reporter = SessionReporter(temp_dir, "Add tests")
        category = reporter._categorize_change("Added pytest unit tests")
        assert category == ChangeCategory.TEST

    def test_categorize_refactor(self, temp_dir: Path):
        """Test that refactor keywords are detected."""
        reporter = SessionReporter(temp_dir, "Refactor")
        category = reporter._categorize_change("Refactored the authentication module")
        assert category == ChangeCategory.REFACTOR

    def test_categorize_docs(self, temp_dir: Path):
        """Test that documentation keywords are detected."""
        reporter = SessionReporter(temp_dir, "Add docs")
        category = reporter._categorize_change("Updated docstrings and README")
        assert category == ChangeCategory.DOCS

    def test_categorize_other(self, temp_dir: Path):
        """Test that unknown changes are categorized as other."""
        reporter = SessionReporter(temp_dir, "Goal")
        category = reporter._categorize_change("Did something")
        assert category == ChangeCategory.OTHER


class TestDescriptionExtraction:
    """Tests for description extraction."""

    def test_extract_first_sentence(self, temp_dir: Path):
        """Test that first sentence is extracted."""
        reporter = SessionReporter(temp_dir, "Goal")
        desc = reporter._extract_description(
            "Added new login feature. This includes OAuth support."
        )
        assert desc == "Added new login feature"

    def test_extract_truncates_long(self, temp_dir: Path):
        """Test that long descriptions are truncated."""
        reporter = SessionReporter(temp_dir, "Goal")
        long_text = "x" * 200
        desc = reporter._extract_description(long_text, max_length=50)
        assert len(desc) == 50
        assert desc.endswith("...")

    def test_handles_goal_complete_prefix(self, temp_dir: Path):
        """Test that GOAL_COMPLETE prefix is removed."""
        reporter = SessionReporter(temp_dir, "Goal")
        desc = reporter._extract_description(
            "GOAL_COMPLETE: All tasks finished successfully"
        )
        assert "GOAL_COMPLETE" not in desc
        assert "finished" in desc.lower()


class TestIterationRecording:
    """Tests for iteration recording."""

    def test_record_iteration(self, temp_dir: Path, mock_iteration):
        """Test that iterations are recorded."""
        reporter = SessionReporter(temp_dir, "Goal")
        iteration = mock_iteration()

        summary = reporter.record_iteration(
            iteration,
            files_changed=["file.py"],
            lines_added=10,
            lines_removed=5,
        )

        assert summary.iteration == 1
        assert summary.success is True
        assert summary.files_changed == ["file.py"]
        assert summary.lines_added == 10

    def test_multiple_iterations_recorded(self, temp_dir: Path, mock_iteration):
        """Test that multiple iterations are accumulated."""
        reporter = SessionReporter(temp_dir, "Goal")

        reporter.record_iteration(mock_iteration(iteration=1))
        reporter.record_iteration(mock_iteration(iteration=2))
        reporter.record_iteration(mock_iteration(iteration=3))

        assert len(reporter._iteration_summaries) == 3


class TestDiffSummary:
    """Tests for diff summary generation."""

    def test_empty_summary(self, temp_dir: Path):
        """Test diff summary with no iterations."""
        reporter = SessionReporter(temp_dir, "Goal")
        summary = reporter.get_diff_summary()

        assert len(summary.features) == 0
        assert summary.total_files_changed == 0

    def test_categorized_summary(self, temp_dir: Path, mock_iteration):
        """Test that changes are properly categorized in summary."""
        reporter = SessionReporter(temp_dir, "Goal")

        # Record various types
        reporter.record_iteration(
            mock_iteration(result="Added new feature"),
            files_changed=["feature.py"],
            lines_added=50,
        )
        reporter.record_iteration(
            mock_iteration(result="Fixed bug in login"),
            files_changed=["login.py"],
            lines_added=5,
            lines_removed=10,
        )
        reporter.record_iteration(
            mock_iteration(result="Added pytest tests"),
            files_changed=["test_feature.py"],
            lines_added=100,
        )

        summary = reporter.get_diff_summary()

        assert len(summary.features) == 1
        assert len(summary.fixes) == 1
        assert len(summary.tests) == 1
        assert summary.total_lines_added == 155

    def test_failed_iterations_excluded(self, temp_dir: Path, mock_iteration):
        """Test that failed iterations are excluded from summary."""
        reporter = SessionReporter(temp_dir, "Goal")

        reporter.record_iteration(
            mock_iteration(result="Added feature", validation_passed=True),
            lines_added=50,
        )
        reporter.record_iteration(
            mock_iteration(result="Failed change", validation_passed=False),
            lines_added=100,
        )

        summary = reporter.get_diff_summary()
        assert summary.total_lines_added == 50  # Only successful one


class TestProgressEstimation:
    """Tests for progress estimation."""

    def test_completed_state(self, temp_dir: Path, mock_state):
        """Test progress for completed goal."""
        reporter = SessionReporter(temp_dir, "Goal")
        state = mock_state(status="completed")

        estimate = reporter.estimate_progress(state)

        assert estimate.percent_complete == 100.0
        assert estimate.iterations_remaining == 0
        assert estimate.confidence == "high"

    def test_no_iterations(self, temp_dir: Path, mock_state):
        """Test progress with no iterations."""
        reporter = SessionReporter(temp_dir, "Goal")
        state = mock_state(status="running", iterations=[])

        estimate = reporter.estimate_progress(state)

        assert estimate.percent_complete == 0.0
        assert estimate.confidence == "low"

    def test_partial_progress(self, temp_dir: Path, mock_state, mock_iteration):
        """Test progress estimation with some iterations."""
        reporter = SessionReporter(temp_dir, "Goal")

        # Create mock iterations
        iterations = [
            mock_iteration(iteration=i, validation_passed=(i % 2 == 0))
            for i in range(1, 6)
        ]
        state = mock_state(status="running", iterations=iterations)

        estimate = reporter.estimate_progress(state)

        assert 0 < estimate.percent_complete < 100
        assert estimate.reasoning != ""


class TestProgressBar:
    """Tests for progress bar generation."""

    def test_empty_progress(self, temp_dir: Path, mock_state):
        """Test progress bar at 0%."""
        reporter = SessionReporter(temp_dir, "Goal")
        state = mock_state(status="running", iterations=[])

        bar = reporter.get_progress_bar(state, width=20)

        assert "░" * 20 in bar
        assert "0%" in bar

    def test_completed_progress(self, temp_dir: Path, mock_state):
        """Test progress bar at 100%."""
        reporter = SessionReporter(temp_dir, "Goal")
        state = mock_state(status="completed")

        bar = reporter.get_progress_bar(state, width=20)

        assert "█" * 20 in bar
        assert "100%" in bar


class TestIterationLog:
    """Tests for iteration log generation."""

    def test_empty_log(self, temp_dir: Path):
        """Test log with no iterations."""
        reporter = SessionReporter(temp_dir, "Goal")
        log = reporter.generate_iteration_log()

        assert "No iterations" in log

    def test_log_with_iterations(self, temp_dir: Path, mock_iteration):
        """Test log generation with iterations."""
        reporter = SessionReporter(temp_dir, "Goal")
        reporter.record_iteration(mock_iteration(iteration=1))
        reporter.record_iteration(mock_iteration(iteration=2, validation_passed=False))

        log = reporter.generate_iteration_log()

        assert "Iteration 1" in log
        assert "Iteration 2" in log
        assert "✓" in log  # Success marker
        assert "✗" in log  # Failure marker


class TestReportGeneration:
    """Tests for full report generation."""

    def test_generate_report(self, temp_dir: Path, mock_state, mock_iteration):
        """Test full report generation."""
        reporter = SessionReporter(temp_dir, "Add feature X")

        # Record some iterations
        reporter.record_iteration(mock_iteration(iteration=1, result="Added feature"))
        reporter.record_iteration(mock_iteration(iteration=2, result="Added tests"))

        iterations = [mock_iteration(i) for i in range(1, 3)]
        state = mock_state(status="completed", iterations=iterations)
        state.summary = "Goal completed successfully"

        report = reporter.generate_report(state, include_git_diff=False)

        assert "# Session Report" in report
        assert "Add feature X" in report
        assert "COMPLETED" in report
        assert "Progress" in report
        assert "Metrics" in report
        assert "Token Usage" in report

    def test_report_includes_summary(self, temp_dir: Path, mock_state):
        """Test that summary is included in report."""
        reporter = SessionReporter(temp_dir, "Goal")
        state = mock_state(status="completed")
        state.summary = "This is the final summary"

        report = reporter.generate_report(state, include_git_diff=False)

        assert "This is the final summary" in report


class TestReportSaving:
    """Tests for report file saving."""

    def test_save_report(self, temp_dir: Path, mock_state):
        """Test saving report to file."""
        reporter = SessionReporter(temp_dir, "Goal")
        state = mock_state(status="completed")

        output_path = reporter.save_report(state)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Session Report" in content

    def test_save_custom_path(self, temp_dir: Path, mock_state):
        """Test saving report to custom path."""
        reporter = SessionReporter(temp_dir, "Goal")
        state = mock_state(status="completed")
        custom_path = temp_dir / "custom-report.md"

        output_path = reporter.save_report(state, output_path=custom_path)

        assert output_path == custom_path
        assert custom_path.exists()
