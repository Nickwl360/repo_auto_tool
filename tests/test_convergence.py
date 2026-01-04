"""Tests for the convergence module."""

from pathlib import Path

from repo_auto_tool.convergence import (
    ChangeMetrics,
    ChangeTracker,
    ConvergenceConfig,
    ConvergenceDetector,
    ConvergenceState,
)


class TestChangeMetrics:
    """Tests for the ChangeMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test ChangeMetrics default values."""
        metrics = ChangeMetrics()
        assert metrics.files_changed == 0
        assert metrics.lines_added == 0
        assert metrics.lines_removed == 0

    def test_custom_values(self) -> None:
        """Test ChangeMetrics with custom values."""
        metrics = ChangeMetrics(files_changed=5, lines_added=100, lines_removed=50)
        assert metrics.files_changed == 5
        assert metrics.lines_added == 100
        assert metrics.lines_removed == 50

    def test_total_lines_changed(self) -> None:
        """Test total_lines_changed property."""
        metrics = ChangeMetrics(lines_added=100, lines_removed=50)
        assert metrics.total_lines_changed == 150

    def test_total_lines_changed_zero(self) -> None:
        """Test total_lines_changed when no changes."""
        metrics = ChangeMetrics()
        assert metrics.total_lines_changed == 0

    def test_is_empty_true(self) -> None:
        """Test is_empty returns True when no changes."""
        metrics = ChangeMetrics()
        assert metrics.is_empty is True

    def test_is_empty_false_with_files(self) -> None:
        """Test is_empty returns False when files changed."""
        metrics = ChangeMetrics(files_changed=1)
        assert metrics.is_empty is False

    def test_is_empty_false_with_lines(self) -> None:
        """Test is_empty returns False when lines changed."""
        metrics = ChangeMetrics(lines_added=1)
        assert metrics.is_empty is False

        metrics2 = ChangeMetrics(lines_removed=1)
        assert metrics2.is_empty is False


class TestConvergenceConfig:
    """Tests for the ConvergenceConfig dataclass."""

    def test_default_values(self) -> None:
        """Test ConvergenceConfig default values."""
        config = ConvergenceConfig()
        assert config.min_iterations == 3
        assert config.convergence_threshold == 0.1
        assert config.plateau_window == 3
        assert config.checkpoint_interval == 5
        assert config.early_stop_enabled is True

    def test_custom_values(self) -> None:
        """Test ConvergenceConfig with custom values."""
        config = ConvergenceConfig(
            min_iterations=5,
            convergence_threshold=0.5,
            plateau_window=4,
            checkpoint_interval=10,
            early_stop_enabled=False,
        )
        assert config.min_iterations == 5
        assert config.convergence_threshold == 0.5
        assert config.plateau_window == 4
        assert config.checkpoint_interval == 10
        assert config.early_stop_enabled is False


class TestConvergenceState:
    """Tests for the ConvergenceState dataclass."""

    def test_default_values(self) -> None:
        """Test ConvergenceState default values."""
        state = ConvergenceState()
        assert state.history == []
        assert state.iteration == 0
        assert state.converged is False
        assert state.convergence_reason == ""

    def test_add_metrics(self) -> None:
        """Test adding metrics increments iteration."""
        state = ConvergenceState()
        metrics = ChangeMetrics(files_changed=1, lines_added=10)
        state.add_metrics(metrics)
        assert state.iteration == 1
        assert len(state.history) == 1
        assert state.history[0] is metrics

    def test_add_multiple_metrics(self) -> None:
        """Test adding multiple metrics."""
        state = ConvergenceState()
        for i in range(5):
            state.add_metrics(ChangeMetrics(lines_added=i * 10))
        assert state.iteration == 5
        assert len(state.history) == 5

    def test_get_recent_metrics_empty(self) -> None:
        """Test get_recent_metrics on empty state."""
        state = ConvergenceState()
        recent = state.get_recent_metrics(3)
        assert recent == []

    def test_get_recent_metrics_partial(self) -> None:
        """Test get_recent_metrics with fewer than window."""
        state = ConvergenceState()
        state.add_metrics(ChangeMetrics(lines_added=10))
        state.add_metrics(ChangeMetrics(lines_added=20))
        recent = state.get_recent_metrics(5)
        assert len(recent) == 2

    def test_get_recent_metrics_full(self) -> None:
        """Test get_recent_metrics with more than window."""
        state = ConvergenceState()
        for i in range(10):
            state.add_metrics(ChangeMetrics(lines_added=i * 10))
        recent = state.get_recent_metrics(3)
        assert len(recent) == 3
        # Should be the last 3
        assert recent[0].lines_added == 70
        assert recent[1].lines_added == 80
        assert recent[2].lines_added == 90

    def test_average_change_rate_empty(self) -> None:
        """Test average_change_rate on empty state."""
        state = ConvergenceState()
        assert state.average_change_rate(3) == 0.0

    def test_average_change_rate_single(self) -> None:
        """Test average_change_rate with single iteration."""
        state = ConvergenceState()
        state.add_metrics(ChangeMetrics(lines_added=10, lines_removed=5))
        assert state.average_change_rate(3) == 15.0

    def test_average_change_rate_multiple(self) -> None:
        """Test average_change_rate with multiple iterations."""
        state = ConvergenceState()
        state.add_metrics(ChangeMetrics(lines_added=10))  # 10 total
        state.add_metrics(ChangeMetrics(lines_added=20))  # 20 total
        state.add_metrics(ChangeMetrics(lines_added=30))  # 30 total
        # Average of (10 + 20 + 30) / 3 = 20
        assert state.average_change_rate(3) == 20.0


class TestConvergenceDetector:
    """Tests for the ConvergenceDetector class."""

    def test_default_config(self) -> None:
        """Test detector uses default config when none provided."""
        detector = ConvergenceDetector()
        assert detector.config.min_iterations == 3

    def test_custom_config(self) -> None:
        """Test detector uses custom config."""
        config = ConvergenceConfig(min_iterations=10)
        detector = ConvergenceDetector(config)
        assert detector.config.min_iterations == 10

    def test_analyze_before_min_iterations(self) -> None:
        """Test analyze returns continue before min iterations."""
        detector = ConvergenceDetector(ConvergenceConfig(min_iterations=5))
        state = ConvergenceState()
        for _ in range(3):
            state.add_metrics(ChangeMetrics())
        action, reason = detector.analyze(state)
        assert action == "continue"
        assert "Minimum iterations not reached" in reason

    def test_analyze_checkpoint_interval(self) -> None:
        """Test analyze returns checkpoint at interval."""
        config = ConvergenceConfig(
            min_iterations=3,
            checkpoint_interval=5,
            convergence_threshold=0.0,  # Never converge by threshold
        )
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        # Add 5 iterations with activity
        for _ in range(5):
            state.add_metrics(ChangeMetrics(lines_added=100))
        action, reason = detector.analyze(state)
        assert action == "checkpoint"
        assert "Checkpoint at iteration 5" in reason

    def test_analyze_converged_low_change_rate(self) -> None:
        """Test analyze detects convergence from low change rate."""
        config = ConvergenceConfig(
            min_iterations=3,
            convergence_threshold=1.0,
            plateau_window=3,
            checkpoint_interval=0,  # Disable checkpoints
        )
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        # Add iterations with very low changes
        for _ in range(5):
            state.add_metrics(ChangeMetrics())  # All empty
        action, reason = detector.analyze(state)
        assert action == "stop"
        assert "Change rate" in reason or "No changes" in reason

    def test_analyze_converged_empty_iterations(self) -> None:
        """Test analyze detects convergence from empty iterations."""
        config = ConvergenceConfig(
            min_iterations=3,
            plateau_window=3,
            checkpoint_interval=0,
        )
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        # Add empty iterations
        for _ in range(5):
            state.add_metrics(ChangeMetrics())
        action, reason = detector.analyze(state)
        assert action == "stop"

    def test_analyze_continue_with_activity(self) -> None:
        """Test analyze continues when changes are being made."""
        config = ConvergenceConfig(
            min_iterations=3,
            convergence_threshold=0.1,
            checkpoint_interval=0,
        )
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        # Add iterations with significant changes
        for _ in range(5):
            state.add_metrics(ChangeMetrics(lines_added=100))
        action, reason = detector.analyze(state)
        assert action == "continue"
        assert "Changes still being made" in reason

    def test_analyze_early_stop_disabled(self) -> None:
        """Test analyze returns checkpoint when early stop disabled."""
        config = ConvergenceConfig(
            min_iterations=3,
            checkpoint_interval=0,
            early_stop_enabled=False,
        )
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        # Add empty iterations (would normally trigger stop)
        for _ in range(5):
            state.add_metrics(ChangeMetrics())
        action, reason = detector.analyze(state)
        assert action == "checkpoint"
        assert "early stop disabled" in reason

    def test_should_stop(self) -> None:
        """Test should_stop convenience method."""
        config = ConvergenceConfig(
            min_iterations=3,
            checkpoint_interval=0,
        )
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        # Add empty iterations
        for _ in range(5):
            state.add_metrics(ChangeMetrics())
        assert detector.should_stop(state) is True

    def test_should_stop_false(self) -> None:
        """Test should_stop returns False when not converged."""
        config = ConvergenceConfig(min_iterations=3, checkpoint_interval=0)
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        # Add iterations with activity
        for _ in range(5):
            state.add_metrics(ChangeMetrics(lines_added=100))
        assert detector.should_stop(state) is False

    def test_should_checkpoint(self) -> None:
        """Test should_checkpoint method."""
        config = ConvergenceConfig(
            min_iterations=3,
            checkpoint_interval=5,
            convergence_threshold=0.0,
        )
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        for _ in range(5):
            state.add_metrics(ChangeMetrics(lines_added=100))
        assert detector.should_checkpoint(state) is True

    def test_should_checkpoint_at_stop(self) -> None:
        """Test should_checkpoint returns True when stopping."""
        config = ConvergenceConfig(min_iterations=3, checkpoint_interval=0)
        detector = ConvergenceDetector(config)
        state = ConvergenceState()
        for _ in range(5):
            state.add_metrics(ChangeMetrics())
        assert detector.should_checkpoint(state) is True


class TestChangeTracker:
    """Tests for the ChangeTracker class."""

    def test_init(self) -> None:
        """Test ChangeTracker initialization."""
        tracker = ChangeTracker(Path("/some/repo"))
        assert tracker.repo_path == Path("/some/repo")

    def test_parse_numstat_empty(self) -> None:
        """Test parsing empty numstat output."""
        tracker = ChangeTracker(Path("/repo"))
        metrics = tracker._parse_numstat("")
        assert metrics.files_changed == 0
        assert metrics.lines_added == 0
        assert metrics.lines_removed == 0

    def test_parse_numstat_single_file(self) -> None:
        """Test parsing numstat with single file."""
        tracker = ChangeTracker(Path("/repo"))
        output = "10\t5\tsrc/main.py"
        metrics = tracker._parse_numstat(output)
        assert metrics.files_changed == 1
        assert metrics.lines_added == 10
        assert metrics.lines_removed == 5

    def test_parse_numstat_multiple_files(self) -> None:
        """Test parsing numstat with multiple files."""
        tracker = ChangeTracker(Path("/repo"))
        output = "10\t5\tsrc/main.py\n20\t15\tsrc/utils.py\n5\t0\ttests/test_main.py"
        metrics = tracker._parse_numstat(output)
        assert metrics.files_changed == 3
        assert metrics.lines_added == 35  # 10 + 20 + 5
        assert metrics.lines_removed == 20  # 5 + 15 + 0

    def test_parse_numstat_binary_file(self) -> None:
        """Test parsing numstat with binary file."""
        tracker = ChangeTracker(Path("/repo"))
        output = "-\t-\timage.png\n10\t5\tsrc/main.py"
        metrics = tracker._parse_numstat(output)
        assert metrics.files_changed == 2
        assert metrics.lines_added == 10  # Binary files don't count
        assert metrics.lines_removed == 5

    def test_parse_numstat_malformed_line(self) -> None:
        """Test parsing numstat with malformed lines."""
        tracker = ChangeTracker(Path("/repo"))
        output = "not valid\n10\t5\tsrc/main.py\nalso invalid"
        metrics = tracker._parse_numstat(output)
        # Should only count the valid line
        assert metrics.files_changed == 1
        assert metrics.lines_added == 10

    def test_get_diff_stats_handles_missing_repo(self) -> None:
        """Test get_diff_stats returns empty metrics on error."""
        tracker = ChangeTracker(Path("/nonexistent/repo"))
        metrics = tracker.get_diff_stats()
        assert metrics.is_empty

    def test_get_staged_stats_handles_missing_repo(self) -> None:
        """Test get_staged_stats returns empty metrics on error."""
        tracker = ChangeTracker(Path("/nonexistent/repo"))
        metrics = tracker.get_staged_stats()
        assert metrics.is_empty

    def test_get_unstaged_stats_handles_missing_repo(self) -> None:
        """Test get_unstaged_stats returns empty metrics on error."""
        tracker = ChangeTracker(Path("/nonexistent/repo"))
        metrics = tracker.get_unstaged_stats()
        assert metrics.is_empty


class TestChangeTrackerWithRealRepo:
    """Tests for ChangeTracker with a real git repository.

    These tests use the sample_repo fixture from conftest.py.
    """

    def test_get_diff_stats_real_repo(self, sample_repo: Path) -> None:
        """Test get_diff_stats on a real git repo."""
        tracker = ChangeTracker(sample_repo)
        # Initially, no changes compared to HEAD~1 (we only have one commit)
        # This will fail gracefully and return empty metrics
        metrics = tracker.get_diff_stats()
        # Should return metrics (possibly empty) without raising
        assert isinstance(metrics, ChangeMetrics)

    def test_get_staged_stats_real_repo(self, sample_repo: Path) -> None:
        """Test get_staged_stats on a real git repo."""
        tracker = ChangeTracker(sample_repo)
        # No staged changes initially
        metrics = tracker.get_staged_stats()
        assert metrics.is_empty

    def test_get_unstaged_stats_real_repo(self, sample_repo: Path) -> None:
        """Test get_unstaged_stats on a real git repo."""
        tracker = ChangeTracker(sample_repo)
        # No unstaged changes initially
        metrics = tracker.get_unstaged_stats()
        assert metrics.is_empty

    def test_get_unstaged_stats_with_changes(self, sample_repo: Path) -> None:
        """Test get_unstaged_stats after modifying a file."""
        # Modify the README
        readme = sample_repo / "README.md"
        readme.write_text("# Updated Test Repo\n\nNew content\n")

        tracker = ChangeTracker(sample_repo)
        metrics = tracker.get_unstaged_stats()
        assert metrics.files_changed >= 1
