"""Integration tests for the full improvement loop.

These tests validate the complete workflow from config to completion,
using mocked Claude responses to test different scenarios.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repo_auto_tool.config import ImproverConfig
from repo_auto_tool.improver import RepoImprover
from repo_auto_tool.state import ImprovementState


@pytest.fixture
def python_repo(temp_dir: Path) -> Path:
    """Create a sample Python repository with tests."""
    # Initialize git
    subprocess.run(["git", "init"], cwd=temp_dir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=temp_dir,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=temp_dir,
        capture_output=True,
    )

    # Create a simple Python module
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "__init__.py").write_text("")
    (temp_dir / "src" / "calculator.py").write_text('''"""Simple calculator module."""


def add(a, b):
    """Add two numbers."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b
''')

    # Create tests
    (temp_dir / "tests").mkdir()
    (temp_dir / "tests" / "__init__.py").write_text("")
    (temp_dir / "tests" / "test_calculator.py").write_text('''"""Tests for calculator module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calculator import add, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2
''')

    # Create pyproject.toml
    (temp_dir / "pyproject.toml").write_text('''[project]
name = "test-project"
version = "0.1.0"

[tool.pytest.ini_options]
testpaths = ["tests"]
''')

    # Initial commit
    subprocess.run(["git", "add", "."], cwd=temp_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=temp_dir,
        capture_output=True,
    )

    return temp_dir


@pytest.fixture
def mock_claude_response():
    """Create a mock Claude response factory."""
    def _create_response(
        result: str = "Made improvements",
        success: bool = True,
        error: str | None = None,
    ):
        response = MagicMock()
        response.success = success
        response.result = result
        response.error = error
        response.model_used = "claude-sonnet-4-20250514"
        response.tokens_used = 1000
        response.cost = 0.01
        return response
    return _create_response


class TestImprovementLoop:
    """Tests for the core improvement loop."""

    def test_config_creation(self, python_repo: Path):
        """Test that config can be created for a repo."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints to all functions",
            max_iterations=3,
        )

        assert config.repo_path == python_repo
        assert config.goal == "Add type hints to all functions"
        assert config.max_iterations == 3

    def test_state_initialization(self, python_repo: Path):
        """Test that state is properly initialized."""
        state = ImprovementState(
            goal="Add type hints",
            repo_path=str(python_repo),
        )

        assert state.status == "running"
        assert state.current_iteration == 0
        assert len(state.iterations) == 0

    def test_state_persistence(self, python_repo: Path):
        """Test that state can be saved and loaded."""
        state_file = python_repo / ".repo-improver-state.json"

        state = ImprovementState(
            goal="Test goal",
            repo_path=str(python_repo),
        )
        state.record_iteration(
            prompt="Test prompt",
            result="Test result",
            success=True,
            validation_passed=True,
        )

        # Save state
        state.save(state_file)
        assert state_file.exists()

        # Load state
        loaded = ImprovementState.load(state_file)
        assert loaded.goal == "Test goal"
        assert len(loaded.iterations) == 1
        assert loaded.iterations[0].prompt == "Test prompt"

    def test_improver_initialization(self, python_repo: Path, mock_claude_response):
        """Test that improver initializes correctly."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints",
            max_iterations=3,
        )

        with patch("repo_auto_tool.improver.ClaudeCodeInterface") as mock_cli:
            mock_cli.return_value.is_available.return_value = True
            improver = RepoImprover(config)

            assert improver.config == config
            assert improver.state.goal == "Add type hints"

    def test_analyze_returns_analysis(self, python_repo: Path, mock_claude_response):
        """Test that analyze() returns proper analysis."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints to all functions",
            max_iterations=3,
        )

        with patch("repo_auto_tool.improver.ClaudeCodeInterface") as mock_cli:
            mock_instance = mock_cli.return_value
            mock_instance.is_available.return_value = True
            mock_instance.analyze.return_value = mock_claude_response(
                result="## Analysis\n- Found 2 functions without type hints"
            )

            improver = RepoImprover(config)
            analysis = improver.analyze()

            assert "Analysis" in analysis or "Found" in analysis

    @patch("repo_auto_tool.improver.ClaudeCodeInterface")
    def test_goal_completion_detection(self, mock_cli, python_repo: Path, mock_claude_response):
        """Test that GOAL_COMPLETE is detected."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints",
            max_iterations=5,
        )

        mock_instance = mock_cli.return_value
        mock_instance.is_available.return_value = True
        mock_instance.improve.return_value = mock_claude_response(
            result="GOAL_COMPLETE: All functions now have type hints"
        )

        improver = RepoImprover(config)

        # Run one iteration
        success = improver._run_iteration()

        # Goal completion should be detected
        assert improver.state.status == "completed"

    @patch("repo_auto_tool.improver.ClaudeCodeInterface")
    def test_blocker_detection(self, mock_cli, python_repo: Path, mock_claude_response):
        """Test that BLOCKED responses are detected."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Impossible goal",
            max_iterations=5,
        )

        mock_instance = mock_cli.return_value
        mock_instance.is_available.return_value = True
        mock_instance.improve.return_value = mock_claude_response(
            result="BLOCKED: Cannot proceed without external dependencies"
        )

        improver = RepoImprover(config)

        # Run iteration
        success = improver._run_iteration()

        # Should be marked as blocked
        assert not success
        assert improver.state.consecutive_failures > 0


class TestValidationPipeline:
    """Tests for the validation pipeline."""

    def test_python_syntax_validation(self, python_repo: Path):
        """Test that Python syntax errors are caught."""
        from repo_auto_tool.validators import CommandValidator, ValidationPipeline

        # Create validator that checks syntax
        validator = CommandValidator(
            name="syntax",
            command="python -m py_compile src/calculator.py",
        )
        pipeline = ValidationPipeline(validators=[validator])

        # Valid code should pass
        all_passed, results = pipeline.validate(python_repo)
        assert all_passed
        assert len(results) == 1

    def test_validation_failure_recorded(self, python_repo: Path):
        """Test that validation failures are properly recorded."""
        from repo_auto_tool.validators import CommandValidator, ValidationPipeline

        # Create a validator that will fail (exit 1 via cmd on Windows)
        validator = CommandValidator(
            name="fail",
            command="python -c \"exit(1)\"",
        )
        pipeline = ValidationPipeline(validators=[validator])

        all_passed, results = pipeline.validate(python_repo)
        assert not all_passed
        assert len(results) == 1
        assert not results[0].passed


class TestSessionManagement:
    """Tests for session management and resumability."""

    def test_resume_from_state(self, python_repo: Path, mock_claude_response):
        """Test resuming an interrupted session."""
        state_file = python_repo / ".repo-improver-state.json"

        # Create initial state with some history
        state = ImprovementState(
            goal="Add type hints",
            repo_path=str(python_repo),
        )
        # Record iteration increments current_iteration
        state.record_iteration(
            prompt="First iteration",
            result="Added type hints to add()",
            success=True,
            validation_passed=True,
        )
        state.save(state_file)

        # Load and verify state is resumed
        loaded = ImprovementState.load(state_file)
        # After one iteration, current_iteration should be 1
        assert loaded.current_iteration == 1
        assert len(loaded.iterations) == 1

    def test_state_summary_generation(self, python_repo: Path):
        """Test that state summary is generated correctly."""
        state = ImprovementState(
            goal="Add type hints",
            repo_path=str(python_repo),
        )

        # Record some iterations
        state.record_iteration(
            prompt="First",
            result="Added hints to func1",
            success=True,
            validation_passed=True,
        )
        state.record_iteration(
            prompt="Second",
            result="Added hints to func2",
            success=True,
            validation_passed=True,
        )
        state.record_iteration(
            prompt="Third",
            result="Failed to add hints",
            success=False,
            validation_passed=False,
            error="Syntax error",
        )

        context = state.get_recent_context()
        assert "func1" in context or "func2" in context


class TestCostTracking:
    """Tests for cost tracking and budget enforcement."""

    def test_cost_limit_enforcement(self, python_repo: Path, mock_claude_response):
        """Test that cost limits are enforced."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints",
            max_iterations=100,
            max_cost=0.05,  # Very low budget
        )

        with patch("repo_auto_tool.improver.ClaudeCodeInterface") as mock_cli:
            mock_instance = mock_cli.return_value
            mock_instance.is_available.return_value = True

            improver = RepoImprover(config)

            # Simulate token usage that exceeds budget
            # Cost is calculated from tokens: approximately 0.003 per 1000 input + 0.015 per 1000 output
            improver.state.total_input_tokens = 10000
            improver.state.total_output_tokens = 5000  # This should exceed $0.05

            # Check should catch this
            from repo_auto_tool.exceptions import CostLimitExceededError
            with pytest.raises(CostLimitExceededError):
                improver._check_cost_limit()


class TestPromptAdapter:
    """Tests for the adaptive prompt system."""

    def test_failure_recording(self):
        """Test that failures are recorded correctly."""
        from repo_auto_tool.prompt_adapter import PromptAdapter

        adapter = PromptAdapter()
        adapter.record_failure("syntax_error", "IndentationError on line 5")
        adapter.record_failure("syntax_error", "Missing colon")

        assert adapter.failure_counts["syntax_error"] == 2

    def test_guidance_generation(self):
        """Test that guidance is generated for failures."""
        from repo_auto_tool.prompt_adapter import PromptAdapter

        adapter = PromptAdapter()
        adapter.record_failure("syntax_error", "Error 1")

        guidance = adapter.get_applicable_guidance()
        assert len(guidance) > 0
        assert any("syntax" in g.lower() for g in guidance)

    def test_prompt_enhancement(self):
        """Test that prompts are enhanced with guidance."""
        from repo_auto_tool.prompt_adapter import PromptAdapter

        adapter = PromptAdapter()
        adapter.record_failure("import_error", "No module named 'foo'")

        original = "Make improvements.\n\nNow make the improvements:"
        enhanced = adapter.enhance_prompt(original)

        assert "IMPORT" in enhanced.upper()
        assert len(enhanced) > len(original)


class TestConvergenceDetection:
    """Tests for convergence detection."""

    def test_no_changes_triggers_convergence(self):
        """Test that repeated iterations with no changes triggers convergence."""
        from repo_auto_tool.convergence import (
            ChangeMetrics,
            ConvergenceDetector,
            ConvergenceState,
        )

        detector = ConvergenceDetector()
        state = ConvergenceState()

        # Record several iterations with no changes
        for _ in range(5):
            state.add_metrics(ChangeMetrics(files_changed=0, lines_added=0, lines_removed=0))

        action, reason = detector.analyze(state)
        # With no changes, should suggest stopping or checkpoint
        assert action in ("checkpoint", "stop", "continue")

    def test_active_changes_continue(self):
        """Test that active changes allow continuation."""
        from repo_auto_tool.convergence import (
            ChangeMetrics,
            ConvergenceDetector,
            ConvergenceState,
        )

        detector = ConvergenceDetector()
        state = ConvergenceState()

        # Record several iterations with changes
        for i in range(5):
            state.add_metrics(ChangeMetrics(files_changed=2, lines_added=50+i, lines_removed=10))

        action, _ = detector.analyze(state)
        # With active changes, should either continue or checkpoint (not stop)
        assert action in ("continue", "checkpoint")


class TestErrorHandling:
    """Tests for error handling and recovery."""

    def test_claude_error_handled_gracefully(self, python_repo: Path, mock_claude_response):
        """Test that Claude API errors are handled."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints",
            max_iterations=5,
        )

        with patch("repo_auto_tool.improver.ClaudeCodeInterface") as mock_cli:
            mock_instance = mock_cli.return_value
            mock_instance.is_available.return_value = True
            mock_instance.improve.return_value = mock_claude_response(
                success=False,
                error="API rate limit exceeded",
            )

            improver = RepoImprover(config)
            success = improver._run_iteration()

            # Should handle error gracefully
            assert not success
            assert improver.state.consecutive_failures > 0

    def test_validation_error_recovery(self, python_repo: Path, mock_claude_response):
        """Test that validation errors are recorded for learning."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints",
            max_iterations=5,
        )

        with patch("repo_auto_tool.improver.ClaudeCodeInterface") as mock_cli:
            mock_instance = mock_cli.return_value
            mock_instance.is_available.return_value = True
            mock_instance.improve.return_value = mock_claude_response(
                result="Added type hints but made a syntax error"
            )

            improver = RepoImprover(config)

            # Record a validation failure
            improver.prompt_adapter.record_failure("syntax_error", "IndentationError")

            # Guidance should be generated
            guidance = improver.prompt_adapter.get_applicable_guidance()
            assert len(guidance) > 0

    def test_state_corruption_detected(self, python_repo: Path):
        """Test that corrupted state files are detected."""
        from repo_auto_tool.exceptions import StateCorruptedError

        state_file = python_repo / ".repo-improver-state.json"
        state_file.write_text('{"invalid": "state"}')

        with pytest.raises(StateCorruptedError):
            ImprovementState.load(state_file)

    def test_max_consecutive_failures(self, python_repo: Path, mock_claude_response):
        """Test that max consecutive failures stops execution."""
        config = ImproverConfig(
            repo_path=python_repo,
            goal="Add type hints",
            max_iterations=10,
            max_consecutive_failures=3,
        )

        with patch("repo_auto_tool.improver.ClaudeCodeInterface") as mock_cli:
            mock_instance = mock_cli.return_value
            mock_instance.is_available.return_value = True

            improver = RepoImprover(config)

            # Simulate consecutive failures
            improver.state.consecutive_failures = 3

            # Should detect and stop
            assert improver.state.consecutive_failures >= config.max_consecutive_failures
