"""Tests for the exceptions module."""

from pathlib import Path

import pytest

from repo_auto_tool.exceptions import (
    BudgetError,
    ClaudeInterfaceError,
    ClaudeNotFoundError,
    ClaudeResponseError,
    ClaudeTimeoutError,
    ConfigurationError,
    CostLimitExceededError,
    GitError,
    GitNotInitializedError,
    GitOperationError,
    InvalidConfigValueError,
    InvalidPathError,
    LintFailureError,
    PromptParseError,
    RepoAutoToolError,
    StateCorruptedError,
    StateError,
    StateLoadError,
    StateSaveError,
    SyntaxError_,
    TestFailureError,
    ValidationError,
)


class TestRepoAutoToolError:
    """Tests for the base exception class."""

    def test_basic_message(self) -> None:
        """Test exception with just a message."""
        error = RepoAutoToolError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.details == {}

    def test_message_with_details(self) -> None:
        """Test exception with message and details."""
        error = RepoAutoToolError("Failed", details={"code": 42, "file": "test.py"})
        assert error.message == "Failed"
        assert error.details == {"code": 42, "file": "test.py"}
        assert "code=42" in str(error)
        assert "file='test.py'" in str(error)

    def test_none_details_becomes_empty_dict(self) -> None:
        """Test that None details becomes empty dict."""
        error = RepoAutoToolError("Error", details=None)
        assert error.details == {}


class TestConfigurationErrors:
    """Tests for configuration-related exceptions."""

    def test_invalid_path_error(self) -> None:
        """Test InvalidPathError with string path."""
        error = InvalidPathError("/some/path", "does not exist")
        assert error.path == Path("/some/path")
        assert error.reason == "does not exist"
        assert "Invalid path" in str(error)
        assert "some" in str(error) and "path" in str(error)
        assert "does not exist" in str(error)

    def test_invalid_path_error_with_path_object(self) -> None:
        """Test InvalidPathError with Path object."""
        path = Path("/another/path")
        error = InvalidPathError(path, "not readable")
        assert error.path == path
        assert error.details["path"] == str(path)

    def test_invalid_config_value_error(self) -> None:
        """Test InvalidConfigValueError."""
        error = InvalidConfigValueError("max_iterations", -5, "must be positive")
        assert error.field == "max_iterations"
        assert error.value == -5
        assert error.reason == "must be positive"
        assert "max_iterations" in str(error)

    def test_inheritance(self) -> None:
        """Test that ConfigurationError inherits from RepoAutoToolError."""
        error = ConfigurationError("Config issue")
        assert isinstance(error, RepoAutoToolError)
        error2 = InvalidPathError("/path", "reason")
        assert isinstance(error2, ConfigurationError)
        assert isinstance(error2, RepoAutoToolError)


class TestStateErrors:
    """Tests for state management exceptions."""

    def test_state_load_error(self) -> None:
        """Test StateLoadError."""
        error = StateLoadError("/state.json", "file not found")
        assert error.path == Path("/state.json")
        assert error.reason == "file not found"
        assert "Failed to load state" in str(error)

    def test_state_save_error(self) -> None:
        """Test StateSaveError."""
        error = StateSaveError("/state.json", "permission denied")
        assert error.path == Path("/state.json")
        assert "Failed to save state" in str(error)

    def test_state_corrupted_error(self) -> None:
        """Test StateCorruptedError."""
        error = StateCorruptedError("/state.json", "invalid JSON")
        assert "Corrupted state file" in str(error)
        assert "invalid JSON" in str(error)

    def test_inheritance(self) -> None:
        """Test StateError inheritance chain."""
        assert isinstance(StateLoadError("/p", "r"), StateError)
        assert isinstance(StateSaveError("/p", "r"), StateError)
        assert isinstance(StateCorruptedError("/p", "r"), StateError)
        assert isinstance(StateError("msg"), RepoAutoToolError)


class TestValidationErrors:
    """Tests for validation-related exceptions."""

    def test_test_failure_error(self) -> None:
        """Test TestFailureError."""
        error = TestFailureError("pytest tests/", "FAILED test_foo.py::test_bar")
        assert error.command == "pytest tests/"
        assert error.output == "FAILED test_foo.py::test_bar"
        assert "Tests failed" in str(error)

    def test_test_failure_truncates_output(self) -> None:
        """Test that TestFailureError truncates long output."""
        long_output = "x" * 1000
        error = TestFailureError("pytest", long_output)
        assert len(error.details["output"]) == 500

    def test_lint_failure_error(self) -> None:
        """Test LintFailureError."""
        error = LintFailureError("ruff check .", "E501: line too long")
        assert error.command == "ruff check ."
        assert "Linting failed" in str(error)

    def test_syntax_error(self) -> None:
        """Test SyntaxError_ (custom named to avoid builtin conflict)."""
        errors = ["line 5: unexpected indent", "line 10: invalid syntax"]
        error = SyntaxError_("test.py", errors)
        assert error.file_path == Path("test.py")
        assert error.errors == errors
        assert "Syntax errors" in str(error)

    def test_inheritance(self) -> None:
        """Test ValidationError inheritance."""
        assert isinstance(TestFailureError("cmd", "out"), ValidationError)
        assert isinstance(LintFailureError("cmd", "out"), ValidationError)
        assert isinstance(SyntaxError_("f.py", []), ValidationError)
        assert isinstance(ValidationError("msg"), RepoAutoToolError)


class TestClaudeInterfaceErrors:
    """Tests for Claude Code CLI integration exceptions."""

    def test_claude_not_found_error(self) -> None:
        """Test ClaudeNotFoundError."""
        error = ClaudeNotFoundError()
        assert "Claude Code CLI not found" in str(error)
        assert "npm install" in error.details["install_command"]

    def test_claude_timeout_error(self) -> None:
        """Test ClaudeTimeoutError."""
        error = ClaudeTimeoutError(300, "Please improve the code...")
        assert error.timeout_seconds == 300
        assert error.prompt == "Please improve the code..."
        assert "300 seconds" in str(error)
        # Prompt preview is truncated to 100 chars
        assert len(error.details["prompt_preview"]) <= 100

    def test_claude_response_error_without_output(self) -> None:
        """Test ClaudeResponseError without raw output."""
        error = ClaudeResponseError("empty response")
        assert error.reason == "empty response"
        assert error.raw_output is None
        assert "output_preview" not in error.details

    def test_claude_response_error_with_output(self) -> None:
        """Test ClaudeResponseError with raw output."""
        error = ClaudeResponseError("parse failed", raw_output="garbage data")
        assert error.raw_output == "garbage data"
        assert "output_preview" in error.details

    def test_inheritance(self) -> None:
        """Test ClaudeInterfaceError inheritance."""
        assert isinstance(ClaudeNotFoundError(), ClaudeInterfaceError)
        assert isinstance(ClaudeTimeoutError(10, "p"), ClaudeInterfaceError)
        assert isinstance(ClaudeResponseError("r"), ClaudeInterfaceError)
        assert isinstance(ClaudeInterfaceError("msg"), RepoAutoToolError)


class TestGitErrors:
    """Tests for Git-related exceptions."""

    def test_git_not_initialized_error(self) -> None:
        """Test GitNotInitializedError."""
        error = GitNotInitializedError("/some/dir")
        assert error.path == Path("/some/dir")
        assert "not a git repository" in str(error)

    def test_git_operation_error_without_output(self) -> None:
        """Test GitOperationError without output."""
        error = GitOperationError("commit", "nothing to commit")
        assert error.operation == "commit"
        assert error.reason == "nothing to commit"
        assert error.output is None
        assert "output" not in error.details

    def test_git_operation_error_with_output(self) -> None:
        """Test GitOperationError with output."""
        error = GitOperationError("push", "rejected", output="non-fast-forward")
        assert error.output == "non-fast-forward"
        assert "output" in error.details

    def test_inheritance(self) -> None:
        """Test GitError inheritance."""
        assert isinstance(GitNotInitializedError("/p"), GitError)
        assert isinstance(GitOperationError("op", "r"), GitError)
        assert isinstance(GitError("msg"), RepoAutoToolError)


class TestBudgetErrors:
    """Tests for budget-related exceptions."""

    def test_cost_limit_exceeded_error(self) -> None:
        """Test CostLimitExceededError."""
        error = CostLimitExceededError(
            current_cost=5.50,
            max_cost=5.00,
            tokens_used=150000,
        )
        assert error.current_cost == 5.50
        assert error.max_cost == 5.00
        assert error.tokens_used == 150000
        assert "$5.5" in str(error) or "5.50" in str(error)
        assert "$5.00" in str(error)

    def test_inheritance(self) -> None:
        """Test BudgetError inheritance."""
        assert isinstance(CostLimitExceededError(1, 1, 1), BudgetError)
        assert isinstance(BudgetError("msg"), RepoAutoToolError)


class TestPromptParseError:
    """Tests for prompt parsing exceptions."""

    def test_without_file_path(self) -> None:
        """Test PromptParseError without file path."""
        error = PromptParseError("empty prompt content")
        assert error.file_path is None
        assert "file_path" not in error.details
        assert "empty prompt content" in str(error)

    def test_with_file_path(self) -> None:
        """Test PromptParseError with file path."""
        error = PromptParseError("invalid YAML", file_path="/goals.yaml")
        assert error.file_path == "/goals.yaml"
        assert error.details["file_path"] == "/goals.yaml"

    def test_inheritance(self) -> None:
        """Test PromptParseError inherits from RepoAutoToolError."""
        assert isinstance(PromptParseError("msg"), RepoAutoToolError)


class TestExceptionCatching:
    """Test that exceptions can be caught at various hierarchy levels."""

    def test_catch_all_repo_auto_tool_errors(self) -> None:
        """Test that all exceptions can be caught as RepoAutoToolError."""
        exceptions = [
            RepoAutoToolError("base"),
            ConfigurationError("config"),
            InvalidPathError("/p", "r"),
            InvalidConfigValueError("f", "v", "r"),
            StateError("state"),
            StateLoadError("/p", "r"),
            StateSaveError("/p", "r"),
            StateCorruptedError("/p", "r"),
            ValidationError("validation"),
            TestFailureError("cmd", "out"),
            LintFailureError("cmd", "out"),
            SyntaxError_("f.py", []),
            ClaudeInterfaceError("claude"),
            ClaudeNotFoundError(),
            ClaudeTimeoutError(10, "p"),
            ClaudeResponseError("r"),
            GitError("git"),
            GitNotInitializedError("/p"),
            GitOperationError("op", "r"),
            BudgetError("budget"),
            CostLimitExceededError(1, 1, 1),
            PromptParseError("prompt"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except RepoAutoToolError as e:
                assert e is exc
            else:
                pytest.fail(f"{type(exc).__name__} not caught as RepoAutoToolError")

    def test_catch_specific_categories(self) -> None:
        """Test that specific error categories can be caught."""
        # Config errors
        with pytest.raises(ConfigurationError):
            raise InvalidPathError("/p", "r")

        # State errors
        with pytest.raises(StateError):
            raise StateLoadError("/p", "r")

        # Validation errors
        with pytest.raises(ValidationError):
            raise TestFailureError("cmd", "out")

        # Claude errors
        with pytest.raises(ClaudeInterfaceError):
            raise ClaudeTimeoutError(10, "p")

        # Git errors
        with pytest.raises(GitError):
            raise GitOperationError("op", "r")

        # Budget errors
        with pytest.raises(BudgetError):
            raise CostLimitExceededError(1, 1, 1)
