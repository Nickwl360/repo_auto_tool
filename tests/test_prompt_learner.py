"""Tests for the PromptLearner intelligent error recovery system."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from repo_auto_tool.prompt_learner import (
    ErrorContext,
    LearnedPrompt,
    PromptLearner,
    PromptSuggestion,
)


class TestErrorContext:
    """Tests for ErrorContext data structure."""

    def test_create_basic_context(self):
        """Test creating a basic error context."""
        context = ErrorContext(
            error_type="import_error",
            error_message="No module named 'foo'",
        )

        assert context.error_type == "import_error"
        assert context.error_message == "No module named 'foo'"
        assert context.file_types == set()
        assert context.iteration_number == 0

    def test_from_error_extracts_patterns(self):
        """Test that from_error extracts code patterns."""
        context = ErrorContext.from_error(
            error_type="type_error",
            error_message="Type error in async function with class decorator",
        )

        assert "async" in context.code_patterns
        assert "class" in context.code_patterns
        assert "decorator" in context.code_patterns

    def test_get_signature(self):
        """Test signature generation for grouping."""
        context1 = ErrorContext(
            error_type="import_error",
            error_message="Different message",
            file_types={".py"},
        )
        context2 = ErrorContext(
            error_type="import_error",
            error_message="Another message",
            file_types={".py"},
        )

        # Same error type and file types -> same signature
        assert context1.get_signature() == context2.get_signature()


class TestLearnedPrompt:
    """Tests for LearnedPrompt data structure."""

    def test_create_learned_prompt(self):
        """Test creating a learned prompt."""
        prompt = LearnedPrompt(
            id="import_error_abc123",
            error_type="import_error",
            error_pattern="No module named",
            context_signature="sig123",
            prompt_template="Fix the import by checking the module exists",
        )

        assert prompt.id == "import_error_abc123"
        assert prompt.success_count == 0
        assert prompt.failure_count == 0
        assert prompt.success_rate == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        prompt = LearnedPrompt(
            id="test",
            error_type="test",
            error_pattern="test",
            context_signature="test",
            prompt_template="test",
            success_count=7,
            failure_count=3,
        )

        assert prompt.success_rate == 0.7  # 7/10

    def test_avg_iterations_calculation(self):
        """Test average iterations calculation."""
        prompt = LearnedPrompt(
            id="test",
            error_type="test",
            error_pattern="test",
            context_signature="test",
            prompt_template="test",
            success_count=5,
            total_iterations=15,
        )

        assert prompt.avg_iterations == 3.0  # 15/5

    def test_matches_error_regex(self):
        """Test error pattern matching with regex."""
        prompt = LearnedPrompt(
            id="test",
            error_type="test",
            error_pattern=r"no module named \w+",
            context_signature="test",
            prompt_template="test",
        )

        assert prompt.matches_error("no module named foo")
        assert prompt.matches_error("No Module Named bar")
        assert not prompt.matches_error("different error")

    def test_matches_error_substring(self):
        """Test error pattern matching with substring."""
        prompt = LearnedPrompt(
            id="test",
            error_type="test",
            error_pattern="import error",
            context_signature="test",
            prompt_template="test",
        )

        assert prompt.matches_error("There was an import error in module X")
        assert not prompt.matches_error("syntax error")

    def test_calculate_similarity_perfect_match(self):
        """Test similarity calculation with perfect match."""
        prompt = LearnedPrompt(
            id="test",
            error_type="import_error",
            error_pattern="No module named",
            context_signature="test",
            prompt_template="test",
            file_types={".py"},
            code_patterns={"import", "class"},
        )

        context = ErrorContext(
            error_type="import_error",
            error_message="No module named foo",
            file_types={".py"},
            code_patterns={"import", "class"},
        )

        similarity = prompt.calculate_similarity(context)
        assert similarity > 0.9  # Very high similarity

    def test_calculate_similarity_partial_match(self):
        """Test similarity calculation with partial match."""
        prompt = LearnedPrompt(
            id="test",
            error_type="import_error",
            error_pattern="No module",
            context_signature="test",
            prompt_template="test",
            file_types={".py"},
        )

        context = ErrorContext(
            error_type="type_error",  # Different type
            error_message="No module named foo",
            file_types={".py"},
        )

        similarity = prompt.calculate_similarity(context)
        assert 0.3 < similarity < 0.7  # Moderate similarity

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        original = LearnedPrompt(
            id="test_123",
            error_type="import_error",
            error_pattern="No module",
            context_signature="sig_abc",
            prompt_template="Fix it",
            success_count=5,
            failure_count=2,
            file_types={".py", ".pyx"},
            code_patterns={"async", "class"},
        )

        data = original.to_dict()
        restored = LearnedPrompt.from_dict(data)

        assert restored.id == original.id
        assert restored.error_type == original.error_type
        assert restored.success_count == original.success_count
        assert restored.file_types == original.file_types
        assert restored.code_patterns == original.code_patterns


class TestPromptSuggestion:
    """Tests for PromptSuggestion."""

    def test_apply_to_context(self):
        """Test applying suggestion to context."""
        suggestion = PromptSuggestion(
            source_id="test",
            prompt="Fix {error_type} by checking {error_message}",
            confidence=0.8,
            similarity=0.9,
            success_rate=0.7,
            reasoning="Test reasoning",
        )

        context = ErrorContext(
            error_type="import_error",
            error_message="No module named foo",
        )

        result = suggestion.apply_to_context(context)
        assert "import_error" in result
        assert "No module named foo" in result


class TestPromptLearner:
    """Tests for PromptLearner main class."""

    def test_create_empty_learner(self, temp_dir):
        """Test creating an empty prompt learner."""
        learner = PromptLearner(repo_path=str(temp_dir))

        assert learner.repo_path == str(temp_dir)
        assert len(learner.learned_prompts) == 0

    def test_learn_from_iteration_history_success_after_failure(self, temp_dir):
        """Test learning from iterations where failure was fixed."""
        from repo_auto_tool.prompt_adapter import PromptAdapter

        learner = PromptLearner(repo_path=str(temp_dir))
        adapter = PromptAdapter()

        iterations = [
            {
                "iteration": 1,
                "validation_passed": False,
                "error": "ImportError: No module named 'foo'",
                "prompt": "Add feature X",
            },
            {
                "iteration": 2,
                "success": True,
                "validation_passed": True,
                "prompt": "Install the foo module and try again",
                "result": "Success",
            },
        ]

        learned_count = learner.learn_from_iteration_history(iterations, adapter)

        assert learned_count == 1
        assert len(learner.learned_prompts) == 1

        # Get the learned prompt
        prompt = list(learner.learned_prompts.values())[0]
        assert prompt.error_type == "import_error"
        assert prompt.success_count == 1
        assert "No module named" in prompt.error_pattern

    def test_learn_from_multiple_similar_errors(self, temp_dir):
        """Test that similar errors update existing prompts."""
        from repo_auto_tool.prompt_adapter import PromptAdapter

        learner = PromptLearner(repo_path=str(temp_dir))
        adapter = PromptAdapter()

        iterations = [
            {
                "iteration": 1,
                "validation_passed": False,
                "error": "ImportError: No module named 'foo'",
                "prompt": "Add feature X to the codebase",
            },
            {
                "iteration": 2,
                "success": True,
                "validation_passed": True,
                "prompt": "Install the foo module using pip install foo",
                "result": "Success",
            },
            {
                "iteration": 3,
                "validation_passed": False,
                "error": "ImportError: No module named 'bar'",
                "prompt": "Add feature Y to the codebase",
            },
            {
                "iteration": 4,
                "success": True,
                "validation_passed": True,
                "prompt": "Install the bar module using pip install bar",
                "result": "Success",
            },
        ]

        learned_count = learner.learn_from_iteration_history(iterations, adapter)

        # Should create 1 learned prompt and update it
        assert len(learner.learned_prompts) == 1
        prompt = list(learner.learned_prompts.values())[0]
        assert prompt.success_count == 2  # Both successes

    def test_suggest_prompts_returns_relevant(self, temp_dir):
        """Test suggesting prompts for similar errors."""
        learner = PromptLearner(repo_path=str(temp_dir))

        # Add a learned prompt
        prompt = LearnedPrompt(
            id="import_001",
            error_type="import_error",
            error_pattern="No module named",
            context_signature="sig",
            prompt_template="Check if module is installed, then import it",
            success_count=5,
            failure_count=1,
            file_types={".py"},
        )
        learner.learned_prompts[prompt.id] = prompt

        # Create a similar error context
        context = ErrorContext(
            error_type="import_error",
            error_message="No module named 'requests'",
            file_types={".py"},
        )

        suggestions = learner.suggest_prompts(context)

        assert len(suggestions) == 1
        assert suggestions[0].source_id == "import_001"
        assert suggestions[0].confidence > 0.5
        assert suggestions[0].success_rate == 5/6

    def test_suggest_prompts_filters_by_similarity(self, temp_dir):
        """Test that low similarity suggestions are filtered out."""
        learner = PromptLearner(repo_path=str(temp_dir))

        # Add a learned prompt for a different error type
        prompt = LearnedPrompt(
            id="type_001",
            error_type="type_error",
            error_pattern="int object",
            context_signature="sig",
            prompt_template="Fix type error",
            success_count=5,
        )
        learner.learned_prompts[prompt.id] = prompt

        # Create a completely different error context
        context = ErrorContext(
            error_type="syntax_error",
            error_message="SyntaxError: invalid syntax",
        )

        suggestions = learner.suggest_prompts(context)

        # Should return no suggestions due to low similarity
        assert len(suggestions) == 0

    def test_suggest_prompts_sorted_by_confidence(self, temp_dir):
        """Test that suggestions are sorted by confidence."""
        learner = PromptLearner(repo_path=str(temp_dir))

        # Add multiple prompts with different success rates
        prompt1 = LearnedPrompt(
            id="import_001",
            error_type="import_error",
            error_pattern="No module",
            context_signature="sig",
            prompt_template="Fix 1",
            success_count=2,
            failure_count=3,  # 40% success rate
        )
        prompt2 = LearnedPrompt(
            id="import_002",
            error_type="import_error",
            error_pattern="No module",
            context_signature="sig",
            prompt_template="Fix 2",
            success_count=8,
            failure_count=2,  # 80% success rate
        )

        learner.learned_prompts[prompt1.id] = prompt1
        learner.learned_prompts[prompt2.id] = prompt2

        context = ErrorContext(
            error_type="import_error",
            error_message="No module named 'foo'",
        )

        suggestions = learner.suggest_prompts(context)

        assert len(suggestions) == 2
        # Higher success rate should come first
        assert suggestions[0].source_id == "import_002"
        assert suggestions[1].source_id == "import_001"

    def test_record_outcome_success(self, temp_dir):
        """Test recording a successful outcome."""
        learner = PromptLearner(repo_path=str(temp_dir))

        prompt = LearnedPrompt(
            id="test_001",
            error_type="test",
            error_pattern="test",
            context_signature="sig",
            prompt_template="test",
            success_count=3,
            failure_count=1,
        )
        learner.learned_prompts[prompt.id] = prompt

        learner.record_outcome("test_001", success=True, iterations=2)

        assert prompt.success_count == 4
        assert prompt.failure_count == 1
        assert prompt.total_iterations == 2

    def test_record_outcome_failure(self, temp_dir):
        """Test recording a failed outcome."""
        learner = PromptLearner(repo_path=str(temp_dir))

        prompt = LearnedPrompt(
            id="test_001",
            error_type="test",
            error_pattern="test",
            context_signature="sig",
            prompt_template="test",
            success_count=3,
            failure_count=1,
        )
        learner.learned_prompts[prompt.id] = prompt

        learner.record_outcome("test_001", success=False)

        assert prompt.success_count == 3
        assert prompt.failure_count == 2

    def test_get_stats_empty(self, temp_dir):
        """Test getting stats for empty learner."""
        learner = PromptLearner(repo_path=str(temp_dir))

        stats = learner.get_stats()

        assert stats["total_prompts"] == 0
        assert stats["total_uses"] == 0
        assert stats["avg_success_rate"] == 0.0
        assert stats["top_prompts"] == []

    def test_get_stats_with_data(self, temp_dir):
        """Test getting stats with learned prompts."""
        learner = PromptLearner(repo_path=str(temp_dir))

        prompt1 = LearnedPrompt(
            id="test_001",
            error_type="import_error",
            error_pattern="test",
            context_signature="sig",
            prompt_template="test",
            success_count=8,
            failure_count=2,
        )
        prompt2 = LearnedPrompt(
            id="test_002",
            error_type="syntax_error",
            error_pattern="test",
            context_signature="sig",
            prompt_template="test",
            success_count=3,
            failure_count=1,
        )

        learner.learned_prompts[prompt1.id] = prompt1
        learner.learned_prompts[prompt2.id] = prompt2

        stats = learner.get_stats()

        assert stats["total_prompts"] == 2
        assert stats["total_uses"] == 14  # 10 + 4
        assert stats["avg_success_rate"] == round(11/14, 2)
        assert len(stats["top_prompts"]) == 2
        assert stats["top_prompts"][0]["error_type"] == "import_error"

    def test_save_and_load(self, temp_dir):
        """Test saving and loading learned prompts."""
        learner = PromptLearner(repo_path=str(temp_dir))

        # Add a learned prompt
        prompt = LearnedPrompt(
            id="test_001",
            error_type="import_error",
            error_pattern="No module",
            context_signature="sig_abc",
            prompt_template="Install the module first",
            success_count=5,
            failure_count=1,
            file_types={".py"},
            code_patterns={"import"},
        )
        learner.learned_prompts[prompt.id] = prompt

        # Save
        learner.save()

        # Load
        loaded = PromptLearner.load(temp_dir)

        assert len(loaded.learned_prompts) == 1
        loaded_prompt = loaded.learned_prompts["test_001"]
        assert loaded_prompt.error_type == "import_error"
        assert loaded_prompt.success_count == 5
        assert loaded_prompt.file_types == {".py"}

    def test_load_nonexistent_creates_new(self, temp_dir):
        """Test loading from nonexistent file creates new learner."""
        learner = PromptLearner.load(temp_dir)

        assert len(learner.learned_prompts) == 0
        assert learner.repo_path == str(temp_dir.resolve())

    def test_load_corrupted_creates_new(self, temp_dir):
        """Test loading corrupted file creates new learner."""
        prompts_file = temp_dir / ".repo-improver-prompts.json"
        prompts_file.write_text("{ invalid json }")

        learner = PromptLearner.load(temp_dir)

        # Should create new learner instead of crashing
        assert len(learner.learned_prompts) == 0

    def test_trimming_on_save(self, temp_dir):
        """Test that save trims to max prompts."""
        learner = PromptLearner(repo_path=str(temp_dir))

        # Add many prompts
        for i in range(150):
            prompt = LearnedPrompt(
                id=f"test_{i:03d}",
                error_type="test",
                error_pattern="test",
                context_signature="sig",
                prompt_template="test",
                success_count=i,  # Different success counts
            )
            learner.learned_prompts[prompt.id] = prompt

        learner.save()

        # Reload and check
        loaded = PromptLearner.load(temp_dir)

        # Should keep only top 100
        assert len(loaded.learned_prompts) == 100

        # Should keep the ones with highest success counts
        assert "test_149" in loaded.learned_prompts
        assert "test_000" not in loaded.learned_prompts


# Integration test
class TestPromptLearnerIntegration:
    """Integration tests for the full PromptLearner workflow."""

    def test_full_learning_cycle(self, temp_dir):
        """Test a complete learning cycle."""
        from repo_auto_tool.prompt_adapter import PromptAdapter

        learner = PromptLearner(repo_path=str(temp_dir))
        adapter = PromptAdapter()

        # Simulate a session with failures and fixes
        iterations = [
            {
                "iteration": 1,
                "validation_passed": False,
                "error": "SyntaxError: invalid syntax at line 5",
                "prompt": "Add type hints",
            },
            {
                "iteration": 2,
                "success": True,
                "validation_passed": True,
                "prompt": "Fix the syntax error by closing the parentheses",
                "result": "Fixed",
            },
            {
                "iteration": 3,
                "validation_passed": False,
                "error": "ImportError: No module named 'pytest'",
                "prompt": "Add tests",
            },
            {
                "iteration": 4,
                "success": True,
                "validation_passed": True,
                "prompt": "Install pytest using pip",
                "result": "Fixed",
            },
        ]

        # Learn from history
        learned = learner.learn_from_iteration_history(iterations, adapter)
        assert learned == 2

        # Save
        learner.save()

        # Load in new session
        new_learner = PromptLearner.load(temp_dir)
        assert len(new_learner.learned_prompts) == 2

        # Test suggestions for similar error
        context = ErrorContext.from_error(
            error_type="import_error",
            error_message="ImportError: No module named 'requests'",
        )

        suggestions = new_learner.suggest_prompts(context)
        assert len(suggestions) >= 1
        assert "pytest" in suggestions[0].prompt or "Install" in suggestions[0].prompt

        # Record outcome
        new_learner.record_outcome(suggestions[0].source_id, success=True, iterations=1)

        # Get stats
        stats = new_learner.get_stats()
        assert stats["total_prompts"] == 2
        assert stats["total_uses"] == 3  # 2 from history + 1 recorded
