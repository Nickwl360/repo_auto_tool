"""Learned prompt library for intelligent error recovery.

This module provides a sophisticated learning system that stores successful
prompts indexed by error patterns and context, automatically suggesting proven
approaches for similar errors.

Unlike PromptAdapter (which provides generic guidance), PromptLearner remembers
specific prompts that worked for specific error scenarios and suggests them
with confidence scores.

Key features:
- Fuzzy matching to find similar errors
- Confidence scoring based on success rate and similarity
- Usage statistics tracking (success rate, avg iterations)
- Cross-session persistence
- Integration with PromptAdapter for enhanced guidance

Example:
    learner = PromptLearner.load(repo_path)

    # After an error occurs
    context = ErrorContext.from_error("import_error", "No module named 'foo'", ...)
    suggestions = learner.suggest_prompts(context, max_suggestions=3)

    # Use the best suggestion
    if suggestions:
        best = suggestions[0]
        prompt = best.apply_to_context(context)
        # ... use prompt with Claude ...

    # After iteration completes
    learner.record_outcome(best.source_id, success=True, iterations=2)
    learner.save()
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants
PROMPTS_FILE = ".repo-improver-prompts.json"
MAX_STORED_PROMPTS = 100
MAX_SUGGESTIONS = 5
MIN_CONFIDENCE_THRESHOLD = 0.3
SIMILARITY_THRESHOLD = 0.4

# Similarity scoring weights
WEIGHT_ERROR_TYPE = 40
WEIGHT_ERROR_MESSAGE = 30
WEIGHT_FILE_TYPES = 20
WEIGHT_CONTEXT = 10


@dataclass
class ErrorContext:
    """Rich context about an error for intelligent matching.

    Attributes:
        error_type: Category of error (syntax_error, import_error, etc.)
        error_message: The actual error message.
        file_types: Set of file extensions involved (e.g., {'.py', '.json'}).
        recent_changes: Brief description of what was changed.
        iteration_number: Current iteration in the session.
        previous_attempts: Number of previous fix attempts for this error.
        code_patterns: Set of code patterns (e.g., {'async', 'class', 'decorator'}).
    """
    error_type: str
    error_message: str
    file_types: set[str] = field(default_factory=set)
    recent_changes: str = ""
    iteration_number: int = 0
    previous_attempts: int = 0
    code_patterns: set[str] = field(default_factory=set)

    def get_signature(self) -> str:
        """Generate a signature for this error context.

        Used for grouping similar errors together.

        Returns:
            Hash string representing this error context.
        """
        components = [
            self.error_type,
            sorted(self.file_types),
            sorted(self.code_patterns),
        ]
        signature_str = str(components)
        return hashlib.md5(signature_str.encode()).hexdigest()[:12]

    @classmethod
    def from_error(
        cls,
        error_type: str,
        error_message: str,
        file_types: set[str] | None = None,
        recent_changes: str = "",
        iteration_number: int = 0,
        previous_attempts: int = 0,
    ) -> ErrorContext:
        """Create ErrorContext from basic error information.

        Args:
            error_type: Category of error.
            error_message: The error message.
            file_types: Set of file extensions involved.
            recent_changes: What was changed that caused this.
            iteration_number: Current iteration number.
            previous_attempts: Number of previous attempts.

        Returns:
            ErrorContext instance with extracted patterns.
        """
        # Extract code patterns from error message
        patterns = set()
        message_lower = error_message.lower()

        # Common patterns to detect
        pattern_keywords = {
            'async', 'await', 'class', 'def', 'import', 'from',
            'decorator', '@', 'generator', 'yield', 'lambda',
            'type hint', 'generic', 'protocol', 'dataclass',
        }

        for keyword in pattern_keywords:
            if keyword in message_lower:
                patterns.add(keyword)

        return cls(
            error_type=error_type,
            error_message=error_message,
            file_types=file_types or set(),
            recent_changes=recent_changes,
            iteration_number=iteration_number,
            previous_attempts=previous_attempts,
            code_patterns=patterns,
        )


@dataclass
class LearnedPrompt:
    """A prompt template proven to fix specific error patterns.

    Attributes:
        id: Unique identifier for this learned prompt.
        error_type: Primary error type this addresses.
        error_pattern: Regex or substring pattern to match errors.
        context_signature: Signature of context where this worked.
        prompt_template: The prompt text (may include {placeholders}).
        success_count: Number of times this successfully fixed the error.
        failure_count: Number of times this failed to fix the error.
        total_iterations: Sum of iterations needed across all uses.
        created_at: When this prompt was first learned.
        last_used_at: When this prompt was last used.
        file_types: Set of file extensions where this worked.
        code_patterns: Set of code patterns this handles.
    """
    id: str
    error_type: str
    error_pattern: str
    context_signature: str
    prompt_template: str
    success_count: int = 0
    failure_count: int = 0
    total_iterations: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used_at: str = ""
    file_types: set[str] = field(default_factory=set)
    code_patterns: set[str] = field(default_factory=set)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def avg_iterations(self) -> float:
        """Average iterations needed when successful."""
        if self.success_count == 0:
            return 0.0
        return self.total_iterations / self.success_count

    @property
    def total_uses(self) -> int:
        """Total number of times this prompt was used."""
        return self.success_count + self.failure_count

    def matches_error(self, error_message: str) -> bool:
        """Check if this prompt's pattern matches an error message.

        Args:
            error_message: The error message to check.

        Returns:
            True if the pattern matches.
        """
        # Try regex match first, fall back to substring
        try:
            return bool(re.search(self.error_pattern, error_message, re.IGNORECASE))
        except re.error:
            return self.error_pattern.lower() in error_message.lower()

    def calculate_similarity(self, context: ErrorContext) -> float:
        """Calculate similarity score with an error context (0.0-1.0).

        Uses weighted scoring across multiple factors:
        - Error type match
        - Error message pattern match
        - File type overlap
        - Code pattern overlap

        Args:
            context: The error context to compare against.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        score = 0.0

        # Error type match (40% weight)
        if self.error_type == context.error_type:
            score += WEIGHT_ERROR_TYPE

        # Error message pattern match (30% weight)
        if self.matches_error(context.error_message):
            # Use fuzzy string matching for better scoring
            pattern_lower = self.error_pattern.lower()
            message_lower = context.error_message.lower()
            ratio = difflib.SequenceMatcher(None, pattern_lower, message_lower).ratio()
            score += WEIGHT_ERROR_MESSAGE * ratio

        # File type overlap (20% weight)
        if self.file_types and context.file_types:
            overlap = len(self.file_types & context.file_types)
            total = len(self.file_types | context.file_types)
            if total > 0:
                score += WEIGHT_FILE_TYPES * (overlap / total)
        elif not self.file_types and not context.file_types:
            # Both empty - perfect match
            score += WEIGHT_FILE_TYPES

        # Code pattern overlap (10% weight)
        if self.code_patterns and context.code_patterns:
            overlap = len(self.code_patterns & context.code_patterns)
            total = len(self.code_patterns | context.code_patterns)
            if total > 0:
                score += WEIGHT_CONTEXT * (overlap / total)
        elif not self.code_patterns and not context.code_patterns:
            score += WEIGHT_CONTEXT

        # Normalize to 0.0-1.0
        return score / 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "error_type": self.error_type,
            "error_pattern": self.error_pattern,
            "context_signature": self.context_signature,
            "prompt_template": self.prompt_template,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_iterations": self.total_iterations,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "file_types": list(self.file_types),
            "code_patterns": list(self.code_patterns),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedPrompt:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            error_type=data["error_type"],
            error_pattern=data["error_pattern"],
            context_signature=data["context_signature"],
            prompt_template=data["prompt_template"],
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            total_iterations=data.get("total_iterations", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_used_at=data.get("last_used_at", ""),
            file_types=set(data.get("file_types", [])),
            code_patterns=set(data.get("code_patterns", [])),
        )


@dataclass
class PromptSuggestion:
    """A suggested prompt with confidence and reasoning.

    Attributes:
        source_id: ID of the LearnedPrompt this came from.
        prompt: The actual prompt text to use.
        confidence: Confidence score (0.0-1.0).
        similarity: Similarity score with current error (0.0-1.0).
        success_rate: Historical success rate of source prompt.
        reasoning: Explanation of why this was suggested.
        avg_iterations: Average iterations needed historically.
    """
    source_id: str
    prompt: str
    confidence: float
    similarity: float
    success_rate: float
    reasoning: str
    avg_iterations: float = 0.0

    def apply_to_context(self, context: ErrorContext) -> str:
        """Apply this suggestion by filling in placeholders.

        Args:
            context: The error context to use for substitution.

        Returns:
            Prompt with placeholders filled in.
        """
        # Simple template substitution
        result = self.prompt
        result = result.replace("{error_type}", context.error_type)
        result = result.replace("{error_message}", context.error_message)
        result = result.replace("{recent_changes}", context.recent_changes)
        return result


@dataclass
class PromptLearner:
    """Intelligent prompt learning and suggestion system.

    Learns from successful error recovery patterns and suggests
    proven prompts for similar errors.

    Attributes:
        repo_path: Path to the repository.
        learned_prompts: Map of prompt ID to LearnedPrompt.
        created_at: When this learner was created.
        updated_at: When this learner was last updated.
    """
    repo_path: str
    learned_prompts: dict[str, LearnedPrompt] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = ""

    def _prompts_file_path(self) -> Path:
        """Get the path to the prompts file."""
        return Path(self.repo_path) / PROMPTS_FILE

    def learn_from_iteration_history(
        self,
        iterations: list[dict[str, Any]],
        prompt_adapter: Any = None,
    ) -> int:
        """Learn patterns from iteration history.

        Analyzes the iteration history to extract successful recovery
        patterns: when an error occurred, what prompt was used next,
        and whether it succeeded.

        Args:
            iterations: List of iteration records.
            prompt_adapter: Optional PromptAdapter for error categorization.

        Returns:
            Number of new patterns learned.
        """
        if not iterations:
            return 0

        from .prompt_adapter import PromptAdapter
        adapter = prompt_adapter or PromptAdapter()

        learned_count = 0

        for i, iteration in enumerate(iterations):
            # Look for failed iterations followed by successful ones
            if iteration.get("validation_passed") or not iteration.get("error"):
                continue

            # Check if next iteration fixed it
            if i + 1 >= len(iterations):
                continue

            next_iteration = iterations[i + 1]
            if not next_iteration.get("success") or not next_iteration.get("validation_passed"):
                continue

            # Extract the error and the fixing prompt
            error = iteration["error"]
            fixing_prompt = next_iteration.get("prompt", "")

            if not fixing_prompt or len(fixing_prompt) < 20:
                continue

            # Categorize error
            error_type = adapter.analyze_error_message(error)

            # Create error context
            context = ErrorContext.from_error(
                error_type=error_type,
                error_message=error,
                iteration_number=iteration.get("iteration", 0),
            )

            # Extract error pattern (first meaningful line)
            error_lines = error.split("\n")
            error_pattern = ""
            for line in error_lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith("#"):
                    error_pattern = line[:100]
                    break

            if not error_pattern:
                error_pattern = error[:100]

            # Check if we already have a similar learned prompt
            existing = self._find_existing_prompt(error_type, error_pattern, context)

            if existing:
                # Update existing prompt
                existing.success_count += 1
                existing.total_iterations += 1
                existing.last_used_at = datetime.now().isoformat()
                # Merge file types and patterns
                existing.file_types.update(context.file_types)
                existing.code_patterns.update(context.code_patterns)
                logger.debug(f"Updated existing prompt {existing.id} (uses: {existing.total_uses})")
            else:
                # Create new learned prompt
                prompt_id = self._generate_prompt_id(error_type, error_pattern)
                learned = LearnedPrompt(
                    id=prompt_id,
                    error_type=error_type,
                    error_pattern=error_pattern,
                    context_signature=context.get_signature(),
                    prompt_template=fixing_prompt[:500],  # Store first 500 chars
                    success_count=1,
                    total_iterations=1,
                    file_types=context.file_types,
                    code_patterns=context.code_patterns,
                )
                self.learned_prompts[prompt_id] = learned
                learned_count += 1
                logger.debug(f"Learned new prompt pattern: {prompt_id} for {error_type}")

        if learned_count > 0:
            logger.info(f"Learned {learned_count} new prompt patterns from iteration history")

        return learned_count

    def _find_existing_prompt(
        self,
        error_type: str,
        error_pattern: str,
        context: ErrorContext,
    ) -> LearnedPrompt | None:
        """Find an existing prompt that matches this error.

        Args:
            error_type: The error type.
            error_pattern: The error pattern.
            context: The error context.

        Returns:
            Matching LearnedPrompt or None.
        """
        for prompt in self.learned_prompts.values():
            if prompt.error_type != error_type:
                continue

            # Check if patterns are very similar
            similarity = difflib.SequenceMatcher(
                None,
                prompt.error_pattern.lower(),
                error_pattern.lower()
            ).ratio()

            if similarity > 0.8:  # 80% similar
                return prompt

        return None

    def _generate_prompt_id(self, error_type: str, error_pattern: str) -> str:
        """Generate a unique ID for a prompt.

        Args:
            error_type: The error type.
            error_pattern: The error pattern.

        Returns:
            Unique prompt ID.
        """
        components = f"{error_type}:{error_pattern}:{datetime.now().isoformat()}"
        hash_str = hashlib.md5(components.encode()).hexdigest()[:12]
        return f"{error_type}_{hash_str}"

    def suggest_prompts(
        self,
        context: ErrorContext,
        max_suggestions: int = MAX_SUGGESTIONS,
    ) -> list[PromptSuggestion]:
        """Suggest prompts for an error context.

        Finds learned prompts that match the error context,
        ranks them by confidence, and returns top suggestions.

        Args:
            context: The error context.
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            List of PromptSuggestion, sorted by confidence (highest first).
        """
        if not self.learned_prompts:
            return []

        suggestions: list[PromptSuggestion] = []

        for prompt in self.learned_prompts.values():
            # Calculate similarity
            similarity = prompt.calculate_similarity(context)

            if similarity < SIMILARITY_THRESHOLD:
                continue

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(
                prompt,
                similarity,
                context.previous_attempts,
            )

            if confidence < MIN_CONFIDENCE_THRESHOLD:
                continue

            # Generate reasoning
            reasoning = self._generate_reasoning(prompt, similarity, context)

            suggestion = PromptSuggestion(
                source_id=prompt.id,
                prompt=prompt.prompt_template,
                confidence=confidence,
                similarity=similarity,
                success_rate=prompt.success_rate,
                reasoning=reasoning,
                avg_iterations=prompt.avg_iterations,
            )
            suggestions.append(suggestion)

        # Sort by confidence (descending)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        return suggestions[:max_suggestions]

    def _calculate_confidence(
        self,
        prompt: LearnedPrompt,
        similarity: float,
        previous_attempts: int,
    ) -> float:
        """Calculate confidence score for a suggestion.

        Factors:
        - Base success rate of the prompt
        - Similarity to current error
        - Recency (recently used = higher confidence)
        - Usage count (more uses = more reliable if successful)
        - Previous attempts (more attempts = lower confidence)

        Args:
            prompt: The learned prompt.
            similarity: Similarity score.
            previous_attempts: Number of previous attempts.

        Returns:
            Confidence score (0.0-1.0).
        """
        # Base confidence from success rate
        confidence = prompt.success_rate

        # Multiply by similarity
        confidence *= similarity

        # Recency factor (prefer recently used prompts)
        if prompt.last_used_at:
            try:
                last_used = datetime.fromisoformat(prompt.last_used_at)
                age_days = (datetime.now() - last_used).days
                recency_factor = max(0.5, 1.0 - (age_days / 365))  # Decay over a year
                confidence *= recency_factor
            except (ValueError, TypeError):
                pass

        # Usage factor (more successful uses = higher confidence)
        if prompt.success_count > 0:
            usage_factor = min(1.2, 1.0 + (prompt.success_count / 20))  # Cap at 1.2x
            confidence *= usage_factor

        # Penalize if many previous attempts
        if previous_attempts > 0:
            attempt_penalty = 1.0 / (1.0 + previous_attempts * 0.2)
            confidence *= attempt_penalty

        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, confidence))

    def _generate_reasoning(
        self,
        prompt: LearnedPrompt,
        similarity: float,
        context: ErrorContext,
    ) -> str:
        """Generate human-readable reasoning for a suggestion.

        Args:
            prompt: The learned prompt.
            similarity: Similarity score.
            context: The error context.

        Returns:
            Reasoning string.
        """
        parts = []

        # Similarity
        parts.append(f"Similarity: {similarity:.0%}")

        # Success rate
        if prompt.total_uses > 0:
            parts.append(
                f"Success rate: {prompt.success_rate:.0%} "
                f"({prompt.success_count}/{prompt.total_uses} times)"
            )

        # Iterations needed
        if prompt.avg_iterations > 0:
            parts.append(f"Avg iterations: {prompt.avg_iterations:.1f}")

        # Matching factors
        matches = []
        if prompt.error_type == context.error_type:
            matches.append("error type")
        if prompt.file_types & context.file_types:
            matches.append("file types")
        if prompt.code_patterns & context.code_patterns:
            patterns = ", ".join(list(prompt.code_patterns & context.code_patterns)[:2])
            matches.append(f"patterns ({patterns})")

        if matches:
            parts.append(f"Matches: {', '.join(matches)}")

        return " | ".join(parts)

    def record_outcome(
        self,
        prompt_id: str,
        success: bool,
        iterations: int = 1,
    ) -> None:
        """Record the outcome of using a learned prompt.

        Args:
            prompt_id: ID of the prompt that was used.
            success: Whether it successfully fixed the error.
            iterations: Number of iterations it took.
        """
        if prompt_id not in self.learned_prompts:
            logger.warning(f"Cannot record outcome for unknown prompt: {prompt_id}")
            return

        prompt = self.learned_prompts[prompt_id]

        if success:
            prompt.success_count += 1
            prompt.total_iterations += iterations
        else:
            prompt.failure_count += 1

        prompt.last_used_at = datetime.now().isoformat()

        logger.debug(
            f"Recorded {'success' if success else 'failure'} for prompt {prompt_id} "
            f"(success rate: {prompt.success_rate:.0%})"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the learned prompts.

        Returns:
            Dictionary with statistics.
        """
        if not self.learned_prompts:
            return {
                "total_prompts": 0,
                "total_uses": 0,
                "avg_success_rate": 0.0,
                "top_prompts": [],
            }

        total_uses = sum(p.total_uses for p in self.learned_prompts.values())
        total_successes = sum(p.success_count for p in self.learned_prompts.values())
        avg_success_rate = total_successes / total_uses if total_uses > 0 else 0.0

        # Get top prompts by success count
        top_prompts = sorted(
            self.learned_prompts.values(),
            key=lambda p: p.success_count,
            reverse=True,
        )[:5]

        top_prompts_data = [
            {
                "error_type": p.error_type,
                "success_count": p.success_count,
                "success_rate": round(p.success_rate, 2),
                "total_uses": p.total_uses,
            }
            for p in top_prompts
        ]

        return {
            "total_prompts": len(self.learned_prompts),
            "total_uses": total_uses,
            "avg_success_rate": round(avg_success_rate, 2),
            "top_prompts": top_prompts_data,
        }

    def save(self) -> None:
        """Save learned prompts to JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        self.updated_at = datetime.now().isoformat()

        # Trim if too many prompts (keep most successful ones)
        if len(self.learned_prompts) > MAX_STORED_PROMPTS:
            sorted_prompts = sorted(
                self.learned_prompts.items(),
                key=lambda x: x[1].success_count,
                reverse=True,
            )[:MAX_STORED_PROMPTS]
            self.learned_prompts = dict(sorted_prompts)
            logger.info(f"Trimmed to {MAX_STORED_PROMPTS} most successful prompts")

        data = {
            "repo_path": self.repo_path,
            "learned_prompts": {
                k: v.to_dict() for k, v in self.learned_prompts.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

        path = self._prompts_file_path()
        try:
            path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved {len(self.learned_prompts)} learned prompts to {path}")
        except OSError as e:
            logger.error(f"Failed to save learned prompts: {e}")
            raise

    @classmethod
    def load(cls, repo_path: str | Path) -> PromptLearner:
        """Load learned prompts from file, or create new if doesn't exist.

        Args:
            repo_path: Path to the repository.

        Returns:
            PromptLearner instance (loaded or newly created).
        """
        repo_path = str(Path(repo_path).resolve())
        prompts_path = Path(repo_path) / PROMPTS_FILE

        if not prompts_path.exists():
            logger.debug(f"No prompts file found, creating new learner for {repo_path}")
            return cls(repo_path=repo_path)

        try:
            content = prompts_path.read_text()
            data = json.loads(content)

            learner = cls(
                repo_path=repo_path,
                created_at=data.get("created_at", datetime.now().isoformat()),
                updated_at=data.get("updated_at", ""),
            )

            # Load learned prompts
            for prompt_id, prompt_data in data.get("learned_prompts", {}).items():
                learner.learned_prompts[prompt_id] = LearnedPrompt.from_dict(prompt_data)

            stats = learner.get_stats()
            logger.info(
                f"Loaded prompt learner: {stats['total_prompts']} prompts, "
                f"{stats['total_uses']} total uses, "
                f"{stats['avg_success_rate']:.0%} avg success rate"
            )

            return learner

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load prompts file (corrupted?): {e}")
            return cls(repo_path=repo_path)
        except OSError as e:
            logger.warning(f"Failed to read prompts file: {e}")
            return cls(repo_path=repo_path)
