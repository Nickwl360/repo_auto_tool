"""Automatic context enrichment for prompts.

Automatically extracts and adds relevant context to prompts to help
Claude make better decisions:

- Recent git commits for understanding code history
- TODO/FIXME comments in modified files
- Code complexity metrics for changed areas
- Related patterns from similar files

Example:
    enricher = ContextEnricher(repo_path)

    # Enrich a prompt with automatic context
    enriched = enricher.enrich_prompt(
        original_prompt="Fix the import errors",
        include_history=True,
        include_todos=True,
    )
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Patterns for extracting useful context
TODO_PATTERN = re.compile(r'#\s*(TODO|FIXME|XXX|HACK|NOTE):\s*(.+)', re.IGNORECASE)
MAX_TODOS_PER_FILE = 5
MAX_COMMITS_IN_CONTEXT = 5
MAX_CONTEXT_LENGTH = 1000


@dataclass
class FileContext:
    """Context extracted from a file.

    Attributes:
        file_path: Path to the file.
        todos: List of TODO comments found.
        complexity_score: Rough complexity score (lines, functions, etc.).
        recent_changes: Recent git changes to this file.
    """
    file_path: Path
    todos: list[str] = field(default_factory=list)
    complexity_score: int = 0
    recent_changes: list[str] = field(default_factory=list)


@dataclass
class EnrichedContext:
    """Enriched context for a prompt.

    Attributes:
        recent_commits: Recent commit messages.
        todos_by_file: Map of file to TODO comments.
        complexity_notes: Notes about code complexity.
        related_changes: Related changes in git history.
        enrichment_summary: Summary of what was enriched.
    """
    recent_commits: list[str] = field(default_factory=list)
    todos_by_file: dict[Path, list[str]] = field(default_factory=dict)
    complexity_notes: list[str] = field(default_factory=list)
    related_changes: list[str] = field(default_factory=list)
    enrichment_summary: str = ""

    def to_prompt_section(self) -> str:
        """Convert enriched context to a prompt section.

        Returns:
            Formatted context string to add to prompts.
        """
        sections = []

        # Recent commits
        if self.recent_commits:
            sections.append("Recent Git History:")
            for commit in self.recent_commits[:3]:
                sections.append(f"  - {commit}")
            sections.append("")

        # TODOs
        if self.todos_by_file:
            sections.append("Active TODOs in Changed Files:")
            for file_path, todos in self.todos_by_file.items():
                sections.append(f"  {file_path.name}:")
                for todo in todos[:3]:
                    sections.append(f"    - {todo}")
            sections.append("")

        # Complexity notes
        if self.complexity_notes:
            sections.append("Code Complexity Notes:")
            for note in self.complexity_notes:
                sections.append(f"  - {note}")
            sections.append("")

        # Related changes
        if self.related_changes:
            sections.append("Related Recent Changes:")
            for change in self.related_changes[:3]:
                sections.append(f"  - {change}")
            sections.append("")

        if not sections:
            return ""

        result = "--- AUTOMATIC CONTEXT ENRICHMENT ---\n\n"
        result += "\n".join(sections)
        result += "--- END CONTEXT ENRICHMENT ---\n\n"

        return result


class ContextEnricher:
    """Automatically enriches prompts with relevant context.

    Attributes:
        repo_path: Path to the repository.
        cache: Cache of extracted context to avoid re-parsing.
    """

    def __init__(self, repo_path: Path | str):
        self.repo_path = Path(repo_path)
        self.cache: dict[Path, FileContext] = {}

    def get_changed_files(self) -> list[Path]:
        """Get list of currently changed files.

        Returns:
            List of changed file paths.
        """
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return []

            paths = result.stdout.strip().split('\n')
            return [
                self.repo_path / p for p in paths
                if p and (self.repo_path / p).suffix == '.py'
            ]

        except (FileNotFoundError, subprocess.SubprocessError) as e:
            logger.debug(f"Could not get changed files: {e}")
            return []

    def extract_todos_from_file(self, file_path: Path) -> list[str]:
        """Extract TODO comments from a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of TODO comment texts.
        """
        todos = []

        try:
            content = file_path.read_text(encoding='utf-8')
            for match in TODO_PATTERN.finditer(content):
                todo_type = match.group(1)
                todo_text = match.group(2).strip()
                todos.append(f"[{todo_type}] {todo_text}")

                if len(todos) >= MAX_TODOS_PER_FILE:
                    break

        except (OSError, UnicodeDecodeError) as e:
            logger.debug(f"Could not read {file_path}: {e}")

        return todos

    def calculate_complexity(self, file_path: Path) -> int:
        """Calculate rough complexity score for a file.

        Args:
            file_path: Path to the file.

        Returns:
            Complexity score (higher = more complex).
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')

            # Simple heuristics
            score = 0
            score += len(lines)  # Raw size
            score += content.count('def ') * 5  # Functions
            score += content.count('class ') * 10  # Classes
            score += content.count('if ') * 2  # Conditionals
            score += content.count('for ') * 2  # Loops
            score += content.count('while ') * 2
            score += content.count('try:') * 3  # Exception handling
            score += content.count('async ') * 3  # Async code

            return score

        except (OSError, UnicodeDecodeError):
            return 0

    def get_recent_commits(self, limit: int = MAX_COMMITS_IN_CONTEXT) -> list[str]:
        """Get recent commit messages.

        Args:
            limit: Maximum number of commits.

        Returns:
            List of commit messages.
        """
        try:
            result = subprocess.run(
                ["git", "log", f"-{limit}", "--oneline", "--no-decorate"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return []

            commits = result.stdout.strip().split('\n')
            return [c for c in commits if c]

        except (FileNotFoundError, subprocess.SubprocessError) as e:
            logger.debug(f"Could not get commits: {e}")
            return []

    def get_file_context(self, file_path: Path) -> FileContext:
        """Get context for a specific file.

        Args:
            file_path: Path to the file.

        Returns:
            FileContext with extracted information.
        """
        # Check cache
        if file_path in self.cache:
            return self.cache[file_path]

        context = FileContext(file_path=file_path)

        # Extract TODOs
        context.todos = self.extract_todos_from_file(file_path)

        # Calculate complexity
        context.complexity_score = self.calculate_complexity(file_path)

        # Cache the result
        self.cache[file_path] = context

        return context

    def enrich_prompt(
        self,
        original_prompt: str,
        include_history: bool = True,
        include_todos: bool = True,
        include_complexity: bool = True,
    ) -> str:
        """Enrich a prompt with automatic context.

        Args:
            original_prompt: The original prompt.
            include_history: Include recent git history.
            include_todos: Include TODO comments.
            include_complexity: Include complexity notes.

        Returns:
            Enriched prompt.
        """
        enriched = EnrichedContext()
        changed_files = self.get_changed_files()

        # Get recent commits
        if include_history:
            enriched.recent_commits = self.get_recent_commits()

        # Get TODOs and complexity from changed files
        if changed_files:
            for file_path in changed_files:
                context = self.get_file_context(file_path)

                if include_todos and context.todos:
                    enriched.todos_by_file[file_path] = context.todos

                if include_complexity and context.complexity_score > 200:
                    enriched.complexity_notes.append(
                        f"{file_path.name} is complex (score: {context.complexity_score})"
                    )

        # Generate enrichment summary
        enrichments = []
        if enriched.recent_commits:
            enrichments.append(f"{len(enriched.recent_commits)} recent commits")
        if enriched.todos_by_file:
            total_todos = sum(len(t) for t in enriched.todos_by_file.values())
            enrichments.append(f"{total_todos} TODOs")
        if enriched.complexity_notes:
            enrichments.append(f"{len(enriched.complexity_notes)} complexity notes")

        if enrichments:
            enriched.enrichment_summary = f"Added context: {', '.join(enrichments)}"
            logger.info(enriched.enrichment_summary)
        else:
            logger.debug("No enrichment context available")

        # Build enriched prompt
        context_section = enriched.to_prompt_section()

        if not context_section:
            return original_prompt

        # Truncate if too long
        if len(context_section) > MAX_CONTEXT_LENGTH:
            context_section = context_section[:MAX_CONTEXT_LENGTH] + "\n...[truncated]\n"

        # Insert before the main instruction
        return context_section + original_prompt

    def clear_cache(self) -> None:
        """Clear the context cache."""
        self.cache.clear()
