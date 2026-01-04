"""Smart prompt parsing for flexible goal input.

This module provides functionality to parse goals from various file formats:
- Plain text files (.txt)
- Markdown files (.md)
- YAML files (.yaml, .yml)
- JSON files (.json)

It intelligently extracts and structures goals from unstructured text.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import PromptParseError


@dataclass
class ParsedPrompt:
    """Represents a parsed prompt with optional structured metadata.

    Attributes:
        goal: The main goal/objective extracted from the prompt.
        sub_goals: Optional list of sub-goals or steps.
        constraints: Optional list of constraints or requirements.
        context: Optional additional context for the goal.
        raw_content: The original raw content from the file.
        source_file: Path to the source file if parsed from a file.
        format_detected: The format detected (text, markdown, yaml, json).
    """

    goal: str
    sub_goals: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    context: str = ""
    raw_content: str = ""
    source_file: Path | None = None
    format_detected: str = "text"

    def to_prompt_string(self) -> str:
        """Convert parsed prompt to a formatted string for Claude.

        Returns:
            A well-formatted prompt string combining all components.
        """
        parts = [f"GOAL: {self.goal}"]

        if self.context:
            parts.append(f"\nCONTEXT:\n{self.context}")

        if self.sub_goals:
            parts.append("\nSUB-GOALS:")
            for i, sub in enumerate(self.sub_goals, 1):
                parts.append(f"  {i}. {sub}")

        if self.constraints:
            parts.append("\nCONSTRAINTS:")
            for constraint in self.constraints:
                parts.append(f"  - {constraint}")

        return "\n".join(parts)


class PromptParser:
    """Parser for extracting goals from various file formats."""

    # Keywords that indicate goals in unstructured text
    GOAL_KEYWORDS = [
        "goal", "objective", "task", "implement", "add", "create",
        "fix", "improve", "refactor", "update", "remove", "change",
        "build", "develop", "enhance", "optimize"
    ]

    # Keywords that indicate constraints
    CONSTRAINT_KEYWORDS = [
        "must", "should", "require", "constraint", "limitation",
        "don't", "do not", "avoid", "without", "keep", "maintain"
    ]

    def __init__(self) -> None:
        """Initialize the prompt parser."""
        pass

    def parse_file(self, file_path: Path) -> ParsedPrompt:
        """Parse a prompt from a file.

        Args:
            file_path: Path to the file containing the prompt.

        Returns:
            ParsedPrompt object with extracted information.

        Raises:
            PromptParseError: If the file cannot be read or parsed.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise PromptParseError(f"Prompt file not found: {file_path}")

        if not file_path.is_file():
            raise PromptParseError(f"Prompt path is not a file: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as e:
            raise PromptParseError(f"Failed to read prompt file: {e}") from e

        if not content.strip():
            raise PromptParseError("Prompt file is empty")

        # Detect format and parse accordingly
        suffix = file_path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            result = self._parse_yaml(content)
        elif suffix == ".json":
            result = self._parse_json(content)
        elif suffix == ".md":
            result = self._parse_markdown(content)
        else:
            result = self._parse_text(content)

        result.source_file = file_path
        result.raw_content = content

        return result

    def parse_string(self, content: str, format_hint: str = "auto") -> ParsedPrompt:
        """Parse a prompt from a string.

        Args:
            content: The string content to parse.
            format_hint: Format hint ('auto', 'text', 'yaml', 'json', 'markdown').

        Returns:
            ParsedPrompt object with extracted information.

        Raises:
            PromptParseError: If the content cannot be parsed.
        """
        if not content.strip():
            raise PromptParseError("Prompt content is empty")

        if format_hint == "auto":
            format_hint = self._detect_format(content)

        if format_hint == "yaml":
            return self._parse_yaml(content)
        elif format_hint == "json":
            return self._parse_json(content)
        elif format_hint == "markdown":
            return self._parse_markdown(content)
        else:
            return self._parse_text(content)

    def _detect_format(self, content: str) -> str:
        """Detect the format of content.

        Args:
            content: The content to analyze.

        Returns:
            Detected format: 'yaml', 'json', 'markdown', or 'text'.
        """
        content_stripped = content.strip()

        # Check for JSON
        if content_stripped.startswith("{") or content_stripped.startswith("["):
            try:
                json.loads(content_stripped)
                return "json"
            except json.JSONDecodeError:
                pass

        # Check for YAML-like structure (key: value patterns)
        yaml_pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*:\s*"
        if re.match(yaml_pattern, content_stripped, re.MULTILINE):
            lines = content_stripped.split("\n")
            yaml_like_lines = sum(1 for line in lines if re.match(yaml_pattern, line))
            if yaml_like_lines >= 2:
                return "yaml"

        # Check for Markdown headers
        if re.search(r"^#+\s+", content, re.MULTILINE):
            return "markdown"

        return "text"

    def _parse_yaml(self, content: str) -> ParsedPrompt:
        """Parse YAML-formatted prompt.

        Args:
            content: YAML content string.

        Returns:
            ParsedPrompt object.

        Raises:
            PromptParseError: If YAML parsing fails.
        """
        # Simple YAML parser (avoid external dependency)
        # Supports basic key: value and key: [list] patterns
        data = self._simple_yaml_parse(content)

        # Extract goal
        goal = data.get("goal") or data.get("objective") or data.get("task")
        if not goal:
            # Fall back to treating content as plain text
            return self._parse_text(content)

        # Extract sub-goals
        sub_goals = data.get("sub_goals") or data.get("subgoals") or data.get("steps") or []
        if isinstance(sub_goals, str):
            sub_goals = [sub_goals]

        # Extract constraints
        constraints = data.get("constraints") or data.get("requirements") or []
        if isinstance(constraints, str):
            constraints = [constraints]

        # Extract context
        context = data.get("context") or data.get("description") or ""

        return ParsedPrompt(
            goal=str(goal),
            sub_goals=[str(s) for s in sub_goals],
            constraints=[str(c) for c in constraints],
            context=str(context),
            format_detected="yaml"
        )

    def _simple_yaml_parse(self, content: str) -> dict[str, Any]:
        """Simple YAML parser for basic structures.

        Supports:
        - key: value
        - key: [item1, item2]
        - key:
            - item1
            - item2

        Args:
            content: YAML-like content.

        Returns:
            Dictionary of parsed values.
        """
        result: dict[str, Any] = {}
        current_key: str | None = None
        current_list: list[str] = []
        in_list = False

        lines = content.split("\n")

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#"):
                continue

            # Check for list item
            if stripped.startswith("- "):
                if current_key and in_list:
                    current_list.append(stripped[2:].strip())
                continue

            # Check for key: value pattern
            match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)", stripped)
            if match:
                # Save previous list if we were in one
                if current_key and in_list and current_list:
                    result[current_key] = current_list
                    current_list = []

                key = match.group(1).lower()
                value = match.group(2).strip()

                if value:
                    # Inline value
                    # Check for inline list [item1, item2]
                    if value.startswith("[") and value.endswith("]"):
                        items = value[1:-1].split(",")
                        result[key] = [item.strip().strip("\"'") for item in items]
                        in_list = False
                        current_key = None
                    else:
                        # Strip quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        result[key] = value
                        in_list = False
                        current_key = None
                else:
                    # Start of list
                    current_key = key
                    current_list = []
                    in_list = True

        # Don't forget the last list
        if current_key and in_list and current_list:
            result[current_key] = current_list

        return result

    def _parse_json(self, content: str) -> ParsedPrompt:
        """Parse JSON-formatted prompt.

        Args:
            content: JSON content string.

        Returns:
            ParsedPrompt object.

        Raises:
            PromptParseError: If JSON parsing fails.
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise PromptParseError(f"Invalid JSON in prompt file: {e}") from e

        if isinstance(data, str):
            # JSON string containing the goal
            return ParsedPrompt(goal=data, format_detected="json")

        if not isinstance(data, dict):
            raise PromptParseError("JSON prompt must be an object or string")

        # Extract goal
        goal = data.get("goal") or data.get("objective") or data.get("task")
        if not goal:
            raise PromptParseError("JSON prompt must contain 'goal', 'objective', or 'task' key")

        # Extract sub-goals
        sub_goals = data.get("sub_goals") or data.get("subgoals") or data.get("steps") or []
        if isinstance(sub_goals, str):
            sub_goals = [sub_goals]

        # Extract constraints
        constraints = data.get("constraints") or data.get("requirements") or []
        if isinstance(constraints, str):
            constraints = [constraints]

        # Extract context
        context = data.get("context") or data.get("description") or ""

        return ParsedPrompt(
            goal=str(goal),
            sub_goals=[str(s) for s in sub_goals],
            constraints=[str(c) for c in constraints],
            context=str(context),
            format_detected="json"
        )

    def _parse_markdown(self, content: str) -> ParsedPrompt:
        """Parse Markdown-formatted prompt.

        Expected structure:
        # Goal
        The main goal text

        ## Sub-goals / Steps
        - Sub-goal 1
        - Sub-goal 2

        ## Constraints
        - Constraint 1

        ## Context
        Additional context...

        Args:
            content: Markdown content string.

        Returns:
            ParsedPrompt object.
        """
        goal = ""
        sub_goals: list[str] = []
        constraints: list[str] = []
        context = ""

        # Split by headers
        sections = re.split(r"^(#+)\s+(.+)$", content, flags=re.MULTILINE)

        current_section = "intro"
        current_content: list[str] = []

        i = 0
        while i < len(sections):
            part = sections[i]

            if part.startswith("#"):
                # This is a header marker, next parts are level and title
                if i + 2 < len(sections):
                    header_title = sections[i + 1].lower()

                    # Process previous section
                    section_text = "\n".join(current_content).strip()
                    self._assign_section_content(
                        current_section, section_text,
                        goal, sub_goals, constraints, context
                    )

                    # Update goal, sub_goals, constraints, context from assignment
                    if current_section == "goal":
                        goal = section_text
                    elif current_section in ("sub-goals", "steps", "tasks"):
                        sub_goals = self._extract_list_items(section_text)
                    elif current_section in ("constraints", "requirements"):
                        constraints = self._extract_list_items(section_text)
                    elif current_section == "context":
                        context = section_text

                    # Determine new section type
                    goal_keywords = ("goal", "objective", "task")
                    if any(kw in header_title for kw in goal_keywords):
                        current_section = "goal"
                    elif any(kw in header_title for kw in ["sub-goal", "subgoal", "step", "task"]):
                        current_section = "sub-goals"
                    elif any(kw in header_title for kw in ["constraint", "requirement", "rule"]):
                        current_section = "constraints"
                    elif "context" in header_title or "description" in header_title:
                        current_section = "context"
                    else:
                        # Unknown section, treat as context
                        current_section = "context"

                    current_content = []
                    i += 2
                else:
                    i += 1
            else:
                # Regular content
                if part.strip():
                    current_content.append(part)
                i += 1

        # Process final section
        section_text = "\n".join(current_content).strip()
        if current_section == "goal":
            goal = section_text
        elif current_section in ("sub-goals", "steps", "tasks"):
            sub_goals = self._extract_list_items(section_text)
        elif current_section in ("constraints", "requirements"):
            constraints = self._extract_list_items(section_text)
        elif current_section == "context":
            context = section_text
        elif current_section == "intro" and section_text and not goal:
            # If no explicit goal section, use intro as goal
            goal = section_text

        # If still no goal, use the entire content
        if not goal:
            goal = self._extract_goal_from_text(content)

        return ParsedPrompt(
            goal=goal,
            sub_goals=sub_goals,
            constraints=constraints,
            context=context,
            format_detected="markdown"
        )

    def _assign_section_content(
        self,
        section: str,
        content: str,
        goal: str,
        sub_goals: list[str],
        constraints: list[str],
        context: str
    ) -> None:
        """Assign content to the appropriate field based on section name.

        Note: This is a helper that would modify the parsed prompt in place,
        but since Python doesn't allow modifying primitives, the actual
        assignment happens in the calling method.
        """
        pass  # Assignment happens in caller due to Python semantics

    def _extract_list_items(self, content: str) -> list[str]:
        """Extract list items from text.

        Handles:
        - Bullet points (-, *, •)
        - Numbered lists (1., 2., etc.)

        Args:
            content: Text containing list items.

        Returns:
            List of extracted items.
        """
        items: list[str] = []

        # Match bullet points and numbered lists
        pattern = r"^[\s]*(?:[-*•]|\d+\.)\s+(.+)$"

        for match in re.finditer(pattern, content, re.MULTILINE):
            item = match.group(1).strip()
            if item:
                items.append(item)

        return items

    def _parse_text(self, content: str) -> ParsedPrompt:
        """Parse plain text prompt with intelligent extraction.

        Args:
            content: Plain text content.

        Returns:
            ParsedPrompt object.
        """
        # Try to identify structured parts even in plain text
        goal = self._extract_goal_from_text(content)
        sub_goals = self._extract_sub_goals_from_text(content)
        constraints = self._extract_constraints_from_text(content)

        return ParsedPrompt(
            goal=goal,
            sub_goals=sub_goals,
            constraints=constraints,
            format_detected="text"
        )

    def _extract_goal_from_text(self, content: str) -> str:
        """Extract the main goal from unstructured text.

        Args:
            content: The text to analyze.

        Returns:
            The extracted goal string.
        """
        lines = content.strip().split("\n")

        # Look for explicit goal markers
        for line in lines:
            line_lower = line.lower()
            for keyword in ["goal:", "objective:", "task:"]:
                if line_lower.startswith(keyword):
                    return line[len(keyword):].strip()

        # Look for sentences starting with goal keywords
        sentences = self._split_sentences(content)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(sentence_lower.startswith(kw) for kw in self.GOAL_KEYWORDS):
                return sentence

        # Use the first non-empty line or sentence as the goal
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped

        # Fallback: use the entire content
        return content.strip()

    def _extract_sub_goals_from_text(self, content: str) -> list[str]:
        """Extract sub-goals from unstructured text.

        Args:
            content: The text to analyze.

        Returns:
            List of extracted sub-goals.
        """
        sub_goals: list[str] = []

        # Look for numbered or bulleted items
        items = self._extract_list_items(content)
        if items:
            return items

        return sub_goals

    def _extract_constraints_from_text(self, content: str) -> list[str]:
        """Extract constraints from unstructured text.

        Args:
            content: The text to analyze.

        Returns:
            List of extracted constraints.
        """
        constraints: list[str] = []

        sentences = self._split_sentences(content)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in self.CONSTRAINT_KEYWORDS):
                # This sentence contains constraint-like language
                constraints.append(sentence)

        return constraints

    def _split_sentences(self, content: str) -> list[str]:
        """Split content into sentences.

        Args:
            content: Text to split.

        Returns:
            List of sentences.
        """
        # Simple sentence splitting by common terminators
        sentences = re.split(r"[.!?]\s+", content)
        return [s.strip() for s in sentences if s.strip()]


def parse_prompt_file(file_path: Path | str) -> ParsedPrompt:
    """Convenience function to parse a prompt file.

    Args:
        file_path: Path to the prompt file.

    Returns:
        ParsedPrompt object with extracted information.

    Raises:
        PromptParseError: If the file cannot be parsed.
    """
    parser = PromptParser()
    return parser.parse_file(Path(file_path))


def parse_prompt_string(content: str, format_hint: str = "auto") -> ParsedPrompt:
    """Convenience function to parse a prompt string.

    Args:
        content: The prompt content.
        format_hint: Optional format hint.

    Returns:
        ParsedPrompt object with extracted information.

    Raises:
        PromptParseError: If the content cannot be parsed.
    """
    parser = PromptParser()
    return parser.parse_string(content, format_hint)
