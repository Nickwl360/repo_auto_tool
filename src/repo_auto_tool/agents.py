"""Agent modes for repo-auto-tool.

This module provides specialized agents for different phases of the improvement process:
- PreAnalysisAgent: Analyzes the codebase and suggests improvement goals
- GoalDecomposerAgent: Breaks down large goals into smaller, actionable steps
- ReviewerAgent: Reviews changes after each iteration and provides feedback

All agents inherit from the base Agent class which provides common functionality.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .claude_interface import ClaudeCodeInterface

logger = logging.getLogger(__name__)

AgentMode = Literal["pre-analysis", "goal-decomposer", "reviewer"]


@dataclass
class AgentResult:
    """Result from an agent operation.

    Attributes:
        success: Whether the agent completed successfully
        output: The main output/result from the agent
        suggestions: List of suggestions (for PreAnalysisAgent/ReviewerAgent)
        steps: List of decomposed steps (for GoalDecomposerAgent)
        metadata: Additional metadata from the agent
        error: Error message if the operation failed
    """
    success: bool
    output: str
    suggestions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Agent(ABC):
    """Base class for all agents.

    Agents are specialized components that use Claude Code CLI to perform
    specific tasks in the improvement process. Each agent has a defined role
    and set of prompts optimized for its purpose.

    Attributes:
        name: Human-readable name for the agent
        working_dir: Directory the agent operates in
        claude: Interface to Claude Code CLI
    """

    name: str = "BaseAgent"

    def __init__(
        self,
        working_dir: Path,
        model: str | None = None,
        timeout: int = 600,
    ):
        """Initialize the agent.

        Args:
            working_dir: The repository directory to work with
            model: Optional model override for Claude CLI
            timeout: Timeout in seconds for Claude CLI calls
        """
        self.working_dir = Path(working_dir).resolve()
        self.model = model
        self.timeout = timeout
        self.claude = ClaudeCodeInterface(
            working_dir=self.working_dir,
            model=model,
            timeout=timeout,
            allowed_tools=self._get_allowed_tools(),
        )
        logger.info(f"Initialized {self.name} for {self.working_dir}")

    def _get_allowed_tools(self) -> list[str]:
        """Return the list of tools this agent is allowed to use.

        Override in subclasses to restrict tool access.
        Default allows read-only tools for safety.
        """
        return [
            "Read(*)",
            "Glob(*)",
            "Grep(*)",
        ]

    @abstractmethod
    def run(self, **kwargs: Any) -> AgentResult:
        """Execute the agent's main task.

        Args:
            **kwargs: Agent-specific arguments

        Returns:
            AgentResult with the outcome
        """
        pass

    def _parse_list_response(self, response: str) -> list[str]:
        """Parse a numbered or bulleted list from Claude's response.

        Args:
            response: Raw response text from Claude

        Returns:
            List of extracted items
        """
        items = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove common list prefixes
            for prefix in ["- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. ",
                          "6. ", "7. ", "8. ", "9. ", "10. "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
                    break
            # Also handle numbered lists like "1) " or "(1) "
            if len(line) > 2 and line[0].isdigit():
                if line[1] in ") .":
                    line = line[2:].strip()
                elif line[1].isdigit() and len(line) > 3 and line[2] in ") .":
                    line = line[3:].strip()
            if line:
                items.append(line)
        return items


class PreAnalysisAgent(Agent):
    """Agent that analyzes a codebase and suggests improvement goals.

    This agent examines the repository structure, code quality, test coverage,
    documentation, and other aspects to identify areas for improvement.
    It produces a list of suggested goals ranked by impact and feasibility.
    """

    name: str = "PreAnalysisAgent"

    ANALYSIS_PROMPT = """You are analyzing a codebase to suggest improvement goals.

Examine this repository thoroughly and identify areas that could be improved.
Consider the following aspects:
1. Code quality and maintainability
2. Test coverage and testing practices
3. Documentation completeness
4. Type safety and type hints
5. Error handling patterns
6. Security considerations
7. Performance optimization opportunities
8. Dependency management
9. CI/CD and automation
10. Code organization and architecture

After your analysis, provide a ranked list of 5-10 specific, actionable improvement goals.
Format each goal as a clear, concise statement that could be used as input to an improvement tool.

IMPORTANT: Focus on concrete, measurable improvements. Avoid vague suggestions.

Example good goals:
- "Add type hints to all public functions in the api/ module"
- "Increase test coverage for the database layer from 45% to 80%"
- "Add input validation to all API endpoints"

Example bad goals:
- "Improve code quality" (too vague)
- "Make it better" (not actionable)

Provide your response as a numbered list of goals, with the highest priority first.
"""

    def run(self, focus_areas: list[str] | None = None, **kwargs: Any) -> AgentResult:
        """Analyze the codebase and suggest improvement goals.

        Args:
            focus_areas: Optional list of specific areas to focus analysis on
            **kwargs: Additional arguments (unused)

        Returns:
            AgentResult with suggested goals in the 'suggestions' field
        """
        logger.info(f"Running pre-analysis on {self.working_dir}")

        prompt = self.ANALYSIS_PROMPT
        if focus_areas:
            focus_str = ", ".join(focus_areas)
            prompt += f"\n\nFocus your analysis particularly on: {focus_str}"

        response = self.claude.call(prompt)

        if not response.success:
            logger.error(f"Pre-analysis failed: {response.error}")
            return AgentResult(
                success=False,
                output="",
                error=response.error or "Pre-analysis failed",
            )

        suggestions = self._parse_list_response(response.result)
        logger.info(f"Pre-analysis complete, found {len(suggestions)} suggestions")

        return AgentResult(
            success=True,
            output=response.result,
            suggestions=suggestions,
            metadata={
                "focus_areas": focus_areas,
                "suggestion_count": len(suggestions),
            },
        )


class GoalDecomposerAgent(Agent):
    """Agent that breaks down large goals into smaller, actionable steps.

    This agent takes a high-level goal and decomposes it into a series of
    smaller tasks that can be executed sequentially. Each step should be
    specific enough to be completed in a single iteration.
    """

    name: str = "GoalDecomposerAgent"

    DECOMPOSE_PROMPT = """You are decomposing a large improvement goal into smaller steps.

GOAL: {goal}

Analyze this goal and break it down into a series of smaller, sequential steps.
Each step should:
1. Be specific and actionable
2. Be completable in a single iteration (typically 5-15 minutes of work)
3. Build on previous steps logically
4. Include clear success criteria

Consider:
- What needs to be done first (prerequisites)?
- What can be done in parallel vs. sequentially?
- What are the dependencies between steps?
- How can we validate each step was completed correctly?

Provide your response as a numbered list of steps in execution order.
For each step, provide a clear, imperative instruction that could be given to a developer.

Example step format:
1. Add type hints to all public functions in src/models/user.py
2. Create unit tests for the User model covering all public methods
3. Add docstrings to User class and all public methods
"""

    def run(self, goal: str, max_steps: int = 10, **kwargs: Any) -> AgentResult:
        """Decompose a goal into smaller steps.

        Args:
            goal: The high-level goal to decompose
            max_steps: Maximum number of steps to generate
            **kwargs: Additional arguments (unused)

        Returns:
            AgentResult with decomposed steps in the 'steps' field
        """
        if not goal:
            return AgentResult(
                success=False,
                output="",
                error="Goal is required for decomposition",
            )

        logger.info(f"Decomposing goal: {goal[:100]}...")

        prompt = self.DECOMPOSE_PROMPT.format(goal=goal)
        prompt += f"\n\nLimit your response to at most {max_steps} steps."

        response = self.claude.call(prompt)

        if not response.success:
            logger.error(f"Goal decomposition failed: {response.error}")
            return AgentResult(
                success=False,
                output="",
                error=response.error or "Goal decomposition failed",
            )

        steps = self._parse_list_response(response.result)

        # Limit to max_steps
        if len(steps) > max_steps:
            steps = steps[:max_steps]
            logger.warning(f"Truncated steps from {len(steps)} to {max_steps}")

        logger.info(f"Goal decomposed into {len(steps)} steps")

        return AgentResult(
            success=True,
            output=response.result,
            steps=steps,
            metadata={
                "original_goal": goal,
                "step_count": len(steps),
                "max_steps": max_steps,
            },
        )


class ReviewerAgent(Agent):
    """Agent that reviews changes after each iteration.

    This agent examines changes made during an improvement iteration and
    provides feedback on quality, correctness, and alignment with the goal.
    It can identify issues that automated tests might miss.
    """

    name: str = "ReviewerAgent"

    REVIEW_PROMPT = """You are reviewing changes made during an improvement iteration.

ORIGINAL GOAL: {goal}
ITERATION TASK: {task}

Review the changes that were just made to this codebase. Consider:

1. **Correctness**: Are the changes logically correct? Any bugs introduced?
2. **Goal Alignment**: Do the changes advance toward the stated goal?
3. **Code Quality**: Is the code clean, readable, and maintainable?
4. **Best Practices**: Are coding standards and best practices followed?
5. **Edge Cases**: Are edge cases handled appropriately?
6. **Security**: Any security concerns introduced?
7. **Performance**: Any performance regressions?
8. **Testing**: Are there adequate tests for the changes?

Provide your review as:
1. A brief summary (2-3 sentences) of what was changed
2. A list of any issues or concerns found
3. A list of suggestions for improvement
4. An overall assessment: APPROVED, NEEDS_CHANGES, or REJECTED

Be constructive and specific in your feedback.
"""

    def run(
        self,
        goal: str,
        task: str,
        changes_summary: str | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Review changes from an iteration.

        Args:
            goal: The overall improvement goal
            task: The specific task for this iteration
            changes_summary: Optional summary of changes made
            **kwargs: Additional arguments (unused)

        Returns:
            AgentResult with review feedback
        """
        if not goal or not task:
            return AgentResult(
                success=False,
                output="",
                error="Both goal and task are required for review",
            )

        logger.info(f"Reviewing changes for task: {task[:100]}...")

        prompt = self.REVIEW_PROMPT.format(goal=goal, task=task)
        if changes_summary:
            prompt += f"\n\nSUMMARY OF CHANGES:\n{changes_summary}"

        # Use git diff to see recent changes
        prompt += "\n\nUse git diff HEAD~1 to examine the recent changes if available."

        response = self.claude.call(prompt)

        if not response.success:
            logger.error(f"Review failed: {response.error}")
            return AgentResult(
                success=False,
                output="",
                error=response.error or "Review failed",
            )

        # Parse the review response for structured data
        result = response.result
        suggestions = []
        assessment = "UNKNOWN"

        # Try to extract assessment
        for line in result.split("\n"):
            line_upper = line.upper()
            if "APPROVED" in line_upper:
                assessment = "APPROVED"
            elif "NEEDS_CHANGES" in line_upper or "NEEDS CHANGES" in line_upper:
                assessment = "NEEDS_CHANGES"
            elif "REJECTED" in line_upper:
                assessment = "REJECTED"

        # Extract suggestions section if present
        in_suggestions = False
        for line in result.split("\n"):
            if "suggestion" in line.lower() and ":" in line:
                in_suggestions = True
                continue
            if in_suggestions:
                if line.strip().startswith(("-", "*", "1", "2", "3", "4", "5")):
                    suggestions.append(line.strip().lstrip("-*0123456789.) "))
                elif line.strip() and not line.strip()[0].isalnum():
                    continue
                elif line.strip() == "" or any(
                    kw in line.lower() for kw in ["assessment", "overall", "summary"]
                ):
                    in_suggestions = False

        logger.info(f"Review complete: {assessment}")

        return AgentResult(
            success=True,
            output=result,
            suggestions=suggestions,
            metadata={
                "goal": goal,
                "task": task,
                "assessment": assessment,
            },
        )


def create_agent(
    mode: AgentMode,
    working_dir: Path,
    model: str | None = None,
    timeout: int = 600,
) -> Agent:
    """Factory function to create an agent by mode.

    Args:
        mode: The agent mode to create
        working_dir: Repository directory for the agent
        model: Optional model override
        timeout: Timeout for Claude CLI calls

    Returns:
        An instance of the appropriate Agent subclass

    Raises:
        ValueError: If the mode is not recognized
    """
    agents: dict[AgentMode, type[Agent]] = {
        "pre-analysis": PreAnalysisAgent,
        "goal-decomposer": GoalDecomposerAgent,
        "reviewer": ReviewerAgent,
    }

    if mode not in agents:
        valid_modes = ", ".join(agents.keys())
        raise ValueError(f"Unknown agent mode '{mode}'. Valid modes: {valid_modes}")

    return agents[mode](working_dir=working_dir, model=model, timeout=timeout)
