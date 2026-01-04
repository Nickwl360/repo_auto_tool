"""Configuration for the repo improver."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ImproverConfig:
    """Configuration for a repo improvement session."""
    
    # Target repository
    repo_path: Path
    
    # The goal to achieve (natural language description)
    goal: str
    
    # Improvement loop settings
    max_iterations: int = 20
    max_consecutive_failures: int = 3
    
    # Claude Code settings
    allowed_tools: list[str] = field(default_factory=lambda: [
        "Bash(*)", "Read(*)", "Edit(*)", "Write(*)", "Glob(*)", "Grep(*)"
    ])
    model: str | None = None  # Use default if None
    
    # Validation settings
    run_tests: bool = True
    test_command: str = "pytest"
    run_linter: bool = True
    lint_command: str = "ruff check ."
    custom_validators: list[str] = field(default_factory=list)  # Shell commands
    
    # Git safety
    use_git: bool = True
    commit_each_iteration: bool = True
    branch_name: str = "repo-improver/auto"
    
    # State persistence
    state_file: Path | None = None  # Auto-set if None
    
    # Output settings
    verbose: bool = True
    log_file: Path | None = None
    output_format: Literal["text", "json"] = "text"
    
    def __post_init__(self):
        self.repo_path = Path(self.repo_path).resolve()
        if self.state_file is None:
            self.state_file = self.repo_path / ".repo-improver-state.json"
        if self.log_file is None:
            self.log_file = self.repo_path / ".repo-improver.log"
