"""State management for improvement sessions."""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Record of a single improvement iteration."""
    iteration: int
    timestamp: str
    prompt: str
    result: str
    success: bool
    validation_passed: bool
    git_commit: str | None = None
    error: str | None = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "IterationRecord":
        return cls(**data)


@dataclass  
class ImprovementState:
    """
    Persistent state for an improvement session.
    
    Tracks progress, allows resumption, and provides history for context.
    """
    goal: str
    repo_path: str
    status: Literal["running", "completed", "failed", "paused"] = "running"
    current_iteration: int = 0
    total_iterations: int = 0
    consecutive_failures: int = 0
    iterations: list[IterationRecord] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    summary: str | None = None
    
    def record_iteration(
        self,
        prompt: str,
        result: str,
        success: bool,
        validation_passed: bool,
        git_commit: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record the results of an iteration."""
        self.current_iteration += 1
        self.total_iterations += 1
        
        record = IterationRecord(
            iteration=self.current_iteration,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            result=result,
            success=success,
            validation_passed=validation_passed,
            git_commit=git_commit,
            error=error,
        )
        self.iterations.append(record)
        
        if success and validation_passed:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
    
    def get_recent_context(self, n: int = 3) -> str:
        """Get context from recent iterations for Claude."""
        if not self.iterations:
            return "No previous iterations."
        
        recent = self.iterations[-n:]
        context_parts = [f"Goal: {self.goal}", "", "Recent progress:"]
        
        for record in recent:
            status = "[OK]" if record.success and record.validation_passed else "[FAIL]"
            context_parts.append(
                f"  [{status}] Iteration {record.iteration}: {record.result[:200]}..."
            )
        
        return "\n".join(context_parts)
    
    def mark_complete(self, summary: str) -> None:
        """Mark the improvement session as complete."""
        self.status = "completed"
        self.completed_at = datetime.now().isoformat()
        self.summary = summary
    
    def mark_failed(self, reason: str) -> None:
        """Mark the improvement session as failed."""
        self.status = "failed"
        self.completed_at = datetime.now().isoformat()
        self.summary = f"Failed: {reason}"
    
    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        data = {
            "goal": self.goal,
            "repo_path": self.repo_path,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "consecutive_failures": self.consecutive_failures,
            "iterations": [it.to_dict() for it in self.iterations],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.debug(f"State saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "ImprovementState":
        """Load state from JSON file."""
        data = json.loads(path.read_text())
        iterations = [IterationRecord.from_dict(it) for it in data.pop("iterations", [])]
        state = cls(**data)
        state.iterations = iterations
        return state
    
    @classmethod
    def load_or_create(cls, path: Path, goal: str, repo_path: str) -> "ImprovementState":
        """Load existing state or create new one."""
        if path.exists():
            logger.info(f"Resuming from existing state: {path}")
            return cls.load(path)
        return cls(goal=goal, repo_path=repo_path)
