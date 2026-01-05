"""Named session management for multiple concurrent investigations.

Allows users to:
- Create named sessions
- Resume by name instead of path
- Track multiple concurrent investigations
- Compare session results
- Archive completed work

Example:
    manager = SessionManager()

    # Create named session
    session = manager.create_session(
        name="type-hints-experiment",
        repo_path="/path/to/repo",
        goal="Add type hints to all functions"
    )

    # Resume by name
    session = manager.resume_session("type-hints-experiment")

    # List all sessions
    sessions = manager.list_sessions()
    for s in sessions:
        print(f"{s.name}: {s.status} ({s.iterations} iterations)")

    # Compare sessions
    report = manager.compare_sessions(["exp1", "exp2"])
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default session storage directory
DEFAULT_SESSION_DIR = Path.home() / ".repo-improver" / "sessions"


@dataclass
class SessionMetadata:
    """Metadata for a session.

    Attributes:
        name: Session name (user-friendly identifier).
        session_id: Unique session ID.
        repo_path: Path to the repository.
        goal: The improvement goal.
        status: Current status (running, completed, failed, paused).
        started_at: ISO timestamp of session start.
        completed_at: ISO timestamp of completion (if done).
        iterations: Number of iterations completed.
        total_cost: Total cost in USD.
        success_rate: Success rate (0.0-1.0).
        tags: Optional tags for categorization.
        is_research: Whether this is a research session.
        research_question: Optional research question.
        hypothesis: Optional hypothesis being tested.
    """
    name: str
    session_id: str
    repo_path: str
    goal: str
    status: str
    started_at: str
    completed_at: str | None = None
    iterations: int = 0
    total_cost: float = 0.0
    success_rate: float = 0.0
    tags: list[str] = field(default_factory=list)
    is_research: bool = False
    research_question: str | None = None
    hypothesis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "session_id": self.session_id,
            "repo_path": self.repo_path,
            "goal": self.goal,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "iterations": self.iterations,
            "total_cost": self.total_cost,
            "success_rate": self.success_rate,
            "tags": self.tags,
            "is_research": self.is_research,
            "research_question": self.research_question,
            "hypothesis": self.hypothesis,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            session_id=data["session_id"],
            repo_path=data["repo_path"],
            goal=data["goal"],
            status=data["status"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            iterations=data.get("iterations", 0),
            total_cost=data.get("total_cost", 0.0),
            success_rate=data.get("success_rate", 0.0),
            tags=data.get("tags", []),
            is_research=data.get("is_research", False),
            research_question=data.get("research_question"),
            hypothesis=data.get("hypothesis"),
        )


@dataclass
class SessionComparison:
    """Comparison between multiple sessions.

    Attributes:
        sessions: List of session names compared.
        metrics: Dictionary of metric comparisons.
        winner: Session with best overall performance.
        summary: Human-readable summary.
    """
    sessions: list[str]
    metrics: dict[str, dict[str, Any]]
    winner: str | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sessions": self.sessions,
            "metrics": self.metrics,
            "winner": self.winner,
            "summary": self.summary,
        }


class SessionManager:
    """Manages named sessions for multiple concurrent investigations.

    Attributes:
        sessions_dir: Directory where sessions are stored.
    """

    def __init__(self, sessions_dir: Path | None = None):
        self.sessions_dir = sessions_dir or DEFAULT_SESSION_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"SessionManager initialized with dir: {self.sessions_dir}")

    def _generate_session_id(self, name: str) -> str:
        """Generate unique session ID.

        Args:
            name: Session name.

        Returns:
            Unique ID like "name_20260104_143022"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return f"{safe_name}_{timestamp}"

    def _get_session_path(self, session_id: str) -> Path:
        """Get path to session directory.

        Args:
            session_id: The session ID.

        Returns:
            Path to session directory.
        """
        return self.sessions_dir / session_id

    def _get_metadata_path(self, session_id: str) -> Path:
        """Get path to metadata file.

        Args:
            session_id: The session ID.

        Returns:
            Path to metadata.json file.
        """
        return self._get_session_path(session_id) / "metadata.json"

    def create_session(
        self,
        name: str,
        repo_path: Path | str,
        goal: str,
        tags: list[str] | None = None,
        is_research: bool = False,
        research_question: str | None = None,
        hypothesis: str | None = None,
    ) -> SessionMetadata:
        """Create a new named session.

        Args:
            name: User-friendly session name.
            repo_path: Path to the repository.
            goal: The improvement goal.
            tags: Optional tags for categorization.
            is_research: Whether this is a research session.
            research_question: Optional research question.
            hypothesis: Optional hypothesis being tested.

        Returns:
            SessionMetadata for the new session.

        Raises:
            ValueError: If session with this name already exists.
        """
        # Check if session with this name already exists
        existing = self.find_session_by_name(name)
        if existing and existing.status not in ("completed", "failed"):
            raise ValueError(
                f"Session '{name}' already exists with status '{existing.status}'. "
                "Choose a different name or resume the existing session."
            )

        # Generate unique ID
        session_id = self._generate_session_id(name)

        # Create session directory
        session_path = self._get_session_path(session_id)
        session_path.mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = SessionMetadata(
            name=name,
            session_id=session_id,
            repo_path=str(Path(repo_path).resolve()),
            goal=goal,
            status="running",
            started_at=datetime.now().isoformat(),
            tags=tags or [],
            is_research=is_research,
            research_question=research_question,
            hypothesis=hypothesis,
        )

        # Save metadata
        self._save_metadata(metadata)

        logger.info(f"Created session '{name}' with ID: {session_id}")

        return metadata

    def _save_metadata(self, metadata: SessionMetadata) -> None:
        """Save session metadata.

        Args:
            metadata: The metadata to save.
        """
        metadata_path = self._get_metadata_path(metadata.session_id)
        metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2))

    def _load_metadata(self, session_id: str) -> SessionMetadata | None:
        """Load session metadata.

        Args:
            session_id: The session ID.

        Returns:
            SessionMetadata or None if not found.
        """
        metadata_path = self._get_metadata_path(session_id)
        if not metadata_path.exists():
            return None

        try:
            data = json.loads(metadata_path.read_text())
            return SessionMetadata.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load metadata for {session_id}: {e}")
            return None

    def find_session_by_name(self, name: str) -> SessionMetadata | None:
        """Find most recent session with given name.

        Args:
            name: Session name to search for.

        Returns:
            SessionMetadata or None if not found.
        """
        sessions = self.list_sessions()

        # Filter by name
        matches = [s for s in sessions if s.name == name]

        if not matches:
            return None

        # Return most recent (by started_at)
        return max(matches, key=lambda s: s.started_at)

    def resume_session(self, name: str) -> SessionMetadata:
        """Resume a session by name.

        Args:
            name: Session name to resume.

        Returns:
            SessionMetadata for the session.

        Raises:
            ValueError: If session not found.
        """
        metadata = self.find_session_by_name(name)

        if not metadata:
            raise ValueError(
                f"Session '{name}' not found. Use --list-sessions to see available sessions."
            )

        logger.info(f"Resuming session '{name}' (ID: {metadata.session_id})")

        return metadata

    def list_sessions(
        self,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[SessionMetadata]:
        """List all sessions, optionally filtered.

        Args:
            status: Filter by status (running, completed, failed, paused).
            tags: Filter by tags (match any).

        Returns:
            List of SessionMetadata.
        """
        sessions: list[SessionMetadata] = []

        # Scan session directories
        if not self.sessions_dir.exists():
            return sessions

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            metadata = self._load_metadata(session_dir.name)
            if metadata:
                # Apply filters
                if status and metadata.status != status:
                    continue
                if tags and not any(t in metadata.tags for t in tags):
                    continue

                sessions.append(metadata)

        # Sort by started_at (most recent first)
        sessions.sort(key=lambda s: s.started_at, reverse=True)

        return sessions

    def get_session_info(self, name: str) -> dict[str, Any]:
        """Get detailed information about a session.

        Args:
            name: Session name.

        Returns:
            Dictionary with session information.

        Raises:
            ValueError: If session not found.
        """
        metadata = self.find_session_by_name(name)

        if not metadata:
            raise ValueError(f"Session '{name}' not found.")

        session_path = self._get_session_path(metadata.session_id)

        # Check what files exist
        has_state = (session_path / "state.json").exists()
        has_history = (session_path / "history.json").exists()
        has_prompts = (session_path / "prompts.json").exists()
        has_report = (session_path / "research_report.md").exists()

        return {
            "metadata": metadata.to_dict(),
            "session_path": str(session_path),
            "has_state": has_state,
            "has_history": has_history,
            "has_prompts": has_prompts,
            "has_report": has_report,
            "files": [f.name for f in session_path.iterdir()],
        }

    def update_session_status(
        self,
        session_id: str,
        status: str,
        iterations: int | None = None,
        total_cost: float | None = None,
        success_rate: float | None = None,
    ) -> None:
        """Update session status and metrics.

        Args:
            session_id: The session ID.
            status: New status.
            iterations: Number of iterations (if known).
            total_cost: Total cost (if known).
            success_rate: Success rate (if known).
        """
        metadata = self._load_metadata(session_id)
        if not metadata:
            logger.warning(f"Cannot update metadata for unknown session: {session_id}")
            return

        metadata.status = status
        if status in ("completed", "failed", "paused"):
            metadata.completed_at = datetime.now().isoformat()

        if iterations is not None:
            metadata.iterations = iterations
        if total_cost is not None:
            metadata.total_cost = total_cost
        if success_rate is not None:
            metadata.success_rate = success_rate

        self._save_metadata(metadata)

    def compare_sessions(self, names: list[str]) -> SessionComparison:
        """Compare multiple sessions.

        Args:
            names: List of session names to compare.

        Returns:
            SessionComparison with analysis.

        Raises:
            ValueError: If any session not found.
        """
        # Load all sessions
        sessions: list[SessionMetadata] = []
        for name in names:
            metadata = self.find_session_by_name(name)
            if not metadata:
                raise ValueError(f"Session '{name}' not found.")
            sessions.append(metadata)

        # Compare metrics
        metrics: dict[str, dict[str, Any]] = {}

        for session in sessions:
            metrics[session.name] = {
                "iterations": session.iterations,
                "cost": session.total_cost,
                "success_rate": session.success_rate,
                "status": session.status,
                "duration": self._calculate_duration(session),
            }

        # Determine winner (lowest cost with completed status)
        completed = [s for s in sessions if s.status == "completed"]
        if completed:
            winner = min(completed, key=lambda s: s.total_cost)
            winner_name = winner.name
        else:
            winner_name = None

        # Generate summary
        summary_lines = []
        summary_lines.append(f"Comparison of {len(sessions)} sessions:")
        for session in sessions:
            m = metrics[session.name]
            summary_lines.append(
                f"  {session.name}: {m['iterations']} iterations, "
                f"${m['cost']:.2f}, {m['success_rate']:.0%} success, "
                f"{m['status']}"
            )

        if winner_name:
            summary_lines.append(f"\nBest: {winner_name} (lowest cost, completed)")

        summary = "\n".join(summary_lines)

        return SessionComparison(
            sessions=names,
            metrics=metrics,
            winner=winner_name,
            summary=summary,
        )

    def _calculate_duration(self, metadata: SessionMetadata) -> str:
        """Calculate session duration.

        Args:
            metadata: Session metadata.

        Returns:
            Duration string like "2h 15m".
        """
        try:
            start = datetime.fromisoformat(metadata.started_at)
            end = (
                datetime.fromisoformat(metadata.completed_at)
                if metadata.completed_at
                else datetime.now()
            )

            delta = end - start
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60

            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"

        except (ValueError, TypeError):
            return "unknown"

    def archive_session(self, name: str, archive_dir: Path | None = None) -> Path:
        """Archive a completed session.

        Args:
            name: Session name to archive.
            archive_dir: Directory to archive to (default: sessions_dir/archive).

        Returns:
            Path to archived session.

        Raises:
            ValueError: If session not found or not completed.
        """
        metadata = self.find_session_by_name(name)

        if not metadata:
            raise ValueError(f"Session '{name}' not found.")

        if metadata.status not in ("completed", "failed"):
            raise ValueError(
                f"Can only archive completed or failed sessions. "
                f"Session '{name}' is '{metadata.status}'."
            )

        # Create archive directory
        archive_base = archive_dir or (self.sessions_dir / "archive")
        archive_base.mkdir(parents=True, exist_ok=True)

        # Move session directory
        session_path = self._get_session_path(metadata.session_id)
        archived_path = archive_base / metadata.session_id

        shutil.move(str(session_path), str(archived_path))

        logger.info(f"Archived session '{name}' to {archived_path}")

        return archived_path

    def delete_session(self, name: str) -> None:
        """Delete a session permanently.

        Args:
            name: Session name to delete.

        Raises:
            ValueError: If session not found.
        """
        metadata = self.find_session_by_name(name)

        if not metadata:
            raise ValueError(f"Session '{name}' not found.")

        session_path = self._get_session_path(metadata.session_id)

        if session_path.exists():
            shutil.rmtree(session_path)
            logger.info(f"Deleted session '{name}' (ID: {metadata.session_id})")

    def get_state_file(self, session_id: str) -> Path:
        """Get path to state file for a session.

        Args:
            session_id: The session ID.

        Returns:
            Path to state.json file.
        """
        return self._get_session_path(session_id) / "state.json"

    def get_prompts_file(self, session_id: str) -> Path:
        """Get path to learned prompts file for a session.

        Args:
            session_id: The session ID.

        Returns:
            Path to prompts.json file.
        """
        return self._get_session_path(session_id) / "prompts.json"

    def get_research_dir(self, session_id: str) -> Path:
        """Get path to research output directory for a session.

        Args:
            session_id: The session ID.

        Returns:
            Path to research directory.
        """
        return self._get_session_path(session_id) / "research"

    def save_research_findings(
        self,
        session_id: str,
        findings: dict[str, Any],
    ) -> None:
        """Save research findings to session.

        Args:
            session_id: The session ID.
            findings: Research findings dictionary.
        """
        research_dir = self.get_research_dir(session_id)
        research_dir.mkdir(exist_ok=True)

        findings_file = research_dir / "findings.json"
        findings_file.write_text(json.dumps(findings, indent=2))

        logger.info(f"Saved research findings to {findings_file}")
