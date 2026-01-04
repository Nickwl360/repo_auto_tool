"""Git utilities for safe repo manipulation.

This module provides safe git operations for the improvement loop,
including branch management, commits, and rollback capabilities.
All git errors are raised as structured exceptions from the exceptions module.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .exceptions import GitNotInitializedError, GitOperationError

logger = logging.getLogger(__name__)


@dataclass
class GitStatus:
    """Current git status."""
    is_repo: bool
    current_branch: str | None
    has_changes: bool
    commit_hash: str | None


class GitHelper:
    """
    Git utilities for the improver.
    
    Provides safety features:
    - Work on a dedicated branch
    - Commit after each successful iteration
    - Rollback failed changes
    """
    
    def __init__(self, repo_path: Path, branch_name: str = "repo-improver/auto"):
        self.repo_path = Path(repo_path).resolve()
        self.branch_name = branch_name
        self.original_branch: str | None = None
    
    def _run(
        self, *args: str, check: bool = True, operation: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command.

        Args:
            *args: Git command arguments (without 'git' prefix).
            check: If True, raise GitOperationError on non-zero exit.
            operation: Human-readable operation name for error messages.
                If not provided, uses the first git argument.

        Returns:
            CompletedProcess with stdout/stderr.

        Raises:
            GitOperationError: If check=True and the command fails.
        """
        cmd = ["git", *args]
        op_name = operation or args[0] if args else "unknown"

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,  # We handle the check ourselves
            )

            if check and result.returncode != 0:
                raise GitOperationError(
                    operation=op_name,
                    reason=f"Exit code {result.returncode}",
                    output=result.stderr or result.stdout,
                )

            return result

        except OSError as e:
            raise GitOperationError(
                operation=op_name,
                reason=str(e),
            ) from e
    
    def get_status(self) -> GitStatus:
        """Get current git status.

        Returns:
            GitStatus with repository information.
            If not a git repo, returns GitStatus with is_repo=False.
        """
        # Check if it's a git repo
        try:
            self._run("rev-parse", "--git-dir", operation="check repository")
        except GitOperationError:
            return GitStatus(
                is_repo=False, current_branch=None, has_changes=False, commit_hash=None
            )

        # Get current branch
        try:
            result = self._run(
                "rev-parse", "--abbrev-ref", "HEAD", operation="get current branch"
            )
            branch = result.stdout.strip()
        except GitOperationError:
            branch = None

        # Check for uncommitted changes
        result = self._run("status", "--porcelain", check=False)
        has_changes = bool(result.stdout.strip())

        # Get current commit hash
        try:
            result = self._run("rev-parse", "HEAD", operation="get commit hash")
            commit_hash = result.stdout.strip()[:8]
        except GitOperationError:
            commit_hash = None

        return GitStatus(
            is_repo=True,
            current_branch=branch,
            has_changes=has_changes,
            commit_hash=commit_hash,
        )
    
    def setup_branch(self) -> str | None:
        """Create and checkout the improvement branch.

        Returns:
            The original branch name for later restoration, or None if not a git repo.

        Raises:
            GitNotInitializedError: If the directory is not a git repository.
            GitOperationError: If branch operations fail.
        """
        status = self.get_status()
        if not status.is_repo:
            raise GitNotInitializedError(self.repo_path)

        self.original_branch = status.current_branch

        # Stash any uncommitted changes
        if status.has_changes:
            logger.info("Stashing uncommitted changes")
            self._run("stash", "push", "-m", "repo-improver: auto-stash", operation="stash")

        # Check if branch exists
        result = self._run("branch", "--list", self.branch_name, check=False)
        branch_exists = bool(result.stdout.strip())

        if branch_exists:
            logger.info(f"Checking out existing branch: {self.branch_name}")
            self._run("checkout", self.branch_name, operation="checkout branch")
        else:
            logger.info(f"Creating new branch: {self.branch_name}")
            self._run("checkout", "-b", self.branch_name, operation="create branch")

        return self.original_branch
    
    def commit(self, message: str) -> str | None:
        """Commit current changes.

        Args:
            message: Commit message (will be prefixed with [repo-improver]).

        Returns:
            Commit hash (short form) or None if nothing to commit.

        Raises:
            GitOperationError: If the commit operation fails.
        """
        status = self.get_status()
        if not status.is_repo:
            return None

        if not status.has_changes:
            logger.debug("No changes to commit")
            return None

        self._run("add", "-A", operation="stage changes")
        self._run("commit", "-m", f"[repo-improver] {message}", operation="commit")

        result = self._run("rev-parse", "HEAD", operation="get commit hash")
        commit_hash = result.stdout.strip()[:8]
        logger.info(f"Committed: {commit_hash} - {message[:50]}")

        return commit_hash
    
    def rollback(self) -> bool:
        """Rollback uncommitted changes.

        Restores tracked files with checkout and removes untracked files with clean.
        Handles failures gracefully (e.g., locked files on Windows).

        Returns:
            True if rollback was performed (even if partial), False if no changes.
        """
        status = self.get_status()
        if not status.is_repo or not status.has_changes:
            return False

        logger.warning("Rolling back uncommitted changes")

        # Restore tracked files
        try:
            self._run("checkout", ".", operation="restore tracked files")
        except GitOperationError as e:
            logger.warning(f"Partial rollback - checkout failed: {e.reason}")

        # Remove untracked files (may fail on Windows with locked files)
        try:
            self._run("clean", "-fd", operation="remove untracked files")
        except GitOperationError as e:
            # Log but don't fail - some files may be locked
            logger.warning(f"Partial rollback - clean failed: {e.reason}")
            # Try a less aggressive clean
            self._run("clean", "-f", check=False)

        return True
    
    def restore_original_branch(self) -> None:
        """Restore the original branch.

        Silently handles failures (e.g., if branch was deleted).
        """
        if self.original_branch:
            logger.info(f"Restoring original branch: {self.original_branch}")
            try:
                self._run(
                    "checkout", self.original_branch, operation="restore original branch"
                )
            except GitOperationError as e:
                logger.warning(f"Could not restore original branch: {e.reason}")

    def get_diff_summary(self) -> str:
        """Get a summary of current changes.

        Returns:
            Git diff --stat output or descriptive message if no changes/not a repo.
        """
        status = self.get_status()
        if not status.is_repo:
            return "Not a git repository"

        result = self._run("diff", "--stat", check=False)
        return result.stdout if result.stdout else "No changes"
