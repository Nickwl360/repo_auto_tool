"""Git utilities for safe repo manipulation."""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
    
    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command."""
        cmd = ["git", *args]
        return subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=check,
        )
    
    def get_status(self) -> GitStatus:
        """Get current git status."""
        # Check if it's a git repo
        try:
            self._run("rev-parse", "--git-dir")
        except subprocess.CalledProcessError:
            return GitStatus(
                is_repo=False, current_branch=None, has_changes=False, commit_hash=None
            )
        
        # Get current branch
        try:
            result = self._run("rev-parse", "--abbrev-ref", "HEAD")
            branch = result.stdout.strip()
        except subprocess.CalledProcessError:
            branch = None
        
        # Check for uncommitted changes
        result = self._run("status", "--porcelain", check=False)
        has_changes = bool(result.stdout.strip())
        
        # Get current commit hash
        try:
            result = self._run("rev-parse", "HEAD")
            commit_hash = result.stdout.strip()[:8]
        except subprocess.CalledProcessError:
            commit_hash = None
        
        return GitStatus(
            is_repo=True,
            current_branch=branch,
            has_changes=has_changes,
            commit_hash=commit_hash,
        )
    
    def setup_branch(self) -> str | None:
        """
        Create and checkout the improvement branch.
        
        Returns the original branch name for later restoration.
        """
        status = self.get_status()
        if not status.is_repo:
            logger.warning("Not a git repository, skipping git operations")
            return None
        
        self.original_branch = status.current_branch
        
        # Stash any uncommitted changes
        if status.has_changes:
            logger.info("Stashing uncommitted changes")
            self._run("stash", "push", "-m", "repo-improver: auto-stash")
        
        # Check if branch exists
        result = self._run("branch", "--list", self.branch_name, check=False)
        branch_exists = bool(result.stdout.strip())
        
        if branch_exists:
            logger.info(f"Checking out existing branch: {self.branch_name}")
            self._run("checkout", self.branch_name)
        else:
            logger.info(f"Creating new branch: {self.branch_name}")
            self._run("checkout", "-b", self.branch_name)
        
        return self.original_branch
    
    def commit(self, message: str) -> str | None:
        """
        Commit current changes.
        
        Returns commit hash or None if nothing to commit.
        """
        status = self.get_status()
        if not status.is_repo:
            return None
        
        if not status.has_changes:
            logger.debug("No changes to commit")
            return None
        
        self._run("add", "-A")
        self._run("commit", "-m", f"[repo-improver] {message}")
        
        result = self._run("rev-parse", "HEAD")
        commit_hash = result.stdout.strip()[:8]
        logger.info(f"Committed: {commit_hash} - {message[:50]}")
        
        return commit_hash
    
    def rollback(self) -> bool:
        """
        Rollback uncommitted changes.

        Restores tracked files with checkout and removes untracked files with clean.
        Handles failures gracefully (e.g., locked files on Windows).

        Returns True if rollback was performed (even if partial).
        """
        status = self.get_status()
        if not status.is_repo or not status.has_changes:
            return False

        logger.warning("Rolling back uncommitted changes")

        # Restore tracked files
        try:
            self._run("checkout", ".")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Partial rollback - checkout failed: {e.stderr}")

        # Remove untracked files (may fail on Windows with locked files)
        try:
            self._run("clean", "-fd")
        except subprocess.CalledProcessError as e:
            # Log but don't fail - some files may be locked
            logger.warning(f"Partial rollback - clean failed: {e.stderr}")
            # Try a less aggressive clean
            try:
                self._run("clean", "-f", check=False)
            except subprocess.CalledProcessError:
                pass

        return True
    
    def restore_original_branch(self) -> None:
        """Restore the original branch."""
        if self.original_branch:
            logger.info(f"Restoring original branch: {self.original_branch}")
            self._run("checkout", self.original_branch, check=False)
    
    def get_diff_summary(self) -> str:
        """Get a summary of current changes."""
        status = self.get_status()
        if not status.is_repo:
            return "Not a git repository"
        
        result = self._run("diff", "--stat", check=False)
        return result.stdout if result.stdout else "No changes"
