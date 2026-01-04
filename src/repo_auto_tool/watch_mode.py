"""Watch mode for continuous repository monitoring and improvement.

This module provides functionality to monitor a repository for file changes
and trigger improvement iterations automatically. It supports:

- File system polling (no external dependencies required)
- Configurable watch patterns (include/exclude)
- Debouncing to prevent rapid-fire triggers
- Graceful error handling and recovery
- Integration with the main RepoImprover

Example:
    from repo_auto_tool.watch_mode import WatchMode
    from repo_auto_tool.config import ImproverConfig

    config = ImproverConfig(repo_path="/path/to/repo", goal="Improve code quality")
    watch = WatchMode(config, debounce_seconds=2.0)
    watch.start()  # Runs until Ctrl+C
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import ImproverConfig
from .exceptions import RepoAutoToolError

logger = logging.getLogger(__name__)


class WatchModeError(RepoAutoToolError):
    """Base exception for watch mode errors."""

    pass


class WatchModeAborted(WatchModeError):
    """Raised when watch mode is aborted by user."""

    pass


@dataclass
class WatchConfig:
    """Configuration for watch mode.

    Attributes:
        debounce_seconds: Minimum time between improvement triggers (default: 2.0)
        poll_interval: How often to check for changes in seconds (default: 1.0)
        include_patterns: Glob patterns for files to watch (default: common source files)
        exclude_patterns: Glob patterns for files to ignore (default: common non-source dirs)
        max_iterations_per_trigger: Max iterations per change detection (default: 5)
        cooldown_seconds: Minimum time between improvement runs (default: 30.0)
    """

    debounce_seconds: float = 2.0
    poll_interval: float = 1.0
    include_patterns: list[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.tsx", "*.jsx",
        "*.java", "*.go", "*.rs", "*.rb", "*.php",
        "*.c", "*.cpp", "*.h", "*.hpp",
        "*.md", "*.txt", "*.yaml", "*.yml", "*.json", "*.toml",
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        ".git/*", "__pycache__/*", "*.pyc", ".venv/*", "venv/*",
        "node_modules/*", ".next/*", "dist/*", "build/*",
        ".repo-improver-*", "*.log", ".pytest_cache/*",
        "*.egg-info/*", ".tox/*", ".mypy_cache/*",
    ])
    max_iterations_per_trigger: int = 5
    cooldown_seconds: float = 30.0


@dataclass
class FileState:
    """Represents the state of a file for change detection.

    Attributes:
        path: Path to the file
        mtime: Last modified time
        size: File size in bytes
        content_hash: MD5 hash of content (optional, for deeper comparison)
    """

    path: Path
    mtime: float
    size: int
    content_hash: str | None = None

    @classmethod
    def from_path(cls, path: Path, compute_hash: bool = False) -> FileState | None:
        """Create FileState from a file path.

        Args:
            path: Path to the file
            compute_hash: Whether to compute content hash (slower but more accurate)

        Returns:
            FileState if file exists and is readable, None otherwise.
        """
        try:
            stat = path.stat()
            content_hash = None

            if compute_hash:
                try:
                    content = path.read_bytes()
                    content_hash = hashlib.md5(content).hexdigest()
                except (OSError, PermissionError):
                    pass

            return cls(
                path=path,
                mtime=stat.st_mtime,
                size=stat.st_size,
                content_hash=content_hash,
            )
        except (OSError, PermissionError, FileNotFoundError):
            return None

    def has_changed(self, other: FileState | None) -> bool:
        """Check if this file state differs from another.

        Args:
            other: Previous file state to compare against

        Returns:
            True if file has changed, False otherwise.
        """
        if other is None:
            return True  # New file

        if self.mtime != other.mtime:
            return True

        if self.size != other.size:
            return True

        if self.content_hash and other.content_hash:
            return self.content_hash != other.content_hash

        return False


class FileWatcher:
    """Watches a directory for file changes using polling.

    This implementation uses polling rather than platform-specific file system
    events to avoid external dependencies and ensure cross-platform compatibility.

    Attributes:
        root: Root directory to watch
        config: Watch configuration
    """

    def __init__(
        self,
        root: Path,
        config: WatchConfig | None = None,
        on_change: Callable[[list[Path]], None] | None = None,
    ):
        """Initialize the file watcher.

        Args:
            root: Root directory to watch
            config: Watch configuration (uses defaults if not provided)
            on_change: Callback function when changes are detected
        """
        self.root = Path(root).resolve()
        self.config = config or WatchConfig()
        self.on_change = on_change

        self._file_states: dict[Path, FileState] = {}
        self._running = False
        self._last_change_time: float = 0
        self._pending_changes: list[Path] = []
        self._lock = threading.Lock()

    def _matches_pattern(self, path: Path, patterns: list[str]) -> bool:
        """Check if a path matches any of the given glob patterns.

        Args:
            path: Path to check
            patterns: List of glob patterns

        Returns:
            True if path matches any pattern.
        """
        # Get path relative to root for pattern matching
        try:
            rel_path = path.relative_to(self.root)
        except ValueError:
            return False

        rel_str = str(rel_path).replace(os.sep, "/")

        for pattern in patterns:
            # Match against filename
            if fnmatch.fnmatch(path.name, pattern):
                return True
            # Match against relative path
            if fnmatch.fnmatch(rel_str, pattern):
                return True
            # For directory patterns ending with /*, match any depth
            if pattern.endswith("/*"):
                dir_pattern = pattern[:-2]
                if rel_str.startswith(dir_pattern + "/") or fnmatch.fnmatch(rel_str, pattern):
                    return True

        return False

    def _should_watch_file(self, path: Path) -> bool:
        """Determine if a file should be watched.

        Args:
            path: Path to check

        Returns:
            True if file should be watched.
        """
        # Must be a file
        if not path.is_file():
            return False

        # Check exclude patterns first
        if self._matches_pattern(path, self.config.exclude_patterns):
            return False

        # Check include patterns
        if self._matches_pattern(path, self.config.include_patterns):
            return True

        return False

    def _scan_files(self) -> dict[Path, FileState]:
        """Scan the directory for watchable files.

        Returns:
            Dictionary mapping paths to their current state.
        """
        states: dict[Path, FileState] = {}

        try:
            for path in self.root.rglob("*"):
                if not self._should_watch_file(path):
                    continue

                state = FileState.from_path(path)
                if state:
                    states[path] = state
        except (OSError, PermissionError) as e:
            logger.warning(f"Error scanning directory: {e}")

        return states

    def _detect_changes(self) -> list[Path]:
        """Detect files that have changed since last scan.

        Returns:
            List of paths that have changed.
        """
        changed: list[Path] = []
        current_states = self._scan_files()

        # Check for new or modified files
        for path, state in current_states.items():
            old_state = self._file_states.get(path)
            if state.has_changed(old_state):
                changed.append(path)

        # Check for deleted files
        for path in self._file_states:
            if path not in current_states:
                changed.append(path)

        self._file_states = current_states
        return changed

    def _handle_changes(self, changes: list[Path]) -> None:
        """Handle detected file changes with debouncing.

        Args:
            changes: List of changed file paths
        """
        current_time = time.time()

        with self._lock:
            # Add to pending changes
            self._pending_changes.extend(changes)
            self._last_change_time = current_time

    def _process_pending_changes(self) -> None:
        """Process pending changes if debounce period has passed."""
        current_time = time.time()

        with self._lock:
            # Check if we have pending changes and debounce period has passed
            if not self._pending_changes:
                return

            time_since_last = current_time - self._last_change_time
            if time_since_last < self.config.debounce_seconds:
                return

            # Get and clear pending changes
            changes = list(set(self._pending_changes))  # Deduplicate
            self._pending_changes = []

        # Trigger callback outside the lock
        if changes and self.on_change:
            try:
                logger.info(f"Detected {len(changes)} file changes")
                for path in changes[:5]:  # Log first 5
                    logger.debug(f"  Changed: {path}")
                if len(changes) > 5:
                    logger.debug(f"  ... and {len(changes) - 5} more")

                self.on_change(changes)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

    def poll_once(self) -> list[Path]:
        """Perform a single poll for changes.

        Returns:
            List of changed files detected.
        """
        changes = self._detect_changes()
        if changes:
            self._handle_changes(changes)
        self._process_pending_changes()
        return changes

    def start(self, blocking: bool = True) -> threading.Thread | None:
        """Start watching for changes.

        Args:
            blocking: If True, blocks until stopped. If False, runs in background thread.

        Returns:
            Thread object if running in background, None if blocking.
        """
        self._running = True

        # Initial scan to establish baseline
        self._file_states = self._scan_files()
        logger.info(f"Watching {len(self._file_states)} files in {self.root}")

        if blocking:
            self._watch_loop()
            return None
        else:
            thread = threading.Thread(target=self._watch_loop, daemon=True)
            thread.start()
            return thread

    def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                self.poll_once()
                time.sleep(self.config.poll_interval)
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
                time.sleep(1)  # Brief pause on error

    def stop(self) -> None:
        """Stop watching for changes."""
        self._running = False
        logger.info("File watcher stopped")


class WatchMode:
    """Watch mode orchestrator that monitors files and triggers improvements.

    This class coordinates between the FileWatcher and RepoImprover to
    provide continuous improvement capabilities.

    Example:
        config = ImproverConfig(repo_path="/path/to/repo", goal="Improve tests")
        watch = WatchMode(config)
        watch.start()
    """

    def __init__(
        self,
        config: ImproverConfig,
        watch_config: WatchConfig | None = None,
    ):
        """Initialize watch mode.

        Args:
            config: Improver configuration
            watch_config: Watch-specific configuration
        """
        self.config = config
        self.watch_config = watch_config or WatchConfig()

        self._watcher: FileWatcher | None = None
        self._running = False
        self._last_run_time: float = 0
        self._total_triggers = 0
        self._total_improvements = 0
        self._original_signal_handler: signal.Handlers | None = None

    def _on_file_changes(self, changes: list[Path]) -> None:
        """Handle detected file changes.

        Args:
            changes: List of changed file paths
        """
        if not self._running:
            return

        # Check cooldown
        current_time = time.time()
        time_since_last = current_time - self._last_run_time

        if time_since_last < self.watch_config.cooldown_seconds:
            remaining = self.watch_config.cooldown_seconds - time_since_last
            logger.info(f"In cooldown period ({remaining:.1f}s remaining), skipping trigger")
            return

        self._total_triggers += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"WATCH MODE TRIGGER #{self._total_triggers}")
        logger.info(f"{'='*60}")
        logger.info(f"Files changed: {len(changes)}")

        # Run improvement iterations
        try:
            self._run_improvement_cycle(changes)
            self._last_run_time = time.time()
        except Exception as e:
            logger.error(f"Error during improvement cycle: {e}")
            # Continue watching despite errors

    def _run_improvement_cycle(self, changed_files: list[Path]) -> None:
        """Run a cycle of improvements triggered by file changes.

        Args:
            changed_files: List of files that triggered this cycle
        """
        # Import here to avoid circular imports
        from .improver import RepoImprover

        # Create a modified config for this cycle
        cycle_config = ImproverConfig(
            repo_path=self.config.repo_path,
            goal=self._enhance_goal_with_context(changed_files),
            max_iterations=self.watch_config.max_iterations_per_trigger,
            max_consecutive_failures=2,  # Be less patient in watch mode
            run_tests=self.config.run_tests,
            test_command=self.config.test_command,
            run_linter=self.config.run_linter,
            lint_command=self.config.lint_command,
            use_git=self.config.use_git,
            branch_name=self.config.branch_name,
            model=self.config.model,
            smart_model_selection=self.config.smart_model_selection,
            verbose=self.config.verbose,
            log_level=self.config.log_level,
        )

        try:
            improver = RepoImprover(cycle_config)
            result = improver.run()

            successful = sum(
                1 for it in result.iterations if it.success and it.validation_passed
            )
            self._total_improvements += successful

            logger.info(f"Cycle complete: {successful} successful improvements")
            logger.info(f"Total improvements in this watch session: {self._total_improvements}")

        except Exception as e:
            logger.error(f"Improvement cycle failed: {e}")
            # Don't raise - continue watching

    def _enhance_goal_with_context(self, changed_files: list[Path]) -> str:
        """Enhance the goal with context about changed files.

        Args:
            changed_files: Files that triggered this cycle

        Returns:
            Enhanced goal string
        """
        base_goal = self.config.goal

        # Add context about what changed
        if changed_files:
            file_names = [f.name for f in changed_files[:5]]
            context = f" Focus on recently changed files: {', '.join(file_names)}"
            if len(changed_files) > 5:
                context += f" (and {len(changed_files) - 5} others)"
            return base_goal + context

        return base_goal

    def _setup_signal_handler(self) -> None:
        """Setup signal handler for graceful shutdown."""
        def handler(signum: int, frame: object) -> None:
            logger.info("\nReceived interrupt signal, stopping watch mode...")
            self.stop()

        try:
            self._original_signal_handler = signal.signal(signal.SIGINT, handler)
        except (ValueError, OSError) as e:
            # Signal handling may not work in all contexts (e.g., threads)
            logger.warning(f"Could not setup signal handler: {e}")

    def _restore_signal_handler(self) -> None:
        """Restore original signal handler."""
        if self._original_signal_handler is not None:
            try:
                signal.signal(signal.SIGINT, self._original_signal_handler)
            except (ValueError, OSError):
                pass

    def start(self, blocking: bool = True) -> None:
        """Start watch mode.

        Args:
            blocking: If True, blocks until stopped. If False, returns immediately.
        """
        self._running = True
        self._setup_signal_handler()

        logger.info(f"\n{'='*60}")
        logger.info("STARTING WATCH MODE")
        logger.info(f"{'='*60}")
        logger.info(f"Repository: {self.config.repo_path}")
        logger.info(f"Goal: {self.config.goal}")
        logger.info(f"Poll interval: {self.watch_config.poll_interval}s")
        logger.info(f"Debounce: {self.watch_config.debounce_seconds}s")
        logger.info(f"Cooldown between runs: {self.watch_config.cooldown_seconds}s")
        logger.info(f"Max iterations per trigger: {self.watch_config.max_iterations_per_trigger}")
        logger.info("")
        logger.info("Press Ctrl+C to stop watching")
        logger.info(f"{'='*60}\n")

        # Create and start watcher
        self._watcher = FileWatcher(
            root=self.config.repo_path,
            config=self.watch_config,
            on_change=self._on_file_changes,
        )

        try:
            if blocking:
                self._watcher.start(blocking=True)
            else:
                self._watcher.start(blocking=False)
        except KeyboardInterrupt:
            logger.info("\nWatch mode interrupted by user")
        finally:
            self._print_summary()
            self._restore_signal_handler()

    def stop(self) -> None:
        """Stop watch mode."""
        self._running = False
        if self._watcher:
            self._watcher.stop()

    def _print_summary(self) -> None:
        """Print summary of watch session."""
        logger.info(f"\n{'='*60}")
        logger.info("WATCH MODE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total triggers: {self._total_triggers}")
        logger.info(f"Total successful improvements: {self._total_improvements}")
        logger.info(f"{'='*60}")


def run_watch_mode(
    config: ImproverConfig,
    debounce_seconds: float = 2.0,
    poll_interval: float = 1.0,
    cooldown_seconds: float = 30.0,
    max_iterations_per_trigger: int = 5,
) -> None:
    """Convenience function to run watch mode.

    Args:
        config: Improver configuration
        debounce_seconds: Minimum time between triggers
        poll_interval: How often to check for changes
        cooldown_seconds: Minimum time between improvement runs
        max_iterations_per_trigger: Max iterations per trigger
    """
    watch_config = WatchConfig(
        debounce_seconds=debounce_seconds,
        poll_interval=poll_interval,
        cooldown_seconds=cooldown_seconds,
        max_iterations_per_trigger=max_iterations_per_trigger,
    )

    watch = WatchMode(config=config, watch_config=watch_config)
    watch.start(blocking=True)
