"""Simple Terminal UI for repo-improver.

Provides a straightforward progress display without complex curses dependencies.
Shows real-time status, progress, and logs.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import ImprovementState, IterationRecord

logger = logging.getLogger(__name__)


class TUIError(Exception):
    """Base exception for TUI-related errors."""
    pass


class TUINotAvailableError(TUIError):
    """Raised when TUI cannot be initialized."""
    pass


class PanelType(Enum):
    """Types of panels in the TUI."""
    STATUS = "status"
    PROGRESS = "progress"
    LOGS = "logs"
    FILES = "files"
    DIFF = "diff"
    HELP = "help"


class KeyAction(Enum):
    """Keyboard actions supported by the TUI."""
    QUIT = "quit"
    PAUSE = "pause"
    RESUME = "resume"
    SKIP = "skip"
    ADJUST_GOAL = "adjust_goal"
    SHOW_HELP = "show_help"
    TOGGLE_LOGS = "toggle_logs"
    TOGGLE_DIFF = "toggle_diff"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    REFRESH = "refresh"


@dataclass
class TUIConfig:
    """Configuration for the TUI."""
    refresh_rate: float = 1.0
    log_buffer_size: int = 100
    show_diff_preview: bool = True
    show_file_changes: bool = True
    enable_colors: bool = True
    min_width: int = 80
    min_height: int = 24

    def __post_init__(self) -> None:
        self.refresh_rate = max(0.5, min(5.0, self.refresh_rate or 1.0))
        self.log_buffer_size = max(10, min(10000, self.log_buffer_size or 100))


@dataclass
class TUIState:
    """Current state of the TUI display."""
    current_iteration: int = 0
    max_iterations: int = 20
    current_task: str = "Initializing..."
    goal: str = ""
    status: str = "starting"
    is_paused: bool = False
    is_running: bool = True
    successful_iterations: int = 0
    failed_iterations: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0
    log_lines: list[str] = field(default_factory=list)
    log_scroll_offset: int = 0
    changed_files: list[str] = field(default_factory=list)
    diff_preview: str = ""
    active_panel: PanelType = PanelType.STATUS
    show_help: bool = False
    last_update: float = field(default_factory=time.time)

    def add_log(self, message: str, max_lines: int = 100) -> None:
        """Add a log message."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted = f"[{timestamp}] {message}"
            self.log_lines.append(formatted)
            if len(self.log_lines) > max_lines:
                self.log_lines = self.log_lines[-max_lines:]
        except Exception:
            pass

    def get_progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.max_iterations <= 0:
            return 0.0
        return min(100.0, (self.current_iteration / self.max_iterations) * 100)

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_iterations + self.failed_iterations
        if total <= 0:
            return 0.0
        return (self.successful_iterations / total) * 100


class TUI:
    """Simple Terminal UI that prints progress updates."""

    def __init__(
        self,
        config: TUIConfig | None = None,
        use_plain_text: bool = False,  # Ignored, always plain text now
    ) -> None:
        self.config = config or TUIConfig()
        self.state = TUIState()
        self._running = False
        self._lock = threading.Lock()
        self._initialized = False
        self._last_status = ""
        self._last_iteration = -1
        self._spinner_idx = 0
        self._spinner = ["|", "/", "-", "\\"]

    def initialize(self) -> bool:
        """Initialize the TUI."""
        self._initialized = True
        return True

    def cleanup(self) -> None:
        """Clean up resources."""
        self._running = False
        self._initialized = False

    def start(self) -> None:
        """Start the TUI (no-op for simple mode)."""
        if not self._initialized:
            self.initialize()
        self._running = True

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False

    def render_once(self) -> None:
        """Print current status if changed."""
        if not self._initialized:
            return

        with self._lock:
            # Only print if something changed
            status_key = f"{self.state.status}:{self.state.current_iteration}"
            if status_key == self._last_status:
                return
            self._last_status = status_key

            # Build status line
            progress = self.state.get_progress_percent()
            spinner = self._spinner[self._spinner_idx % len(self._spinner)]
            self._spinner_idx += 1

            # Progress bar
            bar_width = 20
            filled = int(bar_width * progress / 100)
            bar = "#" * filled + "-" * (bar_width - filled)

            # Status line
            status = self.state.status.upper()
            iter_info = f"{self.state.current_iteration}/{self.state.max_iterations}"

            line = f"\r{spinner} [{bar}] {progress:5.1f}% | {status:<12} | Iter: {iter_info}"

            # Add metrics
            if self.state.successful_iterations or self.state.failed_iterations:
                ok = self.state.successful_iterations
                fail = self.state.failed_iterations
                line += f" | OK:{ok} Fail:{fail}"

            if self.state.total_cost > 0:
                line += f" | ${self.state.total_cost:.2f}"

            # Print with padding to clear previous content
            print(f"{line:<100}", end="", flush=True)

            # Print task on new line if iteration changed
            if self.state.current_iteration != self._last_iteration:
                self._last_iteration = self.state.current_iteration
                if self.state.current_task:
                    task = self.state.current_task[:70]
                    if len(self.state.current_task) > 70:
                        task += "..."
                    print(f"\n  Task: {task}")

    def _print_header(self) -> None:
        """Print header line."""
        print("\n" + "=" * 60)
        print("  REPO-IMPROVER  ".center(60, "="))
        print("=" * 60)
        if self.state.goal:
            goal = self.state.goal[:55] + "..." if len(self.state.goal) > 55 else self.state.goal
            print(f"Goal: {goal}")
        print("-" * 60)

    def _print_summary(self) -> None:
        """Print final summary."""
        print("\n" + "=" * 60)
        print("SESSION SUMMARY".center(60))
        print("=" * 60)
        print(f"  Status: {self.state.status.upper()}")
        print(f"  Iterations: {self.state.current_iteration}/{self.state.max_iterations}")
        print(f"  Successful: {self.state.successful_iterations}")
        print(f"  Failed: {self.state.failed_iterations}")
        if self.state.total_cost > 0:
            print(f"  Total Cost: ${self.state.total_cost:.4f}")
        if self.state.changed_files:
            print(f"  Files Changed: {len(self.state.changed_files)}")
            for f in self.state.changed_files[:10]:
                print(f"    - {f}")
            if len(self.state.changed_files) > 10:
                print(f"    ... and {len(self.state.changed_files) - 10} more")
        print("=" * 60 + "\n")

    # State update methods
    def set_goal(self, goal: str) -> None:
        with self._lock:
            self.state.goal = goal or ""
            self._print_header()

    def set_status(self, status: str) -> None:
        with self._lock:
            old_status = self.state.status
            self.state.status = status or "unknown"
            # Print on significant status changes
            if status in ("complete", "failed", "paused") and status != old_status:
                print(f"\n>>> Status: {status.upper()}")
                if status == "complete":
                    self._print_summary()

    def set_task(self, task: str) -> None:
        with self._lock:
            self.state.current_task = task or ""

    def set_iteration(self, current: int, maximum: int) -> None:
        with self._lock:
            self.state.current_iteration = max(0, current)
            self.state.max_iterations = max(1, maximum)

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.state.is_paused = paused
            if paused:
                print("\n>>> PAUSED - Resume to continue")

    def add_success(self) -> None:
        with self._lock:
            self.state.successful_iterations += 1

    def add_failure(self) -> None:
        with self._lock:
            self.state.failed_iterations += 1

    def set_cost(self, cost: float) -> None:
        with self._lock:
            self.state.total_cost = max(0.0, cost)

    def add_log(self, message: str) -> None:
        with self._lock:
            self.state.add_log(message, self.config.log_buffer_size)
            # Print important logs
            if any(kw in message.lower() for kw in ["error", "fail", "success", "complete"]):
                print(f"\n  {message[:80]}")

    def set_changed_files(self, files: list[str]) -> None:
        with self._lock:
            self.state.changed_files = list(files or [])

    def add_changed_file(self, filepath: str) -> None:
        with self._lock:
            if filepath and filepath not in self.state.changed_files:
                self.state.changed_files.append(filepath)
                print(f"\n  Changed: {filepath}")

    def set_diff_preview(self, diff: str) -> None:
        with self._lock:
            self.state.diff_preview = diff or ""

    def update_from_state(self, improvement_state: ImprovementState) -> None:
        """Update TUI from an ImprovementState object."""
        try:
            with self._lock:
                self.state.goal = getattr(improvement_state, "goal", "") or ""
                self.state.current_iteration = getattr(improvement_state, "current_iteration", 0)
                iterations = getattr(improvement_state, "iterations", []) or []
                self.state.successful_iterations = sum(
                    1 for it in iterations if getattr(it, "status", "") == "success"
                )
                self.state.failed_iterations = sum(
                    1 for it in iterations if getattr(it, "status", "") == "failed"
                )
                changed: set[str] = set()
                for it in iterations:
                    files_changed = getattr(it, "files_changed", []) or []
                    changed.update(files_changed)
                self.state.changed_files = list(changed)
        except Exception as e:
            logger.debug(f"Error updating from state: {e}")

    def update_from_iteration(self, iteration: IterationRecord) -> None:
        """Update TUI from an iteration record."""
        try:
            with self._lock:
                status = getattr(iteration, "status", "unknown")
                self.state.status = status
                if status == "success":
                    self.state.successful_iterations += 1
                elif status == "failed":
                    self.state.failed_iterations += 1
                files_changed = getattr(iteration, "files_changed", []) or []
                for f in files_changed:
                    if f not in self.state.changed_files:
                        self.state.changed_files.append(f)
                summary = getattr(iteration, "summary", "") or ""
                if summary:
                    self.state.add_log(summary[:100], self.config.log_buffer_size)
        except Exception as e:
            logger.debug(f"Error updating from iteration: {e}")

    # Compatibility methods for old TUI interface
    def can_use_curses(self) -> bool:
        return False  # Never use curses

    def register_callback(self, action: KeyAction, callback: Any) -> None:
        pass  # No keyboard callbacks in simple mode


def create_tui(
    use_tui: bool = True,
    config: TUIConfig | None = None,
) -> TUI | None:
    """Create a simple TUI."""
    if not use_tui:
        return None
    try:
        tui = TUI(config=config)
        if tui.initialize():
            return tui
        return None
    except Exception as e:
        logger.debug(f"Failed to create TUI: {e}")
        return None


def create_plain_text_tui(config: TUIConfig | None = None) -> TUI:
    """Create a TUI (always plain text now)."""
    tui = TUI(config=config)
    tui.initialize()
    return tui


class TUILogHandler(logging.Handler):
    """Logging handler that sends messages to the TUI."""

    def __init__(self, tui: TUI, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._tui = tui

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if len(msg) > 200:
                msg = msg[:197] + "..."
            self._tui.add_log(msg)
        except Exception:
            pass


def setup_tui_logging(tui: TUI, level: int = logging.INFO) -> TUILogHandler:
    """Setup a TUI log handler."""
    handler = TUILogHandler(tui, level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler
