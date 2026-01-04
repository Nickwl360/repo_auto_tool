"""Rich Terminal User Interface (TUI) for repo-improver.

Provides an interactive terminal interface with:
- Panels for current task, progress, logs, file changes
- Real-time status updates with live diff preview
- Keyboard shortcuts for common actions
- Fallback to enhanced CLI mode when TUI is unavailable

All code is defensively written with robust error handling.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
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
    """Raised when TUI cannot be initialized (e.g., no terminal)."""

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

    refresh_rate: float = 0.5  # Seconds between screen updates
    log_buffer_size: int = 100  # Max lines to keep in log buffer
    show_diff_preview: bool = True
    show_file_changes: bool = True
    enable_colors: bool = True
    min_width: int = 80
    min_height: int = 24

    def __post_init__(self) -> None:
        """Validate configuration with defensive bounds checking."""
        self.refresh_rate = max(0.1, min(5.0, self.refresh_rate or 0.5))
        self.log_buffer_size = max(10, min(10000, self.log_buffer_size or 100))
        self.min_width = max(40, min(400, self.min_width or 80))
        self.min_height = max(10, min(200, self.min_height or 24))


@dataclass
class TUIState:
    """Current state of the TUI display."""

    # Current task/iteration info
    current_iteration: int = 0
    max_iterations: int = 20
    current_task: str = "Initializing..."
    goal: str = ""

    # Status info
    status: str = "starting"
    is_paused: bool = False
    is_running: bool = True

    # Progress metrics
    successful_iterations: int = 0
    failed_iterations: int = 0
    total_cost: float = 0.0
    elapsed_time: float = 0.0

    # Log buffer
    log_lines: list[str] = field(default_factory=list)
    log_scroll_offset: int = 0

    # File changes
    changed_files: list[str] = field(default_factory=list)
    diff_preview: str = ""

    # UI state
    active_panel: PanelType = PanelType.STATUS
    show_help: bool = False
    last_update: float = field(default_factory=time.time)

    def add_log(self, message: str, max_lines: int = 100) -> None:
        """Add a log message with buffer size limit."""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted = f"[{timestamp}] {message}"
            self.log_lines.append(formatted)

            # Trim buffer if too large
            if len(self.log_lines) > max_lines:
                self.log_lines = self.log_lines[-max_lines:]
        except Exception:
            # Never crash on log operations
            pass

    def get_progress_percent(self) -> float:
        """Calculate progress percentage safely."""
        try:
            if self.max_iterations <= 0:
                return 0.0
            return min(100.0, (self.current_iteration / self.max_iterations) * 100)
        except Exception:
            return 0.0

    def get_success_rate(self) -> float:
        """Calculate success rate safely."""
        try:
            total = self.successful_iterations + self.failed_iterations
            if total <= 0:
                return 0.0
            return (self.successful_iterations / total) * 100
        except Exception:
            return 0.0


class TUIRenderer(ABC):
    """Abstract base class for TUI renderers.

    Allows different rendering backends (curses, plain text, etc.)
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the renderer."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    @abstractmethod
    def render(self, state: TUIState) -> None:
        """Render the current state to the terminal."""
        pass

    @abstractmethod
    def get_key(self, timeout: float = 0.1) -> KeyAction | None:
        """Get keyboard input with timeout. Returns None if no input."""
        pass

    @abstractmethod
    def get_terminal_size(self) -> tuple[int, int]:
        """Get terminal width and height."""
        pass


class PlainTextRenderer(TUIRenderer):
    """Simple plain-text renderer for non-interactive environments.

    Provides a fallback when curses is unavailable or when running
    in non-interactive mode.
    """

    def __init__(self, config: TUIConfig | None = None) -> None:
        self.config = config or TUIConfig()
        self._last_render: str = ""
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Initialize plain text renderer."""
        pass

    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def render(self, state: TUIState) -> None:
        """Render state as plain text output."""
        try:
            with self._lock:
                output = self._build_output(state)
                # Only print if output changed to reduce spam
                if output != self._last_render:
                    self._clear_and_print(output)
                    self._last_render = output
        except Exception as e:
            logger.debug(f"Plain text render error: {e}")

    def _build_output(self, state: TUIState) -> str:
        """Build the text output."""
        lines: list[str] = []

        # Header
        lines.append("=" * 60)
        lines.append("  REPO-IMPROVER  ".center(60, "="))
        lines.append("=" * 60)
        lines.append("")

        # Goal (truncated)
        goal_display = state.goal[:50] + "..." if len(state.goal) > 50 else state.goal
        lines.append(f"Goal: {goal_display}")
        lines.append("")

        # Status
        status_icon = self._get_status_icon(state.status)
        lines.append(f"Status: {status_icon} {state.status.upper()}")
        if state.is_paused:
            lines.append("        [PAUSED - Press 'r' to resume]")
        lines.append("")

        # Progress
        progress = state.get_progress_percent()
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
        lines.append(f"Progress: {bar} {progress:.1f}%")
        lines.append(
            f"Iteration: {state.current_iteration}/{state.max_iterations}"
        )
        lines.append("")

        # Metrics
        lines.append(f"Successes: {state.successful_iterations}")
        lines.append(f"Failures: {state.failed_iterations}")
        lines.append(f"Success Rate: {state.get_success_rate():.1f}%")
        if state.total_cost > 0:
            lines.append(f"Cost: ${state.total_cost:.4f}")
        lines.append("")

        # Current task
        lines.append("-" * 60)
        lines.append("Current Task:")
        lines.append(f"  {state.current_task}")
        lines.append("-" * 60)

        # Changed files (last 5)
        if state.changed_files:
            lines.append("")
            lines.append("Changed Files:")
            for f in state.changed_files[-5:]:
                lines.append(f"  - {f}")

        # Recent logs (last 5)
        if state.log_lines:
            lines.append("")
            lines.append("Recent Activity:")
            for log in state.log_lines[-5:]:
                lines.append(f"  {log}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("Keys: [q]uit  [p]ause  [r]esume  [s]kip  [h]elp")
        lines.append("=" * 60)

        return "\n".join(lines)

    def _get_status_icon(self, status: str) -> str:
        """Get a text icon for status."""
        icons = {
            "starting": "[.]",
            "analyzing": "[?]",
            "improving": "[>]",
            "validating": "[*]",
            "committing": "[+]",
            "success": "[OK]",
            "failed": "[X]",
            "paused": "[||]",
            "complete": "[DONE]",
        }
        return icons.get(status.lower(), "[?]")

    def _clear_and_print(self, text: str) -> None:
        """Clear screen and print text."""
        try:
            # Try to clear screen
            if sys.platform == "win32":
                os.system("cls")
            else:
                # ANSI clear
                print("\033[2J\033[H", end="")
            print(text)
            sys.stdout.flush()
        except Exception:
            # If can't clear, just print
            print(text)

    def get_key(self, timeout: float = 0.1) -> KeyAction | None:
        """Get keyboard input (limited in plain text mode)."""
        # Plain text mode doesn't support real-time key input
        # This is handled by the interrupt handler instead
        return None

    def get_terminal_size(self) -> tuple[int, int]:
        """Get terminal size."""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24


class CursesRenderer(TUIRenderer):
    """Full-featured curses-based TUI renderer.

    Provides rich interactive interface with multiple panels,
    real-time updates, and keyboard shortcuts.
    """

    def __init__(self, config: TUIConfig | None = None) -> None:
        self.config = config or TUIConfig()
        self._stdscr: Any = None
        self._lock = threading.Lock()
        self._initialized = False
        self._key_map: dict[int, KeyAction] = {}

    def initialize(self) -> None:
        """Initialize curses mode."""
        try:
            import curses

            self._stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)  # Hide cursor
            self._stdscr.keypad(True)
            self._stdscr.nodelay(True)  # Non-blocking input

            # Initialize colors if available
            if curses.has_colors() and self.config.enable_colors:
                curses.start_color()
                curses.use_default_colors()
                # Color pairs: 1=green, 2=red, 3=yellow, 4=blue, 5=cyan
                curses.init_pair(1, curses.COLOR_GREEN, -1)
                curses.init_pair(2, curses.COLOR_RED, -1)
                curses.init_pair(3, curses.COLOR_YELLOW, -1)
                curses.init_pair(4, curses.COLOR_BLUE, -1)
                curses.init_pair(5, curses.COLOR_CYAN, -1)

            # Setup key mappings
            self._key_map = {
                ord("q"): KeyAction.QUIT,
                ord("Q"): KeyAction.QUIT,
                ord("p"): KeyAction.PAUSE,
                ord("P"): KeyAction.PAUSE,
                ord("r"): KeyAction.RESUME,
                ord("R"): KeyAction.RESUME,
                ord("s"): KeyAction.SKIP,
                ord("S"): KeyAction.SKIP,
                ord("h"): KeyAction.SHOW_HELP,
                ord("H"): KeyAction.SHOW_HELP,
                ord("?"): KeyAction.SHOW_HELP,
                ord("l"): KeyAction.TOGGLE_LOGS,
                ord("L"): KeyAction.TOGGLE_LOGS,
                ord("d"): KeyAction.TOGGLE_DIFF,
                ord("D"): KeyAction.TOGGLE_DIFF,
                curses.KEY_UP: KeyAction.SCROLL_UP,
                curses.KEY_DOWN: KeyAction.SCROLL_DOWN,
                ord("k"): KeyAction.SCROLL_UP,
                ord("j"): KeyAction.SCROLL_DOWN,
                curses.KEY_RESIZE: KeyAction.REFRESH,
            }

            self._initialized = True
            logger.debug("Curses TUI initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize curses: {e}")
            raise TUINotAvailableError(f"Curses unavailable: {e}") from e

    def cleanup(self) -> None:
        """Restore terminal state."""
        if not self._initialized:
            return

        try:
            import curses

            if self._stdscr:
                self._stdscr.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self._initialized = False
            logger.debug("Curses TUI cleaned up")
        except Exception as e:
            logger.debug(f"Error during curses cleanup: {e}")

    def render(self, state: TUIState) -> None:
        """Render the full TUI."""
        if not self._initialized or not self._stdscr:
            return

        try:
            with self._lock:
                self._stdscr.clear()
                height, width = self._stdscr.getmaxyx()

                # Check minimum size
                if width < self.config.min_width or height < self.config.min_height:
                    self._render_size_warning(width, height)
                    return

                # Render panels
                current_row = 0

                # Header (3 lines)
                current_row = self._render_header(state, current_row, width)

                # Status panel (4 lines)
                current_row = self._render_status(state, current_row, width)

                # Progress bar (3 lines)
                current_row = self._render_progress(state, current_row, width)

                # Main content area
                remaining = height - current_row - 3  # Reserve 3 for footer

                if state.show_help:
                    self._render_help(current_row, width, remaining)
                else:
                    # Split between logs and files/diff
                    log_height = remaining // 2
                    file_height = remaining - log_height

                    self._render_logs(state, current_row, width, log_height)
                    current_row += log_height + 1

                    self._render_files(state, current_row, width, file_height - 1)

                # Footer with key hints
                self._render_footer(state, height - 2, width)

                self._stdscr.refresh()

        except Exception as e:
            logger.debug(f"Curses render error: {e}")

    def _render_size_warning(self, width: int, height: int) -> None:
        """Render a warning when terminal is too small."""
        try:
            msg = f"Terminal too small ({width}x{height})"
            msg2 = f"Need at least {self.config.min_width}x{self.config.min_height}"
            self._stdscr.addstr(height // 2, max(0, (width - len(msg)) // 2), msg)
            self._stdscr.addstr(height // 2 + 1, max(0, (width - len(msg2)) // 2), msg2)
            self._stdscr.refresh()
        except Exception:
            pass

    def _render_header(self, state: TUIState, row: int, width: int) -> int:
        """Render the header with title and goal."""
        import curses

        try:
            # Title bar
            title = " REPO-IMPROVER "
            title_line = "=" * ((width - len(title)) // 2) + title
            title_line += "=" * (width - len(title_line))
            self._safe_addstr(row, 0, title_line[:width], curses.A_BOLD)

            # Goal line
            row += 1
            goal_prefix = "Goal: "
            max_goal_len = width - len(goal_prefix) - 2
            goal_display = state.goal
            if len(goal_display) > max_goal_len:
                goal_display = goal_display[: max_goal_len - 3] + "..."
            self._safe_addstr(row, 0, goal_prefix + goal_display)

            row += 1
            self._safe_addstr(row, 0, "-" * width)

            return row + 1

        except Exception:
            return row + 3

    def _render_status(self, state: TUIState, row: int, width: int) -> int:
        """Render the status panel."""
        import curses

        try:
            # Status with color
            status_text = f"Status: {state.status.upper()}"
            color = self._get_status_color(state.status)
            self._safe_addstr(row, 0, status_text, curses.color_pair(color) | curses.A_BOLD)

            if state.is_paused:
                self._safe_addstr(row, len(status_text) + 2, "[PAUSED]",
                                  curses.color_pair(3) | curses.A_BLINK)

            # Current task
            row += 1
            task_prefix = "Task: "
            max_task_len = width - len(task_prefix)
            task_display = state.current_task
            if len(task_display) > max_task_len:
                task_display = task_display[: max_task_len - 3] + "..."
            self._safe_addstr(row, 0, task_prefix + task_display)

            # Metrics row
            row += 1
            metrics = (
                f"Iter: {state.current_iteration}/{state.max_iterations}  "
                f"OK: {state.successful_iterations}  "
                f"Fail: {state.failed_iterations}  "
            )
            if state.total_cost > 0:
                metrics += f"Cost: ${state.total_cost:.4f}"
            self._safe_addstr(row, 0, metrics[:width])

            return row + 2

        except Exception:
            return row + 4

    def _render_progress(self, state: TUIState, row: int, width: int) -> int:
        """Render the progress bar."""
        import curses

        try:
            progress = state.get_progress_percent()

            # Progress label
            label = f"Progress: {progress:.1f}%"
            self._safe_addstr(row, 0, label)

            # Progress bar
            row += 1
            bar_width = min(width - 4, 60)
            filled = int(bar_width * progress / 100)
            bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"

            color = 1 if progress >= 50 else 3  # Green if >50%, yellow otherwise
            self._safe_addstr(row, 0, bar, curses.color_pair(color))

            return row + 2

        except Exception:
            return row + 3

    def _render_logs(self, state: TUIState, row: int, width: int, height: int) -> None:
        """Render the log panel."""
        import curses

        try:
            # Panel header
            header = " LOGS "
            header_line = "-" * ((width - len(header)) // 2) + header
            header_line += "-" * (width - len(header_line))
            self._safe_addstr(row, 0, header_line[:width], curses.A_DIM)

            row += 1
            available_lines = height - 1

            # Get visible logs with scroll offset
            start_idx = max(0, len(state.log_lines) - available_lines - state.log_scroll_offset)
            end_idx = start_idx + available_lines

            visible_logs = state.log_lines[start_idx:end_idx]

            for i, log in enumerate(visible_logs):
                if row + i >= self._stdscr.getmaxyx()[0] - 3:
                    break
                display = log[:width - 2]
                self._safe_addstr(row + i, 1, display)

        except Exception:
            pass

    def _render_files(self, state: TUIState, row: int, width: int, height: int) -> None:
        """Render the files panel."""
        import curses

        try:
            # Panel header
            header = " CHANGED FILES "
            header_line = "-" * ((width - len(header)) // 2) + header
            header_line += "-" * (width - len(header_line))
            self._safe_addstr(row, 0, header_line[:width], curses.A_DIM)

            row += 1
            available_lines = height - 1

            if not state.changed_files:
                self._safe_addstr(row, 1, "(no changes yet)")
                return

            for i, filepath in enumerate(state.changed_files[-available_lines:]):
                if row + i >= self._stdscr.getmaxyx()[0] - 3:
                    break
                display = f"  {filepath}"[:width - 1]
                self._safe_addstr(row + i, 0, display, curses.color_pair(5))

        except Exception:
            pass

    def _render_help(self, row: int, width: int, height: int) -> None:
        """Render the help panel."""
        try:
            help_lines = [
                "KEYBOARD SHORTCUTS",
                "",
                "  q, Q       Quit the application",
                "  p, P       Pause execution",
                "  r, R       Resume execution",
                "  s, S       Skip current step",
                "  h, H, ?    Toggle this help",
                "  l, L       Toggle logs panel",
                "  d, D       Toggle diff preview",
                "  j, Down    Scroll down",
                "  k, Up      Scroll up",
                "",
                "Press any key to close help...",
            ]

            # Center the help box
            box_width = max(len(line) for line in help_lines) + 4
            box_height = len(help_lines) + 2
            start_col = max(0, (width - box_width) // 2)
            start_row = row + max(0, (height - box_height) // 2)

            # Draw box
            self._safe_addstr(start_row, start_col, "+" + "-" * (box_width - 2) + "+")
            for i, line in enumerate(help_lines):
                padded = f"| {line:<{box_width - 4}} |"
                self._safe_addstr(start_row + 1 + i, start_col, padded)
            self._safe_addstr(
                start_row + len(help_lines) + 1,
                start_col,
                "+" + "-" * (box_width - 2) + "+",
            )

        except Exception:
            pass

    def _render_footer(self, state: TUIState, row: int, width: int) -> None:
        """Render the footer with key hints."""
        import curses

        try:
            self._safe_addstr(row, 0, "=" * width)
            row += 1

            hints = "[q]uit  [p]ause  [r]esume  [s]kip  [h]elp"
            if state.is_paused:
                hints = "PAUSED: " + hints
            self._safe_addstr(row, 0, hints.center(width), curses.A_DIM)

        except Exception:
            pass

    def _safe_addstr(
        self, row: int, col: int, text: str, attrs: int = 0
    ) -> None:
        """Safely add string to screen, handling edge cases."""
        try:
            if self._stdscr is None:
                return
            height, width = self._stdscr.getmaxyx()
            if row < 0 or row >= height or col < 0 or col >= width:
                return
            # Truncate text to fit
            max_len = width - col
            if max_len <= 0:
                return
            display_text = text[:max_len]
            self._stdscr.addstr(row, col, display_text, attrs)
        except Exception:
            pass

    def _get_status_color(self, status: str) -> int:
        """Get color pair for status."""
        color_map = {
            "starting": 4,  # blue
            "analyzing": 5,  # cyan
            "improving": 3,  # yellow
            "validating": 5,  # cyan
            "committing": 4,  # blue
            "success": 1,  # green
            "failed": 2,  # red
            "paused": 3,  # yellow
            "complete": 1,  # green
        }
        return color_map.get(status.lower(), 0)

    def get_key(self, timeout: float = 0.1) -> KeyAction | None:
        """Get keyboard input with timeout."""
        if not self._initialized or not self._stdscr:
            return None

        try:
            # Set timeout for getch
            self._stdscr.timeout(int(timeout * 1000))
            ch = self._stdscr.getch()

            if ch == -1:  # No input
                return None

            return self._key_map.get(ch)

        except Exception:
            return None

    def get_terminal_size(self) -> tuple[int, int]:
        """Get terminal size from curses."""
        if not self._initialized or not self._stdscr:
            return 80, 24

        try:
            height, width = self._stdscr.getmaxyx()
            return width, height
        except Exception:
            return 80, 24


class TUI:
    """Main TUI controller.

    Manages the rendering, state updates, and user input handling.
    Provides both interactive (curses) and fallback (plain text) modes.
    """

    def __init__(
        self,
        config: TUIConfig | None = None,
        use_plain_text: bool = False,
    ) -> None:
        """Initialize TUI.

        Args:
            config: TUI configuration
            use_plain_text: Force plain text mode (--no-tui)
        """
        self.config = config or TUIConfig()
        self.state = TUIState()
        self._renderer: TUIRenderer | None = None
        self._use_plain_text = use_plain_text
        self._running = False
        self._update_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._callbacks: dict[KeyAction, Callable[[], None]] = {}
        self._initialized = False

    def can_use_curses(self) -> bool:
        """Check if curses can be used."""
        try:
            # Check if running in a terminal
            if not sys.stdout.isatty():
                return False

            # Windows curses check
            if sys.platform == "win32":
                try:
                    import curses  # noqa: F401

                    return True
                except ImportError:
                    # Try windows-curses
                    try:
                        import _curses  # noqa: F401

                        return True
                    except ImportError:
                        return False

            # Unix/Mac - should have curses
            import curses  # noqa: F401

            return True

        except Exception:
            return False

    def initialize(self) -> bool:
        """Initialize the TUI. Returns True if successful.

        Automatically falls back to plain text mode if curses unavailable.
        """
        try:
            if self._use_plain_text or not self.can_use_curses():
                logger.debug("Using plain text renderer")
                self._renderer = PlainTextRenderer(self.config)
            else:
                try:
                    self._renderer = CursesRenderer(self.config)
                    self._renderer.initialize()
                    logger.debug("Using curses renderer")
                except TUINotAvailableError:
                    logger.debug("Curses failed, falling back to plain text")
                    self._renderer = PlainTextRenderer(self.config)

            self._initialized = True
            return True

        except Exception as e:
            logger.warning(f"TUI initialization failed: {e}")
            # Create minimal renderer as last resort
            self._renderer = PlainTextRenderer(self.config)
            self._initialized = True
            return True

    def cleanup(self) -> None:
        """Clean up TUI resources."""
        self._running = False

        # Wait for update thread
        if self._update_thread and self._update_thread.is_alive():
            try:
                self._update_thread.join(timeout=1.0)
            except Exception:
                pass

        # Cleanup renderer
        if self._renderer:
            try:
                self._renderer.cleanup()
            except Exception:
                pass

        self._initialized = False

    def start(self) -> None:
        """Start the TUI update loop in a background thread."""
        if not self._initialized:
            self.initialize()

        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="tui-update",
        )
        self._update_thread.start()

    def stop(self) -> None:
        """Stop the TUI."""
        self._running = False

    def _update_loop(self) -> None:
        """Background thread that updates the display."""
        while self._running:
            try:
                # Handle input
                if self._renderer:
                    action = self._renderer.get_key(self.config.refresh_rate / 2)
                    if action:
                        self._handle_action(action)

                # Render
                if self._renderer:
                    self._renderer.render(self.state)

                # Sleep between updates
                time.sleep(self.config.refresh_rate)

            except Exception as e:
                logger.debug(f"TUI update error: {e}")
                time.sleep(0.5)  # Slow down on errors

    def _handle_action(self, action: KeyAction) -> None:
        """Handle a keyboard action."""
        try:
            # Built-in actions
            if action == KeyAction.SHOW_HELP:
                with self._lock:
                    self.state.show_help = not self.state.show_help
            elif action == KeyAction.SCROLL_UP:
                with self._lock:
                    self.state.log_scroll_offset = min(
                        self.state.log_scroll_offset + 1,
                        max(0, len(self.state.log_lines) - 5),
                    )
            elif action == KeyAction.SCROLL_DOWN:
                with self._lock:
                    self.state.log_scroll_offset = max(
                        0, self.state.log_scroll_offset - 1
                    )

            # Call registered callback if any
            callback = self._callbacks.get(action)
            if callback:
                try:
                    callback()
                except Exception as e:
                    logger.debug(f"Callback error for {action}: {e}")

        except Exception as e:
            logger.debug(f"Action handling error: {e}")

    def register_callback(self, action: KeyAction, callback: Callable[[], None]) -> None:
        """Register a callback for a keyboard action."""
        self._callbacks[action] = callback

    def render_once(self) -> None:
        """Render once (for non-threaded updates)."""
        if self._renderer:
            try:
                self._renderer.render(self.state)
            except Exception:
                pass

    # State update methods - thread-safe
    def set_goal(self, goal: str) -> None:
        """Set the goal text."""
        with self._lock:
            self.state.goal = goal or ""

    def set_status(self, status: str) -> None:
        """Set the current status."""
        with self._lock:
            self.state.status = status or "unknown"

    def set_task(self, task: str) -> None:
        """Set the current task description."""
        with self._lock:
            self.state.current_task = task or ""

    def set_iteration(self, current: int, maximum: int) -> None:
        """Set iteration progress."""
        with self._lock:
            self.state.current_iteration = max(0, current)
            self.state.max_iterations = max(1, maximum)

    def set_paused(self, paused: bool) -> None:
        """Set paused state."""
        with self._lock:
            self.state.is_paused = paused

    def add_success(self) -> None:
        """Increment successful iteration count."""
        with self._lock:
            self.state.successful_iterations += 1

    def add_failure(self) -> None:
        """Increment failed iteration count."""
        with self._lock:
            self.state.failed_iterations += 1

    def set_cost(self, cost: float) -> None:
        """Set total cost."""
        with self._lock:
            self.state.total_cost = max(0.0, cost)

    def add_log(self, message: str) -> None:
        """Add a log message."""
        with self._lock:
            self.state.add_log(message, self.config.log_buffer_size)

    def set_changed_files(self, files: list[str]) -> None:
        """Set the list of changed files."""
        with self._lock:
            self.state.changed_files = list(files or [])

    def add_changed_file(self, filepath: str) -> None:
        """Add a changed file."""
        with self._lock:
            if filepath and filepath not in self.state.changed_files:
                self.state.changed_files.append(filepath)

    def set_diff_preview(self, diff: str) -> None:
        """Set the diff preview text."""
        with self._lock:
            self.state.diff_preview = diff or ""

    def update_from_state(self, improvement_state: ImprovementState) -> None:
        """Update TUI from an ImprovementState object."""
        try:
            with self._lock:
                self.state.goal = getattr(improvement_state, "goal", "") or ""
                self.state.current_iteration = getattr(
                    improvement_state, "current_iteration", 0
                )

                iterations = getattr(improvement_state, "iterations", []) or []
                self.state.successful_iterations = sum(
                    1
                    for it in iterations
                    if getattr(it, "status", "") == "success"
                )
                self.state.failed_iterations = sum(
                    1
                    for it in iterations
                    if getattr(it, "status", "") == "failed"
                )

                # Get changed files from iterations
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

                # Update changed files
                files_changed = getattr(iteration, "files_changed", []) or []
                for f in files_changed:
                    if f not in self.state.changed_files:
                        self.state.changed_files.append(f)

                # Add to log
                summary = getattr(iteration, "summary", "") or ""
                if summary:
                    self.state.add_log(summary[:100], self.config.log_buffer_size)

        except Exception as e:
            logger.debug(f"Error updating from iteration: {e}")


def create_tui(
    use_tui: bool = True,
    config: TUIConfig | None = None,
) -> TUI | None:
    """Factory function to create TUI if available.

    Args:
        use_tui: Whether to try creating a TUI (False = --no-tui mode)
        config: Optional TUI configuration

    Returns:
        TUI instance or None if TUI not needed/available
    """
    if not use_tui:
        return None

    try:
        tui = TUI(config=config, use_plain_text=False)
        if tui.initialize():
            return tui
        return None
    except Exception as e:
        logger.debug(f"Failed to create TUI: {e}")
        return None


def create_plain_text_tui(config: TUIConfig | None = None) -> TUI:
    """Create a TUI that uses plain text rendering.

    Always succeeds as plain text mode has no dependencies.
    """
    tui = TUI(config=config, use_plain_text=True)
    tui.initialize()
    return tui


class TUILogHandler(logging.Handler):
    """Logging handler that sends messages to the TUI.

    Allows log messages to appear in the TUI log panel.
    """

    def __init__(self, tui: TUI, level: int = logging.INFO) -> None:
        super().__init__(level)
        self._tui = tui

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the TUI."""
        try:
            msg = self.format(record)
            # Truncate long messages
            if len(msg) > 200:
                msg = msg[:197] + "..."
            self._tui.add_log(msg)
        except Exception:
            pass


def setup_tui_logging(tui: TUI, level: int = logging.INFO) -> TUILogHandler:
    """Setup a TUI log handler for the root logger.

    Args:
        tui: TUI instance to send logs to
        level: Minimum log level

    Returns:
        The created handler (for later removal if needed)
    """
    handler = TUILogHandler(tui, level)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler
