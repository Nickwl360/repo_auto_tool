"""
Interactive interrupt handler for repo-auto-tool.

Provides an interactive menu when users press Ctrl+C, allowing them to:
- Continue execution
- Adjust the goal mid-run
- Skip the current step
- Provide feedback/context
- Abort the session

All code is defensively written with robust error handling.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)


class InterruptAction(Enum):
    """Actions available in the interrupt menu."""
    CONTINUE = "continue"
    ADJUST_GOAL = "adjust_goal"
    SKIP_STEP = "skip_step"
    PROVIDE_FEEDBACK = "provide_feedback"
    ABORT = "abort"


@dataclass
class InterruptResult:
    """Result from the interrupt menu interaction."""
    action: InterruptAction
    new_goal: str | None = None
    feedback: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class InterruptState:
    """State tracked by the interrupt handler."""
    is_paused: bool = False
    pause_count: int = 0
    last_action: InterruptAction | None = None
    accumulated_feedback: list = field(default_factory=list)
    goal_history: list = field(default_factory=list)


class InterruptHandler:
    """
    Manages interactive interrupts during execution.

    When Ctrl+C is pressed, displays an interactive menu allowing
    users to control execution flow without losing progress.

    Thread-safe and handles edge cases gracefully.
    """

    MENU_OPTIONS = [
        ("1", InterruptAction.CONTINUE, "Continue execution"),
        ("2", InterruptAction.ADJUST_GOAL, "Adjust goal (modify the current objective)"),
        ("3", InterruptAction.SKIP_STEP, "Skip current step (move to next iteration)"),
        ("4", InterruptAction.PROVIDE_FEEDBACK, "Provide feedback (add context for Claude)"),
        ("5", InterruptAction.ABORT, "Abort session (save and exit)"),
    ]

    def __init__(
        self,
        on_interrupt: Callable[[InterruptResult], None] | None = None,
        enable_signal_handler: bool = True,
    ):
        """
        Initialize the interrupt handler.

        Args:
            on_interrupt: Optional callback when an interrupt action is taken.
            enable_signal_handler: Whether to register the SIGINT handler.
        """
        self._state = InterruptState()
        self._lock = threading.Lock()
        self._on_interrupt = on_interrupt
        self._original_handler: Any | None = None
        self._enabled = True
        self._in_menu = False

        if enable_signal_handler:
            self._install_signal_handler()

    def _install_signal_handler(self) -> None:
        """Install the custom SIGINT handler."""
        try:
            self._original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._signal_handler)
            logger.debug("Installed custom SIGINT handler")
        except (ValueError, OSError) as e:
            # Can fail if not in main thread or signal not supported
            logger.warning(f"Could not install signal handler: {e}")

    def _restore_signal_handler(self) -> None:
        """Restore the original SIGINT handler."""
        try:
            if self._original_handler is not None:
                signal.signal(signal.SIGINT, self._original_handler)
                logger.debug("Restored original SIGINT handler")
        except (ValueError, OSError) as e:
            logger.warning(f"Could not restore signal handler: {e}")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT signal."""
        if not self._enabled:
            # Pass through to original handler
            if self._original_handler and callable(self._original_handler):
                try:
                    self._original_handler(signum, frame)
                except Exception:
                    pass
            return

        if self._in_menu:
            # Already in menu, ignore second interrupt
            return

        with self._lock:
            self._state.is_paused = True
            self._state.pause_count += 1

        # Don't raise - we'll handle this in the show_menu call
        logger.info("\n[Interrupt received - showing menu]")

    @property
    def is_paused(self) -> bool:
        """Check if execution is currently paused."""
        with self._lock:
            return self._state.is_paused

    @property
    def state(self) -> InterruptState:
        """Get the current interrupt state."""
        with self._lock:
            return InterruptState(
                is_paused=self._state.is_paused,
                pause_count=self._state.pause_count,
                last_action=self._state.last_action,
                accumulated_feedback=list(self._state.accumulated_feedback),
                goal_history=list(self._state.goal_history),
            )

    def enable(self) -> None:
        """Enable interrupt handling."""
        self._enabled = True
        logger.debug("Interrupt handling enabled")

    def disable(self) -> None:
        """Disable interrupt handling (pass through to default)."""
        self._enabled = False
        logger.debug("Interrupt handling disabled")

    def reset_pause(self) -> None:
        """Reset the paused state."""
        with self._lock:
            self._state.is_paused = False

    def check_and_handle(self, current_goal: str = "") -> InterruptResult | None:
        """
        Check if paused and handle interrupt if so.

        This should be called at safe points in the execution loop.

        Args:
            current_goal: The current goal string to display.

        Returns:
            InterruptResult if user took action, None otherwise.
        """
        if not self.is_paused:
            return None

        return self.show_menu(current_goal)

    def show_menu(self, current_goal: str = "") -> InterruptResult:
        """
        Display the interactive interrupt menu and get user choice.

        Args:
            current_goal: The current goal string to display.

        Returns:
            InterruptResult with the user's chosen action.
        """
        self._in_menu = True
        try:
            return self._show_menu_impl(current_goal)
        except Exception as e:
            logger.error(f"Error in interrupt menu: {e}")
            # Default to continue on error
            return InterruptResult(action=InterruptAction.CONTINUE)
        finally:
            self._in_menu = False
            self.reset_pause()

    def _show_menu_impl(self, current_goal: str) -> InterruptResult:
        """Implementation of the menu display."""
        print("\n" + "=" * 60)
        print("  EXECUTION PAUSED - Interactive Menu")
        print("=" * 60)

        if current_goal:
            # Truncate long goals for display
            display_goal = current_goal[:100] + "..." if len(current_goal) > 100 else current_goal
            print(f"\nCurrent goal: {display_goal}")

        print(f"\nPause count this session: {self._state.pause_count}")
        print("\nOptions:")

        for key, _, description in self.MENU_OPTIONS:
            print(f"  [{key}] {description}")

        print()

        # Get user choice
        choice = self._get_user_choice()

        # Handle the choice
        result = self._handle_choice(choice, current_goal)

        with self._lock:
            self._state.last_action = result.action

        # Call callback if set
        if self._on_interrupt:
            try:
                self._on_interrupt(result)
            except Exception as e:
                logger.error(f"Error in interrupt callback: {e}")

        return result

    def _get_user_choice(self) -> str:
        """Get a valid menu choice from the user."""
        valid_keys = [key for key, _, _ in self.MENU_OPTIONS]

        while True:
            try:
                choice = input("Enter choice [1-5]: ").strip()

                if not choice:
                    print("Please enter a choice.")
                    continue

                if choice in valid_keys:
                    return choice

                print(f"Invalid choice '{choice}'. Please enter 1-5.")

            except EOFError:
                # stdin closed, default to abort
                logger.warning("stdin closed, defaulting to abort")
                return "5"
            except KeyboardInterrupt:
                # Another Ctrl+C during menu, treat as abort
                print("\n[Another interrupt received - aborting]")
                return "5"
            except Exception as e:
                logger.error(f"Error reading input: {e}")
                # Default to continue on read error
                return "1"

    def _handle_choice(self, choice: str, current_goal: str) -> InterruptResult:
        """Handle the user's menu choice."""
        # Find the action for this choice
        action = InterruptAction.CONTINUE
        for key, act, _ in self.MENU_OPTIONS:
            if key == choice:
                action = act
                break

        result = InterruptResult(action=action)

        if action == InterruptAction.CONTINUE:
            print("\nResuming execution...")

        elif action == InterruptAction.ADJUST_GOAL:
            result = self._handle_adjust_goal(current_goal)

        elif action == InterruptAction.SKIP_STEP:
            print("\nSkipping current step, moving to next iteration...")

        elif action == InterruptAction.PROVIDE_FEEDBACK:
            result = self._handle_provide_feedback()

        elif action == InterruptAction.ABORT:
            print("\nAborting session (state will be saved)...")

        return result

    def _handle_adjust_goal(self, current_goal: str) -> InterruptResult:
        """Handle the adjust goal action."""
        print("\n" + "-" * 40)
        print("Adjust Goal")
        print("-" * 40)

        if current_goal:
            print(f"\nCurrent goal:\n{current_goal}")

        print("\nEnter the new/adjusted goal (or press Enter to keep current):")
        print("(Type 'done' on a new line when finished)")

        lines = []
        try:
            while True:
                line = input()
                if line.strip().lower() == 'done':
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass

        new_goal = '\n'.join(lines).strip()

        if new_goal:
            with self._lock:
                self._state.goal_history.append(current_goal)
            suffix = "..." if len(new_goal) > 200 else ""
            print(f"\nGoal updated. New goal:\n{new_goal[:200]}{suffix}")
            return InterruptResult(
                action=InterruptAction.ADJUST_GOAL,
                new_goal=new_goal,
            )
        else:
            print("\nKeeping current goal.")
            return InterruptResult(action=InterruptAction.CONTINUE)

    def _handle_provide_feedback(self) -> InterruptResult:
        """Handle the provide feedback action."""
        print("\n" + "-" * 40)
        print("Provide Feedback")
        print("-" * 40)

        print("\nEnter feedback or additional context for Claude:")
        print("(This will be injected into the next prompt)")
        print("(Type 'done' on a new line when finished)")

        lines = []
        try:
            while True:
                line = input()
                if line.strip().lower() == 'done':
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            pass

        feedback = '\n'.join(lines).strip()

        if feedback:
            with self._lock:
                self._state.accumulated_feedback.append(feedback)
            print("\nFeedback recorded. Will be included in next iteration.")
            return InterruptResult(
                action=InterruptAction.PROVIDE_FEEDBACK,
                feedback=feedback,
            )
        else:
            print("\nNo feedback provided. Continuing...")
            return InterruptResult(action=InterruptAction.CONTINUE)

    def get_accumulated_feedback(self) -> list:
        """Get all accumulated feedback and clear it."""
        with self._lock:
            feedback = list(self._state.accumulated_feedback)
            self._state.accumulated_feedback.clear()
            return feedback

    def cleanup(self) -> None:
        """Clean up resources and restore original signal handler."""
        self._restore_signal_handler()
        logger.debug("Interrupt handler cleaned up")

    def __enter__(self) -> InterruptHandler:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Context manager exit."""
        self.cleanup()
        return False


class InterruptContext:
    """
    Context manager for temporary interrupt handling around a code block.

    Usage:
        with InterruptContext() as ctx:
            # Code that can be interrupted
            if ctx.handler.is_paused:
                result = ctx.handler.show_menu(goal)
    """

    def __init__(self, enable_signal_handler: bool = True):
        """
        Initialize interrupt context.

        Args:
            enable_signal_handler: Whether to install signal handler.
        """
        self._enable_signal = enable_signal_handler
        self.handler: InterruptHandler | None = None

    def __enter__(self) -> InterruptContext:
        """Enter the context."""
        try:
            self.handler = InterruptHandler(enable_signal_handler=self._enable_signal)
        except Exception as e:
            logger.error(f"Failed to create interrupt handler: {e}")
            self.handler = None
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the context."""
        if self.handler:
            try:
                self.handler.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up interrupt handler: {e}")
        return False


def create_interrupt_handler(
    on_interrupt: Callable[[InterruptResult], None] | None = None,
    enable_signal_handler: bool = True,
) -> InterruptHandler:
    """
    Factory function to create an interrupt handler.

    Args:
        on_interrupt: Optional callback when an interrupt action is taken.
        enable_signal_handler: Whether to register the SIGINT handler.

    Returns:
        Configured InterruptHandler instance.
    """
    try:
        return InterruptHandler(
            on_interrupt=on_interrupt,
            enable_signal_handler=enable_signal_handler,
        )
    except Exception as e:
        logger.error(f"Failed to create interrupt handler: {e}")
        # Return a minimal handler that does nothing
        return InterruptHandler(enable_signal_handler=False)
