"""Wrapper for Claude Code CLI interactions.

This module provides a Python interface to the Claude Code CLI tool,
handling subprocess execution, JSON parsing, session management,
and automatic retry with exponential backoff for transient failures.
"""

import json
import logging
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ClaudeNotFoundError, ClaudeResponseError, ClaudeTimeoutError

logger = logging.getLogger(__name__)


# Retry configuration constants
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0
DEFAULT_JITTER_FACTOR = 0.25  # +/- 25% jitter


def _calculate_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current retry attempt (0-indexed).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        multiplier: Exponential multiplier for each attempt.
        jitter_factor: Random jitter factor (0.25 = +/- 25%).

    Returns:
        Delay in seconds with jitter applied.
    """
    # Calculate exponential delay
    delay = base_delay * (multiplier ** attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Apply jitter: random value in range [delay * (1 - jitter), delay * (1 + jitter)]
    jitter_range = delay * jitter_factor
    delay = delay + random.uniform(-jitter_range, jitter_range)

    # Ensure non-negative
    return max(0.0, delay)


def _is_retryable_error(error: str | None, returncode: int | None) -> bool:
    """Determine if an error is transient and worth retrying.

    Args:
        error: Error message string.
        returncode: Process return code.

    Returns:
        True if the error appears transient and retryable.
    """
    if error is None:
        return False

    error_lower = error.lower()

    # Network/connection errors (transient)
    transient_patterns = [
        "connection",
        "timeout",
        "network",
        "econnreset",
        "econnrefused",
        "etimedout",
        "rate limit",
        "too many requests",
        "429",
        "503",
        "502",
        "504",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "500",
    ]

    for pattern in transient_patterns:
        if pattern in error_lower:
            return True

    # Specific return codes that may be transient
    if returncode is not None and returncode in (1, 75, 124):
        # 75 = temp failure, 124 = timeout on some systems
        return True

    return False


@dataclass
class ClaudeResponse:
    """Structured response from Claude Code CLI."""
    success: bool
    result: str
    raw_output: dict[str, Any] | None = None
    error: str | None = None
    session_id: str | None = None
    

class ClaudeCodeInterface:
    """
    Interface to Claude Code CLI for scripted/agentic use.
    
    Claude Code CAN and WILL edit files directly when given permission.
    This is the core mechanism for the self-improving loop.
    """
    
    def __init__(
        self,
        working_dir: Path,
        allowed_tools: list[str] | None = None,
        model: str | None = None,
        timeout: int = 600,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.allowed_tools = allowed_tools or [
            "Bash(*)",      # Run any shell command
            "Read(*)",      # Read any file
            "Edit(*)",      # Edit any file (THIS IS THE KEY ONE)
            "Write(*)",     # Create new files
            "Glob(*)",      # Find files by pattern
            "Grep(*)",      # Search file contents
        ]
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session_id: str | None = None

        self._verify_cli()
    
    def _verify_cli(self) -> None:
        """Verify Claude Code CLI is installed and accessible.

        Raises:
            ClaudeNotFoundError: If the Claude CLI is not installed or not in PATH.
            ClaudeResponseError: If the CLI returns an error on version check.
        """
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise ClaudeResponseError(
                    reason="CLI version check failed",
                    raw_output=result.stderr
                )
            logger.info(f"Claude CLI version: {result.stdout.strip()}")
        except FileNotFoundError as err:
            raise ClaudeNotFoundError() from err
    
    def _build_command(
        self,
        prompt: str,
        max_turns: int | None = None,
        resume: bool = False,
    ) -> list[str]:
        """Build the claude CLI command."""
        cmd = [
            "claude",
            "-p", prompt,                          # Non-interactive prompt mode
            "--output-format", "json",             # Machine-readable output
            "--dangerously-skip-permissions",      # Auto-accept all tool use
            "--allowedTools", ",".join(self.allowed_tools),
        ]

        if self.model:
            cmd.extend(["--model", self.model])

        # Resume previous session for context continuity
        if resume and self.session_id:
            cmd.extend(["--resume", self.session_id])

        # Note: max_turns not directly supported, handled by model
        _ = max_turns

        return cmd
    
    def _execute_single_call(
        self,
        cmd: list[str],
        prompt_preview: str = "",
    ) -> ClaudeResponse:
        """Execute a single Claude CLI call without retry logic.

        Args:
            cmd: The command list to execute.
            prompt_preview: First 100 chars of prompt for error context.

        Returns:
            ClaudeResponse from this single execution attempt.
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
            )

            # Parse JSON output
            if result.stdout.strip():
                try:
                    output = json.loads(result.stdout)
                    # Extract session ID for potential resume
                    self.session_id = output.get("session_id")

                    return ClaudeResponse(
                        success=result.returncode == 0,
                        result=output.get("result", output.get("message", str(output))),
                        raw_output=output,
                        session_id=self.session_id,
                    )
                except json.JSONDecodeError as e:
                    # Non-JSON output (shouldn't happen with --output-format json)
                    logger.warning(f"Failed to parse Claude output as JSON: {e}")
                    return ClaudeResponse(
                        success=result.returncode == 0,
                        result=result.stdout,
                        error=f"JSON parse error: {e}" if result.returncode != 0 else None,
                    )
            else:
                error_msg = result.stderr.strip() if result.stderr else "No output from Claude CLI"
                return ClaudeResponse(
                    success=False,
                    result="",
                    error=error_msg,
                )

        except subprocess.TimeoutExpired:
            # Create a structured timeout error for logging context
            timeout_error = ClaudeTimeoutError(self.timeout, prompt_preview)
            logger.error(str(timeout_error))
            return ClaudeResponse(
                success=False,
                result="",
                error=f"Timeout after {self.timeout}s",
            )
        except OSError as e:
            # Handle OS-level errors (permissions, resources, etc.)
            logger.error(f"OS error executing Claude CLI: {e}")
            return ClaudeResponse(
                success=False,
                result="",
                error=f"OS error: {e}",
            )
        except Exception as e:
            # Catch-all for unexpected errors
            logger.exception(f"Unexpected error executing Claude CLI: {e}")
            return ClaudeResponse(
                success=False,
                result="",
                error=f"Unexpected error: {e}",
            )

    def call(
        self,
        prompt: str,
        context: str | None = None,
        max_turns: int | None = None,
        resume: bool = False,
    ) -> ClaudeResponse:
        """Make a call to Claude Code CLI with automatic retry for transient errors.

        Claude WILL edit files in working_dir when instructed.
        Uses exponential backoff with jitter for retries on transient failures.

        Args:
            prompt: The instruction/prompt to send.
            context: Optional context to prepend.
            max_turns: Limit agentic turns (None = unlimited).
            resume: Continue from previous session.

        Returns:
            ClaudeResponse with results.
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        cmd = self._build_command(full_prompt, max_turns, resume)

        # Create prompt preview for error context (used in timeout errors)
        prompt_preview = prompt[:100]

        logger.debug(f"Executing: {' '.join(cmd)}")
        logger.info(f"Prompt: {prompt_preview}...")

        last_response: ClaudeResponse | None = None

        for attempt in range(self.max_retries + 1):
            response = self._execute_single_call(cmd, prompt_preview)

            # Success - return immediately
            if response.success:
                if attempt > 0:
                    logger.info(f"Call succeeded on retry attempt {attempt}")
                return response

            # Check if this is a retryable error
            if not _is_retryable_error(response.error, None):
                # Non-retryable error - return immediately
                logger.debug(f"Non-retryable error: {response.error}")
                return response

            last_response = response

            # Check if we have retries remaining
            if attempt < self.max_retries:
                delay = _calculate_backoff_delay(attempt)
                logger.warning(
                    f"Transient error on attempt {attempt + 1}/{self.max_retries + 1}: "
                    f"{response.error}. Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"All {self.max_retries + 1} attempts failed. "
                    f"Last error: {response.error}"
                )

        # Return the last failed response
        return last_response or ClaudeResponse(
            success=False,
            result="",
            error="All retry attempts exhausted",
        )
    
    def analyze(self, question: str) -> ClaudeResponse:
        """Ask Claude to analyze the repo without making changes."""
        return self.call(
            f"ANALYSIS ONLY - Do not edit any files.\n\n{question}",
            max_turns=5,
        )
    
    def improve(self, instruction: str, max_turns: int = 10) -> ClaudeResponse:
        """
        Ask Claude to make improvements to the repo.
        
        This WILL edit files.
        """
        return self.call(
            f"Make the following improvements to this codebase:\n\n{instruction}",
            max_turns=max_turns,
        )
