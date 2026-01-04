"""Wrapper for Claude Code CLI interactions."""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        self.session_id: str | None = None
        
        self._verify_cli()
    
    def _verify_cli(self) -> None:
        """Verify Claude Code CLI is installed and accessible."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI error: {result.stderr}")
            logger.info(f"Claude CLI version: {result.stdout.strip()}")
        except FileNotFoundError as err:
            raise RuntimeError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            ) from err
    
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
    
    def call(
        self,
        prompt: str,
        context: str | None = None,
        max_turns: int | None = None,
        resume: bool = False,
    ) -> ClaudeResponse:
        """
        Make a single call to Claude Code CLI.
        
        Claude WILL edit files in working_dir when instructed.
        
        Args:
            prompt: The instruction/prompt to send
            context: Optional context to prepend
            max_turns: Limit agentic turns (None = unlimited)
            resume: Continue from previous session
            
        Returns:
            ClaudeResponse with results
        """
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        cmd = self._build_command(full_prompt, max_turns, resume)
        
        logger.debug(f"Executing: {' '.join(cmd)}")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,  # Also set cwd for subprocess
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
                except json.JSONDecodeError:
                    # Non-JSON output (shouldn't happen with --output-format json)
                    return ClaudeResponse(
                        success=result.returncode == 0,
                        result=result.stdout,
                    )
            else:
                return ClaudeResponse(
                    success=False,
                    result="",
                    error=result.stderr or "No output from Claude CLI",
                )
                
        except subprocess.TimeoutExpired:
            return ClaudeResponse(
                success=False,
                result="",
                error=f"Claude CLI timed out after {self.timeout}s",
            )
        except Exception as e:
            return ClaudeResponse(
                success=False,
                result="",
                error=str(e),
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
