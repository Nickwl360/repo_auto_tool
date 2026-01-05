"""Main repo improver orchestrator."""

from __future__ import annotations

import logging

from .claude_interface import ClaudeCodeInterface
from .config import ImproverConfig, get_venv_command
from .convergence import (
    ChangeTracker,
    ConvergenceConfig,
    ConvergenceDetector,
    ConvergenceState,
)
from .exceptions import CostLimitExceededError, GitNotInitializedError
from .git_helper import GitHelper
from .interrupt_handler import (
    InterruptAction,
    InterruptHandler,
    InterruptResult,
    create_interrupt_handler,
)
from .model_selector import ModelSelector
from .prompt_adapter import PromptAdapter
from .safety import SafetyManager
from .session_history import SessionHistory
from .session_metrics import SessionMetrics
from .state import ImprovementState
from .validators import CommandValidator, ValidationPipeline

logger = logging.getLogger(__name__)


class RepoImprover:
    """
    Continuously improve a repository toward a goal using Claude Code.
    
    The loop:
    1. Analyze current state vs goal
    2. Ask Claude to make improvements
    3. Validate changes (tests, linting)
    4. If valid: commit and continue
    5. If invalid: rollback and retry with error context
    6. Repeat until goal is met or max iterations reached
    
    Example:
        config = ImproverConfig(
            repo_path="/path/to/my/project",
            goal="Add comprehensive type hints to all functions",
        )
        improver = RepoImprover(config)
        result = improver.run()
    """
    
    # Prompts for the improvement loop
    ANALYZE_PROMPT = """
Analyze this codebase against the following goal:

GOAL: {goal}

Provide a brief assessment:
1. Current state - what exists now
2. Gap analysis - what's missing or needs improvement  
3. Priority actions - top 3 things to do next

Be specific and actionable. Reference actual files and code.
"""

    IMPROVE_PROMPT = """
You are improving this codebase toward a specific goal.

GOAL: {goal}

CONTEXT FROM PREVIOUS ITERATIONS:
{context}

YOUR TASK FOR THIS ITERATION:
{task}

IMPORTANT INSTRUCTIONS:
- Make real, concrete improvements (edit files, add code, fix issues)
- Focus on ONE clear improvement per iteration
- VERIFY your changes work: after editing, run linters/checks if available
- AUTO-DETECT problems: look for issues in the code you touch and fix them
- If you create new code, ensure it integrates properly with existing code
- Check imports, function signatures, and dependencies are correct
- After making changes, briefly describe what you did
- If you believe the goal is FULLY COMPLETE, start your response with "GOAL_COMPLETE:"
- If you encounter a blocker you cannot resolve, start with "BLOCKED:"

BE ROBUST: Assume your changes will be validated. Write clean, correct code.

Now make the improvements:
"""

    FIX_PROMPT = """
Your previous changes caused validation failures. Fix them.

GOAL: {goal}

VALIDATION FAILURES:
{failures}

WHAT YOU CHANGED:
{changes}

Fix the issues while still making progress toward the goal.
Do NOT simply revert - try to fix the problems properly.
"""

    RECOVERY_PROMPT = """
Previous attempts have failed repeatedly. Take a SIMPLER approach.

GOAL: {goal}

WHAT WENT WRONG:
{errors}

RECOVERY INSTRUCTIONS:
1. Start fresh - don't build on previous failed attempts
2. Make ONE small, safe change that moves toward the goal
3. Ensure the change passes tests and linting before finishing
4. If the codebase has issues, fix those first before adding features
5. Prefer minimal changes over comprehensive ones

Make a single, focused improvement that will definitely work.
"""

    def __init__(self, config: ImproverConfig):
        self.config = config

        # Initialize components
        self.claude = ClaudeCodeInterface(
            working_dir=config.repo_path,
            allowed_tools=config.allowed_tools,
            model=config.model,
            verbose=config.verbose,
        )

        self.validators = self._setup_validators()

        self.git = GitHelper(
            repo_path=config.repo_path,
            branch_name=config.branch_name,
        ) if config.use_git else None

        self.state = ImprovementState.load_or_create(
            path=config.state_file,
            goal=config.goal,
            repo_path=str(config.repo_path),
        )

        # Initialize safety manager for secret redaction and dangerous command detection
        self.safety_manager = SafetyManager(
            redact_secrets=True,
            detect_dangerous=True,
        )

        # Initialize model selector for smart model selection
        # If user specified a model, use it as override (disables smart selection)
        self.model_selector = ModelSelector(
            override_model=config.model if not config.smart_model_selection else None,
        ) if config.smart_model_selection else None

        # Initialize convergence detection components
        self.convergence_config = ConvergenceConfig()
        self.convergence_detector = ConvergenceDetector(self.convergence_config)
        self.convergence_state = ConvergenceState()
        self.change_tracker = ChangeTracker(config.repo_path) if config.use_git else None

        # Initialize adaptive prompt system
        # Pre-populate from existing iteration history if resuming
        self.prompt_adapter = PromptAdapter.from_iteration_history(
            [it.to_dict() for it in self.state.iterations]
        )

        # Load cross-session history for learning from past sessions
        self.session_history = SessionHistory.load(config.repo_path)
        self._apply_historical_learnings()

        # Initialize interactive interrupt handler
        # Allows users to pause, adjust goals, provide feedback during execution
        self.interrupt_handler: InterruptHandler | None = None
        self._user_feedback: list = []  # Accumulated user feedback from interrupts

    def _apply_historical_learnings(self) -> None:
        """Apply learnings from past sessions to the prompt adapter.

        Loads historical error patterns and successful fixes,
        adding them as custom guidance to help avoid past mistakes.
        """
        guidance_items = self.session_history.get_guidance_for_prompt_adapter()

        for error_type, guidance_text, threshold in guidance_items:
            self.prompt_adapter.add_custom_guidance(
                error_type=error_type,
                guidance=guidance_text,
                threshold=threshold,
                priority=4,  # Slightly lower than default guidance
            )

        stats = self.session_history.get_stats()
        if stats["sessions_tracked"] > 0:
            logger.info(
                f"Applied historical learnings: {stats['sessions_tracked']} past sessions, "
                f"{stats['total_fixes_recorded']} recorded fixes"
            )

    def _setup_validators(self) -> ValidationPipeline:
        """Setup validation pipeline from config.

        Automatically resolves commands to use virtual environment paths
        when a venv is detected in the repository.

        Smart detection: Skips test validator if no test files are found.
        Parallel mode: If enabled, validators run concurrently for faster validation.
        """
        pipeline = ValidationPipeline(parallel=self.config.parallel_validation)

        if self.config.run_tests:
            # Smart test detection - skip if no test files exist
            if self._has_test_files():
                test_cmd = get_venv_command(self.config.repo_path, self.config.test_command)
                logger.debug(f"Test command resolved: {test_cmd}")
                pipeline.add(CommandValidator("tests", test_cmd))
            else:
                logger.info("No test files found, skipping test validator")

        if self.config.run_linter:
            lint_cmd = get_venv_command(self.config.repo_path, self.config.lint_command)
            logger.debug(f"Lint command resolved: {lint_cmd}")
            pipeline.add(CommandValidator("linter", lint_cmd))

        for i, cmd in enumerate(self.config.custom_validators):
            resolved_cmd = get_venv_command(self.config.repo_path, cmd)
            pipeline.add(CommandValidator(f"custom_{i}", resolved_cmd))

        return pipeline

    def _has_test_files(self) -> bool:
        """Check if the repository has test files.

        Looks for common test patterns in the src directory (excluding venv).

        Returns:
            True if test files are found, False otherwise.
        """
        skip_dirs = {"venv", ".venv", "env", "node_modules", ".git", "__pycache__"}

        for pattern in ["test_*.py", "*_test.py", "tests.py"]:
            for path in self.config.repo_path.rglob(pattern):
                # Skip venv and other non-source directories
                if not any(part in skip_dirs for part in path.parts):
                    logger.debug(f"Found test file: {path}")
                    return True

        # Also check for tests/ directory
        tests_dir = self.config.repo_path / "tests"
        if tests_dir.is_dir():
            for py_file in tests_dir.rglob("*.py"):
                if py_file.name != "__init__.py":
                    logger.debug(f"Found test file in tests/: {py_file}")
                    return True

        return False
    
    def analyze(self) -> str:
        """Analyze the current state of the repo vs the goal."""
        logger.info("Analyzing repository...")

        # Use smart model selection if enabled
        model_override = None
        if self.model_selector:
            prompt = self.ANALYZE_PROMPT.format(goal=self.config.goal)
            choice = self.model_selector.select_model(prompt, task_type="analyze")
            model_override = choice.model
            logger.info(f"Smart model selection: {choice.reason} -> {choice.model}")

        response = self.claude.analyze(
            self.ANALYZE_PROMPT.format(goal=self.config.goal),
            model_override=model_override,
        )

        if response.success:
            logger.info("Analysis complete")
            if response.model_used:
                logger.debug(f"Used model: {response.model_used}")
            return response.result
        else:
            logger.error(f"Analysis failed: {response.error}")
            return f"Analysis failed: {response.error}"

    def _attempt_recovery(self) -> bool:
        """Attempt to recover from repeated failures.

        Asks Claude to take a simpler approach after multiple failures.
        Returns True if recovery iteration succeeded.
        """
        logger.info("Attempting recovery with simpler approach...")

        # Gather error context from recent failures
        recent_errors = []
        for it in self.state.iterations[-5:]:
            if it.error:
                recent_errors.append(it.error[:500])

        errors_text = "\n".join(recent_errors) if recent_errors else "Multiple validation failures"

        # Build recovery prompt
        prompt = self.RECOVERY_PROMPT.format(
            goal=self.config.goal,
            errors=errors_text,
        )

        # Ask Claude for a simpler approach
        response = self.claude.improve(prompt, max_turns=10)

        if not response.success:
            logger.warning(f"Recovery call failed: {response.error}")
            return False

        result = response.result
        logger.info(f"Recovery response: {result[:200]}...")

        # Validate the recovery attempt
        all_passed, validation_results = self.validators.validate(self.config.repo_path)

        if all_passed:
            logger.info("[OK] Recovery validation passed!")

            # Commit if using git
            commit_hash = None
            if self.git and self.config.commit_each_iteration:
                commit_hash = self.git.commit(f"Recovery: {result[:50]}")

            self.state.record_iteration(
                prompt="[RECOVERY] " + prompt[:200],
                result=result,
                success=True,
                validation_passed=True,
                git_commit=commit_hash,
                token_usage=response.usage.to_dict() if response.usage else None,
            )
            return True
        else:
            logger.warning("[FAIL] Recovery validation failed")
            failure_summary = self.validators.get_failure_summary(validation_results)

            # Rollback recovery attempt
            if self.git:
                self.git.rollback()

            self.state.record_iteration(
                prompt="[RECOVERY] " + prompt[:200],
                result=result,
                success=True,
                validation_passed=False,
                error=failure_summary,
                token_usage=response.usage.to_dict() if response.usage else None,
                counts_as_failure=False,  # Don't count recovery failure
            )
            return False

    def _determine_next_task(self) -> str:
        """Determine what to work on next."""
        # If we have recent failures, focus on fixing
        recent = self.state.iterations[-3:] if self.state.iterations else []
        recent_failures = [it for it in recent if not it.validation_passed]
        
        if recent_failures:
            return "Fix the validation issues from your previous changes, then continue improving."
        
        # Otherwise, continue toward goal
        return "Make the next improvement toward the goal. Focus on one concrete change."
    
    def _run_iteration(self) -> bool:
        """
        Run a single improvement iteration.

        Returns True if iteration was successful.
        """
        iteration_num = self.state.current_iteration + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration_num}")
        logger.info(f"{'='*60}")

        # Determine what to do
        task = self._determine_next_task()
        context = self.state.get_recent_context()

        # Include any user feedback from interrupt menu
        if self._user_feedback:
            feedback_text = "\n\nUSER FEEDBACK (from interactive menu):\n"
            for i, fb in enumerate(self._user_feedback, 1):
                feedback_text += f"{i}. {fb}\n"
            context = context + feedback_text
            # Clear feedback after including it
            self._user_feedback = []

        prompt = self.IMPROVE_PROMPT.format(
            goal=self.config.goal,
            context=context,
            task=task,
        )

        # Apply adaptive guidance based on past failure patterns
        prompt = self.prompt_adapter.enhance_prompt(prompt)

        # Use smart model selection if enabled
        model_override = None
        if self.model_selector:
            # Determine task type based on recent failures
            task_type = "fix" if "fix" in task.lower() else "improve"
            choice = self.model_selector.select_model(prompt, context, task_type)
            model_override = choice.model
            logger.info(f"Smart model selection: {choice.reason}")
            logger.debug(f"  Complexity: {choice.complexity.value}, Model: {choice.model}")

        # Ask Claude to make improvements
        logger.info("Requesting improvements from Claude...")
        response = self.claude.improve(prompt, max_turns=10, model_override=model_override)
        
        if not response.success:
            error_msg = response.error or "Unknown error"
            logger.error(f"Claude call failed: {error_msg}")

            # Don't count timeouts/interrupts as failures - they're infrastructure issues
            is_timeout = "timeout" in error_msg.lower() if error_msg else False
            is_interrupt = error_msg is None or "interrupt" in error_msg.lower()
            counts_as_failure = not (is_timeout or is_interrupt)

            if not counts_as_failure:
                logger.info("(Not counting as failure - timeout/interrupt)")

            self.state.record_iteration(
                prompt=task,
                result=error_msg,
                success=False,
                validation_passed=False,
                error=error_msg,
                token_usage=response.usage.to_dict() if response.usage else None,
                counts_as_failure=counts_as_failure,
            )
            return False
        
        result = response.result

        # Process response through safety manager (redact secrets, detect dangerous commands)
        redacted_result, secret_matches = self.safety_manager.process_text(result)
        if secret_matches:
            logger.warning(
                f"Safety: Found {len(secret_matches)} potential secrets in response"
            )
            for match in secret_matches:
                logger.warning(f"  - {match.secret_type} at position {match.start_position}")
            result = redacted_result

        logger.info(f"Claude response: {result[:200]}...")

        # Check for goal completion
        if result.strip().startswith("GOAL_COMPLETE:"):
            logger.info(":) Claude indicates goal is complete!")
            self.state.record_iteration(
                prompt=task,
                result=result,
                success=True,
                validation_passed=True,
                token_usage=response.usage.to_dict() if response.usage else None,
            )
            self.state.mark_complete(result)
            return True
        
        # Check for blockers
        if result.strip().startswith("BLOCKED:"):
            logger.warning(f"Claude is blocked: {result}")
            self.state.record_iteration(
                prompt=task,
                result=result,
                success=False,
                validation_passed=False,
                error="Blocked",
                token_usage=response.usage.to_dict() if response.usage else None,
            )
            return False
        
        # Validate the changes
        logger.info("Validating changes...")
        all_passed, validation_results = self.validators.validate(self.config.repo_path)
        
        if all_passed:
            logger.info("[OK] All validations passed")

            # Record success for adaptive prompt system (helps decay old guidance)
            self.prompt_adapter.record_success()

            # Commit if using git
            commit_hash = None
            if self.git and self.config.commit_each_iteration:
                commit_hash = self.git.commit(f"Iteration {iteration_num}: {result[:50]}")

                # Track change metrics for convergence detection
                if self.change_tracker:
                    metrics = self.change_tracker.get_diff_stats("HEAD~1")
                    self.convergence_state.add_metrics(metrics)
                    logger.info(
                        f"Change metrics: {metrics.files_changed} files, "
                        f"+{metrics.lines_added}/-{metrics.lines_removed} lines"
                    )

            self.state.record_iteration(
                prompt=task,
                result=result,
                success=True,
                validation_passed=True,
                git_commit=commit_hash,
                token_usage=response.usage.to_dict() if response.usage else None,
            )
            return True

        else:
            logger.warning("[FAIL] Validation failed - asking Claude to fix")
            failure_summary = self.validators.get_failure_summary(validation_results)

            # Record failure pattern for adaptive prompt enhancement
            self.prompt_adapter.record_from_validation_error(failure_summary)

            # Give Claude a chance to fix the issues (don't rollback yet!)
            fix_prompt = self.FIX_PROMPT.format(
                goal=self.config.goal,
                failures=failure_summary,
                changes=result[:500],
            )

            logger.info("Requesting fix from Claude...")
            fix_response = self.claude.improve(fix_prompt, max_turns=10)

            if fix_response.success:
                # Re-validate after fix attempt
                logger.info("Re-validating after fix...")
                fix_passed, fix_results = self.validators.validate(self.config.repo_path)

                if fix_passed:
                    logger.info("[OK] Fix successful! Validation passed")
                    self.prompt_adapter.record_success()

                    # Commit the fixed changes
                    commit_hash = None
                    if self.git and self.config.commit_each_iteration:
                        commit_hash = self.git.commit(
                            f"Iteration {iteration_num}: {fix_response.result[:50]}"
                        )

                    self.state.record_iteration(
                        prompt=task,
                        result=f"Fixed: {fix_response.result}",
                        success=True,
                        validation_passed=True,
                        git_commit=commit_hash,
                        token_usage=fix_response.usage.to_dict() if fix_response.usage else None,
                    )
                    return True
                else:
                    logger.warning("[FAIL] Fix attempt did not resolve issues")
                    failure_summary = self.validators.get_failure_summary(fix_results)

            # Fix failed or didn't work - now rollback
            if self.git:
                self.git.rollback()

            # Record the failure with full error context
            self.state.record_iteration(
                prompt=task,
                result=result,
                success=True,  # Claude succeeded, validation failed
                validation_passed=False,
                error=failure_summary,
                token_usage=response.usage.to_dict() if response.usage else None,
            )
            return False
    
    def run(self) -> ImprovementState:
        """
        Run the full improvement loop.

        Returns the final state with results.
        """
        logger.info("Starting repo improvement")
        logger.info(f"  Repository: {self.config.repo_path}")
        logger.info(f"  Goal: {self.config.goal}")
        logger.info(f"  Max iterations: {self.config.max_iterations}")
        if self.config.max_cost is not None:
            logger.info(f"  Cost budget: ${self.config.max_cost:.2f}")

        # Setup git branch
        if self.git:
            try:
                self.git.setup_branch()
            except GitNotInitializedError:
                logger.warning("Not a git repository, disabling git operations")
                self.git = None
                self.change_tracker = None

        # Initialize interactive interrupt handler for Ctrl+C menu
        try:
            self.interrupt_handler = create_interrupt_handler(
                on_interrupt=self._on_interrupt_callback,
                enable_signal_handler=True,
            )
            logger.info("  Press Ctrl+C anytime to pause and access interactive menu")
        except Exception as e:
            logger.warning(f"Could not initialize interrupt handler: {e}")
            self.interrupt_handler = None

        try:
            while self.state.current_iteration < self.config.max_iterations:
                # Check for interactive interrupt (Ctrl+C was pressed)
                interrupt_result = self._check_interrupt()
                if interrupt_result:
                    if interrupt_result.action == InterruptAction.ABORT:
                        logger.info("Session aborted by user")
                        self.state.status = "paused"
                        self.state.summary = "Aborted by user"
                        break
                    elif interrupt_result.action == InterruptAction.SKIP_STEP:
                        logger.info("Skipping current step by user request")
                        # Continue to next iteration without running current step
                        continue

                # Check for too many consecutive failures - try recovery first
                if self.state.consecutive_failures >= self.config.max_consecutive_failures:
                    failures = self.state.consecutive_failures
                    logger.warning(f"Hit {failures} consecutive failures - attempting recovery")

                    # Try recovery: reset failures and ask Claude to take a simpler approach
                    recovery_success = self._attempt_recovery()
                    if recovery_success:
                        logger.info("Recovery successful - continuing")
                        self.state.consecutive_failures = 0
                        continue
                    else:
                        logger.error("Recovery failed - stopping")
                        self.state.mark_failed("Max consecutive failures reached")
                        break

                # Check if already complete
                if self.state.status == "completed":
                    logger.info("Goal already completed!")
                    break

                # Check cost budget before each iteration
                try:
                    self._check_cost_limit()
                except CostLimitExceededError as e:
                    logger.warning(
                        f"Cost limit reached: ${e.current_cost:.4f} >= ${e.max_cost:.2f}"
                    )
                    logger.info(
                        "Session paused. Resume with --resume and higher --max-cost."
                    )
                    self.state.status = "paused"
                    self.state.summary = f"Cost limit reached: ${e.current_cost:.4f}"
                    break

                # Check convergence before each iteration
                action, reason = self.convergence_detector.analyze(self.convergence_state)
                if action == "stop":
                    logger.info(f"Convergence detected: {reason}")
                    self.state.status = "converged"
                    self.state.summary = f"Converged: {reason}"
                    break
                elif action == "checkpoint":
                    self._handle_checkpoint(reason)

                # Run an iteration
                self._run_iteration()

                # Save state after each iteration
                self.state.save(self.config.state_file)

                # Check for completion
                if self.state.status == "completed":
                    break

            else:
                # Reached max iterations
                logger.warning(f"Reached max iterations ({self.config.max_iterations})")
                self.state.status = "paused"

        except KeyboardInterrupt:
            # Fallback if interrupt handler didn't catch it
            logger.info("\nInterrupted by user")
            self.state.status = "paused"

        finally:
            self.state.save(self.config.state_file)

            # Save session learnings to history for future sessions
            self._save_session_history()

            # Cleanup interrupt handler
            if self.interrupt_handler:
                try:
                    self.interrupt_handler.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up interrupt handler: {e}")

            if self.git and self.state.status != "completed":
                # Optionally restore original branch on non-completion
                pass

        # Final summary
        self._print_summary()

        return self.state

    def _check_interrupt(self) -> InterruptResult | None:
        """
        Check if user has requested an interrupt and handle it.

        Returns:
            InterruptResult if user took action, None otherwise.
        """
        if not self.interrupt_handler:
            return None

        try:
            result = self.interrupt_handler.check_and_handle(current_goal=self.config.goal)
            if result:
                # Handle goal adjustment
                if result.action == InterruptAction.ADJUST_GOAL and result.new_goal:
                    logger.info(f"Goal adjusted to: {result.new_goal[:100]}...")
                    self.config.goal = result.new_goal
                    self.state.goal = result.new_goal

                # Handle feedback - add to accumulated feedback
                if result.action == InterruptAction.PROVIDE_FEEDBACK and result.feedback:
                    self._user_feedback.append(result.feedback)
                    logger.info(f"User feedback recorded ({len(self._user_feedback)} total)")

            return result
        except Exception as e:
            logger.warning(f"Error checking interrupt: {e}")
            return None

    def _on_interrupt_callback(self, result: InterruptResult) -> None:
        """
        Callback when user takes an interrupt action.

        Args:
            result: The result of the interrupt menu interaction.
        """
        logger.debug(f"Interrupt action taken: {result.action.value}")
    
    def _handle_checkpoint(self, reason: str) -> None:
        """Handle a checkpoint during improvement loop.

        Args:
            reason: The reason for the checkpoint.
        """
        logger.info(f"\n{'='*60}")
        logger.info("CHECKPOINT")
        logger.info(f"{'='*60}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Iteration: {self.state.current_iteration}")

        successful = sum(
            1 for it in self.state.iterations if it.success and it.validation_passed
        )
        logger.info(f"Successful iterations so far: {successful}")

        # Log convergence state info
        if self.convergence_state.history:
            avg_rate = self.convergence_state.average_change_rate(
                self.convergence_config.plateau_window
            )
            logger.info(f"Average change rate: {avg_rate:.2f} lines/iteration")

        logger.info(f"{'='*60}\n")

    def _save_session_history(self) -> None:
        """Save session learnings to persistent history.

        Records error patterns, successful fixes, and session outcomes
        to help improve future sessions on this repository.
        """
        try:
            self.session_history.record_session(self.state)
            self.session_history.save()

            stats = self.session_history.get_stats()
            logger.info(
                f"Session history updated: {stats['sessions_tracked']} total sessions, "
                f"{stats['total_fixes_recorded']} recorded fixes"
            )
        except OSError as e:
            # Don't fail the session if history can't be saved
            logger.warning(f"Could not save session history: {e}")

    def _print_summary(self) -> None:
        """Print a summary of the improvement session."""
        logger.info(f"\n{'='*60}")
        logger.info("IMPROVEMENT SESSION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Status: {self.state.status}")
        logger.info(f"Iterations: {self.state.total_iterations}")

        successful = sum(1 for it in self.state.iterations if it.success and it.validation_passed)
        logger.info(f"Successful iterations: {successful}")

        if self.state.summary:
            logger.info(f"Summary: {self.state.summary}")

        # Token usage and cost summary
        self._print_token_summary()

        if self.git:
            logger.info(f"Changes on branch: {self.git.branch_name}")
            logger.info(f"Diff summary:\n{self.git.get_diff_summary()}")

    def get_session_metrics(self) -> SessionMetrics:
        """Get comprehensive metrics for the current session.

        Returns:
            SessionMetrics object with detailed analytics about the session
            including success rates, efficiency metrics, and failure analysis.
        """
        return SessionMetrics.from_state(
            state=self.state,
            model=self.config.model,
        )

    def print_metrics_report(self) -> None:
        """Print a detailed metrics report to the logger.

        Generates and logs a comprehensive report of session metrics
        including success rates, token efficiency, cost analysis, and
        failure pattern analysis.
        """
        metrics = self.get_session_metrics()
        logger.info(metrics.format_report())

    def _print_token_summary(self) -> None:
        """Print token usage and estimated cost summary.

        Uses Claude API pricing (as of 2024) for cost estimates:
        - Claude 3.5 Sonnet: $3/MTok input, $15/MTok output
        - Claude 3 Opus: $15/MTok input, $75/MTok output
        - Cache reads are discounted (10% of normal input cost)
        """
        token_summary = self.state.get_token_summary()
        total = token_summary["total_tokens"]

        if total == 0:
            logger.info("Token usage: No token data recorded")
            return

        logger.info("")
        logger.info("TOKEN USAGE:")
        logger.info(f"  Input tokens:  {token_summary['input_tokens']:,}")
        logger.info(f"  Output tokens: {token_summary['output_tokens']:,}")

        if token_summary["cache_read_tokens"] > 0:
            logger.info(f"  Cache reads:   {token_summary['cache_read_tokens']:,}")
        if token_summary["cache_creation_tokens"] > 0:
            logger.info(f"  Cache writes:  {token_summary['cache_creation_tokens']:,}")

        logger.info(f"  Total tokens:  {total:,}")

        # Estimate cost based on model (default to Sonnet pricing)
        cost = self._estimate_cost(token_summary)
        if cost > 0:
            logger.info(f"  Est. cost:     ${cost:.4f}")

            # Show budget info if max_cost is set
            if self.config.max_cost is not None:
                remaining = self.config.max_cost - cost
                pct_used = (cost / self.config.max_cost) * 100
                logger.info(f"  Budget used:   {pct_used:.1f}% (${remaining:.2f} remaining)")

    def _estimate_cost(self, token_summary: dict[str, int]) -> float:
        """Estimate API cost based on token usage.

        Args:
            token_summary: Dictionary with token counts.

        Returns:
            Estimated cost in USD.

        Note:
            Pricing is approximate and based on public Claude API rates.
            Actual costs may vary based on account, discounts, or API changes.
        """
        # Pricing per million tokens (as of late 2024)
        # These are approximate public rates - actual may vary
        model = self.config.model or "claude-sonnet-4-20250514"
        model_lower = model.lower()

        if "opus" in model_lower:
            input_cost_per_mtok = 15.0
            output_cost_per_mtok = 75.0
        elif "haiku" in model_lower:
            input_cost_per_mtok = 0.25
            output_cost_per_mtok = 1.25
        else:  # Default to Sonnet pricing
            input_cost_per_mtok = 3.0
            output_cost_per_mtok = 15.0

        # Cache reads are typically 10% of normal input cost
        cache_read_cost_per_mtok = input_cost_per_mtok * 0.1

        # Calculate costs
        input_tokens = token_summary["input_tokens"]
        output_tokens = token_summary["output_tokens"]
        cache_read_tokens = token_summary["cache_read_tokens"]

        input_cost = (input_tokens / 1_000_000) * input_cost_per_mtok
        output_cost = (output_tokens / 1_000_000) * output_cost_per_mtok
        cache_cost = (cache_read_tokens / 1_000_000) * cache_read_cost_per_mtok

        return input_cost + output_cost + cache_cost

    def get_current_cost(self) -> float:
        """Get the current estimated cost of the session.

        Returns:
            Estimated cost in USD based on token usage so far.
        """
        return self._estimate_cost(self.state.get_token_summary())

    def _check_cost_limit(self) -> None:
        """Check if the current cost exceeds the configured maximum.

        Raises:
            CostLimitExceededError: If max_cost is set and current cost exceeds it.
        """
        if self.config.max_cost is None:
            return  # No limit set

        current_cost = self.get_current_cost()
        if current_cost >= self.config.max_cost:
            raise CostLimitExceededError(
                current_cost=current_cost,
                max_cost=self.config.max_cost,
                tokens_used=self.state.total_tokens,
            )
