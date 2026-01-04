"""Main repo improver orchestrator."""

import logging

from .claude_interface import ClaudeCodeInterface
from .config import ImproverConfig, get_venv_command
from .convergence import (
    ChangeTracker,
    ConvergenceConfig,
    ConvergenceDetector,
    ConvergenceState,
)
from .exceptions import GitNotInitializedError
from .git_helper import GitHelper
from .model_selector import ModelSelector
from .safety import SafetyManager
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

    def __init__(self, config: ImproverConfig):
        self.config = config

        # Initialize components
        self.claude = ClaudeCodeInterface(
            working_dir=config.repo_path,
            allowed_tools=config.allowed_tools,
            model=config.model,
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

        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.log_file)
                if self.config.log_file else logging.NullHandler(),
            ]
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

        prompt = self.IMPROVE_PROMPT.format(
            goal=self.config.goal,
            context=context,
            task=task,
        )

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
            logger.error(f"Claude call failed: {response.error}")
            self.state.record_iteration(
                prompt=task,
                result=response.error or "Unknown error",
                success=False,
                validation_passed=False,
                error=response.error,
                token_usage=response.usage.to_dict() if response.usage else None,
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
            logger.warning("[FAIL] Validation failed")
            failure_summary = self.validators.get_failure_summary(validation_results)

            # Rollback changes
            if self.git:
                self.git.rollback()

            # Record the failure (Claude will see this in context next iteration)
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
        
        # Setup git branch
        if self.git:
            try:
                self.git.setup_branch()
            except GitNotInitializedError:
                logger.warning("Not a git repository, disabling git operations")
                self.git = None
                self.change_tracker = None
        
        try:
            while self.state.current_iteration < self.config.max_iterations:
                # Check for too many consecutive failures
                if self.state.consecutive_failures >= self.config.max_consecutive_failures:
                    failures = self.state.consecutive_failures
                    logger.error(f"Too many consecutive failures ({failures})")
                    self.state.mark_failed("Max consecutive failures reached")
                    break

                # Check if already complete
                if self.state.status == "completed":
                    logger.info("Goal already completed!")
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
            logger.info("\nInterrupted by user")
            self.state.status = "paused"
        
        finally:
            self.state.save(self.config.state_file)
            
            if self.git and self.state.status != "completed":
                # Optionally restore original branch on non-completion
                pass
        
        # Final summary
        self._print_summary()
        
        return self.state
    
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
