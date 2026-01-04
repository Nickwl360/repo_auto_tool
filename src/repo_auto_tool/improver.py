"""Main repo improver orchestrator."""

import logging

from .claude_interface import ClaudeCodeInterface
from .config import ImproverConfig
from .convergence import (
    ChangeTracker,
    ConvergenceConfig,
    ConvergenceDetector,
    ConvergenceState,
)
from .git_helper import GitHelper
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
- After making changes, briefly describe what you did
- If you believe the goal is FULLY COMPLETE, start your response with "GOAL_COMPLETE:"
- If you encounter a blocker you cannot resolve, start with "BLOCKED:"

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
        """Setup validation pipeline from config."""
        pipeline = ValidationPipeline()
        
        if self.config.run_tests:
            pipeline.add(CommandValidator("tests", self.config.test_command))
        
        if self.config.run_linter:
            pipeline.add(CommandValidator("linter", self.config.lint_command))
        
        for i, cmd in enumerate(self.config.custom_validators):
            pipeline.add(CommandValidator(f"custom_{i}", cmd))
        
        return pipeline
    
    def analyze(self) -> str:
        """Analyze the current state of the repo vs the goal."""
        logger.info("Analyzing repository...")
        
        response = self.claude.analyze(
            self.ANALYZE_PROMPT.format(goal=self.config.goal)
        )
        
        if response.success:
            logger.info("Analysis complete")
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
        
        # Ask Claude to make improvements
        logger.info("Requesting improvements from Claude...")
        response = self.claude.improve(prompt, max_turns=10)
        
        if not response.success:
            logger.error(f"Claude call failed: {response.error}")
            self.state.record_iteration(
                prompt=task,
                result=response.error or "Unknown error",
                success=False,
                validation_passed=False,
                error=response.error,
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
            self.git.setup_branch()
        
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
        
        if self.git:
            logger.info(f"Changes on branch: {self.git.branch_name}")
            logger.info(f"Diff summary:\n{self.git.get_diff_summary()}")
