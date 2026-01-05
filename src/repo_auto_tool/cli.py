"""CLI entry point for repo-improver."""

import argparse
import atexit
import logging
import sys
from pathlib import Path

from .agents import AgentMode, create_agent
from .config import ImproverConfig
from .exceptions import ConfigurationError, PromptParseError
from .foreign_repo import (
    ClonedRepo,
    ForeignRepoError,
    ForeignRepoManager,
    RepoConventions,
)
from .goal_analyzer import GoalAnalyzer, GoalComplexity, GoalRisk
from .improver import RepoImprover
from .logging import setup_logging
from .pr_generator import PRResult, create_pr_from_repo
from .prompt_parser import ParsedPrompt, PromptParser
from .tui import TUI, TUIConfig, create_tui, setup_tui_logging

logger = logging.getLogger(__name__)

# Global manager for cleanup on exit
_foreign_repo_manager: ForeignRepoManager | None = None


def _run_agent_mode(args: argparse.Namespace, config: "ImproverConfig") -> int:
    """Run in agent mode and return exit code.

    Args:
        args: Parsed command line arguments
        config: Improver configuration

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import json as json_module

    mode: AgentMode = args.agent_mode
    agent = create_agent(
        mode=mode,
        working_dir=args.repo_path,
        timeout=600,
    )

    if mode == "pre-analysis":
        result = agent.run()
    elif mode == "goal-decomposer":
        if not args.goal:
            print("Error: --agent-mode goal-decomposer requires a goal", file=sys.stderr)
            return 1
        result = agent.run(goal=args.goal)
    elif mode == "reviewer":
        if not args.goal:
            print("Error: --agent-mode reviewer requires a goal", file=sys.stderr)
            return 1
        # For reviewer, use goal as both goal and task for now
        result = agent.run(goal=args.goal, task=args.goal)
    elif mode == "diagnostics":
        # Diagnostics mode analyzes the session state file
        result = agent.run()
    else:
        print(f"Error: Unknown agent mode '{mode}'", file=sys.stderr)
        return 1

    if not result.success:
        print(f"Agent failed: {result.error}", file=sys.stderr)
        return 1

    # Output results
    if args.json:
        output = {
            "mode": mode,
            "success": result.success,
            "output": result.output,
            "suggestions": result.suggestions,
            "steps": result.steps,
            "metadata": result.metadata,
        }
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n=== {agent.name} Results ===\n")
        print(result.output)

        if result.suggestions:
            print("\n--- Suggestions ---")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"  {i}. {suggestion}")

        if result.steps:
            print("\n--- Steps ---")
            for i, step in enumerate(result.steps, 1):
                print(f"  {i}. {step}")

    return 0


def main() -> int:
    """CLI entry point for repo-improver.

    Returns:
        Exit code: 0 for success/completion, 1 for failure, 2 for paused/resumable.
    """
    parser = argparse.ArgumentParser(
        description="Continuously improve a codebase using Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  repo-improver . "Add type hints"      # Basic usage
  repo-improver . --fix                 # Fix tests/lint only
  repo-improver . --resume              # Resume previous session
  repo-improver . "Goal" -n 50          # Custom iteration limit

Use --help-all for advanced options.
""",
    )

    # Check for --help-all before parsing
    show_all_help = "--help-all" in sys.argv

    # === CORE ARGUMENTS (always shown) ===
    parser.add_argument(
        "repo_path",
        type=str,
        nargs="?",
        default=".",
        help="Path to repository (default: current directory)",
    )

    parser.add_argument(
        "goal",
        type=str,
        nargs="?",
        help="What to improve (natural language)",
    )

    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=20,
        help="Max iterations (default: 20)",
    )

    # === MODES (common) ===
    modes = parser.add_argument_group("Modes")
    modes.add_argument(
        "--fix",
        action="store_true",
        help="Only fix failing tests/lint",
    )
    modes.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous session",
    )
    modes.add_argument(
        "--research",
        action="store_true",
        help="Analyze without changes",
    )
    modes.add_argument(
        "--plan",
        action="store_true",
        help="Create plan, don't execute",
    )
    modes.add_argument(
        "--review",
        action="store_true",
        help="Review changes on improver branch, show how to apply",
    )

    # === OUTPUT (common) ===
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full Claude output",
    )
    output.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    output.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable terminal UI",
    )

    # === HELP ===
    parser.add_argument(
        "--help-all",
        action="store_true",
        help="Show all options including advanced",
    )

    # === ADVANCED OPTIONS (hidden by default) ===
    # Use argparse.SUPPRESS to hide from default help
    HIDE = argparse.SUPPRESS if not show_all_help else None

    advanced = parser.add_argument_group("Advanced" if show_all_help else argparse.SUPPRESS)

    advanced.add_argument(
        "--prompt-file", "-f",
        type=Path,
        default=None,
        help="Read goal from file" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Max consecutive failures (default: 3)" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--test-cmd",
        type=str,
        default=None,
        help="Test command (auto-detected)" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--lint-cmd",
        type=str,
        default=None,
        help="Lint command (auto-detected)" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip tests" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip linter" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--no-git",
        action="store_true",
        help="Don't use git" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--branch",
        type=str,
        default="repo-improver/auto",
        help="Git branch name" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--model",
        type=str,
        default=None,
        help="Claude model to use" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Max cost in USD" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--refactor",
        type=str,
        default=None,
        help="Refactor specific target" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode" if show_all_help else HIDE,
    )
    advanced.add_argument(
        "--create-pr",
        action="store_true",
        help="Create PR when done" if show_all_help else HIDE,
    )

    # === RARELY USED (always hidden unless --help-all) ===
    rarely_used = parser.add_argument_group("Rarely Used" if show_all_help else argparse.SUPPRESS)

    rarely_used.add_argument(
        "--parallel-validation",
        action="store_true",
        help="Run validators in parallel" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--agent-mode",
        type=str,
        choices=["pre-analysis", "goal-decomposer", "reviewer", "diagnostics"],
        help="Run in agent mode" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="State files directory" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--goal-type",
        type=str,
        choices=["open-ended", "bounded", "exploratory"],
        default="open-ended",
        help="Goal type" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Checkpoint interval" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--smart-model-selection",
        action="store_true",
        help="Auto-select model by complexity" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show metrics report" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--tui-refresh-rate",
        type=float,
        default=0.5,
        help="TUI refresh rate" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Clone workspace dir" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--keep-clone",
        action="store_true",
        help="Keep cloned repo" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--shallow-clone",
        action="store_true",
        help="Shallow clone" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--skip-goal-analysis",
        action="store_true",
        help="Skip goal analysis" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--analyze-goal",
        action="store_true",
        help="Analyze goal only" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--pr-preview",
        action="store_true",
        help="Preview PR" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--pr-draft",
        action="store_true",
        help="Create draft PR" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--pr-base",
        type=str,
        default=None,
        help="PR base branch" if show_all_help else HIDE,
    )
    rarely_used.add_argument(
        "--pr-labels",
        type=str,
        default=None,
        help="PR labels" if show_all_help else HIDE,
    )

    args = parser.parse_args()

    # Handle --help-all: show full help and exit
    if args.help_all:
        parser.print_help()
        return 0

    # Resolve repo path (handles URLs and local paths)
    global _foreign_repo_manager
    cloned_repo: ClonedRepo | None = None
    conventions: RepoConventions | None = None
    repo_path: Path

    try:
        # Check if it's a URL or local path
        manager = ForeignRepoManager(
            workspace_dir=args.workspace_dir,
            cleanup_on_exit=not args.keep_clone,
        )
        _foreign_repo_manager = manager  # Keep reference for cleanup

        if manager.is_url(args.repo_path):
            if not args.quiet:
                print(f"Cloning repository: {args.repo_path}")

            cloned_repo = manager.clone(
                args.repo_path,
                depth=1 if args.shallow_clone else None,
            )
            repo_path = cloned_repo.local_path
            conventions = manager.detect_conventions(repo_path)
            cloned_repo.conventions = conventions

            if not args.quiet:
                print(f"  Cloned to: {repo_path}")
                if conventions:
                    print(f"  Detected: {conventions.primary_language} {conventions.project_type}")
                    if conventions.test_framework:
                        print(f"  Test framework: {conventions.test_framework}")
                    if conventions.linter:
                        print(f"  Linter: {conventions.linter}")

            # Register cleanup
            if not args.keep_clone:
                atexit.register(lambda: manager.cleanup(cloned_repo))
        else:
            # Local path
            repo_path = Path(args.repo_path).resolve()
            if not repo_path.exists():
                parser.error(f"Repository path does not exist: {repo_path}")
            if not repo_path.is_dir():
                parser.error(f"Repository path is not a directory: {repo_path}")

            # Detect conventions for local repos too
            try:
                conventions = manager.detect_conventions(repo_path)
            except Exception as e:
                if not args.quiet:
                    print(f"Warning: Could not detect conventions: {e}")

    except ForeignRepoError as e:
        parser.error(f"Failed to clone repository: {e}")
    except ConfigurationError as e:
        parser.error(str(e))
    except Exception as e:
        parser.error(f"Unexpected error resolving repository: {e}")

    # Store resolved path back in args for use by agent modes
    args.repo_path = repo_path

    # Determine the execution mode
    execution_mode = "standard"
    if args.review:
        execution_mode = "review"
    elif args.research:
        execution_mode = "research"
    elif args.fix:
        execution_mode = "fix"
    elif args.refactor:
        execution_mode = "refactor"
    elif args.plan:
        execution_mode = "plan"
    elif args.watch:
        execution_mode = "watch"
    elif args.analyze_only:
        execution_mode = "analyze"

    # Handle --prompt-file for smart prompt input
    parsed_prompt: ParsedPrompt | None = None
    if args.prompt_file:
        try:
            prompt_parser = PromptParser()
            parsed_prompt = prompt_parser.parse_file(args.prompt_file)
            if not args.quiet:
                fmt = parsed_prompt.format_detected
                print(f"Loaded goal from {args.prompt_file} (format: {fmt})")
                if parsed_prompt.sub_goals:
                    print(f"  Found {len(parsed_prompt.sub_goals)} sub-goals")
                if parsed_prompt.constraints:
                    print(f"  Found {len(parsed_prompt.constraints)} constraints")
        except PromptParseError as e:
            parser.error(f"Failed to parse prompt file: {e}")
        except Exception as e:
            parser.error(f"Unexpected error reading prompt file: {e}")

    # Validate arguments
    # Goal not required for certain modes
    goal_not_required_modes = (
        args.resume or args.analyze_only or args.research or args.fix or
        args.review or args.agent_mode in ("pre-analysis", "diagnostics")
    )
    has_goal = args.goal or parsed_prompt or args.refactor

    if not goal_not_required_modes and not has_goal:
        parser.error(
            "Goal is required unless using --resume, --analyze-only, --research, "
            "--fix, --refactor, --review, or --agent-mode pre-analysis/diagnostics"
        )

    # Check for existing state if resuming
    state_file = args.repo_path / ".repo-improver-state.json"
    if args.resume:
        if not state_file.exists():
            parser.error(f"No state file found at {state_file}")
        # Load goal from state
        import json
        try:
            state_data = json.loads(state_file.read_text())
            goal = state_data.get("goal", "")
        except (json.JSONDecodeError, OSError) as e:
            parser.error(f"Failed to load state file: {e}")
    elif parsed_prompt:
        # Use parsed prompt for goal (converts to full prompt string)
        goal = parsed_prompt.to_prompt_string()
    elif args.refactor:
        # Generate refactor-specific goal
        goal = (
            f"Refactor the following target while maintaining existing behavior: "
            f"{args.refactor}"
        )
    elif args.fix:
        goal = (
            "Fix all failing tests and lint errors. "
            "Do not add new features or make other changes."
        )
    elif args.research:
        goal = (
            "Analyze and explore the codebase. Identify patterns, architecture, "
            "potential improvements, and technical debt. "
            "Report findings without making any changes."
        )
    else:
        goal = args.goal or "Analyze and report on code quality"
    
    # Enhance goal with conventions guidance if detected
    goal_with_conventions = goal
    if conventions:
        convention_guidance = conventions.to_prompt_guidance()
        goal_with_conventions = f"{goal}\n\n{convention_guidance}"

    # Auto-configure based on conventions or smart defaults
    test_cmd = args.test_cmd
    lint_cmd = args.lint_cmd

    # Use SmartDefaults for auto-detection if not specified
    if test_cmd is None or lint_cmd is None:
        try:
            from .smart_defaults import SmartDefaults
            analyzer = SmartDefaults(repo_path)
            profile = analyzer.analyze()

            if not args.quiet:
                print(f"  Project type: {profile.project_type}")
                print(f"  Primary language: {profile.primary_language}")

            # Apply detected commands
            if test_cmd is None:
                test_cmd = profile.detected_tools.test_command
                if test_cmd and not args.quiet:
                    print(f"  Auto-detected test command: {test_cmd}")

            if lint_cmd is None:
                lint_cmd = profile.detected_tools.lint_command
                if lint_cmd and not args.quiet:
                    print(f"  Auto-detected lint command: {lint_cmd}")

            # Generate additional guidance from profile
            if profile.code_style:
                profile_guidance = profile.generate_guidance()
                if profile_guidance:
                    goal_with_conventions = f"{goal_with_conventions}\n\n{profile_guidance}"

        except Exception as e:
            if not args.quiet:
                print(f"  Warning: Smart detection failed: {e}")

    # Fall back to conventions-based detection if smart defaults didn't find anything
    if test_cmd is None:
        if conventions and conventions.test_framework:
            if conventions.test_framework == "pytest":
                test_cmd = "pytest"
            elif conventions.test_framework in ("jest", "mocha", "vitest"):
                test_cmd = "npm test"
            else:
                test_cmd = "pytest"
        else:
            test_cmd = "pytest"

    if lint_cmd is None:
        if conventions and conventions.linter:
            if conventions.linter == "ruff":
                lint_cmd = "ruff check ."
            elif conventions.linter == "eslint":
                lint_cmd = "npm run lint"
            elif conventions.linter == "flake8":
                lint_cmd = "flake8 ."
            else:
                lint_cmd = "ruff check ."
        else:
            lint_cmd = "ruff check ."

    # Build config
    config = ImproverConfig(
        repo_path=repo_path,
        goal=goal_with_conventions,
        max_iterations=args.max_iterations,
        max_consecutive_failures=args.max_failures,
        run_tests=not args.no_tests,
        test_command=test_cmd,
        run_linter=not args.no_lint,
        lint_command=lint_cmd,
        parallel_validation=args.parallel_validation,
        use_git=not args.no_git,
        branch_name=args.branch,
        state_dir=args.state_dir,
        model=args.model,
        smart_model_selection=args.smart_model_selection,
        max_cost=args.max_cost,
        goal_type=args.goal_type,
        checkpoint_interval=args.checkpoint_interval,
        early_stop_enabled=not args.no_early_stop,
        verbose=args.verbose if hasattr(args, "verbose") else not args.quiet,
        log_level=args.log_level,
        log_file=args.log_file,
        output_format="json" if args.json else "text",
        conventions=conventions,
        is_foreign_repo=cloned_repo is not None,
        original_url=cloned_repo.original_url if cloned_repo else None,
    )

    # Setup logging based on config
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        console_output=config.verbose,
        json_format=config.output_format == "json",
    )

    # Log startup info
    logger.info(f"Starting repo-auto-tool in {execution_mode} mode")
    logger.info(f"Repository: {repo_path}")
    logger.info(f"Log level: {config.log_level}, log file: {config.log_file}")
    if goal:
        logger.info(f"Goal: {goal[:100]}{'...' if len(goal) > 100 else ''}")

    # Perform goal analysis (unless skipped or in certain modes)
    skip_analysis = (
        args.skip_goal_analysis or
        args.resume or
        args.agent_mode or
        execution_mode in ("analyze", "research")
    )

    if not skip_analysis and goal:
        try:
            goal_analyzer = GoalAnalyzer(repo_path)
            assessment = goal_analyzer.analyze(goal)

            # Handle --analyze-goal mode
            if args.analyze_goal:
                if args.json:
                    import json as json_module
                    print(json_module.dumps(assessment.to_dict(), indent=2))
                else:
                    print("\n" + assessment.get_summary())
                return 0

            # Show assessment if not quiet
            if not args.quiet:
                print("\n--- Goal Analysis ---")
                print(f"  Complexity: {assessment.complexity.value.upper()}")
                print(f"  Risk Level: {assessment.risk.value.upper()}")
                print(f"  Est. Iterations: {assessment.estimated_iterations}")

                if assessment.is_vague:
                    print("  [!] Warning: Goal may be too vague")

                if assessment.warnings:
                    for warning in assessment.warnings[:3]:
                        print(f"  [!] {warning}")

                if assessment.suggestions:
                    print("\n  Suggestions:")
                    for suggestion in assessment.suggestions[:3]:
                        print(f"    - {suggestion}")

                print()  # Blank line before continuing

            # Warn about high-risk goals
            if assessment.risk in (GoalRisk.HIGH, GoalRisk.CRITICAL):
                if not args.quiet:
                    print("  ** High-risk goal detected - changes will be reviewed carefully **\n")

            # Suggest plan mode for complex goals
            if (
                assessment.complexity in (GoalComplexity.COMPLEX, GoalComplexity.MAJOR)
                and not args.plan
                and not args.quiet
            ):
                print("  Tip: Consider using --plan mode for complex goals\n")

        except Exception as e:
            # Goal analysis is non-critical - continue if it fails
            if not args.quiet:
                print(f"  Note: Goal analysis unavailable ({e})")

    # Handle --analyze-goal when analysis was skipped
    if args.analyze_goal and skip_analysis:
        print("Cannot analyze goal in this mode", file=sys.stderr)
        return 1

    # Handle agent modes
    if args.agent_mode:
        return _run_agent_mode(args, config)

    # Run based on execution mode
    improver = RepoImprover(config)

    if execution_mode == "analyze" or args.analyze_only:
        analysis = improver.analyze()
        print(analysis)
        return 0

    if execution_mode == "research":
        # Research mode: analyze without making changes
        if not args.quiet:
            print("Running in research mode - analyzing without making changes...")
        analysis = improver.analyze()
        print("\n=== Research Report ===\n")
        print(analysis)
        return 0

    if execution_mode == "plan":
        # Plan mode: create detailed plan, wait for approval
        if not args.quiet:
            print("Running in plan mode - creating implementation plan...")
        # Use goal-decomposer agent for planning
        agent = create_agent(
            mode="goal-decomposer",
            working_dir=args.repo_path,
            timeout=600,
        )
        result = agent.run(goal=goal)
        if result.success:
            print("\n=== Implementation Plan ===\n")
            print(result.output)
            if result.steps:
                print("\n--- Steps ---")
                for i, step in enumerate(result.steps, 1):
                    print(f"  {i}. {step}")
            print("\nReview the plan above. Run without --plan to execute.")
            return 0
        else:
            print(f"Failed to create plan: {result.error}", file=sys.stderr)
            return 1

    if execution_mode == "review":
        # Review mode: show changes on improver branch and how to apply
        import subprocess

        branch = args.branch
        print(f"\n{'='*60}")
        print(f"REVIEW: Changes on branch '{branch}'")
        print("=" * 60)

        # Check if branch exists
        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch],
            cwd=repo_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"\nNo changes found. Branch '{branch}' doesn't exist yet.")
            print("Run repo-improver first to make improvements.")
            return 0

        # Get base branch
        base_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path, capture_output=True, text=True
        )
        current = base_result.stdout.strip()
        base = "main" if current == branch else current

        # Show commits
        print(f"\n## Commits ({base}..{branch}):\n")
        subprocess.run(
            ["git", "log", f"{base}..{branch}", "--oneline", "--no-decorate"],
            cwd=repo_path
        )

        # Show diff stats
        print("\n## Files Changed:\n")
        subprocess.run(
            ["git", "diff", f"{base}..{branch}", "--stat"],
            cwd=repo_path
        )

        # Show full diff (truncated)
        print("\n## Diff Preview (first 100 lines):\n")
        diff_result = subprocess.run(
            ["git", "diff", f"{base}..{branch}"],
            cwd=repo_path, capture_output=True, text=True
        )
        diff_lines = diff_result.stdout.split("\n")[:100]
        print("\n".join(diff_lines))
        if len(diff_result.stdout.split("\n")) > 100:
            print(f"\n... (truncated, use 'git diff {base}..{branch}' for full diff)")

        # Show how to apply
        print(f"\n{'='*60}")
        print("HOW TO APPLY CHANGES:")
        print("=" * 60)
        print(f"""
Option 1 - Merge the branch:
  git checkout {base}
  git merge {branch}

Option 2 - Cherry-pick specific commits:
  git checkout {base}
  git cherry-pick <commit-hash>

Option 3 - Review and merge interactively:
  git checkout {base}
  git merge --no-commit {branch}
  # Review changes, then:
  git commit -m "Apply repo-improver changes"

Option 4 - Create a PR (if using GitHub):
  git push origin {branch}
  gh pr create --base {base} --head {branch}
""")
        return 0

    if execution_mode == "watch":
        # Watch mode: continuous monitoring
        from .watch_mode import WatchConfig, WatchMode

        if not args.quiet:
            print("Starting watch mode - monitoring repository for changes...")
            print("Press Ctrl+C to stop watching")

        try:
            watch_config = WatchConfig(
                debounce_seconds=2.0,
                poll_interval=1.0,
                cooldown_seconds=30.0,
                max_iterations_per_trigger=min(5, args.max_iterations),
            )
            watch = WatchMode(config=config, watch_config=watch_config)
            watch.start(blocking=True)
            return 0
        except KeyboardInterrupt:
            print("\nWatch mode stopped by user")
            return 0
        except Exception as e:
            print(f"Watch mode error: {e}", file=sys.stderr)
            return 1

    # Standard execution (or fix/refactor modes which use standard loop)
    # Initialize TUI if requested and available
    tui: TUI | None = None
    tui_log_handler = None

    try:
        # Create TUI unless disabled or in quiet/json mode
        use_tui = not args.no_tui and not args.quiet and not args.json
        if use_tui:
            try:
                tui_config = TUIConfig(
                    refresh_rate=max(0.1, min(5.0, args.tui_refresh_rate)),
                    enable_colors=True,
                )
                tui = create_tui(use_tui=True, config=tui_config)
                if tui:
                    # Setup TUI logging
                    import logging
                    tui_log_handler = setup_tui_logging(tui, logging.INFO)

                    # Initialize TUI state
                    tui.set_goal(goal)
                    tui.set_iteration(0, args.max_iterations)
                    tui.set_status("initializing")
                    tui.add_log("Starting repo-improver...")

                    # Start TUI update thread
                    tui.start()
            except Exception as e:
                # TUI initialization failed - continue without it
                if not args.quiet:
                    print(f"Note: TUI unavailable ({e}), using standard output")
                tui = None

        # Run the improver
        result = improver.run()

        # Update TUI with final status
        if tui:
            try:
                if result.status == "completed":
                    tui.set_status("complete")
                    tui.add_log("Goal complete!")
                elif result.status == "paused":
                    tui.set_status("paused")
                    tui.add_log("Session paused - can be resumed")
                else:
                    tui.set_status("failed")
                    tui.add_log(f"Finished with status: {result.status}")
                tui.render_once()
            except Exception:
                pass

    finally:
        # Clean up TUI
        if tui:
            try:
                tui.stop()
                tui.cleanup()
            except Exception:
                pass
        if tui_log_handler:
            try:
                import logging
                logging.getLogger().removeHandler(tui_log_handler)
            except Exception:
                pass

    # Show detailed metrics if requested
    if args.show_metrics:
        improver.print_metrics_report()

    # Handle PR creation/preview after successful run
    if result.status == "completed" and (args.create_pr or args.pr_preview):
        try:
            # Determine base branch
            pr_base = args.pr_base
            if pr_base is None:
                if cloned_repo:
                    pr_base = cloned_repo.default_branch
                else:
                    pr_base = "main"

            # Parse labels
            pr_labels = None
            if args.pr_labels:
                pr_labels = [
                    label.strip() for label in args.pr_labels.split(",") if label.strip()
                ]

            if args.pr_preview:
                # Show preview without creating
                if not args.quiet:
                    print("\n" + "=" * 60)
                    print("PULL REQUEST PREVIEW")
                    print("=" * 60 + "\n")

                preview = create_pr_from_repo(
                    repo_path=repo_path,
                    base_branch=pr_base,
                    goal=goal,
                    labels=pr_labels,
                    preview_only=True,
                )
                print(preview)

            elif args.create_pr:
                # Create the PR
                if not args.quiet:
                    print("\nCreating pull request...")

                pr_result: PRResult = create_pr_from_repo(
                    repo_path=repo_path,
                    base_branch=pr_base,
                    goal=goal,
                    draft=args.pr_draft,
                    labels=pr_labels,
                    push_first=True,
                    preview_only=False,
                )

                if pr_result.success:
                    if args.json:
                        import json as json_module
                        print(json_module.dumps(pr_result.to_dict(), indent=2))
                    else:
                        print("\nPR created successfully!")
                        if pr_result.pr_url:
                            print(f"  URL: {pr_result.pr_url}")
                        if pr_result.pr_number:
                            print(f"  Number: #{pr_result.pr_number}")
                else:
                    print(f"\nFailed to create PR: {pr_result.error}", file=sys.stderr)
                    # Don't change exit code - improvements were successful

        except Exception as e:
            print(f"\nError during PR creation: {e}", file=sys.stderr)
            # Don't change exit code - improvements were successful

    # Exit code based on result
    if result.status == "completed":
        return 0
    elif result.status == "paused":
        return 2  # Can be resumed
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
