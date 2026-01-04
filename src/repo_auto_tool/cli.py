"""CLI entry point for repo-improver."""

import argparse
import atexit
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
from .improver import RepoImprover
from .logging import setup_logging
from .prompt_parser import ParsedPrompt, PromptParser
from .tui import TUI, TUIConfig, create_tui, setup_tui_logging

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
  # Basic usage
  repo-improver /path/to/repo "Add type hints to all functions"

  # With custom settings
  repo-improver . "Improve test coverage to 80%" --max-iterations 50 --test-cmd "pytest --cov"

  # Work on a remote repository (clones automatically)
  repo-improver https://github.com/user/repo "Add documentation"
  repo-improver user/repo "Fix linting issues"  # GitHub shorthand

  # Read goal from a file (supports .txt, .md, .yaml, .json)
  repo-improver . --prompt-file ideas.md

  # Research mode - explore without changes
  repo-improver . --research

  # Fix mode - only fix failing tests/lint
  repo-improver . --fix

  # Refactor a specific target
  repo-improver . --refactor src/auth.py
  repo-improver . --refactor "authentication module"

  # Plan mode - create plan, wait for approval
  repo-improver . "Add caching layer" --plan

  # Resume a paused session
  repo-improver . --resume
""",
    )

    parser.add_argument(
        "repo_path",
        type=str,  # String to support both paths and URLs
        nargs="?",
        default=".",
        help="Path to repository or URL (GitHub/GitLab/etc). Supports: "
             "local path, https://github.com/user/repo, user/repo (GitHub shorthand)",
    )
    
    parser.add_argument(
        "goal",
        type=str,
        nargs="?",
        help="The improvement goal (natural language description)",
    )

    # Smart prompt input
    parser.add_argument(
        "--prompt-file", "-f",
        type=Path,
        default=None,
        metavar="FILE",
        help="Read goal from a file (supports .txt, .md, .yaml, .json). "
             "Intelligently parses structured and unstructured formats.",
    )

    # Iteration settings
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=20,
        help="Maximum improvement iterations (default: 20)",
    )
    
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Max consecutive failures before stopping (default: 3)",
    )
    
    # Validation settings
    parser.add_argument(
        "--test-cmd",
        type=str,
        default=None,
        help="Test command (auto-detected or defaults to pytest)",
    )

    parser.add_argument(
        "--lint-cmd",
        type=str,
        default=None,
        help="Lint command (auto-detected or defaults to ruff check .)",
    )
    
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip running tests",
    )
    
    parser.add_argument(
        "--no-lint",
        action="store_true",
        help="Skip running linter",
    )

    parser.add_argument(
        "--parallel-validation",
        action="store_true",
        help="Run validators (tests, lint) in parallel for faster validation",
    )

    # Git settings
    parser.add_argument(
        "--no-git",
        action="store_true",
        help="Don't use git (no branching/commits)",
    )
    
    parser.add_argument(
        "--branch",
        type=str,
        default="repo-improver/auto",
        help="Git branch name for changes",
    )
    
    # Mode settings
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze, don't make changes",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous session state",
    )

    # New execution modes (Goal #8)
    parser.add_argument(
        "--research",
        action="store_true",
        help="Research mode: explore and report without making changes. "
             "Analyzes the codebase, identifies patterns, and suggests improvements.",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix mode: only fix failing tests and lint errors. "
             "Does not add new features or make other changes.",
    )

    parser.add_argument(
        "--refactor",
        type=str,
        default=None,
        metavar="TARGET",
        help="Refactor mode: focused refactoring on a specific file, module, or pattern. "
             "Example: --refactor src/auth.py or --refactor 'authentication module'",
    )

    parser.add_argument(
        "--plan",
        action="store_true",
        help="Plan mode: create a detailed implementation plan and wait for approval "
             "before executing. Outputs the plan without making changes.",
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode: monitor the repository for changes and continuously improve. "
             "Runs in the background and responds to file changes.",
    )

    parser.add_argument(
        "--agent-mode",
        type=str,
        choices=["pre-analysis", "goal-decomposer", "reviewer", "diagnostics"],
        metavar="MODE",
        help="Run in agent mode: pre-analysis (analyze and suggest goals), "
             "goal-decomposer (break goal into steps), reviewer (review changes), "
             "diagnostics (analyze session history for recurring issues)",
    )
    
    # Output settings
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output",
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: .repo-improver.log in repo)",
    )

    # State and convergence settings
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Directory for state files (default: repo root)",
    )

    parser.add_argument(
        "--goal-type",
        type=str,
        choices=["open-ended", "bounded", "exploratory"],
        default="open-ended",
        help="Goal type: open-ended, bounded, or exploratory (default: open-ended)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Create checkpoint every N iterations (default: 5, 0 to disable)",
    )

    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping when convergence detected",
    )

    parser.add_argument(
        "--smart-model-selection",
        action="store_true",
        help="Auto-select model based on task complexity (Haiku for simple tasks)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Claude model to use (e.g., claude-sonnet-4-20250514). Overrides smart selection.",
    )

    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        metavar="DOLLARS",
        help="Maximum cost in USD before stopping (e.g., --max-cost 5.00 for $5 budget)",
    )

    parser.add_argument(
        "--show-metrics",
        action="store_true",
        help="Show detailed session metrics report at the end of the run",
    )

    # TUI settings
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the interactive terminal UI (use enhanced plain text mode instead)",
    )

    parser.add_argument(
        "--tui-refresh-rate",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="TUI refresh rate in seconds (default: 0.5)",
    )

    # Foreign repository options
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory for cloning remote repositories (default: temp dir)",
    )

    parser.add_argument(
        "--keep-clone",
        action="store_true",
        help="Don't delete cloned repository on exit (useful for inspection)",
    )

    parser.add_argument(
        "--shallow-clone",
        action="store_true",
        help="Use shallow clone (faster, less disk space, but limited git history)",
    )

    args = parser.parse_args()

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
    if args.research:
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
        args.agent_mode in ("pre-analysis", "diagnostics")
    )
    has_goal = args.goal or parsed_prompt or args.refactor

    if not goal_not_required_modes and not has_goal:
        parser.error(
            "Goal is required unless using --resume, --analyze-only, --research, "
            "--fix, --refactor, or --agent-mode pre-analysis/diagnostics"
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
        verbose=not args.quiet,
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

    # Exit code based on result
    if result.status == "completed":
        return 0
    elif result.status == "paused":
        return 2  # Can be resumed
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
