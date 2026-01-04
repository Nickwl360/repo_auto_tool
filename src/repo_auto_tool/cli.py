"""CLI entry point for repo-improver."""

import argparse
import sys
from pathlib import Path

from .agents import AgentMode, create_agent
from .config import ImproverConfig
from .improver import RepoImprover
from .logging import setup_logging


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


def main():
    parser = argparse.ArgumentParser(
        description="Continuously improve a codebase using Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  repo-improver /path/to/repo "Add type hints to all functions"
  
  # With custom settings
  repo-improver . "Improve test coverage to 80%" --max-iterations 50 --test-cmd "pytest --cov"
  
  # Analyze only (no changes)
  repo-improver . "Refactor for better modularity" --analyze-only
  
  # Resume a paused session
  repo-improver . --resume
""",
    )
    
    parser.add_argument(
        "repo_path",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Path to the repository (default: current directory)",
    )
    
    parser.add_argument(
        "goal",
        type=str,
        nargs="?",
        help="The improvement goal (natural language description)",
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
        default="pytest",
        help="Test command (default: pytest)",
    )
    
    parser.add_argument(
        "--lint-cmd",
        type=str,
        default="ruff check .",
        help="Lint command (default: ruff check .)",
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

    parser.add_argument(
        "--agent-mode",
        type=str,
        choices=["pre-analysis", "goal-decomposer", "reviewer"],
        metavar="MODE",
        help="Run in agent mode: pre-analysis (analyze and suggest goals), "
             "goal-decomposer (break goal into steps), reviewer (review changes)",
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

    args = parser.parse_args()
    
    # Validate arguments
    # Goal not required for pre-analysis mode or analyze-only
    if not args.resume and not args.goal and not args.analyze_only:
        if args.agent_mode != "pre-analysis":
            parser.error(
                "Goal is required unless using --resume, --analyze-only, "
                "or --agent-mode pre-analysis"
            )
    
    # Check for existing state if resuming
    state_file = args.repo_path / ".repo-improver-state.json"
    if args.resume:
        if not state_file.exists():
            parser.error(f"No state file found at {state_file}")
        # Load goal from state
        import json
        state_data = json.loads(state_file.read_text())
        goal = state_data["goal"]
    else:
        goal = args.goal or "Analyze and report on code quality"
    
    # Build config
    config = ImproverConfig(
        repo_path=args.repo_path,
        goal=goal,
        max_iterations=args.max_iterations,
        max_consecutive_failures=args.max_failures,
        run_tests=not args.no_tests,
        test_command=args.test_cmd,
        run_linter=not args.no_lint,
        lint_command=args.lint_cmd,
        use_git=not args.no_git,
        branch_name=args.branch,
        verbose=not args.quiet,
        log_level=args.log_level,
        log_file=args.log_file,
        output_format="json" if args.json else "text",
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

    # Run
    improver = RepoImprover(config)
    
    if args.analyze_only:
        analysis = improver.analyze()
        print(analysis)
        return 0
    
    result = improver.run()
    
    # Exit code based on result
    if result.status == "completed":
        return 0
    elif result.status == "paused":
        return 2  # Can be resumed
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
