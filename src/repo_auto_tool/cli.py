"""CLI entry point for repo-improver."""

import argparse
import sys
from pathlib import Path

from .config import ImproverConfig
from .improver import RepoImprover
from .logging import setup_logging


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
    if not args.resume and not args.goal and not args.analyze_only:
        parser.error("Goal is required unless using --resume or --analyze-only")
    
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
