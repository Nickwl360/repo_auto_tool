# repo-improver

**Continuously improve a codebase toward a goal using Claude Code CLI.**

This tool creates an agentic loop that:
1. Analyzes your repo against a goal
2. Makes incremental improvements via Claude Code
3. Validates changes (tests, linting)
4. Commits successful changes, rolls back failures
5. Repeats until the goal is achieved

## Installation

```bash
# Prerequisites: Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Install repo-improver
pip install repo-improver

# Or from source
git clone https://github.com/nick/repo-improver
cd repo-improver
pip install -e .
```

## Quick Start

```bash
# Basic usage
repo-improver /path/to/your/repo "Add comprehensive docstrings to all public functions"

# From within the repo
cd my-project
repo-improver . "Refactor the utils module for better separation of concerns"

# Just analyze (no changes)
repo-improver . "Improve test coverage" --analyze-only
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    IMPROVEMENT LOOP                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Analyze  │───▶│ Claude Code  │───▶│  Validate    │      │
│  │ vs Goal  │    │ Makes Edits  │    │ (tests/lint) │      │
│  └──────────┘    └──────────────┘    └──────┬───────┘      │
│       ▲                                      │              │
│       │         ┌────────────────────────────┴────┐        │
│       │         ▼                                 ▼        │
│       │   ┌──────────┐                    ┌──────────┐     │
│       │   │  PASS    │                    │  FAIL    │     │
│       │   │  Commit  │                    │ Rollback │     │
│       │   └────┬─────┘                    └────┬─────┘     │
│       │        │                               │           │
│       └────────┴───────────────────────────────┘           │
│                                                             │
│  Exit conditions:                                           │
│  • Goal complete (Claude signals GOAL_COMPLETE)            │
│  • Max iterations reached                                   │
│  • Too many consecutive failures                            │
│  • User interrupt (Ctrl+C → pauses, can resume)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### CLI Options

```bash
repo-improver [REPO_PATH] "GOAL" [OPTIONS]

Arguments:
  REPO_PATH           Path to repository (default: current directory)
  GOAL                Natural language description of the improvement goal

Options:
  -n, --max-iterations N    Max iterations (default: 20)
  --max-failures N          Consecutive failures before stopping (default: 3)
  
  --test-cmd CMD            Test command (default: pytest)
  --lint-cmd CMD            Lint command (default: ruff check .)
  --no-tests                Skip tests
  --no-lint                 Skip linting
  
  --no-git                  Don't use git branching/commits
  --branch NAME             Git branch name (default: repo-improver/auto)
  
  --analyze-only            Only analyze, don't make changes
  --resume                  Resume from previous session
  
  -q, --quiet               Minimal output
  --json                    Output as JSON
```

### Python API

```python
from repo_improver import RepoImprover, ImproverConfig

config = ImproverConfig(
    repo_path="/path/to/project",
    goal="Convert all string formatting to f-strings",
    max_iterations=30,
    run_tests=True,
    test_command="pytest -x",  # Stop on first failure
)

improver = RepoImprover(config)

# Just analyze
analysis = improver.analyze()
print(analysis)

# Run the full improvement loop
result = improver.run()

print(f"Status: {result.status}")
print(f"Iterations: {result.total_iterations}")
print(f"Summary: {result.summary}")
```

### Advanced: Custom Validators

```python
from repo_improver import RepoImprover, ImproverConfig
from repo_improver.validators import CommandValidator, ValidationPipeline

config = ImproverConfig(
    repo_path=".",
    goal="Optimize database queries",
    custom_validators=[
        "python scripts/check_n_plus_one.py",
        "python -m pyperformance run --fast",
    ],
)

# Or programmatically
improver = RepoImprover(config)
improver.validators = ValidationPipeline([
    CommandValidator("security", "bandit -r src/"),
    CommandValidator("complexity", "radon cc src/ -a -nc"),
])
```

## Example Goals

Here are some goals that work well:

```bash
# Code quality
"Add type hints to all functions in the src/ directory"
"Add docstrings to all public classes and methods"
"Refactor functions longer than 50 lines into smaller functions"

# Testing
"Increase test coverage to at least 80%"
"Add unit tests for all edge cases in the validation module"
"Convert unittest tests to pytest style"

# Modernization
"Update all code to use Python 3.11+ features"
"Replace requests with httpx for async support"
"Convert the project from setup.py to pyproject.toml"

# Architecture
"Extract the authentication logic into a separate module"
"Implement the repository pattern for database access"
"Add proper error handling and custom exceptions"

# Documentation
"Add a comprehensive README with installation and usage instructions"
"Generate API documentation for all modules"
"Add inline comments explaining complex algorithms"
```

## State & Resumability

The tool saves state to `.repo-improver-state.json` after each iteration:

```bash
# Session interrupted or max iterations reached
repo-improver . "big refactoring goal"
# ... runs for a while, you hit Ctrl+C

# Resume later
repo-improver . --resume
```

## Git Safety

By default, all changes are made on a dedicated branch:

- Creates branch `repo-improver/auto` (configurable)
- Commits after each successful iteration
- Rolls back failed changes before the next attempt
- Your original branch is untouched

After completion, review and merge:

```bash
git checkout repo-improver/auto
git log --oneline  # See all iterations
git diff main      # Review total changes
git checkout main
git merge repo-improver/auto
```

## How Claude Code Edits Files

When `repo-improver` calls Claude Code with:
```bash
claude -p "Add docstring to function X in file.py" \
  --allowedTools "Edit(*),Write(*),Read(*)" \
  --yes \
  --cwd /path/to/repo
```

Claude Code directly edits `/path/to/repo/file.py` using its `Edit` tool. The `--yes` flag auto-accepts all edits without prompting.

## Architecture

```
repo_improver/
├── __init__.py          # Package exports
├── cli.py               # Command-line interface
├── config.py            # Configuration dataclass
├── improver.py          # Main orchestration loop
├── claude_interface.py  # Claude Code CLI wrapper
├── validators.py        # Test/lint validation
├── git_helper.py        # Git operations
└── state.py             # Persistent state management
```

## Requirements

- Python 3.11+
- Claude Code CLI (`npm install -g @anthropic-ai/claude-code`)
- Git (optional but recommended)
- An Anthropic API key configured for Claude Code

## License

MIT
