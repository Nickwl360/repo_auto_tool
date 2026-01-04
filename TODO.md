# repo-auto-tool TODO

## Setup Status
- [x] Initial build & local install
- [x] Push to GitHub
- [x] First self-improvement run

---

## Feature Roadmap

### High Priority (Production Readiness)

#### Testing
- [ ] Add unit tests for core modules (target 80% coverage)
- [ ] Add integration tests for the full improvement loop
- [ ] Add tests for edge cases (network failures, malformed state)

#### Documentation
- [ ] Add docstrings to remaining public functions
- [ ] Document all CLI flags with examples
- [ ] Add CONTRIBUTING.md for contributors

### Medium Priority (Enhanced Functionality)

#### Goal Modes
- [ ] **Macro Goals Mode**: Large architectural changes, multi-phase planning
  - Pre-planning phase: Claude creates a plan before executing
  - Milestone tracking within a single goal
  - Dependency-aware task ordering

- [x] **Short/Research Mode**: Quick exploration, no commits
  - `--research` flag: analyze only, suggest changes, don't apply
  - `--plan` flag: create detailed plan, wait for approval before executing
  - Single-shot mode: one improvement, then stop

- [x] **Fix/Structure Mode**: Targeted repairs
  - `--fix` flag: only fix failing tests/lint, no new features
  - `--refactor <target>`: focus on specific file/module
  - `--watch` flag: (placeholder) monitor for changes, continuously improve

#### Summaries & Reporting
- [ ] Cross-iteration summaries (what changed overall)
- [ ] Validation trend tracking (are we getting better or worse?)
- [ ] Generate markdown report at end of session
- [ ] Diff summary with semantic grouping (tests added, refactors, bug fixes)
- [ ] Progress toward goal estimation (% complete)
- [x] Cost/token usage summary

### Nice to Have (Future)

#### State Management
- [ ] Multiple named sessions per repo
- [ ] State diffing (compare two sessions)
- [ ] Export state to shareable format
- [x] Import goals/config from YAML file -- `--prompt-file` supports .yaml, .json, .md, .txt

#### Integration
- [ ] GitHub Actions workflow for CI improvement
- [ ] Pre-commit hook integration
- [ ] VS Code extension (show improvement suggestions)
- [ ] Slack/Discord notifications on completion

#### Advanced
- [x] Multi-model support (use different models for analysis vs editing) -- `--smart-model-selection` flag
- [x] Parallel validation (run tests and lint simultaneously) -- `--parallel-validation` flag
- [x] Cost budgeting (`--max-cost $5.00`)
- [x] Smart context summarization (efficient history-aware context for longer sessions)
- [x] Learning from past sessions (what worked well?) -- `SessionHistory` class persists learnings across sessions

---

## Completed

### Core Features
- [x] Initial package structure with src-layout
- [x] CLI interface with argparse
- [x] Basic improvement loop (analyze -> improve -> validate -> commit)
- [x] Git safety (branching, commits, rollback)
- [x] Validation pipeline (tests, lint, custom validators)
- [x] State persistence & resumability
- [x] Bootstrap script for self-improvement

### Error Handling (All Done)
- [x] Custom exception hierarchy (RepoAutoToolError and 13+ subclasses)
- [x] Subprocess calls wrapped with proper error catching
- [x] Retry logic with exponential backoff for transient Claude CLI failures
- [x] Graceful degradation when validators fail to run
- [x] Structured error messages with details

### Logging (All Done)
- [x] Structured logging (JSON format option via --json)
- [x] Log levels configurable via CLI (--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL)
- [x] Separate log streams (console vs file)
- [x] JSONFormatter and ConsoleFormatter implementations

### Safety & Robustness (All Done)
- [x] Secret redaction (API keys, tokens, AWS credentials)
- [x] Dangerous command detection
- [x] Convergence detection to prevent infinite loops
- [x] Checkpoint support during long runs
- [x] Context truncation to manage token usage

### Agents (All Done)
- [x] PreAnalysisAgent - analyzes codebase and suggests goals
- [x] GoalDecomposerAgent - breaks goals into actionable steps
- [x] ReviewerAgent - reviews changes and provides feedback
- [x] Agent mode CLI support (--agent-mode)

### Package & Distribution (All Done)
- [x] pyproject.toml with hatchling build
- [x] py.typed marker for type checking support
- [x] Comprehensive __init__.py exports (80+ public items)
- [x] Virtual environment detection and command resolution

### Smart Prompt Input (All Done)
- [x] `--prompt-file` / `-f` flag to read goals from files
- [x] Intelligent parsing of .txt, .md, .yaml, .json formats
- [x] Extract sub-goals and constraints from structured prompts
- [x] ParsedPrompt with to_prompt_string() for Claude-friendly output

### Execution Modes (All Done)
- [x] `--research` mode: explore without making changes
- [x] `--fix` mode: only fix failing tests/lint
- [x] `--refactor <target>` mode: focused refactoring
- [x] `--plan` mode: create plan, wait for approval
- [x] `--watch` mode: continuous file monitoring and improvement

---

## Architecture Decisions
- Keep it simple: single-threaded, synchronous for now
- State is JSON for easy inspection/debugging
- Git is optional but recommended
- Validators are pluggable via CommandValidator
- No external runtime dependencies (pure stdlib)
