# repo-improver TODO

## Setup Status
- [ ] Initial build & local install
- [ ] Push to GitHub
- [ ] First self-improvement run

---

## Feature Roadmap

### 🔴 High Priority

#### Error Handling
- [ ] Add custom exception classes (`ImproverError`, `ValidationError`, `ClaudeAPIError`)
- [ ] Wrap all subprocess calls with proper error catching
- [ ] Add retry logic for transient Claude CLI failures
- [ ] Graceful degradation when validators fail to run (vs fail validation)
- [ ] Better error messages with actionable suggestions

#### Logging
- [ ] Structured logging (JSON format option)
- [ ] Log levels configurable via CLI (`--log-level DEBUG`)
- [ ] Separate log streams (console vs file)
- [ ] Log rotation for long-running sessions
- [ ] Include token/cost tracking in logs (if Claude CLI exposes it)
- [ ] Rich console output with progress bars (optional)

### 🟡 Medium Priority

#### Goal Modes
- [ ] **Macro Goals Mode**: Large architectural changes, multi-phase planning
  - Pre-planning phase: Claude creates a plan before executing
  - Milestone tracking within a single goal
  - Dependency-aware task ordering
  
- [ ] **Short/Research Mode**: Quick exploration, no commits
  - `--research` flag: analyze only, suggest changes, don't apply
  - `--dry-run`: show what would change without doing it
  - Single-shot mode: one improvement, then stop
  
- [ ] **Fix/Structure Mode**: Targeted repairs
  - `--fix-only`: only fix failing tests/lint, no new features
  - `--structure`: reorganize code without changing behavior
  - `--refactor <target>`: focus on specific file/module

#### Summaries & Reporting
- [ ] Cross-iteration summaries (what changed overall)
- [ ] Validation trend tracking (are we getting better or worse?)
- [ ] Generate markdown report at end of session
- [ ] Diff summary with semantic grouping (tests added, refactors, bug fixes)
- [ ] Progress toward goal estimation (% complete)
- [ ] Cost/token usage summary

### 🟢 Nice to Have

#### State Management
- [ ] Multiple named sessions per repo
- [ ] State diffing (compare two sessions)
- [ ] Export state to shareable format
- [ ] Import goals/config from YAML file
- [ ] Checkpoint/restore mid-iteration

#### Integration
- [ ] GitHub Actions workflow for CI improvement
- [ ] Pre-commit hook integration
- [ ] VS Code extension (show improvement suggestions)
- [ ] Slack/Discord notifications on completion

#### Advanced
- [ ] Multi-model support (use different models for analysis vs editing)
- [ ] Parallel validation (run tests and lint simultaneously)
- [ ] Cost budgeting (`--max-cost $5.00`)
- [ ] Learning from past sessions (what worked well?)

---

## Completed
- [x] Initial package structure
- [x] CLI interface
- [x] Basic improvement loop
- [x] Git safety (branching, commits, rollback)
- [x] Validation pipeline (tests, lint)
- [x] State persistence & resumability
- [x] Bootstrap script

---

## Notes

### First Self-Improvement Goals (in order)
1. "Add comprehensive error handling with custom exceptions"
2. "Add structured logging with configurable levels"
3. "Add unit tests for all modules with 80% coverage"
4. "Add --research and --fix-only modes"
5. "Add session summary reports"

### Architecture Decisions
- Keep it simple: single-threaded, synchronous for now
- State is JSON for easy inspection/debugging
- Git is optional but recommended
- Validators are pluggable
