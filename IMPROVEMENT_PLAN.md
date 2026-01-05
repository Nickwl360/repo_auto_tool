# Comprehensive Improvement Plan for Repo-Improver

## Overview
Transform repo-improver into a highly efficient, research-capable agentic system with advanced session management, cost tracking, and intelligent model selection.

---

## Priority 1: Token Budget & Usage Tracking

### Current State
- Token usage is tracked but not displayed as percentage
- No real-time budget awareness during execution
- No warnings before hitting limits

### Implementation: `BudgetTracker`

**Features:**
1. **Percentage-based tracking**
   - Display tokens used / budget as percentage
   - Color-coded warnings (green <50%, yellow 50-80%, red >80%)
   - Real-time updates in TUI

2. **Cost projection**
   - Estimate cost per iteration
   - Project remaining iterations before budget exhaustion
   - Suggest model downgrade if approaching limit

3. **Budget analytics**
   - Cost per successful change
   - Most expensive operation types
   - Efficiency metrics (tokens/LOC changed)

**Files:**
- `src/repo_auto_tool/budget_tracker.py` (new)
- Enhance `src/repo_auto_tool/tui.py` (add budget panel)
- Enhance `src/repo_auto_tool/config.py` (add budget display options)

**CLI:**
```bash
repo-improver --max-cost 10.00 --show-budget-percentage
# During run: "Budget: 35% used ($3.50/$10.00) - 15 iterations remaining"
```

---

## Priority 2: Research Mode with Hypothesis Testing

### Vision
A dedicated research mode for scientific analysis, hypothesis testing, and systematic investigation.

### Implementation: `ResearchMode`

**Features:**

1. **Research Question Framework**
   ```bash
   repo-improver --research-mode --research-question "How does async I/O affect performance?"
   ```

2. **Hypothesis-Driven Investigation**
   - User provides hypothesis OR system generates one
   - Systematic testing and data collection
   - Evidence-based conclusions

3. **Structured Output**
   - **Hypothesis**: Clear statement of what's being tested
   - **Methodology**: How the investigation was conducted
   - **Observations**: Data and findings
   - **Analysis**: Interpretation of results
   - **Conclusion**: Answer to research question
   - **Confidence**: Level of certainty (Low/Medium/High)

4. **Data Visualization**
   - Performance graphs (matplotlib/plotly)
   - Before/after comparisons
   - Complexity analysis charts
   - Token usage trends

5. **Research Report Generation**
   - Markdown report with embedded graphs
   - LaTeX option for papers
   - JSON export for further analysis

**Files:**
- `src/repo_auto_tool/research_mode.py` (new)
- `src/repo_auto_tool/hypothesis_tester.py` (new)
- `src/repo_auto_tool/data_visualizer.py` (new)

**Example Session:**
```bash
repo-improver --research-mode \
  --research-question "Does type hinting improve code quality?" \
  --hypothesis "Adding type hints will reduce type errors by >50%" \
  --output-format markdown \
  /path/to/repo

# Output: research_report_2026-01-04.md with:
# - Hypothesis
# - Baseline measurements
# - Changes made
# - After measurements
# - Statistical analysis
# - Graphs
# - Conclusion
```

---

## Priority 3: Named Session Management

### Current State
- Sessions tied to repository path
- Can't have multiple concurrent investigations
- Resume is repo-based only

### Implementation: `SessionManager`

**Features:**

1. **Named Sessions**
   ```bash
   # Create named session
   repo-improver --session "type-hints-experiment" /path/to/repo "Add type hints"

   # Resume by name
   repo-improver --resume type-hints-experiment

   # List sessions
   repo-improver --list-sessions

   # Session info
   repo-improver --session-info type-hints-experiment
   ```

2. **Session Storage**
   - Default: `~/.repo-improver/sessions/`
   - Each session: `<name>_<timestamp>/`
   - Contains: state, history, learned prompts, reports
   - Metadata: repo path, goal, started, status

3. **Multi-Repo Support**
   - Sessions can reference any repo
   - Archive completed sessions
   - Export/import sessions

4. **Session Comparison**
   ```bash
   repo-improver --compare session1 session2
   # Shows: iterations, cost, success rate, time, quality
   ```

**Files:**
- `src/repo_auto_tool/session_manager.py` (new)
- `src/repo_auto_tool/session_storage.py` (new)
- Enhance `cli.py` for session commands

**Data Structure:**
```
~/.repo-improver/
├── sessions/
│   ├── type-hints-exp_20260104_143022/
│   │   ├── state.json
│   │   ├── history.json
│   │   ├── prompts.json
│   │   ├── research_report.md
│   │   └── metadata.json
│   ├── async-refactor_20260103_091544/
│   └── ...
├── config.json
└── global_stats.json
```

---

## Priority 4: Aggressive Model Efficiency

### Current State
- Smart model selection exists but conservative
- Always uses Sonnet for most tasks

### Implementation: `AggressiveModelSelector`

**Strategy:**

1. **Default to Haiku (claude-3-5-haiku-20241022)**
   - 90% of tasks should use Haiku
   - Only escalate when necessary

2. **Use Sonnet for:**
   - Large refactorings (>5 files)
   - Complex architectural decisions
   - Research mode hypothesis generation
   - After 2+ Haiku failures
   - Code complexity score >500

3. **Use Opus for:**
   - Critical scientific decisions
   - Novel algorithm design
   - Security-sensitive changes
   - User explicitly requests (`--model opus`)

4. **Dynamic Downgrading**
   - If task succeeds with Haiku, prefer Haiku for similar tasks
   - Learn which task types need which models

5. **Speed Optimizations**
   - Shorter prompts for simple tasks
   - Parallel validation where possible
   - Cache repeated analyses

**Files:**
- Enhance `src/repo_auto_tool/model_selector.py`
- Add `src/repo_auto_tool/task_classifier.py`

**Expected Impact:**
- 50-70% cost reduction
- 2-3x faster iterations
- Maintain quality through smart escalation

---

## Priority 5: Objective Deviation & Exploration

### Vision
Allow the system to deviate from strict objectives when it discovers better approaches or critical issues.

### Implementation: `SmartDeviation`

**Features:**

1. **Automatic Issue Detection**
   - Security vulnerabilities found → prioritize fixing
   - Performance bottlenecks discovered → suggest optimization
   - Code quality issues → recommend refactoring

2. **Deviation Modes**
   - `--strict`: Never deviate (current behavior)
   - `--flexible`: Suggest deviations, ask for approval
   - `--autonomous`: Auto-deviate for critical issues
   - `--exploratory`: Actively look for improvements

3. **Deviation Framework**
   ```python
   deviation = Deviation(
       original_goal="Add type hints",
       deviation_reason="Found security vulnerability in auth.py",
       proposed_action="Fix SQL injection before continuing",
       priority="critical",
       estimated_cost=2_iterations,
   )
   ```

4. **User Control**
   - Interactive approval for deviations
   - Whitelist/blacklist deviation types
   - Maximum deviation budget

**Files:**
- `src/repo_auto_tool/deviation_engine.py` (new)
- Enhance `interrupt_handler.py` for deviation prompts

---

## Additional High-Impact Features

### 6. **Parallel Execution Mode**
- Run multiple independent tasks in parallel
- Use multiple Claude API keys
- Process different files simultaneously
- Merge results intelligently

### 7. **Benchmark Suite**
- Standard test repositories
- Automated quality measurement
- Performance regression testing
- Compare against baselines

### 8. **Learning Dashboard**
- Web UI showing learned patterns
- Success rate by error type
- Model usage statistics
- Cost trends over time

### 9. **Export/Import Workflows**
- Save successful workflows
- Share with team
- Version control for prompts
- Template library

### 10. **Integration Tests**
- Test against real repos
- Measure improvement quality
- Track regressions
- CI/CD integration

---

## Implementation Roadmap

### Phase 1: Foundation (Immediate)
1. ✅ BudgetTracker with percentage display
2. ✅ Named SessionManager
3. ✅ AggressiveModelSelector enhancements

### Phase 2: Research Capabilities (Week 1)
4. ResearchMode framework
5. HypothesisTester
6. DataVisualizer with graphs

### Phase 3: Intelligence (Week 2)
7. SmartDeviation engine
8. Task-to-model learning
9. Performance optimizations

### Phase 4: Ecosystem (Week 3)
10. Learning Dashboard
11. Benchmark Suite
12. Workflow templates

---

## Technical Specifications

### BudgetTracker API
```python
class BudgetTracker:
    def __init__(self, max_cost: float, model_costs: dict):
        self.max_cost = max_cost
        self.spent = 0.0

    def track_usage(self, tokens: TokenUsage) -> BudgetStatus:
        """Track token usage and return status."""

    def get_percentage_used(self) -> float:
        """Return 0.0-1.0 percentage of budget used."""

    def project_remaining_iterations(self) -> int:
        """Estimate iterations remaining."""

    def should_downgrade_model(self) -> bool:
        """True if should switch to cheaper model."""

    def get_display_string(self) -> str:
        """Formatted string for display."""
```

### SessionManager API
```python
class SessionManager:
    def create_session(self, name: str, repo_path: Path, goal: str) -> Session:
        """Create a new named session."""

    def resume_session(self, name: str) -> Session:
        """Resume existing session by name."""

    def list_sessions(self) -> list[SessionInfo]:
        """List all sessions with metadata."""

    def compare_sessions(self, names: list[str]) -> ComparisonReport:
        """Compare multiple sessions."""

    def archive_session(self, name: str) -> None:
        """Archive completed session."""
```

### ResearchMode API
```python
class ResearchMode:
    def __init__(self, question: str, hypothesis: str | None = None):
        self.question = question
        self.hypothesis = hypothesis or self.generate_hypothesis()

    def run_investigation(self) -> ResearchReport:
        """Conduct systematic investigation."""

    def collect_baseline(self) -> Metrics:
        """Collect baseline measurements."""

    def collect_after(self) -> Metrics:
        """Collect after-change measurements."""

    def analyze_results(self) -> Analysis:
        """Statistical analysis of results."""

    def generate_report(self, format: str) -> str:
        """Generate research report."""
```

---

## Success Criteria

1. **Budget Tracking**
   - ✅ Real-time percentage display
   - ✅ Accurate cost projection
   - ✅ Warnings before budget exhaustion

2. **Research Mode**
   - ✅ Hypothesis-driven investigation
   - ✅ Automated data collection
   - ✅ Report generation with graphs
   - ✅ Reproducible methodology

3. **Session Management**
   - ✅ Named sessions work reliably
   - ✅ Easy resume by name
   - ✅ Session comparison generates insights

4. **Model Efficiency**
   - ✅ 50%+ cost reduction
   - ✅ 2x+ speed improvement
   - ✅ No quality degradation

5. **Deviations**
   - ✅ Critical issues detected
   - ✅ User approval workflow
   - ✅ Deviation tracking and reporting

---

## Compatibility & Migration

- **Backward compatible**: Old sessions still work
- **Automatic migration**: Convert old state files to new format
- **Default behavior unchanged**: New features opt-in
- **Configuration**: New config options with sensible defaults

---

## Testing Strategy

1. **Unit tests** for each new module
2. **Integration tests** for workflows
3. **Cost simulation** tests (mock Claude API)
4. **Research mode validation** (compare against manual analysis)
5. **Performance benchmarks** (measure speedup)

---

## Documentation

1. **User Guide**: How to use research mode
2. **API Reference**: For programmatic use
3. **Examples**: Research report samples
4. **Tutorial**: Step-by-step walkthrough
5. **Migration Guide**: Upgrading from older versions

---

## Timeline

**Total: ~2-3 weeks for full implementation**

- **Week 1**: Foundation + Research Mode (60 hours)
- **Week 2**: Intelligence + Optimizations (40 hours)
- **Week 3**: Ecosystem + Polish (30 hours)

---

## Next Steps

1. Implement BudgetTracker
2. Implement SessionManager
3. Enhance ModelSelector for aggressive efficiency
4. Build ResearchMode framework
5. Add data visualization
6. Implement SmartDeviation
7. Create learning dashboard
8. Write comprehensive tests
9. Update documentation
10. Release v2.0

---

This plan transforms repo-improver from a good improvement tool into an **intelligent research and development platform**.
