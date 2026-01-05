# Research Mode Guide

## Overview

Research Mode enables hypothesis-driven investigation of code improvements with scientific rigor. It provides:

- **Hypothesis formulation** - User-specified or auto-generated hypotheses
- **Baseline metrics collection** - Capture codebase state before changes
- **Systematic investigation** - Make changes with clear research questions
- **After metrics collection** - Measure impact of changes
- **Statistical analysis** - Calculate percentage changes and significance
- **Report generation** - Comprehensive reports with graphs and visualizations
- **Session integration** - Track research sessions with SessionManager

## Quick Start

### Basic Research Session

```python
from repo_auto_tool import ResearchMode

# Initialize research mode
research = ResearchMode(
    question="Does adding type hints reduce type errors?",
    hypothesis="Type hints will reduce type errors by more than 50%",
    repo_path="/path/to/repo"
)

# Collect baseline metrics
baseline = research.collect_baseline()
# Output: Baseline metrics collected
#   - 5,234 lines of code
#   - 142 functions
#   - 28 classes
#   - 45 lint errors
#   - 23 type errors

# Make your improvements here
# ... add type hints to functions ...

# Collect after metrics
after = research.collect_after()

# Generate report with graphs
report_path = research.generate_report(
    baseline=baseline,
    after=after,
    changes_made="Added type hints to 45 functions across 12 files",
    output_format="markdown"  # or "text"
)

print(f"Report generated: {report_path}")
```

### Integrated with SessionManager

```python
from repo_auto_tool import SessionManager, ResearchMode

# Create a research session
manager = SessionManager()
session = manager.create_session(
    name="type-hints-research",
    repo_path="/path/to/repo",
    goal="Add comprehensive type hints",
    is_research=True,
    research_question="Does adding type hints reduce type errors?",
    hypothesis="Type hints will reduce type errors by >50%",
    tags=["research", "types", "quality"]
)

# Initialize ResearchMode with session
research = ResearchMode(
    question=session.research_question,
    hypothesis=session.hypothesis,
    repo_path=session.repo_path,
    output_dir=manager.get_research_dir(session.session_id),
    session_id=session.session_id
)

# Research workflow
baseline = research.collect_baseline()
# ... make improvements ...
after = research.collect_after()
report = research.generate_report(baseline, after, "Added type hints")

# Findings are automatically saved to the session
# Update session status
manager.update_session_status(
    session.session_id,
    status="completed",
    iterations=5,
    total_cost=2.45,
    success_rate=0.92
)

# View session info
info = manager.get_session_info("type-hints-research")
print(f"Research report: {info['has_report']}")
```

## Research Questions and Hypotheses

### Auto-Generated Hypotheses

If you don't provide a hypothesis, ResearchMode will generate one based on your question:

```python
research = ResearchMode(
    question="Will async I/O improve performance?",
    repo_path="/path/to/repo"
)
# Auto-generated hypothesis: "The changes will improve performance by at least 30%"
```

**Heuristics for auto-generation:**

- Questions with "improve/better" → "measurable improvement"
- Questions with "reduce" → "reduce metric by ≥20%"
- Questions with "increase/add" → "increase metric by ≥20%"
- Questions with "type hint" → "reduce type errors by >50%"
- Questions with "async/performance" → "improve performance by ≥30%"

### Custom Hypotheses

For precise research, specify your own hypothesis:

```python
research = ResearchMode(
    question="How much do docstrings improve code maintainability?",
    hypothesis="Adding comprehensive docstrings will reduce onboarding time by 40% and code review time by 25%",
    repo_path="/path/to/repo"
)
```

## Metrics Collected

ResearchMode automatically collects the following metrics:

### Code Structure Metrics

- **Lines of Code (LOC)** - Total Python lines (excluding venv, build dirs)
- **Functions** - Total function count via AST parsing
- **Classes** - Total class count via AST parsing

### Quality Metrics

- **Error Count** - Lint errors from `ruff check`
- **Type Errors** - Type checking errors from `mypy`

### Future Metrics (Planned)

- **Cyclomatic Complexity** - Average complexity score
- **Test Coverage** - Percentage from pytest-cov
- **Performance Metrics** - Execution time, memory usage

## Report Formats

### Markdown Reports

Markdown reports include:

- Research question and hypothesis
- Methodology section
- Before/after metric tables
- Statistical analysis with percentage changes
- Embedded visualizations (charts)
- Conclusions with confidence levels
- Recommendations

Example output structure:

```
# Research Report

Generated: 2026-01-04 15:30:22

## Research Question
Does adding type hints reduce type errors?

## Hypothesis
Type hints will reduce type errors by more than 50%

## Baseline Metrics
| Metric | Value |
|--------|-------|
| Lines Of Code | 5234 |
| Functions | 142 |
| Type Errors | 23 |
...

## Analysis
| Metric | Change | Percentage |
|--------|--------|------------|
| Type Errors | -15 | -65.2% |
...

## Visualizations
![before_after](charts/before_after.png)
![dashboard](charts/dashboard.png)

## Conclusions
- ✅ Hypothesis CONFIRMED with high confidence
- Type errors reduced by 65.2%
- Codebase grew by 234 lines (type annotations)

## Recommendations
- ✅ Changes are beneficial and should be kept
- Consider applying similar changes to other modules
```

### Text Reports

Simplified text format for terminals:

```
================================================================================
                            RESEARCH REPORT
================================================================================

Generated: 2026-01-04 15:30:22

Research Question: Does adding type hints reduce type errors?
Hypothesis: Type hints will reduce type errors by more than 50%

CONCLUSIONS:
  ✅ Hypothesis CONFIRMED with high confidence
  - Type errors reduced by 65.2%

RECOMMENDATIONS:
  ✅ Changes are beneficial and should be kept
  Consider applying similar changes to other modules

Result: CONFIRMED
Confidence: High

================================================================================
```

## Visualizations

ResearchMode generates multiple visualization types using matplotlib (with text fallback):

### Before/After Comparison Chart

Bar chart comparing metrics before and after changes:
- Error counts
- Type errors
- Functions/classes

### Multi-Metric Dashboard

Grid of subplots showing:
- Error reduction
- Type error trends
- Code structure changes

**Fallback:** When matplotlib is unavailable, text-based ASCII charts are generated.

## Confidence Levels

Hypotheses are confirmed with three confidence levels:

- **High Confidence** - 2+ significant improvements (>20% change)
- **Medium Confidence** - 1 significant improvement
- **Low Confidence** - No significant improvements (hypothesis not confirmed)

Example:

```python
# High confidence
analysis = {
    "significant_improvements": [
        "Errors reduced by 45.3%",
        "Type errors reduced by 65.2%"
    ],
    "confidence": "High",
    "hypothesis_confirmed": True
}

# Low confidence
analysis = {
    "significant_improvements": [],
    "confidence": "Low",
    "hypothesis_confirmed": False
}
```

## File Structure

Research outputs are organized:

```
.research/                          # Default output directory
├── baseline_metrics.json          # Baseline measurements
├── after_metrics.json             # After measurements
├── research_report.md             # Markdown report
├── research_report.txt            # Text report (if generated)
└── charts/                        # Visualizations
    ├── before_after.png
    └── dashboard.png
```

With SessionManager integration:

```
~/.repo-improver/sessions/
└── type-hints_20260104_153022/    # Session directory
    ├── metadata.json              # Session metadata
    ├── state.json                 # Improvement state
    ├── prompts.json               # Learned prompts
    └── research/                  # Research outputs
        ├── baseline_metrics.json
        ├── after_metrics.json
        ├── findings.json          # ResearchFindings
        ├── research_report.md
        └── charts/
            ├── before_after.png
            └── dashboard.png
```

## Advanced Usage

### Custom Output Directory

```python
research = ResearchMode(
    question="Does caching improve performance?",
    repo_path="/path/to/repo",
    output_dir="/path/to/custom/output"
)
```

### Programmatic Analysis

```python
# Collect metrics
baseline = research.collect_baseline()
after = research.collect_after()

# Manual analysis
analysis = research.analyze_results(baseline, after)

print(f"Errors reduced: {analysis['absolute_changes']['error_count']}")
print(f"Percentage change: {analysis['percentage_changes']['error_count']:.1f}%")
print(f"Hypothesis confirmed: {analysis['hypothesis_confirmed']}")
print(f"Confidence: {analysis['confidence']}")

# Access improvements
for improvement in analysis['significant_improvements']:
    print(f"- {improvement}")
```

### Comparing Multiple Research Sessions

```python
manager = SessionManager()

# Run multiple experiments
sessions = [
    "type-hints-approach-1",
    "type-hints-approach-2",
    "type-hints-approach-3"
]

# Compare results
comparison = manager.compare_sessions(sessions)

print(comparison.summary)
print(f"Best approach: {comparison.winner}")

for session_name, metrics in comparison.metrics.items():
    print(f"{session_name}:")
    print(f"  Cost: ${metrics['cost']:.2f}")
    print(f"  Success rate: {metrics['success_rate']:.0%}")
```

## Integration with RepoImprover

Use research mode as part of the improvement loop:

```python
from repo_auto_tool import RepoImprover, ResearchMode, SessionManager

# Create research session
manager = SessionManager()
session = manager.create_session(
    name="error-reduction-research",
    repo_path="/path/to/repo",
    goal="Reduce error count to zero",
    is_research=True,
    research_question="Can we eliminate all errors?",
    hypothesis="Systematic fixes will reduce errors by 100%"
)

# Initialize research
research = ResearchMode(
    question=session.research_question,
    hypothesis=session.hypothesis,
    repo_path=session.repo_path,
    output_dir=manager.get_research_dir(session.session_id),
    session_id=session.session_id
)

# Baseline
baseline = research.collect_baseline()

# Run improvement loop
config = ImproverConfig(
    repo_path=session.repo_path,
    goal="Fix all linting and type errors",
    max_iterations=20
)

improver = RepoImprover(config)
result = improver.run()

# Collect results
after = research.collect_after()
report = research.generate_report(
    baseline=baseline,
    after=after,
    changes_made=f"Fixed {result.iterations} errors across {result.files_changed} files"
)

print(f"Research complete: {report}")
```

## Best Practices

### 1. Clear Research Questions

**Good:**
- "Does adding type hints reduce type errors by >50%?"
- "Will async I/O reduce API response time by >30%?"
- "Can we reduce cyclomatic complexity to <10 average?"

**Avoid:**
- "Make code better" (too vague)
- "Fix stuff" (no measurable hypothesis)

### 2. Measurable Hypotheses

**Good:**
- "Errors will be reduced by at least 20%"
- "Type coverage will increase from 45% to >80%"

**Avoid:**
- "Code will be cleaner" (not measurable)
- "Performance might improve" (no specific target)

### 3. Controlled Changes

- Make one type of change per research session
- Document exactly what was changed
- Use git branches for different approaches
- Compare multiple strategies with SessionManager

### 4. Baseline Before Changes

Always collect baseline BEFORE making any changes:

```python
# ✅ Correct
baseline = research.collect_baseline()
# ... make changes ...
after = research.collect_after()

# ❌ Wrong - baseline after changes
# ... make changes ...
baseline = research.collect_baseline()  # Too late!
```

### 5. Meaningful Tags

Use tags to organize research sessions:

```python
session = manager.create_session(
    name="async-io-experiment-v3",
    repo_path="/path/to/repo",
    goal="Add async I/O",
    tags=["research", "performance", "async", "v3", "high-priority"]
)

# Later, find all performance research
perf_sessions = manager.list_sessions(tags=["performance"])
```

## Troubleshooting

### "ruff not found" or "mypy not found"

Install the tools:

```bash
pip install ruff mypy
```

Or use virtual environment:

```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

pip install ruff mypy
```

### Matplotlib not available

ResearchMode will automatically fall back to text-based charts. To enable graphs:

```bash
pip install matplotlib
```

### No significant changes detected

If your hypothesis is not confirmed:

1. Check if changes were actually made (`git diff`)
2. Verify metric collection is working (check baseline vs after)
3. Lower the significance threshold (default: 20%)
4. Try a different approach and compare sessions

### Session not found

```python
# List all sessions
sessions = manager.list_sessions()
for s in sessions:
    print(f"{s.name}: {s.status}")

# Find by tags
research_sessions = manager.list_sessions(tags=["research"])
```

## CLI Integration (Future)

Planned CLI commands for research mode:

```bash
# Create research session
repo-improver --research \
  --question "Does adding type hints reduce errors?" \
  --hypothesis "Type errors will reduce by >50%" \
  --session "type-hints-exp-1"

# Resume research session
repo-improver --resume type-hints-exp-1

# List research sessions
repo-improver --list-sessions --tags research

# Compare sessions
repo-improver --compare type-hints-exp-1 type-hints-exp-2

# View research report
repo-improver --show-report type-hints-exp-1
```

## Example Workflows

See [RESEARCH_EXAMPLES.md](RESEARCH_EXAMPLES.md) for complete example workflows including:

- Type hints research
- Performance optimization research
- Error reduction research
- Code complexity research
- Multi-strategy comparison

## API Reference

### ResearchMode

```python
class ResearchMode:
    def __init__(
        question: str,
        repo_path: Path | str,
        hypothesis: str | None = None,
        output_dir: Path | str | None = None,
        session_id: str | None = None,
    )

    def collect_baseline() -> CodeMetrics
    def collect_after() -> CodeMetrics
    def analyze_results(baseline: CodeMetrics, after: CodeMetrics) -> dict
    def generate_report(
        baseline: CodeMetrics,
        after: CodeMetrics,
        changes_made: str,
        output_format: str = "markdown"
    ) -> Path
```

### SessionManager

```python
class SessionManager:
    def create_session(
        name: str,
        repo_path: Path | str,
        goal: str,
        tags: list[str] | None = None,
        is_research: bool = False,
        research_question: str | None = None,
        hypothesis: str | None = None,
    ) -> SessionMetadata

    def get_research_dir(session_id: str) -> Path
    def save_research_findings(session_id: str, findings: dict) -> None
```

### CodeMetrics

```python
@dataclass
class CodeMetrics:
    lines_of_code: int = 0
    functions: int = 0
    classes: int = 0
    complexity: float = 0.0
    test_coverage: float = 0.0
    error_count: int = 0
    type_errors: int = 0
```

### ResearchFindings

```python
@dataclass
class ResearchFindings:
    hypothesis: str
    baseline_metrics: CodeMetrics
    after_metrics: CodeMetrics
    changes_made: str
    hypothesis_confirmed: bool
    confidence: str  # Low/Medium/High
    analysis: dict
    conclusions: list[str]
    recommendations: list[str]
```

## See Also

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) - Full feature roadmap
- [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) - Phase 1 implementation details
- [RESEARCH_EXAMPLES.md](RESEARCH_EXAMPLES.md) - Example workflows
- [SessionManager docs](src/repo_auto_tool/session_manager.py)
- [ResearchMode docs](src/repo_auto_tool/research_mode.py)
- [DataVisualizer docs](src/repo_auto_tool/data_visualizer.py)
