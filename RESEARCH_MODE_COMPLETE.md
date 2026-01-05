# Research Mode Implementation - Complete

## Summary

Research Mode with comprehensive data visualization has been successfully implemented for repo-improver. This provides a complete framework for hypothesis-driven investigation of code improvements with scientific rigor.

**Status:** ✅ **COMPLETE**

**Date Completed:** 2026-01-04

---

## What Was Implemented

### 1. Core Research Framework

**File:** `src/repo_auto_tool/research_mode.py` (700 lines)

- **ResearchMode class** - Orchestrates research investigations
  - Hypothesis generation (auto or user-specified)
  - Baseline metric collection
  - After-change metric collection
  - Statistical analysis with percentage changes
  - Report generation (Markdown and text formats)
  - Session integration

- **MetricsCollector class** - Collects code metrics
  - Lines of code counting
  - Function and class counting (AST-based)
  - Lint error counting (via ruff)
  - Type error counting (via mypy)

- **CodeMetrics dataclass** - Structured metric storage
  - Lines of code, functions, classes
  - Error counts, type errors
  - Extensible for future metrics

- **ResearchFindings dataclass** - Analysis results
  - Hypothesis confirmation
  - Confidence levels (Low/Medium/High)
  - Conclusions and recommendations

### 2. Data Visualization System

**File:** `src/repo_auto_tool/data_visualizer.py` (500 lines)

- **DataVisualizer class** - Complete visualization toolkit
  - Automatic matplotlib detection with text fallback
  - Before/after comparison bar charts
  - Trend line charts for time-series data
  - Success rate visualization
  - Cost breakdown pie charts
  - Multi-metric dashboards with subplots
  - Customizable colors, labels, and formatting

### 3. Session Manager Integration

**File:** `src/repo_auto_tool/session_manager.py` (enhanced)

Enhanced SessionMetadata with research fields:
- `is_research: bool` - Flag for research sessions
- `research_question: str | None` - The research question
- `hypothesis: str | None` - The hypothesis being tested

New methods:
- `get_research_dir(session_id)` - Get research output directory
- `save_research_findings(session_id, findings)` - Store findings
- `create_session()` - Now accepts research parameters

Research outputs are automatically saved to:
```
~/.repo-improver/sessions/<session_id>/research/
├── baseline_metrics.json
├── after_metrics.json
├── findings.json
├── research_report.md
└── charts/
    ├── before_after.png
    └── dashboard.png
```

### 4. Complete Documentation

#### RESEARCH_MODE_GUIDE.md (comprehensive guide)
- Overview and quick start
- Research questions and hypotheses
- Metrics collected
- Report formats (Markdown and text)
- Visualizations
- Confidence levels
- File structure
- Advanced usage
- Best practices
- Troubleshooting
- API reference

#### RESEARCH_EXAMPLES.md (5 complete examples)
1. **Type Hints Research** - Reduce type errors
2. **Performance Optimization** - Async I/O improvements
3. **Error Reduction Study** - Systematic error elimination
4. **Multi-Strategy Comparison** - Compare approaches
5. **Code Complexity Research** - Reduce cyclomatic complexity

Each example includes:
- Complete, runnable code
- Expected output
- Custom metrics where applicable
- Integration with improvement loop

---

## Key Features

### Hypothesis-Driven Investigation

```python
research = ResearchMode(
    question="Does adding type hints reduce type errors?",
    hypothesis="Type hints will reduce type errors by more than 50%",
    repo_path="/path/to/repo"
)

baseline = research.collect_baseline()
# ... make changes ...
after = research.collect_after()
report = research.generate_report(baseline, after, "Added type hints")
```

### Automatic Hypothesis Generation

```python
# Auto-generates hypothesis from question
research = ResearchMode(
    question="Will async I/O improve performance?",
    repo_path="/path/to/repo"
)
# Hypothesis: "The changes will improve performance by at least 30%"
```

### Statistical Analysis

- Absolute changes (before - after)
- Percentage changes
- Significance detection (>20% threshold)
- Confidence levels (High/Medium/Low)
- Hypothesis confirmation

### Comprehensive Reports

**Markdown reports include:**
- Research question and hypothesis
- Methodology
- Before/after metric tables
- Statistical analysis
- Embedded visualizations
- Conclusions with confidence levels
- Recommendations

**Text reports** for terminal viewing with ASCII formatting

### Data Visualizations

**Before/After Charts:**
- Bar charts comparing metrics
- Clear visual comparison
- Customizable colors

**Trend Charts:**
- Line charts for time-series
- Multiple metrics on one chart
- Legend and grid

**Dashboards:**
- Multi-subplot layouts
- Comprehensive overview
- Professional appearance

**Text Fallback:**
- ASCII bar charts when matplotlib unavailable
- Terminal-friendly output
- No dependency failures

### Session Integration

```python
manager = SessionManager()

# Create research session
session = manager.create_session(
    name="type-hints-research",
    repo_path="/path/to/repo",
    goal="Add type hints",
    is_research=True,
    research_question="Does adding type hints reduce errors?",
    hypothesis="Type errors will reduce by >50%",
    tags=["research", "types"]
)

# Link ResearchMode to session
research = ResearchMode(
    question=session.research_question,
    hypothesis=session.hypothesis,
    repo_path=session.repo_path,
    output_dir=manager.get_research_dir(session.session_id),
    session_id=session.session_id
)

# Findings automatically saved to session
report = research.generate_report(...)
```

### Multi-Strategy Comparison

```python
# Test different approaches
strategies = ["conservative", "aggressive", "incremental"]

for strategy in strategies:
    session = manager.create_session(name=strategy, ...)
    research = ResearchMode(...)
    # ... run experiment ...

# Compare results
comparison = manager.compare_sessions(strategies)
print(f"Winner: {comparison.winner}")
```

---

## Integration with Existing Features

### Works with RepoImprover

```python
# Collect baseline
baseline = research.collect_baseline()

# Run improvement loop
improver = RepoImprover(config)
result = improver.run()

# Analyze results
after = research.collect_after()
report = research.generate_report(baseline, after, result.summary)
```

### Works with PromptLearner

- Research findings can inform prompt learning
- Successful strategies are documented
- Failed approaches are recorded
- Build knowledge base of effective techniques

### Works with BudgetTracker

- Track research costs
- Compare cost-effectiveness of strategies
- Optimize for budget constraints

### Works with ModelSelector

- Research can inform model selection
- Track which models work best for which tasks
- Build success rate data

---

## Files Modified

### New Files Created
1. `src/repo_auto_tool/research_mode.py` (700 lines)
2. `src/repo_auto_tool/data_visualizer.py` (500 lines)
3. `RESEARCH_MODE_GUIDE.md` (comprehensive documentation)
4. `RESEARCH_EXAMPLES.md` (5 complete examples)
5. `RESEARCH_MODE_COMPLETE.md` (this file)

### Files Enhanced
1. `src/repo_auto_tool/session_manager.py`
   - Added research fields to SessionMetadata
   - Added `get_research_dir()` method
   - Added `save_research_findings()` method
   - Updated `create_session()` with research parameters

2. `src/repo_auto_tool/__init__.py`
   - Added exports for ResearchMode
   - Added exports for DataVisualizer
   - Added exports for research-related dataclasses

---

## Usage Examples

### Basic Research

```python
from repo_auto_tool import ResearchMode

research = ResearchMode(
    question="Does adding docstrings improve maintainability?",
    repo_path="."
)

baseline = research.collect_baseline()
# Add docstrings...
after = research.collect_after()
report = research.generate_report(baseline, after, "Added comprehensive docstrings")
```

### With Session Management

```python
from repo_auto_tool import SessionManager, ResearchMode

manager = SessionManager()
session = manager.create_session(
    name="docstring-research",
    repo_path=".",
    goal="Add docstrings",
    is_research=True,
    research_question="Does adding docstrings improve maintainability?"
)

research = ResearchMode(
    question=session.research_question,
    repo_path=session.repo_path,
    session_id=session.session_id
)

# Research workflow...
```

### Comparing Strategies

```python
strategies = ["approach-a", "approach-b", "approach-c"]
results = []

for strategy in strategies:
    session = manager.create_session(name=strategy, is_research=True, ...)
    research = ResearchMode(session_id=session.session_id, ...)
    baseline = research.collect_baseline()
    # Apply strategy...
    after = research.collect_after()
    results.append(research.generate_report(...))

# Compare
comparison = manager.compare_sessions(strategies)
print(comparison.summary)
```

---

## Testing

All research mode components are ready for testing:

1. **Unit tests** can be added to:
   - `tests/test_research_mode.py`
   - `tests/test_data_visualizer.py`
   - `tests/test_session_manager.py` (update existing)

2. **Integration tests** for:
   - ResearchMode + SessionManager
   - ResearchMode + RepoImprover
   - Multi-strategy comparison workflows

3. **Example workflows** in RESEARCH_EXAMPLES.md can serve as:
   - Integration test cases
   - Documentation examples
   - User tutorials

---

## Future Enhancements

### CLI Integration (Next Phase)

```bash
# Create research session
repo-improver --research \
  --question "Does adding type hints reduce errors?" \
  --hypothesis "Type errors will reduce by >50%" \
  --session "type-hints-exp"

# Resume research session
repo-improver --resume type-hints-exp

# Compare sessions
repo-improver --compare exp-1 exp-2 exp-3

# View report
repo-improver --show-report type-hints-exp
```

### Additional Metrics

- Cyclomatic complexity (via radon)
- Test coverage (via pytest-cov)
- Performance metrics (execution time, memory)
- Code duplication (via pylint)
- Security vulnerabilities (via bandit)

### Advanced Analysis

- Multi-variate analysis
- Correlation detection
- Trend prediction
- Cost-benefit analysis
- Statistical significance testing (t-tests, p-values)

### Visualization Enhancements

- Interactive charts (plotly)
- HTML reports with embedded charts
- Time-series animations
- Heatmaps for complexity
- Dependency graphs

---

## Documentation Links

- **User Guide:** [RESEARCH_MODE_GUIDE.md](RESEARCH_MODE_GUIDE.md)
- **Examples:** [RESEARCH_EXAMPLES.md](RESEARCH_EXAMPLES.md)
- **Implementation Plan:** [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md)
- **Phase 1 Complete:** [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md)

---

## Success Criteria

All success criteria have been met:

✅ **Hypothesis-driven investigation** - Full support for user or auto-generated hypotheses
✅ **Baseline collection** - Comprehensive metric collection before changes
✅ **After collection** - Metric collection after changes
✅ **Statistical analysis** - Percentage changes, significance detection, confidence levels
✅ **Report generation** - Markdown and text formats with embedded visualizations
✅ **Data visualization** - Multiple chart types with matplotlib and text fallback
✅ **Session integration** - Full integration with SessionManager
✅ **Documentation** - Comprehensive guide and 5 complete examples
✅ **Extensibility** - Easy to add custom metrics and visualizations

---

## Conclusion

Research Mode is now fully implemented and ready for use. Users can:

1. Create research sessions with clear hypotheses
2. Collect baseline and after metrics automatically
3. Get statistical analysis with confidence levels
4. Generate professional reports with visualizations
5. Compare multiple strategies systematically
6. Track all research through SessionManager
7. Build a knowledge base of effective improvements

The implementation provides a solid foundation for scientific, data-driven code improvements.

**Next steps:** Continue with additional features from IMPROVEMENT_PLAN.md or integrate research mode into the main CLI workflow.
