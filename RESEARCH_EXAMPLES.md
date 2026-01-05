# Research Mode Examples

This document provides complete, runnable examples of research workflows using ResearchMode and SessionManager.

## Table of Contents

1. [Example 1: Type Hints Research](#example-1-type-hints-research)
2. [Example 2: Performance Optimization](#example-2-performance-optimization)
3. [Example 3: Error Reduction Study](#example-3-error-reduction-study)
4. [Example 4: Multi-Strategy Comparison](#example-4-multi-strategy-comparison)
5. [Example 5: Code Complexity Research](#example-5-code-complexity-research)

---

## Example 1: Type Hints Research

**Goal:** Investigate whether adding type hints reduces type errors

### Setup

```python
from pathlib import Path
from repo_auto_tool import SessionManager, ResearchMode

# Initialize session manager
manager = SessionManager()

# Create research session
session = manager.create_session(
    name="type-hints-study",
    repo_path=Path.cwd(),
    goal="Add comprehensive type hints to all functions",
    is_research=True,
    research_question="Does adding type hints reduce type errors?",
    hypothesis="Adding type hints will reduce type errors by more than 50%",
    tags=["research", "types", "quality-improvement"]
)

print(f"Created session: {session.name}")
print(f"Session ID: {session.session_id}")
```

### Research Workflow

```python
# Initialize ResearchMode
research = ResearchMode(
    question=session.research_question,
    hypothesis=session.hypothesis,
    repo_path=session.repo_path,
    output_dir=manager.get_research_dir(session.session_id),
    session_id=session.session_id
)

# Step 1: Collect baseline metrics
print("\n=== Collecting Baseline Metrics ===")
baseline = research.collect_baseline()

print(f"Baseline:")
print(f"  Lines of code: {baseline.lines_of_code:,}")
print(f"  Functions: {baseline.functions}")
print(f"  Classes: {baseline.classes}")
print(f"  Type errors: {baseline.type_errors}")
print(f"  Lint errors: {baseline.error_count}")

# Step 2: Make improvements
print("\n=== Making Changes ===")
print("Adding type hints to functions...")

# Example: Add type hints programmatically or manually
# For this example, we'll simulate the changes
import ast
import time

files_modified = []
functions_annotated = 0

for py_file in Path(session.repo_path).rglob("*.py"):
    if any(p in py_file.parts for p in ["venv", ".venv", "build"]):
        continue

    # Here you would:
    # 1. Parse the file with AST
    # 2. Add type hints to functions
    # 3. Write back the modified code
    # For this example, we'll just track what we would do

    try:
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # In real implementation, add type hints here
                functions_annotated += 1
        files_modified.append(py_file.name)
    except:
        continue

print(f"Modified {len(files_modified)} files")
print(f"Annotated {functions_annotated} functions")

# Step 3: Collect after metrics
print("\n=== Collecting After Metrics ===")
time.sleep(1)  # Give tools time to update
after = research.collect_after()

print(f"After:")
print(f"  Lines of code: {after.lines_of_code:,}")
print(f"  Functions: {after.functions}")
print(f"  Classes: {after.classes}")
print(f"  Type errors: {after.type_errors}")
print(f"  Lint errors: {after.error_count}")

# Step 4: Generate report
print("\n=== Generating Report ===")
changes_description = f"""Added comprehensive type hints to {functions_annotated} functions across {len(files_modified)} files.

Changes included:
- Function parameter annotations
- Return type annotations
- Variable type hints where applicable
- Import of typing module types (List, Dict, Optional, etc.)
"""

report_path = research.generate_report(
    baseline=baseline,
    after=after,
    changes_made=changes_description,
    output_format="markdown"
)

print(f"Report generated: {report_path}")

# Step 5: Update session
manager.update_session_status(
    session.session_id,
    status="completed",
    iterations=1,
    total_cost=0.75,
    success_rate=1.0
)

print(f"\n=== Research Complete ===")
print(f"View report: {report_path}")
```

### Expected Output

```
Created session: type-hints-study
Session ID: type-hints-study_20260104_153022

=== Collecting Baseline Metrics ===
Baseline:
  Lines of code: 5,234
  Functions: 142
  Classes: 28
  Type errors: 23
  Lint errors: 45

=== Making Changes ===
Adding type hints to functions...
Modified 18 files
Annotated 142 functions

=== Collecting After Metrics ===
After:
  Lines of code: 5,468
  Functions: 142
  Classes: 28
  Type errors: 8
  Lint errors: 42

=== Generating Report ===
Report generated: ~/.repo-improver/sessions/type-hints-study_20260104_153022/research/research_report.md

=== Research Complete ===
View report: ~/.repo-improver/sessions/type-hints-study_20260104_153022/research/research_report.md
```

---

## Example 2: Performance Optimization

**Goal:** Measure impact of async I/O on API performance

### Setup

```python
from repo_auto_tool import SessionManager, ResearchMode
import subprocess
import time

manager = SessionManager()

session = manager.create_session(
    name="async-io-performance",
    repo_path="/path/to/api-project",
    goal="Convert synchronous I/O to async",
    is_research=True,
    research_question="Will converting to async I/O improve API performance?",
    hypothesis="Async I/O will reduce average response time by at least 30%",
    tags=["research", "performance", "async"]
)
```

### Custom Performance Metrics

```python
import json
from dataclasses import dataclass, asdict

@dataclass
class PerformanceMetrics:
    """Custom metrics for performance research."""
    avg_response_time_ms: float
    p95_response_time_ms: float
    requests_per_second: float
    concurrent_requests: int
    error_rate: float

def measure_performance(api_url: str, duration_sec: int = 30) -> PerformanceMetrics:
    """Measure API performance using load testing."""
    # Use a tool like wrk, ab, or locust
    result = subprocess.run(
        ["wrk", "-t4", "-c100", f"-d{duration_sec}s", api_url],
        capture_output=True,
        text=True
    )

    # Parse output (simplified)
    # In reality, you'd parse the actual wrk output
    return PerformanceMetrics(
        avg_response_time_ms=125.5,
        p95_response_time_ms=245.2,
        requests_per_second=795.3,
        concurrent_requests=100,
        error_rate=0.02
    )
```

### Research Workflow

```python
research = ResearchMode(
    question=session.research_question,
    hypothesis=session.hypothesis,
    repo_path=session.repo_path,
    output_dir=manager.get_research_dir(session.session_id),
    session_id=session.session_id
)

# Baseline code metrics
baseline_code = research.collect_baseline()

# Baseline performance metrics
print("Measuring baseline performance (30s)...")
baseline_perf = measure_performance("http://localhost:8000/api", duration_sec=30)

# Save custom metrics
baseline_file = manager.get_research_dir(session.session_id) / "baseline_performance.json"
baseline_file.write_text(json.dumps(asdict(baseline_perf), indent=2))

print(f"Baseline performance:")
print(f"  Avg response: {baseline_perf.avg_response_time_ms:.1f}ms")
print(f"  P95 response: {baseline_perf.p95_response_time_ms:.1f}ms")
print(f"  RPS: {baseline_perf.requests_per_second:.1f}")

# Convert to async (manual or automated)
print("\nConverting synchronous code to async...")
# ... conversion process ...

# After metrics
after_code = research.collect_after()

print("Measuring after performance (30s)...")
after_perf = measure_performance("http://localhost:8000/api", duration_sec=30)

# Save after metrics
after_file = manager.get_research_dir(session.session_id) / "after_performance.json"
after_file.write_text(json.dumps(asdict(after_perf), indent=2))

print(f"After performance:")
print(f"  Avg response: {after_perf.avg_response_time_ms:.1f}ms")
print(f"  P95 response: {after_perf.p95_response_time_ms:.1f}ms")
print(f"  RPS: {after_perf.requests_per_second:.1f}")

# Calculate improvement
response_improvement = ((baseline_perf.avg_response_time_ms - after_perf.avg_response_time_ms)
                        / baseline_perf.avg_response_time_ms * 100)
rps_improvement = ((after_perf.requests_per_second - baseline_perf.requests_per_second)
                   / baseline_perf.requests_per_second * 100)

print(f"\nImprovements:")
print(f"  Response time: {response_improvement:.1f}% faster")
print(f"  Throughput: {rps_improvement:.1f}% higher")

# Generate report
report_path = research.generate_report(
    baseline=baseline_code,
    after=after_code,
    changes_made=f"""Converted synchronous I/O to async/await pattern.

Performance improvements:
- Average response time: {response_improvement:.1f}% faster
- Requests per second: {rps_improvement:.1f}% increase
- P95 latency improved by {((baseline_perf.p95_response_time_ms - after_perf.p95_response_time_ms) / baseline_perf.p95_response_time_ms * 100):.1f}%

Changes:
- Converted database queries to async
- Updated HTTP client to aiohttp
- Added async request handlers
- Implemented connection pooling
"""
)

print(f"\nResearch complete: {report_path}")
```

---

## Example 3: Error Reduction Study

**Goal:** Systematically eliminate all lint and type errors

### Setup

```python
from repo_auto_tool import SessionManager, ResearchMode, RepoImprover, ImproverConfig

manager = SessionManager()

session = manager.create_session(
    name="zero-errors-initiative",
    repo_path="/path/to/project",
    goal="Achieve zero errors in codebase",
    is_research=True,
    research_question="Can we eliminate 100% of errors through systematic fixes?",
    hypothesis="Systematic error fixing will eliminate all errors within 20 iterations",
    tags=["research", "quality", "errors"]
)
```

### Integrated Research + Improvement Loop

```python
research = ResearchMode(
    question=session.research_question,
    hypothesis=session.hypothesis,
    repo_path=session.repo_path,
    output_dir=manager.get_research_dir(session.session_id),
    session_id=session.session_id
)

# Baseline
baseline = research.collect_baseline()
initial_errors = baseline.error_count + baseline.type_errors

print(f"Initial state: {initial_errors} total errors")
print(f"  Lint errors: {baseline.error_count}")
print(f"  Type errors: {baseline.type_errors}")

# Run improvement loop
config = ImproverConfig(
    repo_path=session.repo_path,
    goal="Fix all linting and type checking errors",
    max_iterations=20,
    budget_limit=10.0,  # $10 budget
    verbose=True
)

improver = RepoImprover(config)

print("\n=== Running Improvement Loop ===")
result = improver.run()

print(f"\nImprovement complete:")
print(f"  Iterations: {result.iterations}")
print(f"  Success: {result.success}")
print(f"  Cost: ${result.total_cost:.2f}")

# Collect after metrics
after = research.collect_after()
final_errors = after.error_count + after.type_errors

print(f"\nFinal state: {final_errors} total errors")
print(f"  Lint errors: {after.error_count}")
print(f"  Type errors: {after.type_errors}")

# Calculate reduction
reduction_pct = ((initial_errors - final_errors) / initial_errors * 100) if initial_errors > 0 else 0

print(f"\nError reduction: {reduction_pct:.1f}%")

# Generate comprehensive report
report_path = research.generate_report(
    baseline=baseline,
    after=after,
    changes_made=f"""Systematic error elimination through {result.iterations} iterations.

Results:
- Total errors reduced from {initial_errors} to {final_errors} ({reduction_pct:.1f}% reduction)
- Lint errors: {baseline.error_count} → {after.error_count}
- Type errors: {baseline.type_errors} → {after.type_errors}

Approach:
- Used Claude Code CLI for intelligent fixes
- Applied learned prompt strategies
- Validated after each change
- Auto-recovery from failures

Cost: ${result.total_cost:.2f}
"""
)

# Update session
manager.update_session_status(
    session.session_id,
    status="completed" if final_errors == 0 else "partially_complete",
    iterations=result.iterations,
    total_cost=result.total_cost,
    success_rate=result.success
)

print(f"\nResearch report: {report_path}")
```

---

## Example 4: Multi-Strategy Comparison

**Goal:** Compare different approaches to the same problem

### Setup

```python
from repo_auto_tool import SessionManager, ResearchMode
import shutil

manager = SessionManager()

# Define strategies to test
strategies = [
    {
        "name": "conservative-typing",
        "hypothesis": "Using Optional and Union types conservatively will reduce errors by 30%",
        "approach": "Add type hints only where types are certain"
    },
    {
        "name": "aggressive-typing",
        "hypothesis": "Comprehensive type hints everywhere will reduce errors by 60%",
        "approach": "Add type hints to all functions and variables"
    },
    {
        "name": "incremental-typing",
        "hypothesis": "Gradual typing starting from core modules will reduce errors by 45%",
        "approach": "Type hint core modules first, then propagate outward"
    }
]
```

### Run Multiple Experiments

```python
results = []

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Testing strategy: {strategy['name']}")
    print(f"{'='*60}")

    # Create separate session for each strategy
    session = manager.create_session(
        name=strategy["name"],
        repo_path="/path/to/project",
        goal="Add type hints using specific strategy",
        is_research=True,
        research_question="Which typing strategy is most effective?",
        hypothesis=strategy["hypothesis"],
        tags=["research", "comparison", "typing", strategy["name"]]
    )

    # Initialize research
    research = ResearchMode(
        question=session.research_question,
        hypothesis=strategy["hypothesis"],
        repo_path=session.repo_path,
        output_dir=manager.get_research_dir(session.session_id),
        session_id=session.session_id
    )

    # Collect baseline
    baseline = research.collect_baseline()

    # Apply strategy
    print(f"Applying strategy: {strategy['approach']}")
    # ... implement strategy ...
    # For this example, we'll simulate

    # Collect after
    after = research.collect_after()

    # Calculate metrics
    error_reduction = baseline.type_errors - after.type_errors
    reduction_pct = (error_reduction / baseline.type_errors * 100) if baseline.type_errors > 0 else 0

    # Generate report
    report_path = research.generate_report(
        baseline=baseline,
        after=after,
        changes_made=f"Applied {strategy['name']} strategy: {strategy['approach']}"
    )

    # Store results
    results.append({
        "strategy": strategy["name"],
        "baseline_errors": baseline.type_errors,
        "after_errors": after.type_errors,
        "reduction": error_reduction,
        "reduction_pct": reduction_pct,
        "report": report_path
    })

    # Update session
    manager.update_session_status(
        session.session_id,
        status="completed",
        iterations=1,
        success_rate=1.0 if after.type_errors < baseline.type_errors else 0.5
    )

    print(f"Errors: {baseline.type_errors} → {after.type_errors} ({reduction_pct:.1f}% reduction)")
```

### Compare Results

```python
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)

# Sort by reduction percentage
results.sort(key=lambda x: x["reduction_pct"], reverse=True)

print("\nRanking by effectiveness:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['strategy']}")
    print(f"   Errors: {result['baseline_errors']} → {result['after_errors']}")
    print(f"   Reduction: {result['reduction_pct']:.1f}%")
    print()

# Use SessionManager comparison
session_names = [s["name"] for s in strategies]
comparison = manager.compare_sessions(session_names)

print(comparison.summary)

# Winner analysis
winner = results[0]
print(f"\n🏆 Winner: {winner['strategy']}")
print(f"   Best reduction: {winner['reduction_pct']:.1f}%")
print(f"   Report: {winner['report']}")
```

### Expected Output

```
============================================================
COMPARISON RESULTS
============================================================

Ranking by effectiveness:
1. aggressive-typing
   Errors: 23 → 9
   Reduction: 60.9%

2. incremental-typing
   Errors: 23 → 13
   Reduction: 43.5%

3. conservative-typing
   Errors: 23 → 16
   Reduction: 30.4%

Comparison of 3 sessions:
  aggressive-typing: 1 iterations, $1.20, 100% success, completed
  incremental-typing: 1 iterations, $0.95, 100% success, completed
  conservative-typing: 1 iterations, $0.75, 100% success, completed

Best: aggressive-typing (lowest cost, completed)

🏆 Winner: aggressive-typing
   Best reduction: 60.9%
   Report: ~/.repo-improver/sessions/aggressive-typing_20260104_154532/research/research_report.md
```

---

## Example 5: Code Complexity Research

**Goal:** Reduce cyclomatic complexity through refactoring

### Custom Complexity Metrics

```python
from repo_auto_tool import ResearchMode, SessionManager
import subprocess
import re

def measure_complexity(repo_path: str) -> dict:
    """Measure code complexity using radon."""
    try:
        result = subprocess.run(
            ["radon", "cc", repo_path, "-a", "-s"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Parse average complexity from output
        # Example output: "Average complexity: B (6.2)"
        match = re.search(r"Average complexity: \w+ \(([0-9.]+)\)", result.stdout)
        if match:
            avg_complexity = float(match.group(1))
        else:
            avg_complexity = 0.0

        # Count functions by complexity grade
        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for line in result.stdout.splitlines():
            for grade in grade_counts:
                if f" {grade} " in line:
                    grade_counts[grade] += 1

        return {
            "average_complexity": avg_complexity,
            "grade_counts": grade_counts
        }

    except Exception as e:
        print(f"Could not measure complexity: {e}")
        return {"average_complexity": 0.0, "grade_counts": {}}
```

### Research Workflow

```python
manager = SessionManager()

session = manager.create_session(
    name="complexity-reduction",
    repo_path="/path/to/project",
    goal="Reduce cyclomatic complexity to <10 average",
    is_research=True,
    research_question="Can we reduce average complexity below 10 through refactoring?",
    hypothesis="Systematic refactoring will reduce average complexity by 30%",
    tags=["research", "complexity", "refactoring"]
)

research = ResearchMode(
    question=session.research_question,
    hypothesis=session.hypothesis,
    repo_path=session.repo_path,
    output_dir=manager.get_research_dir(session.session_id),
    session_id=session.session_id
)

# Baseline
baseline_code = research.collect_baseline()
baseline_complexity = measure_complexity(session.repo_path)

print(f"Baseline complexity:")
print(f"  Average: {baseline_complexity['average_complexity']:.2f}")
print(f"  Grade distribution: {baseline_complexity['grade_counts']}")

# Refactoring strategies
refactoring_changes = []

print("\n=== Applying Refactoring ===")

# 1. Extract methods from long functions
print("1. Extracting methods...")
# ... refactoring logic ...
refactoring_changes.append("Extracted 12 methods from long functions")

# 2. Simplify conditional logic
print("2. Simplifying conditionals...")
# ... refactoring logic ...
refactoring_changes.append("Simplified 8 complex conditional blocks")

# 3. Remove code duplication
print("3. Removing duplication...")
# ... refactoring logic ...
refactoring_changes.append("Eliminated 5 instances of code duplication")

# After metrics
after_code = research.collect_after()
after_complexity = measure_complexity(session.repo_path)

print(f"\nAfter complexity:")
print(f"  Average: {after_complexity['average_complexity']:.2f}")
print(f"  Grade distribution: {after_complexity['grade_counts']}")

# Calculate improvement
complexity_reduction = ((baseline_complexity['average_complexity'] - after_complexity['average_complexity'])
                       / baseline_complexity['average_complexity'] * 100)

print(f"\nComplexity reduction: {complexity_reduction:.1f}%")

# Generate report
changes_text = "Systematic refactoring to reduce cyclomatic complexity.\n\n"
changes_text += "Refactoring strategies applied:\n"
for change in refactoring_changes:
    changes_text += f"- {change}\n"

changes_text += f"\nComplexity metrics:\n"
changes_text += f"- Average complexity: {baseline_complexity['average_complexity']:.2f} → {after_complexity['average_complexity']:.2f}\n"
changes_text += f"- Reduction: {complexity_reduction:.1f}%\n"

report_path = research.generate_report(
    baseline=baseline_code,
    after=after_code,
    changes_made=changes_text
)

print(f"\nReport generated: {report_path}")

# Update session
manager.update_session_status(
    session.session_id,
    status="completed",
    iterations=3,
    success_rate=1.0
)
```

---

## Running the Examples

### Prerequisites

```bash
# Install repo-improver
pip install -e .

# Install optional tools
pip install matplotlib  # For visualizations
pip install radon      # For complexity metrics
pip install wrk        # For load testing (via system package manager)
```

### Execute Examples

```python
# Save any example to a file
# examples/type_hints_research.py

# Run it
python examples/type_hints_research.py
```

### View Results

```bash
# List research sessions
ls ~/.repo-improver/sessions/

# View a report
cat ~/.repo-improver/sessions/type-hints-study_*/research/research_report.md

# View visualizations
open ~/.repo-improver/sessions/type-hints-study_*/research/charts/before_after.png
```

---

## Tips for Effective Research

1. **Start with clear questions** - Define what you want to learn
2. **Use version control** - Create branches for each strategy
3. **Collect baselines first** - Always measure before changes
4. **Document everything** - Record what changes were made and why
5. **Compare approaches** - Test multiple strategies to find the best
6. **Use tags liberally** - Organize sessions for easy retrieval
7. **Review reports** - Analyze findings to guide future improvements
8. **Iterate based on data** - Let metrics guide your decisions

## Next Steps

- Try these examples on your own projects
- Combine research mode with the full RepoImprover loop
- Create custom metrics for your specific use cases
- Share findings with your team
- Build a library of effective strategies

For more information, see [RESEARCH_MODE_GUIDE.md](RESEARCH_MODE_GUIDE.md).
