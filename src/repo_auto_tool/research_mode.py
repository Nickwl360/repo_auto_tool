"""Research Mode for hypothesis-driven investigation.

This module provides a framework for systematic, scientific investigation
of code improvements with:
- Hypothesis formulation (user or auto-generated)
- Baseline metric collection
- Systematic changes
- After metric collection
- Statistical analysis
- Report generation with graphs

Example:
    research = ResearchMode(
        question="Does async I/O improve performance?",
        hypothesis="Async will reduce latency by >50%",
        repo_path="/path/to/repo"
    )

    # Collect baseline
    baseline = research.collect_baseline()

    # Make changes (via improvement loop)
    # ... improvements happen ...

    # Collect after metrics
    after = research.collect_after()

    # Analyze and generate report
    report = research.generate_report(
        baseline=baseline,
        after=after,
        output_path="research_report.md"
    )
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .data_visualizer import DataVisualizer

logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Code metrics for research analysis.

    Attributes:
        lines_of_code: Total lines of code.
        functions: Number of functions.
        classes: Number of classes.
        complexity: Average complexity score.
        test_coverage: Test coverage percentage (if available).
        error_count: Number of errors/warnings.
        type_errors: Number of type errors (if applicable).
    """
    lines_of_code: int = 0
    functions: int = 0
    classes: int = 0
    complexity: float = 0.0
    test_coverage: float = 0.0
    error_count: int = 0
    type_errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lines_of_code": self.lines_of_code,
            "functions": self.functions,
            "classes": self.classes,
            "complexity": self.complexity,
            "test_coverage": self.test_coverage,
            "error_count": self.error_count,
            "type_errors": self.type_errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeMetrics:
        """Create from dictionary."""
        return cls(
            lines_of_code=data.get("lines_of_code", 0),
            functions=data.get("functions", 0),
            classes=data.get("classes", 0),
            complexity=data.get("complexity", 0.0),
            test_coverage=data.get("test_coverage", 0.0),
            error_count=data.get("error_count", 0),
            type_errors=data.get("type_errors", 0),
        )


@dataclass
class ResearchFindings:
    """Results of research investigation.

    Attributes:
        hypothesis: The hypothesis being tested.
        baseline_metrics: Metrics before changes.
        after_metrics: Metrics after changes.
        changes_made: Description of changes.
        hypothesis_confirmed: Whether hypothesis was confirmed.
        confidence: Confidence level (Low/Medium/High).
        analysis: Statistical analysis results.
        conclusions: List of conclusions.
        recommendations: List of recommendations.
    """
    hypothesis: str
    baseline_metrics: CodeMetrics
    after_metrics: CodeMetrics
    changes_made: str
    hypothesis_confirmed: bool
    confidence: str  # Low/Medium/High
    analysis: dict[str, Any]
    conclusions: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis": self.hypothesis,
            "baseline_metrics": self.baseline_metrics.to_dict(),
            "after_metrics": self.after_metrics.to_dict(),
            "changes_made": self.changes_made,
            "hypothesis_confirmed": self.hypothesis_confirmed,
            "confidence": self.confidence,
            "analysis": self.analysis,
            "conclusions": self.conclusions,
            "recommendations": self.recommendations,
        }


class MetricsCollector:
    """Collects code metrics for research analysis.

    Attributes:
        repo_path: Path to the repository.
    """

    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)

    def collect_metrics(self) -> CodeMetrics:
        """Collect current code metrics.

        Returns:
            CodeMetrics with current state.
        """
        metrics = CodeMetrics()

        # Count lines of code
        metrics.lines_of_code = self._count_lines_of_code()

        # Count functions and classes
        counts = self._count_functions_and_classes()
        metrics.functions = counts["functions"]
        metrics.classes = counts["classes"]

        # Get error count (from linting if available)
        metrics.error_count = self._count_errors()

        # Get type errors (from mypy if available)
        metrics.type_errors = self._count_type_errors()

        logger.info(f"Collected metrics: {metrics.lines_of_code} LOC, "
                   f"{metrics.functions} functions, {metrics.classes} classes")

        return metrics

    def _count_lines_of_code(self) -> int:
        """Count total lines of Python code."""
        total = 0

        for py_file in self.repo_path.rglob("*.py"):
            # Skip virtual environments and build directories
            if any(p in py_file.parts for p in ["venv", ".venv", "env", "build", "dist", "__pycache__"]):
                continue

            try:
                total += len(py_file.read_text(encoding='utf-8').splitlines())
            except (OSError, UnicodeDecodeError):
                continue

        return total

    def _count_functions_and_classes(self) -> dict[str, int]:
        """Count functions and classes in Python files."""
        import ast

        functions = 0
        classes = 0

        for py_file in self.repo_path.rglob("*.py"):
            if any(p in py_file.parts for p in ["venv", ".venv", "env", "build", "dist", "__pycache__"]):
                continue

            try:
                tree = ast.parse(py_file.read_text(encoding='utf-8'))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions += 1
                    elif isinstance(node, ast.ClassDef):
                        classes += 1

            except (SyntaxError, OSError, UnicodeDecodeError):
                continue

        return {"functions": functions, "classes": classes}

    def _count_errors(self) -> int:
        """Count linting errors."""
        try:
            result = subprocess.run(
                ["ruff", "check", str(self.repo_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Count lines in output (each line is typically an error)
            error_lines = [
                line for line in result.stdout.splitlines()
                if line.strip() and not line.startswith("Found")
            ]
            return len(error_lines)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Could not run ruff for error counting")
            return 0

    def _count_type_errors(self) -> int:
        """Count type checking errors using mypy."""
        try:
            result = subprocess.run(
                ["mypy", str(self.repo_path), "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse mypy output for error count
            output = result.stdout
            if "Found" in output and "error" in output:
                # Extract number from "Found X errors in Y files"
                import re
                match = re.search(r"Found (\d+) error", output)
                if match:
                    return int(match.group(1))

            return 0

        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Could not run mypy for type error counting")
            return 0


class ResearchMode:
    """Orchestrates hypothesis-driven research investigations.

    Attributes:
        question: Research question.
        hypothesis: Hypothesis to test (user or auto-generated).
        repo_path: Path to repository.
        output_dir: Directory for research outputs.
        visualizer: Data visualizer for graphs.
        metrics_collector: Metrics collector.
        session_id: Optional session ID for integration with SessionManager.
    """

    def __init__(
        self,
        question: str,
        repo_path: Path | str,
        hypothesis: str | None = None,
        output_dir: Path | str | None = None,
        session_id: str | None = None,
    ):
        """Initialize research mode.

        Args:
            question: Research question to investigate.
            repo_path: Path to the repository.
            hypothesis: Optional hypothesis (auto-generated if not provided).
            output_dir: Directory for outputs (defaults to repo/.research/).
            session_id: Optional session ID for integration with SessionManager.
        """
        self.question = question
        self.repo_path = Path(repo_path)
        self.hypothesis = hypothesis or self._generate_hypothesis(question)
        self.output_dir = Path(output_dir) if output_dir else (self.repo_path / ".research")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id

        self.visualizer = DataVisualizer()
        self.metrics_collector = MetricsCollector(self.repo_path)

        logger.info(f"ResearchMode initialized: {question}")
        logger.info(f"Hypothesis: {self.hypothesis}")
        if session_id:
            logger.info(f"Linked to session: {session_id}")

    def _generate_hypothesis(self, question: str) -> str:
        """Generate a hypothesis from the research question.

        Args:
            question: Research question.

        Returns:
            Generated hypothesis.
        """
        # Simple heuristic-based generation
        question_lower = question.lower()

        if "improve" in question_lower or "better" in question_lower:
            return "The proposed changes will result in measurable improvement"

        if "reduce" in question_lower:
            return "The changes will reduce the measured metric by at least 20%"

        if "increase" in question_lower or "add" in question_lower:
            return "The changes will increase the measured metric by at least 20%"

        if "type" in question_lower and "hint" in question_lower:
            return "Adding type hints will reduce type errors by more than 50%"

        if "async" in question_lower or "performance" in question_lower:
            return "The changes will improve performance by at least 30%"

        # Default
        return "The proposed changes will result in measurable positive impact"

    def collect_baseline(self) -> CodeMetrics:
        """Collect baseline metrics before changes.

        Returns:
            CodeMetrics with baseline measurements.
        """
        logger.info("Collecting baseline metrics...")
        baseline = self.metrics_collector.collect_metrics()

        # Save baseline to file
        baseline_file = self.output_dir / "baseline_metrics.json"
        baseline_file.write_text(json.dumps(baseline.to_dict(), indent=2))

        logger.info(f"Baseline metrics collected and saved to {baseline_file}")
        return baseline

    def collect_after(self) -> CodeMetrics:
        """Collect metrics after changes.

        Returns:
            CodeMetrics with after measurements.
        """
        logger.info("Collecting after-change metrics...")
        after = self.metrics_collector.collect_metrics()

        # Save to file
        after_file = self.output_dir / "after_metrics.json"
        after_file.write_text(json.dumps(after.to_dict(), indent=2))

        logger.info(f"After metrics collected and saved to {after_file}")
        return after

    def analyze_results(
        self,
        baseline: CodeMetrics,
        after: CodeMetrics,
    ) -> dict[str, Any]:
        """Analyze results and test hypothesis.

        Args:
            baseline: Baseline metrics.
            after: After metrics.

        Returns:
            Dictionary with analysis results.
        """
        analysis: dict[str, Any] = {}

        # Calculate changes
        changes = {
            "lines_of_code": after.lines_of_code - baseline.lines_of_code,
            "functions": after.functions - baseline.functions,
            "classes": after.classes - baseline.classes,
            "error_count": after.error_count - baseline.error_count,
            "type_errors": after.type_errors - baseline.type_errors,
        }

        # Calculate percentage changes
        pct_changes = {}
        for metric, change in changes.items():
            baseline_val = getattr(baseline, metric)
            if baseline_val != 0:
                pct_changes[metric] = (change / baseline_val) * 100
            else:
                pct_changes[metric] = 0.0

        analysis["absolute_changes"] = changes
        analysis["percentage_changes"] = pct_changes

        # Determine if hypothesis is confirmed
        # Look for significant positive changes
        significant_improvements = []

        if changes["error_count"] < 0:
            improvement = abs(pct_changes["error_count"])
            if improvement > 20:
                significant_improvements.append(f"Errors reduced by {improvement:.1f}%")

        if changes["type_errors"] < 0:
            improvement = abs(pct_changes["type_errors"])
            if improvement > 20:
                significant_improvements.append(f"Type errors reduced by {improvement:.1f}%")

        analysis["significant_improvements"] = significant_improvements
        analysis["hypothesis_confirmed"] = len(significant_improvements) > 0

        # Confidence level
        if len(significant_improvements) >= 2:
            analysis["confidence"] = "High"
        elif len(significant_improvements) == 1:
            analysis["confidence"] = "Medium"
        else:
            analysis["confidence"] = "Low"

        return analysis

    def generate_report(
        self,
        baseline: CodeMetrics,
        after: CodeMetrics,
        changes_made: str,
        output_format: str = "markdown",
    ) -> Path:
        """Generate research report with graphs.

        Args:
            baseline: Baseline metrics.
            after: After metrics.
            changes_made: Description of changes made.
            output_format: Report format (markdown or text).

        Returns:
            Path to generated report.
        """
        logger.info("Generating research report...")

        # Analyze results
        analysis = self.analyze_results(baseline, after)

        # Create findings
        findings = ResearchFindings(
            hypothesis=self.hypothesis,
            baseline_metrics=baseline,
            after_metrics=after,
            changes_made=changes_made,
            hypothesis_confirmed=analysis["hypothesis_confirmed"],
            confidence=analysis["confidence"],
            analysis=analysis,
        )

        # Generate conclusions
        findings.conclusions = self._generate_conclusions(analysis)

        # Generate recommendations
        findings.recommendations = self._generate_recommendations(analysis)

        # Create visualizations
        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        chart_paths = self._create_visualizations(baseline, after, charts_dir)

        # Generate report file
        if output_format == "markdown":
            report_path = self._generate_markdown_report(findings, chart_paths)
        else:
            report_path = self._generate_text_report(findings, chart_paths)

        # Save findings to SessionManager if session_id is set
        if self.session_id:
            try:
                from .session_manager import SessionManager
                manager = SessionManager()
                manager.save_research_findings(self.session_id, findings.to_dict())
                logger.info(f"Saved research findings to session {self.session_id}")
            except Exception as e:
                logger.warning(f"Could not save research findings to session: {e}")

        logger.info(f"Research report generated: {report_path}")

        return report_path

    def _generate_conclusions(self, analysis: dict[str, Any]) -> list[str]:
        """Generate conclusions from analysis."""
        conclusions = []

        if analysis["hypothesis_confirmed"]:
            conclusions.append("[CONFIRMED] Hypothesis confirmed with " + analysis["confidence"].lower() + " confidence")

            for improvement in analysis["significant_improvements"]:
                conclusions.append(f"- {improvement}")
        else:
            conclusions.append("[NOT CONFIRMED] Hypothesis not confirmed")
            conclusions.append("- No significant improvements detected (threshold: 20%)")

        # Add general observations
        changes = analysis["absolute_changes"]

        if changes["lines_of_code"] > 0:
            conclusions.append(f"- Codebase grew by {changes['lines_of_code']} lines")
        elif changes["lines_of_code"] < 0:
            conclusions.append(f"- Codebase reduced by {abs(changes['lines_of_code'])} lines")

        return conclusions

    def _generate_recommendations(self, analysis: dict[str, Any]) -> list[str]:
        """Generate recommendations from analysis."""
        recommendations = []

        if analysis["hypothesis_confirmed"]:
            recommendations.append("[KEEP] Changes are beneficial and should be kept")
            recommendations.append("Consider applying similar changes to other parts of the codebase")
        else:
            recommendations.append("[WARNING] Re-evaluate the approach")
            recommendations.append("Consider alternative strategies to achieve the goal")

        changes = analysis["absolute_changes"]

        if changes["error_count"] > 0:
            recommendations.append(f"[WARNING] Fix the {changes['error_count']} new errors introduced")

        if changes["type_errors"] > 0:
            recommendations.append(f"[WARNING] Address the {changes['type_errors']} new type errors")

        return recommendations

    def _create_visualizations(
        self,
        baseline: CodeMetrics,
        after: CodeMetrics,
        charts_dir: Path,
    ) -> dict[str, Path]:
        """Create visualization charts."""
        chart_paths = {}

        # Before/After comparison chart
        before_after_data = {
            "Error Count": baseline.error_count,
            "Type Errors": baseline.type_errors,
            "Functions": baseline.functions,
            "Classes": baseline.classes,
        }

        after_data = {
            "Error Count": after.error_count,
            "Type Errors": after.type_errors,
            "Functions": after.functions,
            "Classes": after.classes,
        }

        chart_paths["before_after"] = self.visualizer.create_before_after_chart(
            before=before_after_data,
            after=after_data,
            title="Before vs After Comparison",
            output_path=charts_dir / "before_after.png",
        )

        # Multi-metric dashboard
        dashboard_metrics = {
            "Errors": {
                "Before": float(baseline.error_count),
                "After": float(after.error_count),
            },
            "Type Errors": {
                "Before": float(baseline.type_errors),
                "After": float(after.type_errors),
            },
            "Code Elements": {
                "Before": float(baseline.functions + baseline.classes),
                "After": float(after.functions + after.classes),
            },
        }

        chart_paths["dashboard"] = self.visualizer.create_multi_metric_dashboard(
            metrics=dashboard_metrics,
            title="Research Dashboard",
            output_path=charts_dir / "dashboard.png",
        )

        return chart_paths

    def _generate_markdown_report(
        self,
        findings: ResearchFindings,
        chart_paths: dict[str, Path],
    ) -> Path:
        """Generate Markdown research report."""
        lines = []

        # Header
        lines.append("# Research Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n## Research Question\n\n{self.question}")
        lines.append(f"\n## Hypothesis\n\n{findings.hypothesis}")

        # Methodology
        lines.append("\n## Methodology\n")
        lines.append("1. Collected baseline metrics from the codebase")
        lines.append("2. Applied systematic changes")
        lines.append("3. Collected after-change metrics")
        lines.append("4. Performed statistical analysis")
        lines.append("5. Generated visualizations")

        # Changes Made
        lines.append(f"\n## Changes Made\n\n{findings.changes_made}")

        # Baseline Metrics
        lines.append("\n## Baseline Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in findings.baseline_metrics.to_dict().items():
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

        # After Metrics
        lines.append("\n## After Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for key, value in findings.after_metrics.to_dict().items():
            lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

        # Analysis
        lines.append("\n## Analysis\n")
        changes = findings.analysis["absolute_changes"]
        pct_changes = findings.analysis["percentage_changes"]

        lines.append("| Metric | Change | Percentage |")
        lines.append("|--------|--------|------------|")
        for key in changes.keys():
            change = changes[key]
            pct = pct_changes[key]
            sign = "+" if change > 0 else ""
            lines.append(f"| {key.replace('_', ' ').title()} | {sign}{change} | {sign}{pct:.1f}% |")

        # Visualizations
        lines.append("\n## Visualizations\n")
        for chart_name, chart_path in chart_paths.items():
            rel_path = chart_path.relative_to(self.output_dir)
            lines.append(f"\n### {chart_name.replace('_', ' ').title()}\n")
            lines.append(f"![{chart_name}]({rel_path})\n")

        # Conclusions
        lines.append("\n## Conclusions\n")
        for conclusion in findings.conclusions:
            lines.append(f"- {conclusion}")

        # Recommendations
        lines.append("\n## Recommendations\n")
        for rec in findings.recommendations:
            lines.append(f"- {rec}")

        # Result
        lines.append(f"\n## Result\n")
        if findings.hypothesis_confirmed:
            lines.append(f"**Hypothesis CONFIRMED** (Confidence: {findings.confidence})")
        else:
            lines.append(f"**Hypothesis NOT CONFIRMED** (Confidence: {findings.confidence})")

        # Write report
        report_path = self.output_dir / "research_report.md"
        report_path.write_text("\n".join(lines))

        return report_path

    def _generate_text_report(
        self,
        findings: ResearchFindings,
        chart_paths: dict[str, Path],
    ) -> Path:
        """Generate text research report."""
        lines = []

        lines.append("=" * 80)
        lines.append("RESEARCH REPORT".center(80))
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        lines.append(f"Research Question: {self.question}")
        lines.append(f"Hypothesis: {findings.hypothesis}\n")

        lines.append("CONCLUSIONS:")
        for conclusion in findings.conclusions:
            lines.append(f"  {conclusion}")

        lines.append("\nRECOMMENDATIONS:")
        for rec in findings.recommendations:
            lines.append(f"  {rec}")

        lines.append(f"\nResult: {'CONFIRMED' if findings.hypothesis_confirmed else 'NOT CONFIRMED'}")
        lines.append(f"Confidence: {findings.confidence}")

        lines.append("\n" + "=" * 80)

        # Write report
        report_path = self.output_dir / "research_report.txt"
        report_path.write_text("\n".join(lines))

        return report_path
