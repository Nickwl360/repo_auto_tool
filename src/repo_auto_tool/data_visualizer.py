"""Data visualization for research mode.

Creates graphs, charts, and visual reports for research investigations.

Uses matplotlib for graph generation with fallback to text-based charts
if matplotlib is not available.

Example:
    visualizer = DataVisualizer()

    # Create performance comparison
    visualizer.create_performance_chart(
        before_metrics={"time": 10.5, "errors": 5},
        after_metrics={"time": 6.2, "errors": 1},
        title="Performance Improvement",
        output_path="performance.png"
    )

    # Create trend chart
    visualizer.create_trend_chart(
        data={"iterations": [1, 2, 3, 4], "success_rate": [0.5, 0.6, 0.75, 0.8]},
        title="Success Rate Over Time",
        output_path="trend.png"
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import matplotlib, fall back to text mode if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, using text-based charts")


@dataclass
class ChartData:
    """Data for creating a chart.

    Attributes:
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        data: Dictionary of series name to list of values.
        x_values: X-axis values (optional, defaults to indices).
    """
    title: str
    x_label: str
    y_label: str
    data: dict[str, list[float]]
    x_values: list[Any] | None = None
    colors: dict[str, str] = field(default_factory=dict)


class DataVisualizer:
    """Creates visualizations for research mode.

    Attributes:
        use_matplotlib: Whether matplotlib is available.
        default_colors: Default color scheme.
    """

    DEFAULT_COLORS = {
        "before": "#e74c3c",  # Red
        "after": "#2ecc71",   # Green
        "baseline": "#3498db", # Blue
        "target": "#f39c12",  # Orange
        "actual": "#9b59b6",  # Purple
    }

    def __init__(self, use_matplotlib: bool | None = None):
        """Initialize visualizer.

        Args:
            use_matplotlib: Force matplotlib on/off. None = auto-detect.
        """
        if use_matplotlib is None:
            self.use_matplotlib = MATPLOTLIB_AVAILABLE
        else:
            self.use_matplotlib = use_matplotlib and MATPLOTLIB_AVAILABLE

        if self.use_matplotlib:
            logger.info("DataVisualizer initialized with matplotlib support")
        else:
            logger.info("DataVisualizer initialized in text mode (matplotlib not available)")

    def create_before_after_chart(
        self,
        before: dict[str, float],
        after: dict[str, float],
        title: str,
        output_path: Path | str,
        y_label: str = "Value",
    ) -> Path:
        """Create before/after comparison chart.

        Args:
            before: Dictionary of metric name to value (before).
            after: Dictionary of metric name to value (after).
            title: Chart title.
            output_path: Path to save chart.
            y_label: Y-axis label.

        Returns:
            Path to saved chart.
        """
        output_path = Path(output_path)

        if self.use_matplotlib:
            return self._create_matplotlib_before_after(
                before, after, title, output_path, y_label
            )
        else:
            return self._create_text_before_after(
                before, after, title, output_path, y_label
            )

    def _create_matplotlib_before_after(
        self,
        before: dict[str, float],
        after: dict[str, float],
        title: str,
        output_path: Path,
        y_label: str,
    ) -> Path:
        """Create matplotlib before/after chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics = list(before.keys())
        before_values = [before[m] for m in metrics]
        after_values = [after[m] for m in metrics]

        x = range(len(metrics))
        width = 0.35

        bars1 = ax.bar(
            [i - width/2 for i in x],
            before_values,
            width,
            label='Before',
            color=self.DEFAULT_COLORS["before"],
            alpha=0.8
        )
        bars2 = ax.bar(
            [i + width/2 for i in x],
            after_values,
            width,
            label='After',
            color=self.DEFAULT_COLORS["after"],
            alpha=0.8
        )

        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9
            )
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Created before/after chart: {output_path}")
        return output_path

    def _create_text_before_after(
        self,
        before: dict[str, float],
        after: dict[str, float],
        title: str,
        output_path: Path,
        y_label: str,
    ) -> Path:
        """Create text-based before/after chart."""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
        lines.append("")

        for metric in before.keys():
            before_val = before[metric]
            after_val = after[metric]
            change = after_val - before_val
            change_pct = (change / before_val * 100) if before_val != 0 else 0

            lines.append(f"{metric}:")
            lines.append(f"  Before: {before_val:.2f}")
            lines.append(f"  After:  {after_val:.2f}")
            lines.append(f"  Change: {change:+.2f} ({change_pct:+.1f}%)")
            lines.append("")

        content = "\n".join(lines)
        output_path.write_text(content)

        logger.info(f"Created text-based before/after chart: {output_path}")
        return output_path

    def create_trend_chart(
        self,
        data: dict[str, list[float]],
        title: str,
        output_path: Path | str,
        x_label: str = "Iteration",
        y_label: str = "Value",
        x_values: list[Any] | None = None,
    ) -> Path:
        """Create trend/line chart showing values over time.

        Args:
            data: Dictionary of series name to list of values.
            title: Chart title.
            output_path: Path to save chart.
            x_label: X-axis label.
            y_label: Y-axis label.
            x_values: Optional x-axis values (defaults to indices).

        Returns:
            Path to saved chart.
        """
        output_path = Path(output_path)

        if self.use_matplotlib:
            return self._create_matplotlib_trend(
                data, title, output_path, x_label, y_label, x_values
            )
        else:
            return self._create_text_trend(
                data, title, output_path, x_label, y_label, x_values
            )

    def _create_matplotlib_trend(
        self,
        data: dict[str, list[float]],
        title: str,
        output_path: Path,
        x_label: str,
        y_label: str,
        x_values: list[Any] | None,
    ) -> Path:
        """Create matplotlib trend chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Default x values to indices
        if x_values is None:
            x_values = list(range(1, len(next(iter(data.values()))) + 1))

        # Plot each series
        colors = list(self.DEFAULT_COLORS.values())
        for i, (series_name, values) in enumerate(data.items()):
            color = colors[i % len(colors)]
            ax.plot(
                x_values,
                values,
                marker='o',
                label=series_name,
                color=color,
                linewidth=2,
                markersize=6
            )

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Created trend chart: {output_path}")
        return output_path

    def _create_text_trend(
        self,
        data: dict[str, list[float]],
        title: str,
        output_path: Path,
        x_label: str,
        y_label: str,
        x_values: list[Any] | None,
    ) -> Path:
        """Create text-based trend chart."""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
        lines.append("")

        # Default x values to indices
        if x_values is None:
            x_values = list(range(1, len(next(iter(data.values()))) + 1))

        # Create table
        lines.append(f"{x_label:>10} | " + " | ".join(f"{name:>10}" for name in data.keys()))
        lines.append("-" * (12 + len(data) * 13))

        for i, x in enumerate(x_values):
            row = f"{x:>10} | "
            row += " | ".join(f"{values[i]:>10.2f}" for values in data.values())
            lines.append(row)

        content = "\n".join(lines)
        output_path.write_text(content)

        logger.info(f"Created text-based trend chart: {output_path}")
        return output_path

    def create_success_rate_chart(
        self,
        iterations: list[int],
        success_rates: list[float],
        title: str,
        output_path: Path | str,
    ) -> Path:
        """Create success rate over iterations chart.

        Args:
            iterations: List of iteration numbers.
            success_rates: List of success rates (0.0-1.0).
            title: Chart title.
            output_path: Path to save chart.

        Returns:
            Path to saved chart.
        """
        output_path = Path(output_path)

        if self.use_matplotlib:
            return self._create_matplotlib_success_rate(
                iterations, success_rates, title, output_path
            )
        else:
            return self._create_text_success_rate(
                iterations, success_rates, title, output_path
            )

    def _create_matplotlib_success_rate(
        self,
        iterations: list[int],
        success_rates: list[float],
        title: str,
        output_path: Path,
    ) -> Path:
        """Create matplotlib success rate chart."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convert to percentages
        percentages = [rate * 100 for rate in success_rates]

        ax.plot(
            iterations,
            percentages,
            marker='o',
            color=self.DEFAULT_COLORS["actual"],
            linewidth=2,
            markersize=6,
            label='Success Rate'
        )

        # Add horizontal line at 50%
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Created success rate chart: {output_path}")
        return output_path

    def _create_text_success_rate(
        self,
        iterations: list[int],
        success_rates: list[float],
        title: str,
        output_path: Path,
    ) -> Path:
        """Create text-based success rate chart."""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
        lines.append("")

        lines.append("Iteration | Success Rate")
        lines.append("----------|-------------")

        for iter_num, rate in zip(iterations, success_rates):
            bar = "█" * int(rate * 20)  # 20-char bar
            lines.append(f"{iter_num:>9} | {rate:>5.1%} {bar}")

        content = "\n".join(lines)
        output_path.write_text(content)

        logger.info(f"Created text-based success rate chart: {output_path}")
        return output_path

    def create_cost_breakdown_chart(
        self,
        cost_by_model: dict[str, float],
        title: str,
        output_path: Path | str,
    ) -> Path:
        """Create pie chart showing cost breakdown by model.

        Args:
            cost_by_model: Dictionary of model name to cost.
            title: Chart title.
            output_path: Path to save chart.

        Returns:
            Path to saved chart.
        """
        output_path = Path(output_path)

        if self.use_matplotlib:
            return self._create_matplotlib_pie(
                cost_by_model, title, output_path
            )
        else:
            return self._create_text_pie(
                cost_by_model, title, output_path
            )

    def _create_matplotlib_pie(
        self,
        data: dict[str, float],
        title: str,
        output_path: Path,
    ) -> Path:
        """Create matplotlib pie chart."""
        fig, ax = plt.subplots(figsize=(10, 8))

        labels = list(data.keys())
        values = list(data.values())
        colors = list(self.DEFAULT_COLORS.values())[:len(labels)]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Created pie chart: {output_path}")
        return output_path

    def _create_text_pie(
        self,
        data: dict[str, float],
        title: str,
        output_path: Path,
    ) -> Path:
        """Create text-based pie chart."""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
        lines.append("")

        total = sum(data.values())

        for name, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
            percentage = (value / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 5)  # 20-char max
            lines.append(f"{name:>15}: ${value:>6.2f} ({percentage:>5.1f}%) {bar}")

        lines.append("")
        lines.append(f"{'Total':>15}: ${total:>6.2f}")

        content = "\n".join(lines)
        output_path.write_text(content)

        logger.info(f"Created text-based pie chart: {output_path}")
        return output_path

    def create_multi_metric_dashboard(
        self,
        metrics: dict[str, dict[str, float]],
        title: str,
        output_path: Path | str,
    ) -> Path:
        """Create dashboard with multiple metrics.

        Args:
            metrics: Nested dictionary like:
                {
                    "Performance": {"Before": 10.5, "After": 6.2},
                    "Quality": {"Before": 75, "After": 92},
                }
            title: Dashboard title.
            output_path: Path to save dashboard.

        Returns:
            Path to saved dashboard.
        """
        output_path = Path(output_path)

        if self.use_matplotlib:
            return self._create_matplotlib_dashboard(
                metrics, title, output_path
            )
        else:
            return self._create_text_dashboard(
                metrics, title, output_path
            )

    def _create_matplotlib_dashboard(
        self,
        metrics: dict[str, dict[str, float]],
        title: str,
        output_path: Path,
    ) -> Path:
        """Create matplotlib dashboard with subplots."""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_name, metric_data) in zip(axes, metrics.items()):
            categories = list(metric_data.keys())
            values = list(metric_data.values())

            colors = [
                self.DEFAULT_COLORS.get(cat.lower(), self.DEFAULT_COLORS["actual"])
                for cat in categories
            ]

            bars = ax.bar(categories, values, color=colors, alpha=0.8)

            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10
                )

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Created dashboard: {output_path}")
        return output_path

    def _create_text_dashboard(
        self,
        metrics: dict[str, dict[str, float]],
        title: str,
        output_path: Path,
    ) -> Path:
        """Create text-based dashboard."""
        lines = []
        lines.append(f"\n{title}")
        lines.append("=" * len(title))
        lines.append("")

        for metric_name, metric_data in metrics.items():
            lines.append(f"\n{metric_name}:")
            lines.append("-" * len(metric_name))

            for category, value in metric_data.items():
                lines.append(f"  {category:>10}: {value:>8.2f}")

        content = "\n".join(lines)
        output_path.write_text(content)

        logger.info(f"Created text-based dashboard: {output_path}")
        return output_path
