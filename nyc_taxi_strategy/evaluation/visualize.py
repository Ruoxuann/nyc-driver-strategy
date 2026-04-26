"""Visualization functions for strategy evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nyc_taxi_strategy.simulation.parallel import MonteCarloResult


def plot_revenue_distributions(
    results: dict[str, MonteCarloResult],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot revenue distributions for each strategy as overlapping histograms.

    Args:
        results: Dict mapping strategy name to MonteCarloResult.
        output_path: Path to save the figure. None to display.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, mc in results.items():
        revenues = [r.net_revenue for r in mc.results]
        ax.hist(revenues, bins=50, alpha=0.5, label=f"{name} (μ=${mc.mean_revenue:.0f})")

    ax.set_xlabel("Net Revenue ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Revenue Distribution by Strategy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_strategy_boxplot(
    results: dict[str, MonteCarloResult],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Box plot comparing strategy revenues.

    Args:
        results: Dict mapping strategy name to MonteCarloResult.
        output_path: Path to save the figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    for name, mc in results.items():
        revenues = [r.net_revenue for r in mc.results]
        data.append(revenues)
        labels.append(f"{name}\n(μ=${mc.mean_revenue:.0f})")

    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_ylabel("Net Revenue ($)")
    ax.set_title("Strategy Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_zone_heatmap(
    zone_values: dict[int, float],
    zone_positions: dict[int, tuple[float, float]] | None = None,
    title: str = "Zone Values",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Heatmap of values across taxi zones.

    Args:
        zone_values: zone_id -> value (e.g., expected revenue).
        zone_positions: zone_id -> (x, y) coordinates. If None, uses grid layout.
        title: Plot title.
        output_path: Path to save.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    zones = sorted(zone_values.keys())
    values = [zone_values[z] for z in zones]

    if zone_positions is not None:
        x = [zone_positions[z][0] for z in zones]
        y = [zone_positions[z][1] for z in zones]
        scatter = ax.scatter(x, y, c=values, cmap="YlOrRd", s=80, edgecolors="gray")
        plt.colorbar(scatter, ax=ax, label="Value")
    else:
        # Grid layout fallback
        n = len(zones)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        grid = np.full((rows, cols), np.nan)
        for i, v in enumerate(values):
            grid[i // cols, i % cols] = v
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Value")

    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hourly_pattern(
    hourly_stats: dict[int, float],
    ylabel: str = "Value",
    title: str = "Hourly Pattern",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Line plot showing a value across hours of the day.

    Args:
        hourly_stats: hour (0-23) -> value.
        ylabel: Y-axis label.
        title: Plot title.
        output_path: Path to save.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    hours = sorted(hourly_stats.keys())
    values = [hourly_stats[h] for h in hours]

    ax.plot(hours, values, marker="o", linewidth=2)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
