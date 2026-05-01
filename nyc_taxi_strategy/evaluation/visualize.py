"""Visualization functions for strategy evaluation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def plot_strategy_summary(
    results: dict[str, MonteCarloResult],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of mean revenue ± std for each strategy, with trips and idle % as subplots."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    names = list(results.keys())
    means = [results[n].mean_revenue for n in names]
    stds = [results[n].std_revenue for n in names]
    trips = [results[n].mean_trips for n in names]
    idle = [results[n].mean_idle_pct for n in names]

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    axes[0].bar(names, means, yerr=stds, capsize=5, color=colors)
    axes[0].set_title("Mean Net Revenue ± Std")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(names, trips, color=colors)
    axes[1].set_title("Mean Trips Completed")
    axes[1].set_ylabel("Trips")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(names, idle, color=colors)
    axes[2].set_title("Mean Idle Time %")
    axes[2].set_ylabel("Idle %")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Strategy Comparison Summary", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trips_vs_revenue(
    results: dict[str, MonteCarloResult],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Scatter plot of trips completed vs net revenue for each simulation run."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    rng = np.random.default_rng(42)
    n = len(results)

    for (name, mc), color in zip(results.items(), colors[:n]):
        trips = np.array([r.trips_completed for r in mc.results], dtype=float)
        revenues = [r.net_revenue for r in mc.results]
        jitter = rng.uniform(-0.25, 0.25, size=len(trips))
        ax.scatter(trips + jitter, revenues, alpha=0.4, s=12, color=color, label=name)

    ax.set_xlabel("Trips Completed")
    ax.set_ylabel("Net Revenue ($)")
    ax.set_title("Trips Completed vs Net Revenue (per simulation run)")
    ax.legend()
    ax.grid(True, alpha=0.3)

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


def plot_route_timeline(
    zone: int,
    time_str: str,
    route: list,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Line chart of expected remaining revenue along the optimal route.

    Args:
        zone: Starting zone ID.
        time_str: Query time string (HH:MM).
        route: List of (time, zone, action, value) tuples from run_query.
        output_path: Path to save. None to display.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    times = [r[0] for r in route]
    values = [r[3] for r in route]

    ax.plot(range(len(times)), values, marker="o", linewidth=2, color="steelblue")
    for i, (_, z, a, v) in enumerate(route):
        if "go to" in a:
            ax.axvline(i, color="orange", alpha=0.4, linewidth=1.5, linestyle="--")
            ax.annotate(f"→z{z}", (i, v), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="darkorange")
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Expected Remaining Revenue ($)")
    ax.set_title(f"Optimal Route Value — Starting zone {zone} at {time_str}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import argparse
    import pickle
    from nyc_taxi_strategy.utils.config import load_config

    parser = argparse.ArgumentParser(description="Visualize strategy or route results")
    parser.add_argument("--type", choices=["strategies", "route"], required=True,
                        help="'strategies': compare all strategies; 'route': optimal route for a query")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save figures (default: show interactively)")
    parser.add_argument("--zone", type=int, help="Zone ID (required for --type route)")
    parser.add_argument("--time", help="Time in HH:MM (required for --type route)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir) if args.output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    if args.type == "strategies":
        results_path = Path(cfg.evaluation.output_dir) / "simulation_results.pkl"
        if not results_path.exists():
            raise FileNotFoundError(f"No results at {results_path}. Run simulation first.")
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        plots = [
            ("revenue_distributions.png", lambda: plot_revenue_distributions(results, out / "revenue_distributions.png" if out else None)),
            ("strategy_boxplot.png",      lambda: plot_strategy_boxplot(results,       out / "strategy_boxplot.png"      if out else None)),
            ("strategy_summary.png",      lambda: plot_strategy_summary(results,       out / "strategy_summary.png"      if out else None)),
            ("trips_vs_revenue.png",      lambda: plot_trips_vs_revenue(results,       out / "trips_vs_revenue.png"      if out else None)),
        ]
        for name, fn in plots:
            fn()
            print(f"  {'Saved' if out else 'Generated'}: {name}")

    elif args.type == "route":
        if not args.zone or not args.time:
            parser.error("--type route requires --zone and --time")
        from nyc_taxi_strategy.query import run_query
        result = run_query(cfg, args.zone, args.time)
        if result is not None:
            route, _, _, _ = result
            p = out / "route_timeline.png" if out else None
            plot_route_timeline(args.zone, args.time, route, output_path=p)
            print(f"  {'Saved: ' + str(p) if out else 'Generated: route_timeline'}")

    if not out:
        plt.show()
