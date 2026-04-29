"""Evaluate and compare different driver strategies."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from nyc_taxi_strategy.simulation.parallel import MonteCarloResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two strategies."""

    strategy_a: str
    strategy_b: str
    mean_diff: float
    p_value: float
    significant: bool
    effect_size: float  # Cohen's d

    def __str__(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"{self.strategy_a} vs {self.strategy_b}: "
            f"mean diff=${self.mean_diff:.2f}, p={self.p_value:.4f} ({sig}), "
            f"Cohen's d={self.effect_size:.3f}"
        )


def compare_strategies(
    results: dict[str, MonteCarloResult],
    alpha: float = 0.05,
) -> list[ComparisonResult]:
    """Compare all pairs of strategies for statistical significance.

    Uses Welch's t-test (unequal variance t-test) to determine if
    revenue differences are statistically significant.

    Args:
        results: Dict mapping strategy name to MonteCarloResult.
        alpha: Significance level.

    Returns:
        List of ComparisonResult for each pair.
    """
    names = list(results.keys())
    comparisons = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a_rev = [r.net_revenue for r in results[a_name].results]
            b_rev = [r.net_revenue for r in results[b_name].results]

            t_stat, p_val = scipy_stats.ttest_ind(a_rev, b_rev, equal_var=False)

            # Cohen's d
            pooled_std = np.sqrt(
                (np.std(a_rev) ** 2 + np.std(b_rev) ** 2) / 2
            )
            cohens_d = (np.mean(a_rev) - np.mean(b_rev)) / pooled_std if pooled_std > 0 else 0

            comp = ComparisonResult(
                strategy_a=a_name,
                strategy_b=b_name,
                mean_diff=float(np.mean(a_rev) - np.mean(b_rev)),
                p_value=float(p_val),
                significant=bool(p_val < alpha),
                effect_size=float(cohens_d),
            )
            comparisons.append(comp)
            logger.info(str(comp))

    return comparisons


def build_summary_table(results: dict[str, MonteCarloResult]) -> list[dict]:
    """Build a summary table comparing all strategies.

    Args:
        results: Dict mapping strategy name to MonteCarloResult.

    Returns:
        List of dicts, each representing a row in the summary table.
    """
    rows = []
    for name, mc in results.items():
        row = {"strategy": name}
        row.update(mc.summary())
        rows.append(row)

    # Sort by mean revenue descending
    rows.sort(key=lambda r: r["mean_revenue"], reverse=True)
    return rows


if __name__ == "__main__":
    import argparse
    import pickle
    from pathlib import Path
    from nyc_taxi_strategy.utils.config import load_config

    parser = argparse.ArgumentParser(description="Compare strategy simulation results")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strategies", nargs="+",
                        help="Subset of strategies to compare (default: all in results file)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    results_path = Path(cfg.evaluation.output_dir) / "simulation_results.pkl"

    if not results_path.exists():
        raise FileNotFoundError(f"No results found at {results_path}. Run simulation first.")

    with open(results_path, "rb") as f:
        results = pickle.load(f)

    if args.strategies:
        results = {k: v for k, v in results.items() if k in args.strategies}

    summary = build_summary_table(results)
    print("\n" + "=" * 70)
    print(f"{'Strategy':<20} {'Mean Revenue':>13} {'Std':>8} {'Trips':>7} {'Idle%':>7}")
    print("=" * 70)
    for row in summary:
        print(f"{row['strategy']:<20} ${row['mean_revenue']:>12.2f} "
              f"${row['std_revenue']:>7.2f} {row['mean_trips']:>7.1f} "
              f"{row['mean_idle_pct']:>6.1f}%")
    print("=" * 70)

    print("\nStatistical comparisons (Welch t-test, α=0.05):")
    for c in compare_strategies(results):
        sig = "sig" if c.significant else "n.s"
        print(f"  [{sig}] {c.strategy_a} vs {c.strategy_b}: "
              f"Δ=${c.mean_diff:+.2f}  p={c.p_value:.4f}  d={c.effect_size:.2f}")
