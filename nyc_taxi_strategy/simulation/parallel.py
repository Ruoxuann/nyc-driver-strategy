"""Parallel simulation runner using multiprocessing.

Runs N simulations across multiple CPU cores to estimate the distribution
of shift outcomes under a given strategy.
"""

import logging
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

from nyc_taxi_strategy.simulation.simulator import (
    ShiftResult,
    ShiftSimulator,
    Strategy,
)

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Aggregated results from many simulation runs."""

    results: list[ShiftResult]

    @property
    def n_runs(self) -> int:
        return len(self.results)

    @property
    def mean_revenue(self) -> float:
        return float(np.mean([r.net_revenue for r in self.results]))

    @property
    def std_revenue(self) -> float:
        return float(np.std([r.net_revenue for r in self.results]))

    @property
    def median_revenue(self) -> float:
        return float(np.median([r.net_revenue for r in self.results]))

    @property
    def mean_trips(self) -> float:
        return float(np.mean([r.trips_completed for r in self.results]))

    @property
    def mean_idle_pct(self) -> float:
        return float(np.mean([r.idle_time_pct for r in self.results]))

    @property
    def revenue_percentiles(self) -> dict[str, float]:
        revenues = [r.net_revenue for r in self.results]
        return {
            "p5": float(np.percentile(revenues, 5)),
            "p25": float(np.percentile(revenues, 25)),
            "p50": float(np.percentile(revenues, 50)),
            "p75": float(np.percentile(revenues, 75)),
            "p95": float(np.percentile(revenues, 95)),
        }

    def summary(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "mean_revenue": round(self.mean_revenue, 2),
            "std_revenue": round(self.std_revenue, 2),
            "median_revenue": round(self.median_revenue, 2),
            "mean_trips": round(self.mean_trips, 1),
            "mean_idle_pct": round(self.mean_idle_pct, 1),
            "percentiles": {
                k: round(v, 2) for k, v in self.revenue_percentiles.items()
            },
        }


def _run_single(
    seed: int,
    simulator: ShiftSimulator,
    strategy: Strategy,
    start_zone: int,
) -> ShiftResult:
    """Worker function for a single simulation run."""
    return simulator.run(strategy, start_zone, seed=seed)


def run_monte_carlo(
    simulator: ShiftSimulator,
    strategy: Strategy,
    start_zone: int,
    n_simulations: int = 1000,
    n_workers: int | None = None,
    base_seed: int = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulations in parallel.

    Args:
        simulator: Configured ShiftSimulator.
        strategy: Repositioning strategy.
        start_zone: Starting zone for all runs.
        n_simulations: Number of simulation runs.
        n_workers: Number of parallel workers. None = cpu_count.
        base_seed: Base random seed (each run uses base_seed + i).

    Returns:
        MonteCarloResult with aggregated statistics.
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    seeds = [base_seed + i for i in range(n_simulations)]

    logger.info(
        f"Running {n_simulations} simulations with {n_workers} workers"
    )

    worker_fn = partial(
        _run_single,
        simulator=simulator,
        strategy=strategy,
        start_zone=start_zone,
    )

    if n_workers == 1:
        results = [worker_fn(s) for s in seeds]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(worker_fn, seeds)

    mc_result = MonteCarloResult(results=results)
    logger.info(
        f"Monte Carlo complete: mean revenue=${mc_result.mean_revenue:.2f} "
        f"± ${mc_result.std_revenue:.2f}"
    )
    return mc_result
