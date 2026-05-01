"""Tests for strategy comparison and evaluation."""

import pytest

from nyc_taxi_strategy.evaluation.compare import (
    build_summary_table,
    compare_strategies,
)
from nyc_taxi_strategy.simulation.parallel import MonteCarloResult
from nyc_taxi_strategy.simulation.simulator import ShiftResult


def _make_mc_result(revenues: list[float]) -> MonteCarloResult:
    """Create a MonteCarloResult from a list of revenue values."""
    results = []
    for rev in revenues:
        r = ShiftResult(total_revenue=rev, total_reposition_cost=0)
        results.append(r)
    return MonteCarloResult(results=results)


class TestCompareStrategies:
    def test_significant_difference(self):
        a = _make_mc_result([100 + i for i in range(50)])
        b = _make_mc_result([200 + i for i in range(50)])
        comparisons = compare_strategies({"low": a, "high": b})
        assert len(comparisons) == 1
        assert comparisons[0].significant is True

    def test_no_significant_difference(self):
        a = _make_mc_result([100.0] * 50)
        b = _make_mc_result([100.0] * 50)
        comparisons = compare_strategies({"a": a, "b": b})
        assert comparisons[0].significant is False

    def test_multiple_strategies(self):
        results = {
            "s1": _make_mc_result([100.0] * 20),
            "s2": _make_mc_result([150.0] * 20),
            "s3": _make_mc_result([200.0] * 20),
        }
        comparisons = compare_strategies(results)
        assert len(comparisons) == 3  # C(3,2) = 3 pairs


class TestBuildSummaryTable:
    def test_sorted_by_revenue(self):
        results = {
            "low": _make_mc_result([50.0] * 10),
            "high": _make_mc_result([200.0] * 10),
            "mid": _make_mc_result([100.0] * 10),
        }
        table = build_summary_table(results)
        revenues = [row["mean_revenue"] for row in table]
        assert revenues == sorted(revenues, reverse=True)

    def test_has_expected_keys(self):
        results = {"a": _make_mc_result([100.0] * 10)}
        table = build_summary_table(results)
        row = table[0]
        assert "strategy" in row
        assert "mean_revenue" in row
        assert "n_runs" in row


class TestMonteCarloResult:
    def test_statistics(self):
        mc = _make_mc_result([100, 200, 300])
        assert mc.mean_revenue == pytest.approx(200.0)
        assert mc.median_revenue == pytest.approx(200.0)
        assert mc.n_runs == 3

    def test_percentiles(self):
        mc = _make_mc_result(list(range(100)))
        pcts = mc.revenue_percentiles
        assert "p5" in pcts
        assert "p95" in pcts
        assert pcts["p5"] < pcts["p95"]
