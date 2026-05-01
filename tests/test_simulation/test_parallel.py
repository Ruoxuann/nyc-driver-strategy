"""Tests for Monte Carlo parallel runner."""

import pytest

from nyc_taxi_strategy.simulation.parallel import MonteCarloResult, run_monte_carlo
from nyc_taxi_strategy.simulation.simulator import (
    ShiftResult,
    ShiftSimulator,
    StayPutStrategy,
    TripEvent,
    ZoneModel,
)


def _make_shift_result(revenue: float) -> ShiftResult:
    return ShiftResult(
        trips=[TripEvent(
            pickup_zone=1, dropoff_zone=1, pickup_time_slot=0,
            fare=revenue, wait_time_s=300, trip_duration_s=600,
        )],
        total_revenue=revenue,
        total_reposition_cost=0.0,
        total_idle_time_s=300.0,
        total_driving_time_s=600.0,
    )


@pytest.fixture
def simple_simulator():
    zone_models = {
        1: {
            slot: ZoneModel(
                expected_wait_s=300,
                expected_fare=15.0,
                fare_std=2.0,
                expected_duration_s=600,
                dropoff_probs={1: 1.0},
            )
            for slot in range(4)
        }
    }
    return ShiftSimulator(
        n_time_slots=4,
        time_slot_minutes=30,
        zone_models=zone_models,
        travel_times={},
        travel_distances={},
        fuel_cost_per_mile=0.15,
    )


class TestMonteCarloResult:
    def test_n_runs(self):
        mc = MonteCarloResult(results=[_make_shift_result(r) for r in [100, 200, 300]])
        assert mc.n_runs == 3

    def test_mean_revenue(self):
        mc = MonteCarloResult(results=[_make_shift_result(r) for r in [100, 200, 300]])
        assert mc.mean_revenue == pytest.approx(200.0)

    def test_std_revenue(self):
        mc = MonteCarloResult(results=[_make_shift_result(r) for r in [100, 100, 100]])
        assert mc.std_revenue == pytest.approx(0.0)

    def test_median_revenue(self):
        mc = MonteCarloResult(results=[_make_shift_result(r) for r in [100, 200, 300]])
        assert mc.median_revenue == pytest.approx(200.0)

    def test_mean_trips(self):
        mc = MonteCarloResult(results=[_make_shift_result(r) for r in [100, 200]])
        assert mc.mean_trips == pytest.approx(1.0)

    def test_mean_idle_pct(self):
        mc = MonteCarloResult(results=[_make_shift_result(r) for r in [100]])
        assert 0 <= mc.mean_idle_pct <= 100

    def test_revenue_percentiles(self):
        revenues = list(range(100))
        mc = MonteCarloResult(results=[_make_shift_result(float(r)) for r in revenues])
        pcts = mc.revenue_percentiles
        assert pcts["p5"] < pcts["p50"] < pcts["p95"]

    def test_summary_keys(self):
        mc = MonteCarloResult(results=[_make_shift_result(100.0)])
        s = mc.summary()
        for key in ["n_runs", "mean_revenue", "std_revenue", "median_revenue",
                    "mean_trips", "mean_idle_pct", "percentiles"]:
            assert key in s


class TestRunMonteCarlo:
    def test_returns_correct_n_runs(self, simple_simulator):
        mc = run_monte_carlo(
            simulator=simple_simulator,
            strategy=StayPutStrategy(),
            start_zone=1,
            n_simulations=10,
            n_workers=1,
        )
        assert mc.n_runs == 10

    def test_revenue_non_negative(self, simple_simulator):
        mc = run_monte_carlo(
            simulator=simple_simulator,
            strategy=StayPutStrategy(),
            start_zone=1,
            n_simulations=5,
            n_workers=1,
        )
        assert mc.mean_revenue >= 0

    def test_different_seeds_vary(self, simple_simulator):
        mc = run_monte_carlo(
            simulator=simple_simulator,
            strategy=StayPutStrategy(),
            start_zone=1,
            n_simulations=20,
            n_workers=1,
        )
        revenues = [r.net_revenue for r in mc.results]
        assert len(set(revenues)) > 1
