"""Tests for visualization functions."""

import matplotlib
matplotlib.use("Agg")

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from nyc_taxi_strategy.evaluation.visualize import (
    plot_hourly_pattern,
    plot_revenue_distributions,
    plot_route_timeline,
    plot_strategy_boxplot,
    plot_strategy_summary,
    plot_trips_vs_revenue,
    plot_zone_heatmap,
)
from nyc_taxi_strategy.simulation.parallel import MonteCarloResult
from nyc_taxi_strategy.simulation.simulator import ShiftResult, TripEvent


def _make_trip() -> TripEvent:
    return TripEvent(
        pickup_zone=1, dropoff_zone=2, pickup_time_slot=0,
        fare=15.0, wait_time_s=300, trip_duration_s=600,
    )


def _make_mc(revenues: list[float], n_trips: int = 2) -> MonteCarloResult:
    results = []
    for rev in revenues:
        r = ShiftResult(
            trips=[_make_trip() for _ in range(n_trips)],
            total_revenue=rev,
            total_reposition_cost=0.5,
            total_idle_time_s=600.0,
            total_driving_time_s=1200.0,
        )
        results.append(r)
    return MonteCarloResult(results=results)


@pytest.fixture
def sample_results():
    return {
        "dp_optimal": _make_mc([150 + i for i in range(20)]),
        "stay_put": _make_mc([120 + i for i in range(20)]),
        "random": _make_mc([100 + i for i in range(20)]),
    }


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestPlotRevenueDistributions:
    def test_returns_figure(self, sample_results):
        fig = plot_revenue_distributions(sample_results)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, sample_results, tmp_path):
        out = tmp_path / "dist.png"
        plot_revenue_distributions(sample_results, output_path=out)
        assert out.exists()

    def test_single_strategy(self):
        results = {"only": _make_mc([100.0] * 10)}
        fig = plot_revenue_distributions(results)
        assert isinstance(fig, plt.Figure)


class TestPlotStrategyBoxplot:
    def test_returns_figure(self, sample_results):
        fig = plot_strategy_boxplot(sample_results)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, sample_results, tmp_path):
        out = tmp_path / "box.png"
        plot_strategy_boxplot(sample_results, output_path=out)
        assert out.exists()


class TestPlotZoneHeatmap:
    def test_without_positions(self):
        zone_values = {i: float(i * 10) for i in range(1, 20)}
        fig = plot_zone_heatmap(zone_values)
        assert isinstance(fig, plt.Figure)

    def test_with_positions(self):
        zone_values = {1: 100.0, 2: 200.0, 3: 150.0}
        positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, 1.0)}
        fig = plot_zone_heatmap(zone_values, zone_positions=positions, title="Test")
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path):
        zone_values = {i: float(i) for i in range(1, 10)}
        out = tmp_path / "heatmap.png"
        plot_zone_heatmap(zone_values, output_path=out)
        assert out.exists()


class TestPlotStrategySummary:
    def test_returns_figure(self, sample_results):
        fig = plot_strategy_summary(sample_results)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, sample_results, tmp_path):
        out = tmp_path / "summary.png"
        plot_strategy_summary(sample_results, output_path=out)
        assert out.exists()

    def test_single_strategy(self):
        results = {"s1": _make_mc([80.0] * 5)}
        fig = plot_strategy_summary(results)
        assert isinstance(fig, plt.Figure)


class TestPlotTripsVsRevenue:
    def test_returns_figure(self, sample_results):
        fig = plot_trips_vs_revenue(sample_results)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, sample_results, tmp_path):
        out = tmp_path / "trips.png"
        plot_trips_vs_revenue(sample_results, output_path=out)
        assert out.exists()


class TestPlotHourlyPattern:
    def test_returns_figure(self):
        hourly = {h: float(h * 5) for h in range(24)}
        fig = plot_hourly_pattern(hourly, ylabel="Demand", title="Test")
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path):
        hourly = {h: float(h) for h in range(24)}
        out = tmp_path / "hourly.png"
        plot_hourly_pattern(hourly, output_path=out)
        assert out.exists()

    def test_partial_hours(self):
        hourly = {6: 10.0, 12: 20.0, 18: 15.0}
        fig = plot_hourly_pattern(hourly)
        assert isinstance(fig, plt.Figure)


class TestPlotRouteTimeline:
    def test_returns_figure(self):
        route = [
            ("14:00", 161, "go to zone 226", 88.42),
            ("14:30", 226, "stay", 89.01),
            ("15:00", 226, "go to zone 28", 62.30),
            ("15:30", 28, "stay", 55.00),
        ]
        fig = plot_route_timeline(zone=161, time_str="14:00", route=route)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path):
        route = [
            ("08:00", 100, "stay", 50.0),
            ("08:30", 100, "go to zone 200", 45.0),
        ]
        out = tmp_path / "route.png"
        plot_route_timeline(100, "08:00", route, output_path=out)
        assert out.exists()

    def test_all_stay_actions(self):
        route = [("10:00", 50, "stay", 30.0), ("10:30", 50, "stay", 20.0)]
        fig = plot_route_timeline(50, "10:00", route)
        assert isinstance(fig, plt.Figure)
