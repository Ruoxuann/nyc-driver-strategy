"""Tests for driver shift simulator."""

import numpy as np
import pytest

from nyc_taxi_strategy.simulation.simulator import (
    DPStrategy,
    GreedyDemandStrategy,
    RandomStrategy,
    ShiftResult,
    ShiftSimulator,
    StayPutStrategy,
    TripEvent,
    ZoneModel,
)


@pytest.fixture
def zone_models():
    """Simple zone models for 2 zones, 10 time slots."""
    models = {}
    for zone in [1, 2]:
        models[zone] = {}
        for t in range(10):
            fare = 15.0 if zone == 1 else 30.0
            models[zone][t] = ZoneModel(
                expected_wait_s=180,
                expected_fare=fare,
                fare_std=5.0,
                expected_duration_s=600,
                dropoff_probs={1: 0.5, 2: 0.5},
            )
    return models


@pytest.fixture
def travel_data():
    return {
        "times": {(1, 2): 300, (2, 1): 300},
        "distances": {(1, 2): 2.0, (2, 1): 2.0},
    }


@pytest.fixture
def simulator(zone_models, travel_data):
    return ShiftSimulator(
        n_time_slots=10,
        time_slot_minutes=30,
        zone_models=zone_models,
        travel_times=travel_data["times"],
        travel_distances=travel_data["distances"],
        fuel_cost_per_mile=0.15,
    )


class TestShiftResult:
    def test_empty_result(self):
        r = ShiftResult()
        assert r.trips_completed == 0
        assert r.net_revenue == 0

    def test_net_revenue(self):
        r = ShiftResult(total_revenue=100, total_reposition_cost=15)
        assert r.net_revenue == 85

    def test_idle_pct(self):
        r = ShiftResult(total_idle_time_s=3600, total_driving_time_s=3600)
        assert r.idle_time_pct == pytest.approx(50.0)


class TestStayPutStrategy:
    def test_always_returns_minus_one(self):
        s = StayPutStrategy()
        assert s.choose_action(1, 0) == -1
        assert s.choose_action(100, 5) == -1


class TestRandomStrategy:
    def test_sometimes_moves(self):
        s = RandomStrategy(zones=[1, 2, 3], move_probability=1.0)
        action = s.choose_action(1, 0)
        assert action in [1, 2, 3]

    def test_never_moves_when_prob_zero(self):
        s = RandomStrategy(zones=[1, 2, 3], move_probability=0.0)
        for _ in range(20):
            assert s.choose_action(1, 0) == -1


class TestGreedyDemandStrategy:
    def test_goes_to_highest_demand(self):
        demand = {
            1: {0: 10},
            2: {0: 50},
            3: {0: 30},
        }
        s = GreedyDemandStrategy(demand_lookup=demand)
        action = s.choose_action(1, 0)
        assert action == 2


class TestDPStrategy:
    def test_follows_policy(self):
        policy = {1: {0: 2, 1: -1}, 2: {0: -1}}
        s = DPStrategy(policy=policy)
        assert s.choose_action(1, 0) == 2
        assert s.choose_action(1, 1) == -1
        assert s.choose_action(2, 0) == -1

    def test_missing_state_returns_stay(self):
        policy = {1: {0: 2}}
        s = DPStrategy(policy=policy)
        assert s.choose_action(99, 0) == -1


class TestZoneModel:
    def test_sample_wait_positive(self):
        zm = ZoneModel(
            expected_wait_s=300, expected_fare=20,
            fare_std=5, expected_duration_s=600,
            dropoff_probs={1: 1.0},
        )
        rng = np.random.default_rng(42)
        wait = zm.sample_wait(rng)
        assert wait > 0

    def test_sample_fare_non_negative(self):
        zm = ZoneModel(
            expected_wait_s=300, expected_fare=20,
            fare_std=5, expected_duration_s=600,
            dropoff_probs={1: 1.0},
        )
        rng = np.random.default_rng(42)
        for _ in range(100):
            assert zm.sample_fare(rng) >= 0

    def test_sample_dropoff_valid(self):
        zm = ZoneModel(
            expected_wait_s=300, expected_fare=20,
            fare_std=5, expected_duration_s=600,
            dropoff_probs={1: 0.3, 2: 0.7},
        )
        rng = np.random.default_rng(42)
        dropoff = zm.sample_dropoff(rng)
        assert dropoff in [1, 2]


class TestShiftSimulator:
    def test_completes_without_error(self, simulator):
        strategy = StayPutStrategy()
        result = simulator.run(strategy, start_zone=1, seed=42)
        assert isinstance(result, ShiftResult)
        assert result.trips_completed >= 0

    def test_deterministic_with_seed(self, simulator):
        strategy = StayPutStrategy()
        r1 = simulator.run(strategy, start_zone=1, seed=42)
        r2 = simulator.run(strategy, start_zone=1, seed=42)
        assert r1.total_revenue == r2.total_revenue
        assert r1.trips_completed == r2.trips_completed

    def test_different_seeds_differ(self, simulator):
        strategy = StayPutStrategy()
        r1 = simulator.run(strategy, start_zone=1, seed=1)
        r2 = simulator.run(strategy, start_zone=1, seed=999)
        # Very unlikely to be identical
        assert r1.total_revenue != r2.total_revenue or r1.trips_completed != r2.trips_completed

    def test_revenue_non_negative(self, simulator):
        strategy = StayPutStrategy()
        result = simulator.run(strategy, start_zone=1, seed=42)
        assert result.total_revenue >= 0
