"""Integration tests for run_strategy and build_zone_models edge cases."""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_strategy.data.etl import _insert_demand, _insert_transitions, init_db
from nyc_taxi_strategy.models.train import save_models, train_models
from nyc_taxi_strategy.simulation.run import build_zone_models, _shift_slots


def _make_demand_df(n_zones=5, n_hours=48) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    base = pd.Timestamp("2024-01-01 06:00")
    for z in range(1, n_zones + 1):
        for h in range(n_hours):
            rows.append({
                "pickup_zone": z,
                "hour_start": base + pd.Timedelta(hours=h),
                "trip_count": rng.integers(5, 30),
                "avg_fare": rng.uniform(10, 40),
                "median_fare": rng.uniform(10, 40),
                "avg_duration_s": rng.uniform(300, 1200),
                "avg_distance": rng.uniform(1, 8),
            })
    return pd.DataFrame(rows)


def _make_transitions_df(n_zones=5) -> pd.DataFrame:
    rows = []
    for src in range(1, n_zones + 1):
        for dst in range(1, n_zones + 1):
            if src != dst:
                rows.append({
                    "pickup_zone": src,
                    "dropoff_zone": dst,
                    "hour_of_day": 6,
                    "day_of_week": 0,
                    "trip_count": 10,
                    "avg_duration_s": 600.0,
                    "avg_distance": 3.0,
                    "avg_fare": 15.0,
                })
    return pd.DataFrame(rows)


def _base_cfg(tmp_path, shift_end="08:00", use_holidays=False):
    return SimpleNamespace(
        data=SimpleNamespace(db_path=str(tmp_path / "taxi.db")),
        features=SimpleNamespace(
            lag_hours=[1],
            rolling_windows=[3],
            use_holidays=use_holidays,
        ),
        model=SimpleNamespace(
            wait_time=SimpleNamespace(
                algorithm="gradient_boosting",
                params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1},
            ),
            fare=SimpleNamespace(
                algorithm="gradient_boosting",
                params={"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1},
            ),
        ),
        evaluation=SimpleNamespace(output_dir=str(tmp_path)),
        simulation=SimpleNamespace(
            shift_start="06:00",
            shift_end=shift_end,
            fuel_cost_per_mile=0.15,
            n_simulations=2,
            parallel_workers=1,
            strategies=["stay_put", "random", "greedy_demand", "greedy_revenue", "dp_optimal"],
        ),
        graph=SimpleNamespace(travel_time_source="median"),
    )


@pytest.fixture(scope="module")
def run_setup(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("run_strategy")
    db_path = tmp_path / "taxi.db"
    conn = init_db(db_path)
    _insert_demand(conn, _make_demand_df())
    _insert_transitions(conn, _make_transitions_df())
    conn.close()

    cfg = _base_cfg(tmp_path)
    wait_model, fare_model, feature_cols = train_models(cfg)
    save_models(wait_model, fare_model, feature_cols, tmp_path / "models")
    return cfg, tmp_path


class TestRunStrategy:
    def test_stay_put_saves_pickle(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "stay_put", start_zone=1)
        assert (tmp_path / "simulation_results.pkl").exists()

    def test_pickle_contains_strategy_key(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "stay_put", start_zone=1)
        with open(tmp_path / "simulation_results.pkl", "rb") as f:
            results = pickle.load(f)
        assert "stay_put" in results

    def test_random_strategy(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "random", start_zone=1)
        with open(tmp_path / "simulation_results.pkl", "rb") as f:
            results = pickle.load(f)
        assert "random" in results

    def test_greedy_demand_strategy(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "greedy_demand", start_zone=1)
        with open(tmp_path / "simulation_results.pkl", "rb") as f:
            results = pickle.load(f)
        assert "greedy_demand" in results

    def test_greedy_revenue_strategy(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "greedy_revenue", start_zone=1)
        with open(tmp_path / "simulation_results.pkl", "rb") as f:
            results = pickle.load(f)
        assert "greedy_revenue" in results

    def test_dp_optimal_strategy(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "dp_optimal", start_zone=1)
        with open(tmp_path / "simulation_results.pkl", "rb") as f:
            results = pickle.load(f)
        assert "dp_optimal" in results

    def test_all_strategies(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "all", start_zone=1)
        with open(tmp_path / "simulation_results.pkl", "rb") as f:
            results = pickle.load(f)
        for name in cfg.simulation.strategies:
            assert name in results

    def test_default_start_zone(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        # start_zone=None should not raise
        run_strategy(cfg, "stay_put", start_zone=None)

    def test_invalid_zone_raises(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, _ = run_setup
        with pytest.raises(ValueError, match="Zone 9999"):
            run_strategy(cfg, "stay_put", start_zone=9999)

    def test_invalid_strategy_raises(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, _ = run_setup
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_strategy(cfg, "not_a_strategy", start_zone=1)

    def test_result_has_positive_revenue(self, run_setup):
        from nyc_taxi_strategy.simulation.run import run_strategy
        cfg, tmp_path = run_setup
        run_strategy(cfg, "stay_put", start_zone=1)
        with open(tmp_path / "simulation_results.pkl", "rb") as f:
            results = pickle.load(f)
        assert results["stay_put"].mean_revenue >= 0


class TestBuildZoneModelsEdgeCases:
    def test_use_holidays_flag(self, run_setup):
        """Line 44: HolidayFeature appended when use_holidays=True."""
        cfg, _ = run_setup
        from types import SimpleNamespace
        cfg_h = SimpleNamespace(
            data=cfg.data,
            features=SimpleNamespace(
                lag_hours=cfg.features.lag_hours,
                rolling_windows=cfg.features.rolling_windows,
                use_holidays=True,
            ),
            simulation=cfg.simulation,
        )
        from nyc_taxi_strategy.models.train import load_models
        wait_model, fare_model, feature_cols = load_models(
            Path(cfg.evaluation.output_dir) / "models"
        )
        n_slots = _shift_slots(cfg.simulation.shift_start, cfg.simulation.shift_end)
        zone_models = build_zone_models(
            cfg_h, wait_model, fare_model, feature_cols, list(range(1, 6)), n_slots
        )
        assert len(zone_models) > 0

    def test_unknown_zone_skipped(self, run_setup):
        """Line 64: zone with no demand data is skipped (continue)."""
        cfg, _ = run_setup
        from nyc_taxi_strategy.models.train import load_models
        wait_model, fare_model, feature_cols = load_models(
            Path(cfg.evaluation.output_dir) / "models"
        )
        n_slots = _shift_slots(cfg.simulation.shift_start, cfg.simulation.shift_end)
        active_zones = list(range(1, 6)) + [999]
        zone_models = build_zone_models(
            cfg, wait_model, fare_model, feature_cols, active_zones, n_slots
        )
        assert 999 not in zone_models
        assert len(zone_models) > 0
