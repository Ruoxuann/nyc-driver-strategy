"""Integration tests for build_zone_models using a temporary database."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_strategy.data.etl import _insert_demand, _insert_transitions, init_db
from nyc_taxi_strategy.models.train import save_models, train_models
from nyc_taxi_strategy.simulation.run import build_zone_models, _shift_slots


def _make_zone_hour_df(n_zones=5, n_hours=48) -> pd.DataFrame:
    rng = np.random.default_rng(1)
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


@pytest.fixture(scope="module")
def trained_setup(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("data")
    db_path = tmp_path / "taxi.db"
    conn = init_db(db_path)
    _insert_demand(conn, _make_zone_hour_df())
    _insert_transitions(conn, _make_transitions_df())
    conn.close()

    cfg = SimpleNamespace(
        data=SimpleNamespace(db_path=str(db_path)),
        features=SimpleNamespace(
            lag_hours=[1],
            rolling_windows=[3],
            use_holidays=False,
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
        simulation=SimpleNamespace(shift_start="06:00", shift_end="10:00"),
    )

    wait_model, fare_model, feature_cols = train_models(cfg)
    model_dir = tmp_path / "models"
    save_models(wait_model, fare_model, feature_cols, model_dir)

    return cfg, wait_model, fare_model, feature_cols


class TestBuildZoneModels:
    def test_returns_zone_dict(self, trained_setup):
        cfg, wait_model, fare_model, feature_cols = trained_setup
        active_zones = list(range(1, 6))
        n_slots = _shift_slots(cfg.simulation.shift_start, cfg.simulation.shift_end)

        zone_models = build_zone_models(
            cfg, wait_model, fare_model, feature_cols, active_zones, n_slots
        )
        assert isinstance(zone_models, dict)
        assert len(zone_models) > 0

    def test_each_zone_has_slots(self, trained_setup):
        cfg, wait_model, fare_model, feature_cols = trained_setup
        active_zones = list(range(1, 6))
        n_slots = _shift_slots(cfg.simulation.shift_start, cfg.simulation.shift_end)

        zone_models = build_zone_models(
            cfg, wait_model, fare_model, feature_cols, active_zones, n_slots
        )
        for zone, slot_dict in zone_models.items():
            assert len(slot_dict) == n_slots

    def test_wait_times_positive(self, trained_setup):
        cfg, wait_model, fare_model, feature_cols = trained_setup
        active_zones = list(range(1, 6))
        n_slots = _shift_slots(cfg.simulation.shift_start, cfg.simulation.shift_end)

        zone_models = build_zone_models(
            cfg, wait_model, fare_model, feature_cols, active_zones, n_slots
        )
        for zone, slot_dict in zone_models.items():
            for slot, zm in slot_dict.items():
                assert zm.expected_wait_s >= 60.0
                assert zm.expected_fare >= 5.0
