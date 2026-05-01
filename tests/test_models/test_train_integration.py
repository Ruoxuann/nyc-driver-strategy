"""Integration tests for train_models using a temporary database."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_strategy.data.etl import _insert_demand, _insert_transitions, init_db
from nyc_taxi_strategy.models.train import load_models, save_models, train_models


def _make_zone_hour_df(n_zones=5, n_hours=48) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    base = pd.Timestamp("2024-01-01")
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
                    "hour_of_day": 8,
                    "day_of_week": 0,
                    "trip_count": 10,
                    "avg_duration_s": 600.0,
                    "avg_distance": 3.0,
                    "avg_fare": 15.0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "taxi.db"
    conn = init_db(db_path)
    _insert_demand(conn, _make_zone_hour_df())
    _insert_transitions(conn, _make_transitions_df())
    conn.close()
    return str(db_path)


def _make_cfg(db_path, model_dir):
    return SimpleNamespace(
        data=SimpleNamespace(db_path=db_path),
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
        evaluation=SimpleNamespace(output_dir=str(model_dir)),
    )


class TestTrainModels:
    def test_returns_three_items(self, tmp_db, tmp_path):
        cfg = _make_cfg(tmp_db, tmp_path)
        result = train_models(cfg)
        assert len(result) == 3

    def test_models_can_predict(self, tmp_db, tmp_path):
        cfg = _make_cfg(tmp_db, tmp_path)
        wait_model, fare_model, feature_cols = train_models(cfg)
        X = pd.DataFrame({c: [0.0] for c in feature_cols})
        assert len(wait_model.predict(X)) == 1
        assert len(fare_model.predict(X)) == 1

    def test_feature_cols_nonempty(self, tmp_db, tmp_path):
        cfg = _make_cfg(tmp_db, tmp_path)
        _, _, feature_cols = train_models(cfg)
        assert len(feature_cols) > 0

    def test_save_and_load_roundtrip(self, tmp_db, tmp_path):
        cfg = _make_cfg(tmp_db, tmp_path)
        wait_model, fare_model, feature_cols = train_models(cfg)
        model_dir = tmp_path / "models"
        save_models(wait_model, fare_model, feature_cols, model_dir)
        loaded_wait, loaded_fare, loaded_cols = load_models(model_dir)
        assert loaded_cols == feature_cols
