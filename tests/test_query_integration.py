"""Integration tests for run_query using a real temporary database and trained models."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_strategy.data.etl import _insert_demand, _insert_transitions, init_db
from nyc_taxi_strategy.models.train import save_models, train_models
from nyc_taxi_strategy.query import run_query


def _make_zone_hour_df(n_zones=5, n_hours=72) -> pd.DataFrame:
    rng = np.random.default_rng(7)
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
                    "trip_count": 15,
                    "avg_duration_s": 600.0,
                    "avg_distance": 3.0,
                    "avg_fare": 15.0,
                })
    return pd.DataFrame(rows)


def _make_cfg(db_path, model_dir):
    return SimpleNamespace(
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
        evaluation=SimpleNamespace(output_dir=str(model_dir)),
        simulation=SimpleNamespace(
            shift_start="06:00",
            shift_end="10:00",
            fuel_cost_per_mile=0.15,
        ),
        graph=SimpleNamespace(travel_time_source="median"),
    )


@pytest.fixture(scope="module")
def query_setup(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("query")
    db_path = tmp_path / "taxi.db"
    conn = init_db(db_path)
    _insert_demand(conn, _make_zone_hour_df())
    _insert_transitions(conn, _make_transitions_df())
    conn.close()

    cfg = _make_cfg(db_path, tmp_path)
    wait_model, fare_model, feature_cols = train_models(cfg)
    model_dir = tmp_path / "models"
    save_models(wait_model, fare_model, feature_cols, model_dir)
    cfg.evaluation = SimpleNamespace(output_dir=str(tmp_path))

    return cfg


class TestRunQuery:
    def test_returns_result_for_valid_zone(self, query_setup, capsys):
        cfg = query_setup
        result = run_query(cfg, zone=1, time_str="06:00")
        assert result is not None

    def test_result_has_four_elements(self, query_setup):
        cfg = query_setup
        result = run_query(cfg, zone=1, time_str="06:00")
        assert len(result) == 4

    def test_route_is_list(self, query_setup):
        cfg = query_setup
        route, dp_result, zone_stats, n_slots = run_query(cfg, zone=1, time_str="06:00")
        assert isinstance(route, list)
        assert len(route) > 0

    def test_route_entries_have_correct_shape(self, query_setup):
        cfg = query_setup
        route, _, _, _ = run_query(cfg, zone=1, time_str="06:00")
        for entry in route:
            assert len(entry) == 4  # (time, zone, action, value)

    def test_invalid_zone_returns_none(self, query_setup, capsys):
        cfg = query_setup
        result = run_query(cfg, zone=9999, time_str="06:00")
        assert result is None

    def test_time_at_shift_end_returns_none(self, query_setup, capsys):
        cfg = query_setup
        result = run_query(cfg, zone=1, time_str="10:00")
        assert result is None

    def test_prints_recommendation(self, query_setup, capsys):
        cfg = query_setup
        run_query(cfg, zone=1, time_str="06:00")
        output = capsys.readouterr().out
        assert "Recommendation" in output

    def test_value_table_populated(self, query_setup):
        cfg = query_setup
        _, dp_result, _, _ = run_query(cfg, zone=1, time_str="06:00")
        assert len(dp_result.value_table) > 0

    def test_n_slots_correct(self, query_setup):
        cfg = query_setup
        _, _, _, n_slots = run_query(cfg, zone=1, time_str="06:00")
        assert n_slots == 8  # 06:00 to 10:00 = 4 hours = 8 slots
