"""Tests for ETL pipeline."""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from unittest.mock import patch, MagicMock

from nyc_taxi_strategy.data.etl import (
    _insert_demand,
    _insert_transitions,
    aggregate_zone_hour,
    aggregate_zone_transitions,
    init_db,
    query_demand,
    run_etl,
)


def _make_clean_df():
    """Create a minimal cleaned trip DataFrame."""
    return pd.DataFrame({
        "pickup_datetime": pd.to_datetime([
            "2024-01-15 08:15:00",
            "2024-01-15 08:30:00",
            "2024-01-15 08:45:00",
            "2024-01-15 09:10:00",
            "2024-01-15 09:20:00",
        ]),
        "dropoff_datetime": pd.to_datetime([
            "2024-01-15 08:30:00",
            "2024-01-15 08:50:00",
            "2024-01-15 09:05:00",
            "2024-01-15 09:25:00",
            "2024-01-15 09:40:00",
        ]),
        "pickup_zone": [100, 100, 100, 200, 200],
        "dropoff_zone": [200, 150, 200, 100, 100],
        "fare_amount": [15.0, 20.0, 18.0, 25.0, 22.0],
        "trip_distance": [3.0, 4.0, 3.5, 5.0, 4.5],
        "tip_amount": [2.0, 3.0, 2.5, 4.0, 3.5],
        "total_amount": [17.0, 23.0, 20.5, 29.0, 25.5],
        "passenger_count": [1, 2, 1, 1, 3],
        "duration_seconds": [900, 1200, 1200, 900, 1200],
    })


class TestAggregateZoneHour:
    def test_groups_by_zone_and_hour(self):
        df = _make_clean_df()
        result = aggregate_zone_hour(df)
        # Zone 100 has 3 trips in hour 08:00, zone 200 has 2 trips in hour 09:00
        z100_h8 = result[(result["pickup_zone"] == 100) & (result["hour_start"].dt.hour == 8)]
        assert len(z100_h8) == 1
        assert z100_h8.iloc[0]["trip_count"] == 3

    def test_avg_fare_correct(self):
        df = _make_clean_df()
        result = aggregate_zone_hour(df)
        z200_h9 = result[(result["pickup_zone"] == 200) & (result["hour_start"].dt.hour == 9)]
        expected_avg = (25.0 + 22.0) / 2
        assert z200_h9.iloc[0]["avg_fare"] == pytest.approx(expected_avg)


class TestAggregateZoneTransitions:
    def test_groups_by_od_and_time(self):
        df = _make_clean_df()
        result = aggregate_zone_transitions(df)
        # Zone 100->200 at hour 8, weekday Monday (Jan 15, 2024 is a Monday)
        mask = (
            (result["pickup_zone"] == 100)
            & (result["dropoff_zone"] == 200)
            & (result["hour_of_day"] == 8)
        )
        matching = result[mask]
        assert len(matching) == 1
        assert matching.iloc[0]["trip_count"] == 2


class TestInitDb:
    def test_creates_tables(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "zone_hour_demand" in tables
        assert "zone_transitions" in tables
        conn.close()

    def test_creates_parent_dirs(self, tmp_path):
        db_path = tmp_path / "nested" / "dir" / "test.db"
        conn = init_db(db_path)
        assert db_path.exists()
        conn.close()


class TestQueryDemand:
    def test_query_all(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        demand = aggregate_zone_hour(_make_clean_df())
        _insert_demand(conn, demand)
        conn.close()

        result = query_demand(db_path)
        assert len(result) == len(demand)

    def test_query_by_zone(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        demand = aggregate_zone_hour(_make_clean_df())
        _insert_demand(conn, demand)
        conn.close()

        result = query_demand(db_path, zone=100)
        assert all(result["pickup_zone"] == 100)


class TestInsertTransitions:
    def test_inserts_rows(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        df = _make_clean_df()
        transitions = aggregate_zone_transitions(df)
        _insert_transitions(conn, transitions)
        conn.close()

        conn2 = sqlite3.connect(str(db_path))
        count = conn2.execute("SELECT COUNT(*) FROM zone_transitions").fetchone()[0]
        conn2.close()
        assert count == len(transitions)

    def test_idempotent_insert(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        df = _make_clean_df()
        transitions = aggregate_zone_transitions(df)
        _insert_transitions(conn, transitions)
        _insert_transitions(conn, transitions)
        conn.close()

        conn2 = sqlite3.connect(str(db_path))
        count = conn2.execute("SELECT COUNT(*) FROM zone_transitions").fetchone()[0]
        conn2.close()
        assert count == len(transitions)


class TestRunEtl:
    def test_run_etl_populates_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        clean_df = _make_clean_df()

        with patch("nyc_taxi_strategy.data.etl.load_raw_parquet") as mock_load, \
             patch("nyc_taxi_strategy.data.etl.clean_trips") as mock_clean:
            mock_load.return_value = clean_df
            mock_clean.return_value = clean_df

            fake_path = tmp_path / "fake.parquet"
            fake_path.touch()
            run_etl([fake_path], db_path)

        conn = sqlite3.connect(str(db_path))
        demand_count = conn.execute("SELECT COUNT(*) FROM zone_hour_demand").fetchone()[0]
        transition_count = conn.execute("SELECT COUNT(*) FROM zone_transitions").fetchone()[0]
        conn.close()
        assert demand_count > 0
        assert transition_count > 0

    def test_run_etl_multiple_files(self, tmp_path):
        db_path = tmp_path / "test.db"
        clean_df = _make_clean_df()

        with patch("nyc_taxi_strategy.data.etl.load_raw_parquet") as mock_load, \
             patch("nyc_taxi_strategy.data.etl.clean_trips") as mock_clean:
            mock_load.return_value = clean_df
            mock_clean.return_value = clean_df

            paths = [tmp_path / f"fake_{i}.parquet" for i in range(2)]
            for p in paths:
                p.touch()
            run_etl(paths, db_path)

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM zone_hour_demand").fetchone()[0]
        conn.close()
        assert count > 0


class TestQueryDemandFilters:
    def _setup_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        demand = aggregate_zone_hour(_make_clean_df())
        _insert_demand(conn, demand)
        conn.close()
        return db_path

    def test_filter_start_time(self, tmp_path):
        db_path = self._setup_db(tmp_path)
        result = query_demand(db_path, start_time="2024-01-15 09:00:00")
        assert all(result["hour_start"] >= pd.Timestamp("2024-01-15 09:00:00"))

    def test_filter_end_time(self, tmp_path):
        db_path = self._setup_db(tmp_path)
        result = query_demand(db_path, end_time="2024-01-15 09:00:00")
        assert all(result["hour_start"] < pd.Timestamp("2024-01-15 09:00:00"))
