"""Tests for ETL pipeline."""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nyc_taxi_strategy.data.etl import (
    aggregate_zone_hour,
    aggregate_zone_transitions,
    init_db,
    query_demand,
    _insert_demand,
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
