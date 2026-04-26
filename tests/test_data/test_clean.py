"""Tests for data cleaning and validation."""

import pandas as pd
import pytest

from nyc_taxi_strategy.data.clean import (
    FARE_MAX,
    FARE_MIN,
    TRIP_DISTANCE_MAX,
    TRIP_DISTANCE_MIN,
    TRIP_DURATION_MAX_SECONDS,
    TRIP_DURATION_MIN_SECONDS,
    VALID_ZONE_RANGE,
    clean_trips,
)


def _make_trip(
    pickup_zone=100,
    dropoff_zone=200,
    fare=15.0,
    distance=3.0,
    duration_min=10,
    passengers=1,
    pickup_time="2024-01-15 08:00:00",
):
    """Helper to create a single trip row."""
    pickup = pd.Timestamp(pickup_time)
    dropoff = pickup + pd.Timedelta(minutes=duration_min)
    return {
        "pickup_datetime": pickup,
        "dropoff_datetime": dropoff,
        "pickup_zone": pickup_zone,
        "dropoff_zone": dropoff_zone,
        "fare_amount": fare,
        "trip_distance": distance,
        "tip_amount": 2.0,
        "total_amount": fare + 2.0,
        "passenger_count": passengers,
    }


def _make_df(trips):
    return pd.DataFrame(trips)


class TestCleanTrips:
    def test_valid_trip_kept(self):
        df = _make_df([_make_trip()])
        result = clean_trips(df)
        assert len(result) == 1

    def test_negative_fare_removed(self):
        df = _make_df([_make_trip(fare=-5.0)])
        result = clean_trips(df)
        assert len(result) == 0

    def test_extreme_fare_removed(self):
        df = _make_df([_make_trip(fare=FARE_MAX + 1)])
        result = clean_trips(df)
        assert len(result) == 0

    def test_fare_at_boundary_kept(self):
        df = _make_df([_make_trip(fare=FARE_MIN)])
        result = clean_trips(df)
        assert len(result) == 1

    def test_invalid_zone_removed(self):
        df = _make_df([_make_trip(pickup_zone=0)])
        result = clean_trips(df)
        assert len(result) == 0

    def test_zone_above_range_removed(self):
        df = _make_df([_make_trip(pickup_zone=300)])
        result = clean_trips(df)
        assert len(result) == 0

    def test_zero_passengers_removed(self):
        df = _make_df([_make_trip(passengers=0)])
        result = clean_trips(df)
        assert len(result) == 0

    def test_too_short_trip_removed(self):
        df = _make_df([_make_trip(duration_min=0.5)])  # 30 seconds
        result = clean_trips(df)
        assert len(result) == 0

    def test_too_long_trip_removed(self):
        df = _make_df([_make_trip(duration_min=200)])  # > 3 hours
        result = clean_trips(df)
        assert len(result) == 0

    def test_zero_distance_removed(self):
        df = _make_df([_make_trip(distance=0.0)])
        result = clean_trips(df)
        assert len(result) == 0

    def test_duration_column_added(self):
        df = _make_df([_make_trip()])
        result = clean_trips(df)
        assert "duration_seconds" in result.columns
        assert result.iloc[0]["duration_seconds"] == 600  # 10 min

    def test_missing_datetime_removed(self):
        trip = _make_trip()
        trip["pickup_datetime"] = None
        df = _make_df([trip])
        result = clean_trips(df)
        assert len(result) == 0

    def test_multiple_trips_mixed(self):
        trips = [
            _make_trip(fare=20.0),      # valid
            _make_trip(fare=-1.0),      # invalid
            _make_trip(fare=30.0),      # valid
            _make_trip(distance=0.0),   # invalid
        ]
        df = _make_df(trips)
        result = clean_trips(df)
        assert len(result) == 2
