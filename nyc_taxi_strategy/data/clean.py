"""Clean and validate raw TLC trip records."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Reasonable bounds for filtering outliers
FARE_MIN = 2.50
FARE_MAX = 500.0
TRIP_DISTANCE_MIN = 0.1
TRIP_DISTANCE_MAX = 100.0
TRIP_DURATION_MIN_SECONDS = 60
TRIP_DURATION_MAX_SECONDS = 3 * 3600  # 3 hours
VALID_ZONE_RANGE = (1, 263)


def load_raw_parquet(path: str | Path) -> pd.DataFrame:
    """Load a raw TLC parquet file with only needed columns.

    Args:
        path: Path to the parquet file.

    Returns:
        DataFrame with standardized column names.
    """
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "passenger_count",
    ]
    df = pd.read_parquet(path, columns=columns)
    df = df.rename(columns={
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "PULocationID": "pickup_zone",
        "DOLocationID": "dropoff_zone",
    })
    return df


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning rules to remove invalid or anomalous trips.

    Removes trips with:
    - Missing pickup/dropoff times
    - Invalid zone IDs (outside 1-263)
    - Negative or extreme fares
    - Unreasonable distances or durations
    - Zero passengers

    Args:
        df: Raw trip DataFrame.

    Returns:
        Cleaned DataFrame with a 'duration_seconds' column added.
    """
    n_before = len(df)

    # Drop missing timestamps
    df = df.dropna(subset=["pickup_datetime", "dropoff_datetime"])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    # Compute duration
    df = df.assign(
        duration_seconds=(
            (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
        )
    )

    # Apply filters
    mask = (
        df["pickup_zone"].between(*VALID_ZONE_RANGE)
        & df["dropoff_zone"].between(*VALID_ZONE_RANGE)
        & df["fare_amount"].between(FARE_MIN, FARE_MAX)
        & df["trip_distance"].between(TRIP_DISTANCE_MIN, TRIP_DISTANCE_MAX)
        & df["duration_seconds"].between(TRIP_DURATION_MIN_SECONDS, TRIP_DURATION_MAX_SECONDS)
        & (df["passenger_count"] > 0)
    )

    df = df.loc[mask].copy()

    n_after = len(df)
    pct_removed = (1 - n_after / n_before) * 100 if n_before > 0 else 0
    logger.info(f"Cleaning: {n_before} -> {n_after} trips ({pct_removed:.1f}% removed)")

    return df
