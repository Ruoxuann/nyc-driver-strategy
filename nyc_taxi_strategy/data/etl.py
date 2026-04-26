"""ETL pipeline: aggregate trip-level data into zone-hour demand table in SQLite."""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from nyc_taxi_strategy.data.clean import clean_trips, load_raw_parquet

logger = logging.getLogger(__name__)

SCHEMA_ZONE_HOUR = """
CREATE TABLE IF NOT EXISTS zone_hour_demand (
    pickup_zone     INTEGER,
    hour_start      TEXT,
    trip_count       INTEGER,
    avg_fare        REAL,
    median_fare     REAL,
    avg_duration_s  REAL,
    avg_distance    REAL,
    PRIMARY KEY (pickup_zone, hour_start)
);
"""

SCHEMA_ZONE_TRANSITIONS = """
CREATE TABLE IF NOT EXISTS zone_transitions (
    pickup_zone     INTEGER,
    dropoff_zone    INTEGER,
    hour_of_day     INTEGER,
    day_of_week     INTEGER,
    trip_count       INTEGER,
    avg_duration_s  REAL,
    avg_distance    REAL,
    avg_fare        REAL,
    PRIMARY KEY (pickup_zone, dropoff_zone, hour_of_day, day_of_week)
);
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Initialize SQLite database with required tables.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        SQLite connection.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute(SCHEMA_ZONE_HOUR)
    conn.execute(SCHEMA_ZONE_TRANSITIONS)
    conn.commit()
    return conn


def aggregate_zone_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip records into zone-hour demand stats.

    Args:
        df: Cleaned trip DataFrame with pickup_datetime, pickup_zone, etc.

    Returns:
        DataFrame with one row per (pickup_zone, hour_start).
    """
    df = df.assign(hour_start=df["pickup_datetime"].dt.floor("h"))

    agg = (
        df.groupby(["pickup_zone", "hour_start"])
        .agg(
            trip_count=("fare_amount", "count"),
            avg_fare=("fare_amount", "mean"),
            median_fare=("fare_amount", "median"),
            avg_duration_s=("duration_seconds", "mean"),
            avg_distance=("trip_distance", "mean"),
        )
        .reset_index()
    )
    return agg


def aggregate_zone_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip records into zone-to-zone transition stats.

    Args:
        df: Cleaned trip DataFrame.

    Returns:
        DataFrame with one row per (pickup_zone, dropoff_zone, hour_of_day, day_of_week).
    """
    df = df.assign(
        hour_of_day=df["pickup_datetime"].dt.hour,
        day_of_week=df["pickup_datetime"].dt.dayofweek,
    )

    agg = (
        df.groupby(["pickup_zone", "dropoff_zone", "hour_of_day", "day_of_week"])
        .agg(
            trip_count=("fare_amount", "count"),
            avg_duration_s=("duration_seconds", "mean"),
            avg_distance=("trip_distance", "mean"),
            avg_fare=("fare_amount", "mean"),
        )
        .reset_index()
    )
    return agg


def _insert_demand(conn: sqlite3.Connection, demand: pd.DataFrame) -> None:
    """Insert zone-hour demand rows into SQLite."""
    rows = []
    for _, r in demand.iterrows():
        rows.append((
            int(r["pickup_zone"]),
            str(r["hour_start"]),
            int(r["trip_count"]),
            float(r["avg_fare"]),
            float(r["median_fare"]),
            float(r["avg_duration_s"]),
            float(r["avg_distance"]),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO zone_hour_demand VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def _insert_transitions(conn: sqlite3.Connection, transitions: pd.DataFrame) -> None:
    """Insert zone transition rows into SQLite."""
    rows = []
    for _, r in transitions.iterrows():
        rows.append((
            int(r["pickup_zone"]),
            int(r["dropoff_zone"]),
            int(r["hour_of_day"]),
            int(r["day_of_week"]),
            int(r["trip_count"]),
            float(r["avg_duration_s"]),
            float(r["avg_distance"]),
            float(r["avg_fare"]),
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO zone_transitions VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def run_etl(parquet_paths: list[Path], db_path: str | Path) -> None:
    """Run full ETL pipeline: load, clean, aggregate, store.

    Args:
        parquet_paths: List of paths to raw parquet files.
        db_path: Path to SQLite database.
    """
    conn = init_db(db_path)

    for path in parquet_paths:
        logger.info(f"Processing {path.name}")

        df = load_raw_parquet(path)
        df = clean_trips(df)

        demand = aggregate_zone_hour(df)
        _insert_demand(conn, demand)

        transitions = aggregate_zone_transitions(df)
        _insert_transitions(conn, transitions)

        logger.info(f"  Loaded {len(demand)} zone-hour rows, {len(transitions)} transition rows")

    conn.close()
    logger.info(f"ETL complete. Database: {db_path}")


def query_demand(
    db_path: str | Path,
    zone: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    """Query zone-hour demand from the database.

    Args:
        db_path: Path to SQLite database.
        zone: Filter by zone ID.
        start_time: Filter by hour_start >= this (ISO string).
        end_time: Filter by hour_start < this (ISO string).

    Returns:
        DataFrame with demand data.
    """
    conn = sqlite3.connect(str(db_path))
    query = "SELECT * FROM zone_hour_demand WHERE 1=1"
    params: list = []

    if zone is not None:
        query += " AND pickup_zone = ?"
        params.append(zone)
    if start_time is not None:
        query += " AND hour_start >= ?"
        params.append(start_time)
    if end_time is not None:
        query += " AND hour_start < ?"
        params.append(end_time)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if "hour_start" in df.columns:
        df["hour_start"] = pd.to_datetime(df["hour_start"])
    return df
