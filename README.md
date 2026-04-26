# NYC Taxi Driver Strategy Optimizer

A Python package that helps NYC taxi drivers maximize daily revenue by combining **demand prediction**, **graph-based city modeling**, and **dynamic programming** to recommend optimal repositioning strategies after each trip.

## Problem Statement

After completing a trip, a taxi driver faces a decision: stay in the current zone and wait for the next passenger, or drive empty to another zone where demand might be higher. This project builds a data-driven decision engine that recommends the best action at each step, considering:

- Expected waiting time for a pickup in each zone
- Expected trip fare from each zone
- Travel time and fuel cost to reposition between zones
- Time-of-day and day-of-week demand patterns
- Weather and holiday effects

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Data Layer  │────▶│ Feature Eng. │────▶│   ML Models     │
│  (SQL + ETL) │     │  (Pipeline)  │     │ (Scikit-Learn)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
┌─────────────┐     ┌──────────────┐              │
│  Simulator  │◀────│   DP Engine  │◀─────────────┘
│  (Evaluate) │     │ (Graph + DP) │
└─────────────┘     └──────────────┘
```

### Modules

| Module | Description | Course Topics |
|--------|-------------|---------------|
| `data` | Download TLC parquet files, clean, aggregate into SQLite | SQL, Data Streams |
| `graph` | Model taxi zones as a weighted graph (travel time, cost) | Graphs, DP |
| `features` | Build feature pipelines (time, lag, weather, holidays) | Data Processing |
| `models` | Predict wait time and fare per zone/hour | Scikit-Learn, ML |
| `simulation` | Simulate a driver's shift under different strategies | Multiprocessing |
| `evaluation` | Compare strategies, statistical analysis, visualization | — |

## Dataset

- **NYC TLC Trip Records**: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
  - Yellow taxi trip data (parquet format)
  - ~3M trips/month, fields include pickup/dropoff zone, datetime, fare, tip, distance
- **Taxi Zone Shapefile**: zone boundaries and borough mapping (~260 zones)
- **Weather Data**: Historical hourly weather for NYC (scraped from Open-Meteo)
- **US Holidays**: Generated via `holidays` Python library

## Installation

```bash
git clone https://github.com/<your-username>/nyc-taxi-driver-strategy.git
cd nyc-taxi-driver-strategy
pip install -e ".[dev]"
```

## Quick Start

### 1. Download and prepare data
```bash
python -m nyc_taxi_strategy.data.download --months 2024-01 2024-02 2024-03
python -m nyc_taxi_strategy.data.etl --config configs/default.yaml
```

### 2. Train models
```bash
python -m nyc_taxi_strategy.models.train --config configs/default.yaml
```

### 3. Run strategy optimization
```bash
python -m nyc_taxi_strategy.simulation.run --config configs/default.yaml --strategy dp_optimal
```

### 4. Compare strategies
```bash
python -m nyc_taxi_strategy.evaluation.compare --strategies random naive_best_zone dp_optimal
```

## Configuration

All experiments are driven by YAML config files in `configs/`:

```yaml
data:
  months: ["2024-01", "2024-02", "2024-03"]
  db_path: "data/nyc_taxi.db"
  boroughs: ["Manhattan"]  # or null for all

features:
  time_encoding: "cyclic"  # cyclic | onehot
  lag_hours: [1, 2, 3, 24, 168]
  rolling_windows: [3, 6, 12, 24]
  use_weather: true
  use_holidays: true

model:
  target: "wait_time"  # wait_time | fare
  algorithm: "gradient_boosting"  # linear | random_forest | gradient_boosting
  cv_strategy: "expanding_window"
  test_months: ["2024-03"]

simulation:
  shift_start: "06:00"
  shift_end: "18:00"
  fuel_cost_per_mile: 0.15
  n_simulations: 1000
  parallel_workers: 4
```

## Project Structure

```
nyc-taxi-driver-strategy/
├── nyc_taxi_strategy/          # Main package
│   ├── data/                   # Data download, cleaning, SQL ETL
│   ├── graph/                  # Zone graph, shortest paths, travel costs
│   ├── features/               # Feature transformers and pipeline
│   ├── models/                 # ML models with unified interface
│   ├── simulation/             # Driver shift simulator
│   ├── evaluation/             # Strategy comparison and visualization
│   └── utils/                  # Config loader, logging, constants
├── tests/                      # Unit tests (>80% coverage target)
├── configs/                    # YAML experiment configs
├── scripts/                    # Convenience scripts
├── notebooks/                  # Exploratory analysis (not graded)
├── docs/                       # Additional documentation
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
pytest tests/ -v --cov=nyc_taxi_strategy --cov-report=term-missing
```

## License

MIT
