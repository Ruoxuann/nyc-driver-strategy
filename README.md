# NYC Taxi Driver Strategy Optimizer

A Python package that helps NYC taxi drivers maximize daily revenue by combining **demand prediction**, **graph-based city modeling**, and **dynamic programming** to recommend optimal repositioning strategies after each trip.

## Overview

After completing a trip, a driver faces a decision: wait in the current zone or reposition to a higher-demand zone. This project builds a data-driven decision engine that recommends the best action at each step by modeling:

- Expected pickup wait time per zone
- Expected fare and tip from each zone
- Repositioning travel time and fuel cost
- Time-of-day and day-of-week demand patterns
- Weather and public holiday effects

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

| Module | Description |
|--------|-------------|
| `data` | Download TLC parquet files, clean, and aggregate into SQLite |
| `graph` | Model taxi zones as a weighted directed graph (travel time, cost) |
| `features` | Build feature pipelines (time encoding, lag features, weather, holidays) |
| `models` | Predict wait time and expected fare per zone and hour |
| `simulation` | Simulate a driver's shift under different repositioning strategies |
| `evaluation` | Compare strategies with statistical analysis and visualization |

## Dataset

This project uses three publicly available data sources:

| Source | Description |
|--------|-------------|
| [NYC TLC Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Yellow taxi trip data in Parquet format (~3M trips/month). Fields include pickup/dropoff zone, datetime, fare, tip, and trip distance. |
| [NYC Taxi Zone Shapefile](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc) | Zone boundary polygons and borough mapping (~260 zones). |
| [Open-Meteo](https://open-meteo.com/) | Historical hourly weather data for NYC (temperature, precipitation, wind). |

US public holidays are generated programmatically via the [`holidays`](https://pypi.org/project/holidays/) Python library.

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Ruoxuann/nyc-taxi-driver-strategy.git
cd nyc-taxi-driver-strategy
pip install -e ".[dev]"
```

## Usage

### 1. Download and prepare data

```bash
python -m nyc_taxi_strategy.data.download --months 2024-01 2024-02 2024-03
python -m nyc_taxi_strategy.data.etl --config configs/default.yaml
```

### 2. Train demand and fare models

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

### Importing as a library

```python
from nyc_taxi_strategy.graph import ZoneGraph
from nyc_taxi_strategy.models import WaitTimeModel, FareModel
from nyc_taxi_strategy.simulation import ShiftSimulator

graph = ZoneGraph.from_config("configs/default.yaml")
simulator = ShiftSimulator(graph=graph, strategy="dp_optimal")
result = simulator.run(shift_start="06:00", shift_end="18:00")
print(result.summary())
```

## Configuration

Experiments are driven by YAML config files in `configs/`:

```yaml
data:
  months: ["2024-01", "2024-02", "2024-03"]
  db_path: "data/nyc_taxi.db"
  boroughs: ["Manhattan"]  # set to null for all boroughs

features:
  time_encoding: "cyclic"       # cyclic | onehot
  lag_hours: [1, 2, 3, 24, 168]
  rolling_windows: [3, 6, 12, 24]
  use_weather: true
  use_holidays: true

model:
  target: "wait_time"           # wait_time | fare
  algorithm: "gradient_boosting"
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
├── tests/                      # Unit tests
├── configs/                    # YAML experiment configs
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
pytest tests/ -v --cov=nyc_taxi_strategy --cov-report=term-missing
```

## License

MIT
