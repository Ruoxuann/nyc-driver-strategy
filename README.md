# NYC Taxi Driver Strategy Optimizer

A Python package that helps NYC taxi drivers maximize daily revenue. Given a current zone and time of day, it uses **demand prediction**, **graph-based city modeling**, and **dynamic programming** to recommend the optimal repositioning action for the rest of the shift.

## How It Works

After dropping off a passenger, a driver asks: *should I wait here, or drive empty to another zone?*

This project answers that question by:

1. Learning historical demand and fare patterns from NYC TLC trip data
2. Modeling all 260+ taxi zones as a weighted directed graph (travel time, fuel cost)
3. Solving a dynamic programming problem over the remaining shift to find the globally optimal sequence of actions — not just the best next move

**Example query:**

```
$ python -m nyc_taxi_strategy.query --zone 161 --time 14:00

  Current zone : 161
  Current time : 14:00  (slot 16 of 24)
  Shift ends   : 18:00

  [Stay in current zone]
    Expected wait     : 1.0 min
    Expected fare     : $13.69
    Expected value    : $81.34  (remaining shift)

  [Reposition to zone 226]
    Travel time       : 21.6 min  (arrive at 14:30)
    Fuel cost         : $0.59
    Expected wait     : 29.5 min
    Expected fare     : $20.82
    Expected value    : $88.42  (remaining shift after travel cost)

  Recommendation: REPOSITION to zone 226
  Expected gain over staying: $+7.08

  Optimal policy from current time onwards:
  Time     Zone     Action                Expected Value
  -----------------------------------------------------
  14:00    161      go to zone 226       $         88.42
  14:30    226      stay                 $         89.01
  15:00    226      go to zone 28        $         62.30
  ...
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Data Layer  │────▶│ Feature Eng. │────▶│   ML Models     │
│  (SQL + ETL) │     │  (Pipeline)  │     │ (Scikit-Learn)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
┌─────────────┐     ┌──────────────┐              │
│  Query / Sim│◀────│   DP Engine  │◀─────────────┘
│  (Evaluate) │     │ (Graph + DP) │
└─────────────┘     └──────────────┘
```

| Module | Description |
|--------|-------------|
| `data` | Download TLC parquet files, clean, and aggregate into SQLite |
| `graph` | Model taxi zones as a weighted directed graph (travel time, fuel cost) |
| `features` | Feature pipelines: cyclic time encoding, lag features, rolling stats, holidays |
| `models` | Train and save gradient boosting models for wait time and fare prediction |
| `simulation` | Monte Carlo simulation of full shifts under five repositioning strategies |
| `evaluation` | Statistical comparison of strategies (Welch t-test, Cohen's d) |
| `query` | Real-time decision engine: given zone + time, output the optimal route |

## Dataset

| Source | Description |
|--------|-------------|
| [NYC TLC Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Yellow taxi trips in Parquet format (~3M trips/month). Fields: pickup/dropoff zone, datetime, fare, tip, distance. |
| [NYC Taxi Zone Shapefile](https://data.cityofnewyork.us/Transportation/NYC-Taxi-Zones/d3c5-ddgc) | Zone boundary polygons and borough mapping (~260 zones). |
| [Open-Meteo](https://open-meteo.com/) | Historical hourly weather for NYC (temperature, precipitation, wind). |

US public holidays are generated via the [`holidays`](https://pypi.org/project/holidays/) Python library.

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Ruoxuann/nyc-taxi-driver-strategy.git
cd nyc-taxi-driver-strategy
pip install -e ".[dev]"
```

## Usage

Steps 1–3 are a one-time setup pipeline. Step 4 is the real-time decision tool.

### 1. Download and prepare data

```bash
python -m nyc_taxi_strategy.data.download --months 2024-01 2024-02 2024-03
python -m nyc_taxi_strategy.data.etl --config configs/default.yaml
```

Downloads raw parquet files to `data/raw/` and aggregates them into a SQLite database at `data/nyc_taxi.db`. Missing months are skipped with a warning.

### 2. Train demand and fare models

```bash
python -m nyc_taxi_strategy.models.train --config configs/default.yaml
```

Trains two gradient boosting models — one for expected pickup wait time, one for expected fare — and saves them to `results/models/`.

### 3. Run strategy simulation (optional)

```bash
python -m nyc_taxi_strategy.simulation.run --config configs/default.yaml --strategy all --start-zone 161
```

Runs Monte Carlo simulations (1000 shifts per strategy) for all five strategies starting from the specified zone, and saves results to `results/simulation_results.pkl`. `--start-zone` defaults to zone 1 if omitted. Use `--strategy dp_optimal` to run a single strategy.

Available strategies: `random`, `stay_put`, `greedy_demand`, `greedy_revenue`, `dp_optimal`.

### 4. Query the optimal action

```bash
python -m nyc_taxi_strategy.query --zone <ZONE_ID> --time <HH:MM>
```

Given your current zone and time, solves the DP and outputs:
- Whether to stay or reposition, and to which zone
- Expected gain over staying
- The full planned route for the rest of the shift

```bash
python -m nyc_taxi_strategy.query --zone 161 --time 08:00
python -m nyc_taxi_strategy.query --zone 79  --time 17:30
```

### 5. Compare strategies

```bash
python -m nyc_taxi_strategy.evaluation.compare --config configs/default.yaml
```

Loads simulation results and prints a ranked summary table with Welch t-test comparisons between all strategy pairs.

## Configuration

All settings are controlled by `configs/default.yaml`:

```yaml
data:
  months: ["2024-01", "2024-02", "2024-03"]
  db_path: "data/nyc_taxi.db"
  raw_dir: "data/raw"
  boroughs: null  # null = all boroughs, or ["Manhattan", "Brooklyn"]

features:
  time_encoding: "cyclic"       # cyclic | onehot
  lag_hours: [1, 2, 3, 24, 168]
  rolling_windows: [3, 6, 12, 24]
  use_weather: true
  use_holidays: true

model:
  wait_time:
    algorithm: "gradient_boosting"  # gradient_boosting | random_forest | linear
    params:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1
  fare:
    algorithm: "gradient_boosting"
    params:
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1

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
├── nyc_taxi_strategy/
│   ├── data/                   # Download, cleaning, ETL
│   ├── graph/                  # Zone graph, Dijkstra, DP engine
│   ├── features/               # Feature transformers and pipeline
│   ├── models/                 # ML models, cross-validation, train/save
│   ├── simulation/             # Shift simulator, Monte Carlo runner
│   ├── evaluation/             # Strategy comparison and statistics
│   ├── query.py                # Real-time decision query
│   └── utils/                  # Config loader, logging
├── tests/                      # Unit tests (86 tests, 71% coverage)
├── configs/
│   └── default.yaml
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
pytest tests/ -v --cov=nyc_taxi_strategy --cov-report=term-missing
```

## License

MIT
