"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    months: list[str]
    db_path: str = "data/nyc_taxi.db"
    raw_dir: str = "data/raw"
    boroughs: list[str] | None = None


@dataclass
class FeatureConfig:
    time_encoding: str = "cyclic"
    lag_hours: list[int] = field(default_factory=lambda: [1, 2, 3, 24, 168])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 6, 12, 24])
    use_weather: bool = True
    use_holidays: bool = True


@dataclass
class ModelSpec:
    algorithm: str = "gradient_boosting"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    wait_time: ModelSpec = field(default_factory=ModelSpec)
    fare: ModelSpec = field(default_factory=ModelSpec)
    cv_strategy: str = "expanding_window"
    n_splits: int = 3
    test_months: list[str] = field(default_factory=list)


@dataclass
class GraphConfig:
    travel_time_source: str = "median"
    max_reposition_zones: int = 10


@dataclass
class SimulationConfig:
    shift_start: str = "06:00"
    shift_end: str = "18:00"
    fuel_cost_per_mile: float = 0.15
    n_simulations: int = 1000
    parallel_workers: int = 4
    strategies: list[str] = field(
        default_factory=lambda: ["random", "stay_put", "greedy_demand", "dp_optimal"]
    )


@dataclass
class EvalConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["total_revenue", "trips_completed", "idle_time_pct"]
    )
    output_dir: str = "results"


@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    graph: GraphConfig
    simulation: SimulationConfig
    evaluation: EvalConfig


def _build_dataclass(cls, raw: dict) -> Any:
    """Recursively build a dataclass from a dict, handling nested dataclasses."""
    if raw is None:
        return cls()
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in raw.items():
        if key in fieldtypes:
            ft = cls.__dataclass_fields__[key].type
            # Check if the field type is itself a dataclass
            if isinstance(ft, type) and hasattr(ft, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[key] = _build_dataclass(ft, value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load experiment configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A Config dataclass with all settings.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if "data" not in raw or "months" not in raw.get("data", {}):
        raise ValueError("Config must specify data.months")

    return Config(
        data=_build_dataclass(DataConfig, raw.get("data", {})),
        features=_build_dataclass(FeatureConfig, raw.get("features", {})),
        model=_build_dataclass(ModelConfig, raw.get("model", {})),
        graph=_build_dataclass(GraphConfig, raw.get("graph", {})),
        simulation=_build_dataclass(SimulationConfig, raw.get("simulation", {})),
        evaluation=_build_dataclass(EvalConfig, raw.get("evaluation", {})),
    )
