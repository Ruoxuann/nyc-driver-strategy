"""ML models for demand and fare prediction with a unified interface.

All models follow the same fit/predict interface so they can be swapped
via configuration without changing downstream code.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train the model.

        Args:
            X: Feature matrix.
            y: Target variable.

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            X: Feature matrix (same columns as fit).

        Returns:
            Array of predictions.
        """
        ...

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            X: Feature matrix.
            y: True values.

        Returns:
            Dict with 'mae', 'rmse', 'mape' keys.
        """
        preds = self.predict(X)
        mae = mean_absolute_error(y, preds)
        rmse = np.sqrt(mean_squared_error(y, preds))
        # MAPE with protection against zeros
        nonzero = y != 0
        mape = np.mean(np.abs((y[nonzero] - preds[nonzero]) / y[nonzero])) * 100
        return {"mae": mae, "rmse": rmse, "mape": mape}


class LinearModel(BaseModel):
    """Ridge regression baseline."""

    def __init__(self, alpha: float = 1.0, **kwargs):
        self._model = Ridge(alpha=alpha, **kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)


class RandomForestModel(BaseModel):
    """Random forest regressor."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10, **kwargs):
        self._model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, **kwargs
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_


class GradientBoostingModel(BaseModel):
    """Gradient boosting regressor."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        self._model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            **kwargs,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_


class HistoricalAverageModel(BaseModel):
    """Baseline: predict the historical average for each (zone, hour, day_of_week).

    No ML involved — just lookup tables. Useful as a sanity-check baseline.
    """

    def __init__(self):
        self._lookup: dict[tuple[int, int, int], float] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalAverageModel":
        df = X[["pickup_zone", "hour", "day_of_week"]].copy()
        df["target"] = y.values

        grouped = df.groupby(["pickup_zone", "hour", "day_of_week"])["target"].mean()
        self._lookup = grouped.to_dict()
        self._global_mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            key = (int(row["pickup_zone"]), int(row["hour"]), int(row["day_of_week"]))
            preds.append(self._lookup.get(key, self._global_mean))
        return np.array(preds)


# ---- Model Registry ----

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "linear": LinearModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "historical_average": HistoricalAverageModel,
}


def create_model(algorithm: str, **params) -> BaseModel:
    """Create a model instance from the registry.

    Args:
        algorithm: Name of the algorithm (must be in MODEL_REGISTRY).
        **params: Model hyperparameters.

    Returns:
        An instance of the requested model.

    Raises:
        ValueError: If the algorithm is not registered.
    """
    if algorithm not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[algorithm](**params)
