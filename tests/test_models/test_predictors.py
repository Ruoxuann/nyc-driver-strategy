"""Tests for prediction models."""

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_strategy.models.predictors import (
    BaseModel,
    GradientBoostingModel,
    HistoricalAverageModel,
    LinearModel,
    RandomForestModel,
    create_model,
)
from nyc_taxi_strategy.models.cv import ExpandingWindowCV, SlidingWindowCV


@pytest.fixture
def regression_data():
    """Simple regression dataset."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.uniform(0, 10, n),
    })
    y = pd.Series(3 * X["x1"] + 2 * X["x2"] + 0.5 * X["x3"] + rng.normal(0, 1, n))
    return X, y


class TestModelInterface:
    """All models should follow the same interface."""

    @pytest.mark.parametrize("model_cls", [LinearModel, RandomForestModel, GradientBoostingModel])
    def test_fit_predict(self, model_cls, regression_data):
        X, y = regression_data
        model = model_cls()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert isinstance(preds, np.ndarray)

    @pytest.mark.parametrize("model_cls", [LinearModel, RandomForestModel, GradientBoostingModel])
    def test_evaluate_returns_metrics(self, model_cls, regression_data):
        X, y = regression_data
        model = model_cls()
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0

    @pytest.mark.parametrize("model_cls", [LinearModel, RandomForestModel, GradientBoostingModel])
    def test_predictions_reasonable(self, model_cls, regression_data):
        X, y = regression_data
        model = model_cls()
        model.fit(X, y)
        preds = model.predict(X)
        # In-sample predictions should correlate with targets
        corr = np.corrcoef(preds, y)[0, 1]
        assert corr > 0.5


class TestHistoricalAverageModel:
    def test_predicts_mean(self):
        X = pd.DataFrame({
            "pickup_zone": [1, 1, 1, 2, 2, 2],
            "hour": [8, 8, 8, 8, 8, 8],
            "day_of_week": [0, 0, 0, 0, 0, 0],
        })
        y = pd.Series([10, 20, 30, 100, 200, 300])

        model = HistoricalAverageModel()
        model.fit(X, y)
        preds = model.predict(X[:2])

        assert preds[0] == pytest.approx(20.0)  # mean of [10, 20, 30]

    def test_fallback_to_global_mean(self):
        X_train = pd.DataFrame({
            "pickup_zone": [1, 1],
            "hour": [8, 8],
            "day_of_week": [0, 0],
        })
        y_train = pd.Series([10, 20])

        X_test = pd.DataFrame({
            "pickup_zone": [99],  # unseen zone
            "hour": [8],
            "day_of_week": [0],
        })

        model = HistoricalAverageModel()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds[0] == pytest.approx(15.0)  # global mean


class TestCreateModel:
    def test_valid_algorithm(self):
        model = create_model("linear")
        assert isinstance(model, LinearModel)

    def test_with_params(self):
        model = create_model("gradient_boosting", n_estimators=50, max_depth=3)
        assert isinstance(model, GradientBoostingModel)

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_model("deep_quantum_nn")


class TestExpandingWindowCV:
    def test_splits_count(self):
        df = pd.DataFrame({
            "hour_start": pd.date_range("2024-01-01", periods=720, freq="h"),
            "value": range(720),
        })
        cv = ExpandingWindowCV(n_splits=3)
        splits = cv.split(df)
        assert len(splits) <= 3

    def test_no_future_leakage(self):
        df = pd.DataFrame({
            "hour_start": pd.date_range("2024-01-01", periods=720, freq="h"),
            "value": range(720),
        })
        cv = ExpandingWindowCV(n_splits=3)
        splits = cv.split(df)
        for split in splits:
            train_max = df.iloc[split.train_idx]["hour_start"].max()
            test_min = df.iloc[split.test_idx]["hour_start"].min()
            assert train_max < test_min, "Training data leaks into test period"

    def test_expanding_train_size(self):
        df = pd.DataFrame({
            "hour_start": pd.date_range("2024-01-01", periods=720, freq="h"),
            "value": range(720),
        })
        cv = ExpandingWindowCV(n_splits=3)
        splits = cv.split(df)
        if len(splits) >= 2:
            assert len(splits[1].train_idx) >= len(splits[0].train_idx)
