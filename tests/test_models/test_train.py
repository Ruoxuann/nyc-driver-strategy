"""Tests for model save/load utilities."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_strategy.models.predictors import GradientBoostingModel, create_model
from nyc_taxi_strategy.models.train import load_models, save_models


@pytest.fixture
def trained_models(tmp_path):
    X = pd.DataFrame({"x1": np.arange(20, dtype=float), "x2": np.ones(20)})
    y = pd.Series(np.arange(20, dtype=float))
    wait_model = create_model("gradient_boosting", n_estimators=10, max_depth=2)
    fare_model = create_model("gradient_boosting", n_estimators=10, max_depth=2)
    wait_model.fit(X, y)
    fare_model.fit(X, y)
    feature_cols = ["x1", "x2"]
    return wait_model, fare_model, feature_cols


class TestSaveModels:
    def test_creates_pkl_files(self, trained_models, tmp_path):
        wait_model, fare_model, feature_cols = trained_models
        save_models(wait_model, fare_model, feature_cols, tmp_path)
        assert (tmp_path / "wait_model.pkl").exists()
        assert (tmp_path / "fare_model.pkl").exists()

    def test_creates_output_dir(self, trained_models, tmp_path):
        wait_model, fare_model, feature_cols = trained_models
        out = tmp_path / "new_dir" / "models"
        save_models(wait_model, fare_model, feature_cols, out)
        assert out.exists()


class TestLoadModels:
    def test_roundtrip(self, trained_models, tmp_path):
        wait_model, fare_model, feature_cols = trained_models
        save_models(wait_model, fare_model, feature_cols, tmp_path)

        loaded_wait, loaded_fare, loaded_cols = load_models(tmp_path)
        assert loaded_cols == feature_cols

    def test_loaded_model_predicts(self, trained_models, tmp_path):
        wait_model, fare_model, feature_cols = trained_models
        save_models(wait_model, fare_model, feature_cols, tmp_path)

        loaded_wait, loaded_fare, loaded_cols = load_models(tmp_path)
        X = pd.DataFrame({"x1": [5.0], "x2": [1.0]})
        preds = loaded_wait.predict(X)
        assert len(preds) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_models(tmp_path / "nonexistent")
