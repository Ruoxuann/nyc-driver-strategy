"""Tests for configuration loading."""

import pytest
import tempfile
from pathlib import Path

import yaml

from nyc_taxi_strategy.utils.config import load_config


@pytest.fixture
def valid_config_path(tmp_path):
    config = {
        "data": {"months": ["2024-01"], "db_path": "test.db"},
        "features": {"time_encoding": "cyclic"},
        "model": {"cv_strategy": "expanding_window"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return path


class TestLoadConfig:
    def test_loads_valid_config(self, valid_config_path):
        cfg = load_config(valid_config_path)
        assert cfg.data.months == ["2024-01"]
        assert cfg.data.db_path == "test.db"
        assert cfg.features.time_encoding == "cyclic"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_missing_months_raises(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"data": {}}))
        with pytest.raises(ValueError, match="months"):
            load_config(path)

    def test_defaults_applied(self, tmp_path):
        config = {"data": {"months": ["2024-01"]}}
        path = tmp_path / "minimal.yaml"
        path.write_text(yaml.dump(config))
        cfg = load_config(path)
        # Should use default values
        assert cfg.features.lag_hours == [1, 2, 3, 24, 168]
        assert cfg.simulation.n_simulations == 1000
        assert cfg.graph.max_reposition_zones == 10
