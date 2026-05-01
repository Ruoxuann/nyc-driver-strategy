"""Tests for SlidingWindowCV."""

import pandas as pd
import pytest

from nyc_taxi_strategy.models.cv import ExpandingWindowCV, SlidingWindowCV


@pytest.fixture
def time_df():
    return pd.DataFrame({
        "hour_start": pd.date_range("2024-01-01", periods=720, freq="h"),
        "value": range(720),
    })


class TestSlidingWindowCV:
    def test_splits_count(self, time_df):
        cv = SlidingWindowCV(n_splits=3, train_size_hours=240)
        splits = cv.split(time_df)
        assert len(splits) <= 3

    def test_no_future_leakage(self, time_df):
        cv = SlidingWindowCV(n_splits=3, train_size_hours=240)
        splits = cv.split(time_df)
        for split in splits:
            train_max = time_df.iloc[split.train_idx]["hour_start"].max()
            test_min = time_df.iloc[split.test_idx]["hour_start"].min()
            assert train_max < test_min

    def test_fixed_train_size(self, time_df):
        cv = SlidingWindowCV(n_splits=2, train_size_hours=240)
        splits = cv.split(time_df)
        for split in splits:
            n_train_hours = len(split.train_idx)
            assert n_train_hours > 0

    def test_too_small_data_raises(self):
        tiny_df = pd.DataFrame({
            "hour_start": pd.date_range("2024-01-01", periods=10, freq="h"),
        })
        cv = SlidingWindowCV(n_splits=3, train_size_hours=720)
        with pytest.raises(ValueError, match="Training window"):
            cv.split(tiny_df)


class TestExpandingWindowCVWithTestSize:
    def test_custom_test_size_hours(self, time_df):
        cv = ExpandingWindowCV(n_splits=3, test_size_hours=48)
        splits = cv.split(time_df)
        assert len(splits) > 0
        for split in splits:
            assert len(split.test_idx) > 0
