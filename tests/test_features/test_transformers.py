"""Tests for feature transformers."""

import numpy as np
import pandas as pd
import pytest

from nyc_taxi_strategy.features.transformers import (
    CyclicTimeEncoder,
    FeaturePipeline,
    HolidayFeature,
    LagFeatureBuilder,
    OneHotTimeEncoder,
    RollingStatsBuilder,
)


@pytest.fixture
def sample_demand_df():
    """Create a sample zone-hour demand DataFrame."""
    dates = pd.date_range("2024-01-01", periods=72, freq="h")  # 3 days
    zones = [100, 200]
    rows = []
    for zone in zones:
        for dt in dates:
            rows.append({
                "pickup_zone": zone,
                "hour_start": dt,
                "trip_count": np.random.randint(5, 50),
                "avg_fare": np.random.uniform(10, 40),
            })
    return pd.DataFrame(rows)


class TestCyclicTimeEncoder:
    def test_output_columns(self, sample_demand_df):
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(sample_demand_df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert col in result.columns

    def test_sin_cos_range(self, sample_demand_df):
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(sample_demand_df)
        for col in ["hour_sin", "hour_cos"]:
            assert result[col].min() >= -1.0
            assert result[col].max() <= 1.0

    def test_midnight_and_23_are_close(self):
        """Hour 23 and hour 0 should have similar encodings."""
        df = pd.DataFrame({
            "hour_start": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 23:00"]),
        })
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(df)
        # Euclidean distance in sin/cos space should be small
        h0 = result.iloc[0]
        h23 = result.iloc[1]
        dist = np.sqrt((h0["hour_sin"] - h23["hour_sin"])**2 + (h0["hour_cos"] - h23["hour_cos"])**2)
        assert dist < 0.5  # should be very close

    def test_no_hour_column_raises(self):
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        encoder = CyclicTimeEncoder()
        with pytest.raises(ValueError):
            encoder.transform(df)

    def test_does_not_modify_original(self, sample_demand_df):
        original_cols = set(sample_demand_df.columns)
        encoder = CyclicTimeEncoder()
        encoder.fit_transform(sample_demand_df)
        assert set(sample_demand_df.columns) == original_cols


class TestLagFeatureBuilder:
    def test_lag_columns_created(self, sample_demand_df):
        builder = LagFeatureBuilder(lag_hours=[1, 24])
        result = builder.fit_transform(sample_demand_df)
        assert "trip_count_lag_1h" in result.columns
        assert "trip_count_lag_24h" in result.columns

    def test_lag_values_correct(self):
        df = pd.DataFrame({
            "pickup_zone": [1] * 5,
            "hour_start": pd.date_range("2024-01-01", periods=5, freq="h"),
            "trip_count": [10, 20, 30, 40, 50],
        })
        builder = LagFeatureBuilder(lag_hours=[1])
        result = builder.fit_transform(df)
        assert pd.isna(result.iloc[0]["trip_count_lag_1h"])
        assert result.iloc[1]["trip_count_lag_1h"] == 10
        assert result.iloc[4]["trip_count_lag_1h"] == 40

    def test_lag_respects_zone_groups(self):
        df = pd.DataFrame({
            "pickup_zone": [1, 1, 2, 2],
            "hour_start": pd.to_datetime([
                "2024-01-01 00:00", "2024-01-01 01:00",
                "2024-01-01 00:00", "2024-01-01 01:00",
            ]),
            "trip_count": [10, 20, 100, 200],
        })
        builder = LagFeatureBuilder(lag_hours=[1])
        result = builder.fit_transform(df)
        # Zone 1 lag: NaN, 10; Zone 2 lag: NaN, 100
        zone2_lag = result[result["pickup_zone"] == 2].iloc[1]["trip_count_lag_1h"]
        assert zone2_lag == 100


class TestRollingStatsBuilder:
    def test_rolling_columns_created(self, sample_demand_df):
        builder = RollingStatsBuilder(windows=[3, 6])
        result = builder.fit_transform(sample_demand_df)
        assert "trip_count_roll_mean_3h" in result.columns
        assert "trip_count_roll_std_6h" in result.columns


class TestHolidayFeature:
    def test_holiday_detected(self):
        df = pd.DataFrame({
            "hour_start": pd.to_datetime([
                "2024-01-01 12:00",  # New Year's Day
                "2024-01-02 12:00",  # Regular day
            ]),
        })
        holiday = HolidayFeature(years=[2024])
        result = holiday.fit_transform(df)
        assert result.iloc[0]["is_holiday"] == 1
        assert result.iloc[1]["is_holiday"] == 0

    def test_non_holiday(self):
        df = pd.DataFrame({
            "hour_start": pd.to_datetime(["2024-03-15 12:00"]),
        })
        holiday = HolidayFeature(years=[2024])
        result = holiday.fit_transform(df)
        assert result.iloc[0]["is_holiday"] == 0


class TestFeaturePipeline:
    def test_pipeline_chains_transformers(self, sample_demand_df):
        pipeline = FeaturePipeline([
            CyclicTimeEncoder(),
            LagFeatureBuilder(lag_hours=[1]),
            HolidayFeature(),
        ])
        result = pipeline.fit_transform(sample_demand_df)
        assert "hour_sin" in result.columns
        assert "trip_count_lag_1h" in result.columns
        assert "is_holiday" in result.columns

    def test_fit_then_transform(self, sample_demand_df):
        pipeline = FeaturePipeline([CyclicTimeEncoder()])
        pipeline.fit(sample_demand_df)
        result = pipeline.transform(sample_demand_df)
        assert "hour_sin" in result.columns


class TestOneHotTimeEncoder:
    def test_output_columns(self, sample_demand_df):
        encoder = OneHotTimeEncoder()
        result = encoder.fit_transform(sample_demand_df)
        assert "is_weekend" in result.columns

    def test_weekend_flag(self):
        df = pd.DataFrame({
            "hour_start": pd.to_datetime(["2024-01-06 12:00", "2024-01-07 12:00",
                                          "2024-01-08 12:00"]),
        })
        encoder = OneHotTimeEncoder()
        result = encoder.fit_transform(df)
        assert result.iloc[0]["is_weekend"] == 1  # Saturday
        assert result.iloc[1]["is_weekend"] == 1  # Sunday
        assert result.iloc[2]["is_weekend"] == 0  # Monday

    def test_hour_dummies_created(self, sample_demand_df):
        encoder = OneHotTimeEncoder()
        result = encoder.fit_transform(sample_demand_df)
        hour_cols = [c for c in result.columns if c.startswith("hour_")]
        assert len(hour_cols) > 0


class TestHolidayFeatureFallback:
    def test_no_hour_start_col(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        holiday = HolidayFeature(years=[2024])
        result = holiday.fit_transform(df)
        assert "is_holiday" not in result.columns

    def test_years_inferred_from_data(self):
        df = pd.DataFrame({
            "hour_start": pd.to_datetime(["2024-01-01", "2024-07-04"]),
        })
        holiday = HolidayFeature()
        result = holiday.fit_transform(df)
        assert "is_holiday" in result.columns


class TestCyclicTimeEncoderWithHourColumn:
    def test_hour_column_fallback(self):
        df = pd.DataFrame({"hour": [0, 6, 12, 18]})
        encoder = CyclicTimeEncoder()
        result = encoder.fit_transform(df)
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
