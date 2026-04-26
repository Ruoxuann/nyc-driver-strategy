"""Feature transformers for demand and fare prediction.

Each transformer follows a consistent interface: fit(df) -> self, transform(df) -> df.
Transformers can be composed into a pipeline.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False


class BaseTransformer(ABC):
    """Base class for feature transformers."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseTransformer":
        """Fit the transformer (learn parameters if any)."""
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class CyclicTimeEncoder(BaseTransformer):
    """Encode time features as sin/cos pairs to capture cyclical patterns.

    Converts hour (0-23) and day_of_week (0-6) into smooth cyclical features
    so the model understands that hour 23 is close to hour 0.
    """

    def fit(self, df: pd.DataFrame) -> "CyclicTimeEncoder":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "hour_start" in df.columns:
            hour = df["hour_start"].dt.hour
            dow = df["hour_start"].dt.dayofweek
            month = df["hour_start"].dt.month
        elif "hour" in df.columns:
            hour = df["hour"]
            dow = df.get("day_of_week", 0)
            month = df.get("month", 1)
        else:
            raise ValueError("DataFrame must have 'hour_start' or 'hour' column")

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        return df


class OneHotTimeEncoder(BaseTransformer):
    """Encode time features as one-hot vectors."""

    def fit(self, df: pd.DataFrame) -> "OneHotTimeEncoder":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "hour_start" in df.columns:
            df["hour"] = df["hour_start"].dt.hour
            df["day_of_week"] = df["hour_start"].dt.dayofweek

        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        hour_dummies = pd.get_dummies(df["hour"], prefix="hour")
        dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow")

        return pd.concat([df, hour_dummies, dow_dummies], axis=1)


class LagFeatureBuilder(BaseTransformer):
    """Create lagged demand features.

    For each zone, adds columns like trip_count_lag_1h, trip_count_lag_24h, etc.
    These are typically the most predictive features for demand forecasting.
    """

    def __init__(self, lag_hours: list[int], target_col: str = "trip_count"):
        self.lag_hours = lag_hours
        self.target_col = target_col

    def fit(self, df: pd.DataFrame) -> "LagFeatureBuilder":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["pickup_zone", "hour_start"])

        for lag in self.lag_hours:
            col_name = f"{self.target_col}_lag_{lag}h"
            df[col_name] = df.groupby("pickup_zone")[self.target_col].shift(lag)

        return df


class RollingStatsBuilder(BaseTransformer):
    """Compute rolling statistics (mean, std) over recent demand history."""

    def __init__(self, windows: list[int], target_col: str = "trip_count"):
        self.windows = windows
        self.target_col = target_col

    def fit(self, df: pd.DataFrame) -> "RollingStatsBuilder":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values(["pickup_zone", "hour_start"])

        for w in self.windows:
            mean_col = f"{self.target_col}_roll_mean_{w}h"
            std_col = f"{self.target_col}_roll_std_{w}h"

            df[mean_col] = (
                df.groupby("pickup_zone")[self.target_col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
            df[std_col] = (
                df.groupby("pickup_zone")[self.target_col]
                .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std())
            )

        return df


class HolidayFeature(BaseTransformer):
    """Add US holiday indicator feature."""

    def __init__(self, years: list[int] | None = None):
        self.years = years
        self._holidays: set = set()

    def fit(self, df: pd.DataFrame) -> "HolidayFeature":
        if self.years is None:
            if "hour_start" in df.columns:
                self.years = df["hour_start"].dt.year.unique().tolist()
            else:
                self.years = [2024]

        self._holidays = set()
        if HAS_HOLIDAYS:
            for year in self.years:
                for date in holidays_lib.US(years=year).keys():
                    self._holidays.add(date)
        else:
            # Fallback: major US holidays (fixed dates only)
            from datetime import date
            for year in self.years:
                self._holidays.update([
                    date(year, 1, 1),   # New Year
                    date(year, 7, 4),   # Independence Day
                    date(year, 12, 25), # Christmas
                    date(year, 11, 11), # Veterans Day
                    date(year, 6, 19),  # Juneteenth
                ])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "hour_start" in df.columns:
            df["is_holiday"] = df["hour_start"].dt.date.isin(self._holidays).astype(int)
        return df


class FeaturePipeline:
    """Chain multiple transformers into a single pipeline.

    Example:
        pipeline = FeaturePipeline([
            CyclicTimeEncoder(),
            LagFeatureBuilder(lag_hours=[1, 24, 168]),
            RollingStatsBuilder(windows=[6, 24]),
            HolidayFeature(),
        ])
        df_features = pipeline.fit_transform(df)
    """

    def __init__(self, transformers: list[BaseTransformer]):
        self.transformers = transformers

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        for t in self.transformers:
            t.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in self.transformers:
            df = t.transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in self.transformers:
            df = t.fit_transform(df)
        return df
