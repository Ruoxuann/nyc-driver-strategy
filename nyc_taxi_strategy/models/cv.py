"""Time-series aware cross-validation strategies.

Standard k-fold CV is invalid for time series because it leaks future
information into training. These splitters respect temporal ordering.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CVSplit:
    """A single train/test split with indices."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    fold: int


class ExpandingWindowCV:
    """Expanding window cross-validation for time series.

    Each fold uses all data up to a cutoff for training and the next
    chunk for testing. Training set grows with each fold.

    Example with 3 folds over months [Jan, Feb, Mar, Apr, May, Jun]:
        Fold 1: train=[Jan, Feb, Mar, Apr], test=[May]
        Fold 2: train=[Jan, Feb, Mar],      test=[Apr]
        Fold 3: train=[Jan, Feb],           test=[Mar]
    """

    def __init__(self, n_splits: int = 3, test_size_hours: int | None = None):
        self.n_splits = n_splits
        self.test_size_hours = test_size_hours

    def split(
        self, df: pd.DataFrame, time_col: str = "hour_start"
    ) -> list[CVSplit]:
        """Generate train/test splits.

        Args:
            df: DataFrame sorted by time.
            time_col: Column containing timestamps.

        Returns:
            List of CVSplit objects.
        """
        df = df.sort_values(time_col).reset_index(drop=True)
        timestamps = df[time_col]
        t_min, t_max = timestamps.min(), timestamps.max()
        total_hours = (t_max - t_min).total_seconds() / 3600

        if self.test_size_hours is not None:
            test_h = self.test_size_hours
        else:
            test_h = total_hours / (self.n_splits + 1)

        splits = []
        for fold in range(self.n_splits):
            test_end = t_max - pd.Timedelta(hours=test_h * fold)
            test_start = test_end - pd.Timedelta(hours=test_h)

            train_mask = timestamps < test_start
            test_mask = (timestamps >= test_start) & (timestamps < test_end)

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            splits.append(CVSplit(
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
                fold=fold,
            ))

        splits.reverse()  # chronological order
        logger.info(f"ExpandingWindowCV: {len(splits)} folds")
        return splits


class SlidingWindowCV:
    """Sliding window cross-validation.

    Unlike expanding window, the training set has a fixed size and
    slides forward.
    """

    def __init__(self, n_splits: int = 3, train_size_hours: int = 720):
        self.n_splits = n_splits
        self.train_size_hours = train_size_hours

    def split(
        self, df: pd.DataFrame, time_col: str = "hour_start"
    ) -> list[CVSplit]:
        df = df.sort_values(time_col).reset_index(drop=True)
        timestamps = df[time_col]
        t_min, t_max = timestamps.min(), timestamps.max()
        total_hours = (t_max - t_min).total_seconds() / 3600

        remaining = total_hours - self.train_size_hours
        if remaining <= 0:
            raise ValueError("Training window is larger than total data")

        test_h = remaining / self.n_splits

        splits = []
        for fold in range(self.n_splits):
            train_start = t_min + pd.Timedelta(hours=test_h * fold)
            train_end = train_start + pd.Timedelta(hours=self.train_size_hours)
            test_end = train_end + pd.Timedelta(hours=test_h)

            train_mask = (timestamps >= train_start) & (timestamps < train_end)
            test_mask = (timestamps >= train_end) & (timestamps < test_end)

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            splits.append(CVSplit(
                train_idx=np.where(train_mask)[0],
                test_idx=np.where(test_mask)[0],
                fold=fold,
            ))

        logger.info(f"SlidingWindowCV: {len(splits)} folds")
        return splits
