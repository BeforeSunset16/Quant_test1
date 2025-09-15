from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterator, Tuple

class PurgedKFold:
    """Time-series split with purge and embargo.

    Splits by *unique dates* in chronological order. For each fold, removes
    an embargo window around the test block from the training set.
    """

    def __init__(self, n_splits: int = 5, embargo_days: int = 0):
        assert n_splits >= 2
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        # Expect DatetimeIndex; if MultiIndex, use level 0.
        if isinstance(X.index, pd.MultiIndex):
            dates = X.index.get_level_values(0).normalize()
        else:
            dates = pd.to_datetime(X.index).normalize()
        uniq = dates.unique().sort_values()
        folds = np.array_split(uniq, self.n_splits)

        # Build map date -> row indices
        by_date = {}
        for d in uniq:
            by_date[d] = np.where(dates == d)[0]

        for i in range(self.n_splits):
            test_dates = pd.DatetimeIndex(folds[i])
            test_idx = np.concatenate([by_date[d] for d in test_dates])

            # Embargo range
            if self.embargo_days > 0:
                left = test_dates.min() - pd.Timedelta(days=self.embargo_days)
                right = test_dates.max() + pd.Timedelta(days=self.embargo_days)
                mask = (dates < left) | (dates > right)
            else:
                mask = ~dates.isin(test_dates)

            train_idx = np.where(mask)[0]
            yield train_idx, test_idx