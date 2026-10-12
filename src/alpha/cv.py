"""Cross-validation schemes for overlapping financial labels.

Why the usual K-fold is wrong here
----------------------------------
Each label covers 21 trading days. The samples for RELIANCE on 3 March and 4 March
share 20 of those 21 days, so their labels are almost identical. If shuffled K-fold
puts 3 March in training and 4 March in testing, the model can simply memorise the
answer from its neighbour. Test scores then measure memory, not prediction.

Purged K-fold (López de Prado, *Advances in Financial Machine Learning*, ch. 7)
1. splits time into contiguous blocks instead of shuffling,
2. **purges** training samples whose label window overlaps the test block, and
3. adds an **embargo**: also drops training samples that start just after the test
   block, because features like 12-month momentum are serially correlated.

All splitters yield (train_idx, test_idx) as positional numpy arrays.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def naive_kfold(dates: pd.Series, n_splits: int = 5, seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Shuffled K-fold over rows: the textbook default, and leaky here."""
    yield from KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(len(dates)))


def purged_kfold(dates: pd.Series, label_end: pd.Series, n_splits: int = 5,
                 embargo_days: int = 21) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Contiguous-block K-fold with purging and an embargo (in trading days)."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    label_end = pd.to_datetime(pd.Series(label_end).reset_index(drop=True))
    uniq = np.sort(dates.unique())
    for block in np.array_split(uniq, n_splits):
        test_start, test_end = block[0], block[-1]
        test_mask = (dates >= test_start) & (dates <= test_end)
        # last date any test label depends on
        test_label_end = label_end[test_mask].max()
        pos_end = np.searchsorted(uniq, test_end)
        embargo_end = uniq[min(pos_end + embargo_days, len(uniq) - 1)]
        cutoff = max(test_label_end, embargo_end)
        # keep a training sample only if its whole label window ends before the test
        # block starts, or it starts after the test labels and embargo are over
        train_mask = (label_end < test_start) | (dates > cutoff)
        yield np.flatnonzero(train_mask.values), np.flatnonzero(test_mask.values)


def walk_forward(dates: pd.Series, label_end: pd.Series, first_test_year: int,
                 last_test_year: int | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding window: train on everything known before year Y starts, test on year Y.

    This is how the model would actually have been used: retrained every January on
    all labels that were already *realised* (label_end before 1 January).
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    label_end = pd.to_datetime(pd.Series(label_end).reset_index(drop=True))
    last_test_year = last_test_year or dates.dt.year.max()
    for year in range(first_test_year, last_test_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        train_mask = label_end < start
        test_mask = dates.dt.year == year
        if test_mask.any() and train_mask.any():
            yield np.flatnonzero(train_mask.values), np.flatnonzero(test_mask.values)
