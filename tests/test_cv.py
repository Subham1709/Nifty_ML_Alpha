"""Purged K-fold must leave no training label overlapping the test period."""
import numpy as np
import pandas as pd

from alpha.cv import purged_kfold, walk_forward
from alpha.labels import build_labels


def test_purged_no_overlap(fake_prices):
    lab = build_labels(fake_prices, horizon=21)
    uniq = np.sort(lab["date"].unique())
    for tr, te in purged_kfold(lab["date"], lab["label_end"], n_splits=5, embargo_days=10):
        test_start = lab["date"].iloc[te].min()
        test_label_end = lab["label_end"].iloc[te].max()
        tr_dates, tr_ends = lab["date"].iloc[tr], lab["label_end"].iloc[tr]
        overlap = (tr_ends >= test_start) & (tr_dates <= test_label_end)
        assert not overlap.any()
        assert len(np.intersect1d(tr, te)) == 0
        # embargo: no training sample in the 10 trading days after the test block
        pos = np.searchsorted(uniq, lab["date"].iloc[te].max())
        emb = uniq[pos + 1: pos + 11]
        assert not tr_dates.isin(emb).any()


def test_walk_forward_uses_only_realised_labels(fake_prices):
    lab = build_labels(fake_prices, horizon=21)
    for tr, te in walk_forward(lab["date"], lab["label_end"], first_test_year=2016):
        test_start = lab["date"].iloc[te].min()
        assert lab["label_end"].iloc[tr].max() < test_start
