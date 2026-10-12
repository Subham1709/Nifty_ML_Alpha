"""Labels: what we are trying to predict.

For each stock on day t we predict the return from day t's close to day t+h's close
(h = 21 trading days, about one month). The model's target is that return *ranked
across stocks* on day t: we only need to know which stocks will do better than the
others, not where the whole market is going.

`label_end` is the date the label is finally known (day t+h). Purged cross-validation
needs it: a training sample whose label window [t, label_end] overlaps the test period
has, in effect, seen part of the test answers.
"""
from __future__ import annotations

import pandas as pd

HORIZON = 21


def build_labels(prices: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    parts = []
    for ticker, df in prices.groupby("ticker", sort=False):
        df = df.sort_values("date")
        px = df["adj_close"]
        out = pd.DataFrame({
            "date": df["date"].values,
            "ticker": ticker,
            "fwd_ret": (px.shift(-horizon) / px - 1).values,
            "label_end": df["date"].shift(-horizon).values,
        })
        parts.append(out)
    lab = pd.concat(parts, ignore_index=True).dropna(subset=["fwd_ret"])
    lab["target"] = lab.groupby("date")["fwd_ret"].rank(pct=True) - 0.5
    return lab
