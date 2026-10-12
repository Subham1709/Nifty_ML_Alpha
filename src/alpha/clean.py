"""Basic cleaning of Yahoo Finance prices for NSE stocks.

Yahoo's Indian data is usable but not perfect. The common problems are:
- duplicate rows for the same date,
- rows with zero volume (days the stock did not trade, filled with the last price),
- corporate actions (splits, bonus issues) that Yahoo failed to adjust, which show up
  as a fake one-day crash of 50% or more in the adjusted price.

`clean_prices` fixes the first two. `suspicious_jumps` lists the third kind so that
`scripts/01b_data_report.py` can show them; whether to drop them is a judgement call
recorded in the learning log.
"""
from __future__ import annotations

import pandas as pd


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.drop_duplicates(["ticker", "date"], keep="last")
    df = df[(df["adj_close"] > 0) & (df["close"] > 0) & df["adj_close"].notna()]
    df = df[df["volume"] > 0]
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def suspicious_jumps(prices: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Stock-days whose adjusted close moved more than `threshold` in one day."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()
    df["raw_ret"] = df.groupby("ticker")["close"].pct_change()
    out = df[df["ret"].abs() > threshold]
    return out[["ticker", "date", "close", "adj_close", "ret", "raw_ret"]].reset_index(drop=True)
