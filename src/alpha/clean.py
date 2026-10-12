"""Basic cleaning of Yahoo Finance prices for NSE stocks.

Yahoo's Indian data is usable but not perfect. The common problems are:
- duplicate rows for the same date,
- rows with zero volume (days the stock did not trade, filled with the last price),
- corporate actions (splits, bonus issues) that Yahoo failed to adjust, which show up
  as a fake one-day crash of 50% or more in the adjusted price.

`clean_prices` fixes the first two, and the third for the events listed in
`UNADJUSTED_ACTIONS`. Those were found with `suspicious_jumps`, which
`scripts/01b_data_report.py` runs on every download, and then checked by hand: a big
one-day move is only removed when it is a corporate action, never when it is a real
market move.
"""
from __future__ import annotations

import pandas as pd


# (ticker, date) -> why the jump is not a real return. Shareholders received shares of
# the demerged companies, so their wealth did not fall the way the price did.
UNADJUSTED_ACTIONS = {
    ("ADANIENT", "2015-06-03"): "demerger of Adani Transmission (2015)",
    ("TATAMOTORS", "2025-10-14"): "demerger of the commercial-vehicle business (2025)",
    ("VEDL", "2026-04-30"): "demerger of Vedanta's businesses (inferred from the -65% unadjusted drop)",
}

# Large one-day moves that are real and stay in the data
REAL_MOVES = {
    ("CANBK", "2017-10-25"): "government's public-sector bank recapitalisation plan",
    ("PNB", "2017-10-25"): "government's public-sector bank recapitalisation plan",
    ("INDUSINDBK", "2020-03-26"): "rebound after the COVID crash",
}


def neutralise_jumps(prices: pd.DataFrame, events=UNADJUSTED_ACTIONS) -> pd.DataFrame:
    """Rescale adj_close before each event so the event day's adjusted return is zero.

    This is what a proper corporate-action adjustment does. Raw close is left untouched,
    so the adj_close/close ratio (used to adjust high and low) stays consistent.
    """
    df = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    for ticker, date in events:
        rows = df.index[df["ticker"] == ticker]
        on = rows[df.loc[rows, "date"] == pd.Timestamp(date)]
        if len(on) == 0 or on[0] == rows[0]:
            continue
        i = on[0]
        factor = df.at[i, "adj_close"] / df.at[i - 1, "adj_close"]
        before = rows[rows < i]
        df.loc[before, "adj_close"] *= factor
    return df


def clean_prices(prices: pd.DataFrame, fix_actions: bool = True) -> pd.DataFrame:
    df = prices.drop_duplicates(["ticker", "date"], keep="last")
    df = df[(df["adj_close"] > 0) & (df["close"] > 0) & df["adj_close"].notna()]
    df = df[df["volume"] > 0]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return neutralise_jumps(df) if fix_actions else df


def suspicious_jumps(prices: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Stock-days whose adjusted close moved more than `threshold` in one day."""
    df = prices.sort_values(["ticker", "date"]).copy()
    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()
    df["raw_ret"] = df.groupby("ticker")["close"].pct_change()
    out = df[df["ret"].abs() > threshold]
    return out[["ticker", "date", "close", "adj_close", "ret", "raw_ret"]].reset_index(drop=True)
