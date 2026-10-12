"""Features: what the model is allowed to know about each stock on day t.

Rule: every feature for day t uses only prices up to and including day t's close.
`tests/test_no_lookahead.py` checks this by changing future prices and asserting that
past features do not move.

Each raw feature is then ranked across stocks on the same day (cross-sectional rank),
scaled to [-0.5, 0.5]. We care about "is this stock's momentum high *relative to the
other stocks today*", not its absolute level, which mostly tracks the whole market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "rev_5d",        # 1-week return: short-term reversal
    "ret_21d",       # 1-month return
    "mom_63d",       # 3-month momentum
    "mom_126d",      # 6-month momentum
    "mom_12_1",      # 12-month momentum skipping the last month (classic Jegadeesh-Titman)
    "vol_21d",       # 1-month volatility of daily returns
    "vol_63d",       # 3-month volatility
    "vol_ratio",     # vol_21d / vol_63d: is volatility rising?
    "hl_range_21d",  # average (high-low)/close: intraday range, another volatility proxy
    "dist_52w_high", # close / 52-week high
    "ma_50_ratio",   # close / 50-day moving average
    "ma_200_ratio",  # close / 200-day moving average
    "rsi_14",        # relative strength index
    "log_dollar_vol",# log of average daily traded value: size/liquidity proxy
    "volume_surge",  # 5-day vs 63-day average volume
]


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / down)


def _per_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (un-ranked) features for one stock, indexed by date."""
    px = df["adj_close"]
    r = px.pct_change()
    # adj_close / close converts raw high/low into the adjusted price scale
    adj = df["adj_close"] / df["close"]
    high, low = df["high"] * adj, df["low"] * adj
    out = pd.DataFrame(index=df.index)
    out["rev_5d"] = px.pct_change(5)
    out["ret_21d"] = px.pct_change(21)
    out["mom_63d"] = px.pct_change(63)
    out["mom_126d"] = px.pct_change(126)
    out["mom_12_1"] = px.shift(21) / px.shift(252) - 1
    out["vol_21d"] = r.rolling(21).std()
    out["vol_63d"] = r.rolling(63).std()
    out["vol_ratio"] = out["vol_21d"] / out["vol_63d"]
    out["hl_range_21d"] = ((high - low) / px).rolling(21).mean()
    out["dist_52w_high"] = px / px.rolling(252).max()
    out["ma_50_ratio"] = px / px.rolling(50).mean()
    out["ma_200_ratio"] = px / px.rolling(200).mean()
    out["rsi_14"] = _rsi(px)
    dollar_vol = df["close"] * df["volume"]
    out["log_dollar_vol"] = np.log1p(dollar_vol.rolling(21).mean())
    out["volume_surge"] = df["volume"].rolling(5).mean() / df["volume"].rolling(63).mean()
    return out


def compute_raw_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Long table of raw features: one row per (date, ticker)."""
    parts = []
    for ticker, df in prices.groupby("ticker", sort=False):
        f = _per_stock(df.set_index("date").sort_index())
        f["ticker"] = ticker
        parts.append(f.reset_index())
    out = pd.concat(parts, ignore_index=True)
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


def cross_sectional_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rank each column across stocks on each date, scaled to [-0.5, 0.5]."""
    ranked = df.groupby("date")[cols].rank(pct=True) - 0.5
    out = df.copy()
    out[cols] = ranked
    return out


def build_features(prices: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    raw = compute_raw_features(prices)
    # require a full year of history so mom_12_1 and dist_52w_high exist
    raw = raw.dropna(subset=["mom_12_1", "dist_52w_high", "ma_200_ratio"])
    return cross_sectional_rank(raw, FEATURES)
