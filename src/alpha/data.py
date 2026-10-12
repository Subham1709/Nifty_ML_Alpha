"""Download and load daily prices.

We store one long ("tidy") table with columns
    date, ticker, open, high, low, close, adj_close, volume
because a long table makes cross-sectional operations (rank all stocks on a date)
a simple groupby('date').
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"


def download_prices(tickers: list[str], start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance for NSE tickers, returned as a long table."""
    import yfinance as yf

    from .universe import yahoo_symbol

    symbols = {yahoo_symbol(t): t for t in tickers}
    raw = yf.download(
        list(symbols), start=start, end=end, auto_adjust=False,
        group_by="ticker", progress=True, threads=True,
    )
    frames = []
    for sym, name in symbols.items():
        if sym not in raw.columns.get_level_values(0):
            print(f"  skipped {name}: no data")
            continue
        df = raw[sym].dropna(subset=["Close"]).copy()
        if df.empty:
            print(f"  skipped {name}: empty")
            continue
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["ticker"] = name
        frames.append(df.reset_index().rename(columns={"Date": "date"}))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return out[cols].sort_values(["ticker", "date"]).reset_index(drop=True)


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)
