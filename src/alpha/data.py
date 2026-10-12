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
INDEX_PATH = DATA_DIR / "indices.parquet"


def _download(symbols: dict[str, str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
    """Yahoo symbol -> our name. Returns {name: OHLCV frame} for symbols that have data."""
    import yfinance as yf

    raw = yf.download(list(symbols), start=start, end=end, auto_adjust=False,
                      group_by="ticker", progress=False, threads=True)
    out = {}
    for sym, name in symbols.items():
        if sym not in raw.columns.get_level_values(0):
            continue
        df = raw[sym].dropna(subset=["Close"]).copy()
        if not df.empty:
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            out[name] = df
    return out


def download_prices(tickers: list[str], start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance for NSE tickers, returned as a long table.

    Tickers missing from the batch download are retried one by one, then under any
    alias in `universe.ALIASES` (symbols that changed after a rename or demerger).
    """
    from .universe import ALIASES, yahoo_symbol

    got = _download({yahoo_symbol(t): t for t in tickers}, start, end)
    for t in [t for t in tickers if t not in got]:
        for candidate in [t, *ALIASES.get(t, [])]:
            one = _download({yahoo_symbol(candidate): t}, start, end)
            if one:
                print(f"  {t}: found on retry as {candidate}")
                got.update(one)
                break
        else:
            print(f"  skipped {t}: no data on Yahoo")
    frames = []
    for name, df in got.items():
        df["ticker"] = name
        frames.append(df.reset_index().rename(columns={"Date": "date"}))
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return out[cols].sort_values(["ticker", "date"]).reset_index(drop=True)


def download_indices(symbols: dict[str, str], start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Index levels (e.g. {"^CNX100": "NIFTY100"}) as a long table of date, ticker, close."""
    frames = []
    for name, df in _download(symbols, start, end).items():
        frames.append(pd.DataFrame({"date": df.index.tz_localize(None) if df.index.tz else df.index,
                                    "ticker": name, "close": df["close"].values}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "close"])


def load_prices(path: Path = PRICES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_indices(path: Path = INDEX_PATH) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["date", "ticker", "close"])
