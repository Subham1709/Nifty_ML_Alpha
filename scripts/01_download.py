"""Step 1: download daily prices for the Nifty 100 stocks, plus the Nifty 100 and Nifty 50 indices.

Run from the repo root:  python scripts/01_download.py
Works the same in Google Colab:  !python scripts/01_download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha.data import DATA_DIR, INDEX_PATH, PRICES_PATH, download_indices, download_prices  # noqa: E402
from alpha.universe import INDICES, NIFTY_100  # noqa: E402

# Fixed window so results are reproducible (yfinance treats `end` as exclusive).
START, END = "2010-01-01", "2026-10-01"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    prices = download_prices(NIFTY_100, start=START, end=END)
    prices.to_parquet(PRICES_PATH, index=False)
    n = prices["ticker"].nunique()
    print(f"saved {len(prices):,} rows for {n} stocks, "
          f"{prices['date'].min().date()} to {prices['date'].max().date()} -> {PRICES_PATH}")
    indices = download_indices(INDICES, start=START, end=END)
    indices.to_parquet(INDEX_PATH, index=False)
    print(f"saved indices: {sorted(indices['ticker'].unique())} -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
