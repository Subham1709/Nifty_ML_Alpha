"""Step 1: download daily prices for the Nifty 100 (and the Nifty 50 index).

Run from the repo root:  python scripts/01_download.py
Works the same in Google Colab:  !python scripts/01_download.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha.data import DATA_DIR, PRICES_PATH, download_prices  # noqa: E402
from alpha.universe import NIFTY_100  # noqa: E402


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    prices = download_prices(NIFTY_100, start="2010-01-01")
    prices.to_parquet(PRICES_PATH, index=False)
    n = prices["ticker"].nunique()
    print(f"saved {len(prices):,} rows for {n} stocks, "
          f"{prices['date'].min().date()} to {prices['date'].max().date()} -> {PRICES_PATH}")


if __name__ == "__main__":
    main()
