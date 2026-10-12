"""Step 1b: check the downloaded data before modelling.  -> results/data_report.md"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from alpha.clean import clean_prices, suspicious_jumps  # noqa: E402
from alpha.data import load_prices  # noqa: E402
from alpha.universe import NIFTY_100  # noqa: E402


def main() -> None:
    raw = load_prices()
    prices = clean_prices(raw)
    per = prices.groupby("ticker").agg(first=("date", "min"), last=("date", "max"), rows=("date", "size"))
    missing = sorted(set(NIFTY_100) - set(per.index))
    jumps = suspicious_jumps(prices)
    by_year = prices.groupby(prices["date"].dt.year)["ticker"].nunique()

    lines = [
        "# Data report",
        "",
        f"- Raw rows: {len(raw):,}; after cleaning: {len(prices):,} "
        f"({len(raw) - len(prices):,} duplicate, zero-volume or bad-price rows removed)",
        f"- Stocks with data: {per.shape[0]} of {len(NIFTY_100)}",
        f"- Missing from Yahoo: {', '.join(missing) if missing else 'none'}",
        f"- Date range: {prices['date'].min().date()} to {prices['date'].max().date()}",
        "",
        "## Stocks with data, by year",
        "",
        by_year.rename("stocks").to_frame().T.to_markdown(),
        "",
        "## Late listings (first date after 2011)",
        "",
        per[per["first"] > "2011-01-01"].sort_values("first").to_markdown(),
        "",
        f"## One-day moves larger than 35% in adjusted close ({len(jumps)})",
        "",
        "`ret` is the adjusted return, `raw_ret` the unadjusted one. A big `raw_ret` with an",
        "equally big `ret` on a known split/bonus date means Yahoo failed to adjust.",
        "",
        jumps.to_markdown(index=False, floatfmt=".3f") if len(jumps) else "none",
        "",
    ]
    out = ROOT / "results" / "data_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
