"""Step 1b: check the downloaded data before modelling.  -> results/data_report.md"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from alpha.clean import REAL_MOVES, UNADJUSTED_ACTIONS, clean_prices, suspicious_jumps  # noqa: E402
from alpha.data import load_indices, load_prices  # noqa: E402
from alpha.universe import NIFTY_100  # noqa: E402


def survivorship_check(prices: pd.DataFrame, indices: pd.DataFrame, start: str = "2011-01-01") -> str:
    px = prices[prices["date"] >= start]
    daily = px.pivot(index="date", columns="ticker", values="adj_close").pct_change(fill_method=None)
    ew = daily.mean(axis=1).fillna(0)
    years = (ew.index[-1] - ew.index[0]).days / 365.25
    rows = [{"series": "equal-weight, today's constituents (total return)",
             "annual_growth": (1 + ew).prod() ** (1 / years) - 1}]
    for name, g in indices[indices["date"] >= start].groupby("ticker"):
        g = g.sort_values("date")
        y = (g["date"].iloc[-1] - g["date"].iloc[0]).days / 365.25
        rows.append({"series": f"{name} index (price only)",
                     "annual_growth": (g["close"].iloc[-1] / g["close"].iloc[0]) ** (1 / y) - 1})
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    raw = load_prices()
    unfixed = clean_prices(raw, fix_actions=False)
    prices = clean_prices(raw)
    per = prices.groupby("ticker").agg(first=("date", "min"), last=("date", "max"), rows=("date", "size"))
    missing = sorted(set(NIFTY_100) - set(per.index))
    jumps = suspicious_jumps(unfixed)
    keys = list(zip(jumps["ticker"], jumps["date"].dt.strftime("%Y-%m-%d")))
    jumps["status"] = [
        f"fixed: {UNADJUSTED_ACTIONS[k]}" if k in UNADJUSTED_ACTIONS
        else f"kept, real: {REAL_MOVES[k]}" if k in REAL_MOVES else "NOT REVIEWED"
        for k in keys
    ]
    jumps["date"] = jumps["date"].dt.date
    unreviewed = int((jumps["status"] == "NOT REVIEWED").sum())
    by_year = prices.groupby(prices["date"].dt.year)["ticker"].nunique()
    surv = survivorship_check(prices, load_indices())

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
        f"Unreviewed jumps: {unreviewed}",
        "",
        "## Survivorship check",
        "",
        "Annual growth of an equal-weight portfolio of *today's* Nifty 100 stocks against the",
        "actual Nifty 100 index over the same dates. Part of the gap is equal weighting and",
        "dividends (the index here is price-only, about 1-1.5% a year), and the rest is",
        "survivorship bias: the stock list already knows which companies would succeed.",
        "",
        surv,
        "",
    ]
    out = ROOT / "results" / "data_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
