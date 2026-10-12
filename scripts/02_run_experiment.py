"""Step 2: train, validate three ways, backtest, and write results/.

    python scripts/02_run_experiment.py

Outputs
- results/summary.csv            IC and backtest stats for every (validation, model)
- results/summary.md             the same as a markdown table
- results/cumulative_returns.png long-short net equity curves, naive vs purged vs walk-forward
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from alpha.backtest import information_coefficient, performance, run_backtest  # noqa: E402
from alpha.cv import naive_kfold, purged_kfold, walk_forward  # noqa: E402
from alpha.data import load_prices  # noqa: E402
from alpha.features import FEATURES, build_features  # noqa: E402
from alpha.labels import HORIZON, build_labels  # noqa: E402
from alpha.models import oos_predictions  # noqa: E402

RESULTS = ROOT / "results"
FIRST_TEST_YEAR = 2015  # walk-forward: train on 2011-2014 first, then expand yearly


def load_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(prices)
    labels = build_labels(prices, HORIZON)
    df = feats.merge(labels, on=["date", "ticker"], how="inner")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def main(prices: pd.DataFrame | None = None, out: Path = RESULTS) -> pd.DataFrame:
    out.mkdir(exist_ok=True)
    prices = load_prices() if prices is None else prices
    df = load_dataset(prices)
    print(f"dataset: {len(df):,} stock-days, {df['ticker'].nunique()} stocks, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    schemes = {
        "naive_kfold": lambda: naive_kfold(df["date"], 5),
        "purged_kfold": lambda: purged_kfold(df["date"], df["label_end"], 5, embargo_days=HORIZON),
        "walk_forward": lambda: walk_forward(df["date"], df["label_end"], FIRST_TEST_YEAR),
    }
    rows, curves = [], {}
    for scheme, splits in schemes.items():
        for model in ["ridge", "lightgbm"]:
            df["pred"] = oos_predictions(df, FEATURES, splits(), model)
            ic = information_coefficient(df)
            bt = run_backtest(df)
            # IC on rebalance dates only, so its t-stat isn't inflated by overlapping labels
            ic_reb = ic.reindex(bt.index).dropna()
            row = {
                "validation": scheme, "model": model,
                "mean_ic": ic_reb.mean(),
                "ic_tstat": ic_reb.mean() / ic_reb.std() * np.sqrt(len(ic_reb)),
            }
            for port in ["long_short", "long_only"]:
                for kind in ["gross", "net"]:
                    p = performance(bt[f"{port}_{kind}"])
                    row[f"{port}_{kind}_sharpe"] = p["sharpe"]
                    row[f"{port}_{kind}_ann_return"] = p["ann_return"]
                row[f"{port}_max_dd"] = performance(bt[f"{port}_net"])["max_drawdown"]
                row[f"{port}_turnover"] = bt[f"{port}_turnover"].iloc[1:].mean()
            row["equal_weight_sharpe"] = performance(bt["equal_weight_gross"])["sharpe"]
            rows.append(row)
            curves[(scheme, model)] = (1 + bt["long_short_net"]).cumprod()
            print(f"  {scheme:13s} {model:9s} IC={row['mean_ic']:+.4f} "
                  f"L/S net Sharpe={row['long_short_net_sharpe']:+.2f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    cols = ["validation", "model", "mean_ic", "ic_tstat", "long_short_gross_sharpe",
            "long_short_net_sharpe", "long_short_net_ann_return", "long_short_max_dd",
            "long_short_turnover", "long_only_net_sharpe", "equal_weight_sharpe"]
    (out / "summary.md").write_text(summary[cols].to_markdown(index=False, floatfmt=".3f"))

    fig, ax = plt.subplots(figsize=(9, 5))
    for (scheme, model), c in curves.items():
        if model == "lightgbm":
            ax.plot(c.index, c.values, label=scheme.replace("_", " "))
    ax.set_yscale("log")
    ax.set_title("LightGBM long-short quintiles, net of 15 bps costs")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cumulative_returns.png", dpi=130)
    return summary


if __name__ == "__main__":
    main()
