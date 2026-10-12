"""Step 2: train, validate three ways, backtest, and write results/.

    python scripts/02_run_experiment.py

Outputs
- results/summary.csv / summary.md   IC and backtest stats for every (validation, model)
- results/cost_sensitivity.md        walk-forward LightGBM at 0, 15 and 30 bps
- results/ic_by_year.md              walk-forward IC year by year
- results/feature_importance.md      LightGBM gain importance (walk-forward models)
- results/cumulative_returns.png     long-short net equity curves for the three schemes
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
from alpha.clean import clean_prices  # noqa: E402
from alpha.cv import naive_kfold, purged_kfold, walk_forward  # noqa: E402
from alpha.data import load_prices  # noqa: E402
from alpha.features import FEATURES, build_features  # noqa: E402
from alpha.labels import HORIZON, build_labels  # noqa: E402
from alpha.models import oos_predictions  # noqa: E402

RESULTS = ROOT / "results"
FIRST_TEST_YEAR = 2015  # walk-forward: first model trains on 2011-2014, then expands yearly
COST_BPS = 15.0


def load_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    feats = build_features(prices)
    labels = build_labels(prices, HORIZON)
    df = feats.merge(labels, on=["date", "ticker"], how="inner")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def evaluate(df: pd.DataFrame, cost_bps: float = COST_BPS) -> tuple[dict, pd.DataFrame, pd.Series]:
    ic = information_coefficient(df)
    bt = run_backtest(df, cost_bps=cost_bps)
    # IC on rebalance dates only, so its t-stat isn't inflated by overlapping labels
    ic_reb = ic.reindex(bt.index).dropna()
    row = {"mean_ic": ic_reb.mean(), "ic_tstat": ic_reb.mean() / ic_reb.std() * np.sqrt(len(ic_reb))}
    for port in ["long_short", "long_only"]:
        for kind in ["gross", "net"]:
            p = performance(bt[f"{port}_{kind}"])
            row[f"{port}_{kind}_sharpe"] = p["sharpe"]
            row[f"{port}_{kind}_ann_return"] = p["ann_return"]
        row[f"{port}_max_dd"] = performance(bt[f"{port}_net"])["max_drawdown"]
        row[f"{port}_turnover"] = bt[f"{port}_turnover"].iloc[1:].mean()
    ew = performance(bt["equal_weight_gross"])
    row["equal_weight_sharpe"], row["equal_weight_ann_return"] = ew["sharpe"], ew["ann_return"]
    row["test_start"], row["test_end"] = bt.index.min().date(), bt.index.max().date()
    return row, bt, ic


def main(prices: pd.DataFrame | None = None, out: Path = RESULTS) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    prices = clean_prices(load_prices() if prices is None else prices)
    df = load_dataset(prices)
    print(f"dataset: {len(df):,} stock-days, {df['ticker'].nunique()} stocks, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    schemes = {
        "naive_kfold": lambda: naive_kfold(df["date"], 5),
        "purged_kfold": lambda: purged_kfold(df["date"], df["label_end"], 5, embargo_days=HORIZON),
        "walk_forward": lambda: walk_forward(df["date"], df["label_end"], FIRST_TEST_YEAR),
    }
    rows, curves, importances = [], {}, []
    for scheme, splits in schemes.items():
        for model in ["ridge", "lightgbm"]:
            imp = importances if (scheme, model) == ("walk_forward", "lightgbm") else None
            df["pred"] = oos_predictions(df, FEATURES, splits(), model, importances=imp)
            row, bt, ic = evaluate(df)
            rows.append({"validation": scheme, "model": model, **row})
            curves[(scheme, model)] = (1 + bt["long_short_net"]).cumprod()
            print(f"  {scheme:13s} {model:9s} IC={row['mean_ic']:+.4f} "
                  f"L/S net Sharpe={row['long_short_net_sharpe']:+.2f}", flush=True)
            if (scheme, model) == ("walk_forward", "lightgbm"):
                wf_df, wf_ic, wf_bt = df.copy(), ic, bt

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    cols = ["validation", "model", "mean_ic", "ic_tstat", "long_short_gross_sharpe",
            "long_short_net_sharpe", "long_short_net_ann_return", "long_short_max_dd",
            "long_short_turnover", "long_only_net_sharpe", "equal_weight_sharpe"]
    (out / "summary.md").write_text(summary[cols].to_markdown(index=False, floatfmt=".3f") + "\n")

    # costs: how much of the walk-forward result survives realistic trading costs?
    cost_rows = []
    for bps in [0, 15, 30]:
        r, _, _ = evaluate(wf_df, cost_bps=bps)
        cost_rows.append({"cost_bps": bps, **{k: r[k] for k in [
            "long_short_net_sharpe", "long_short_net_ann_return",
            "long_only_net_sharpe", "long_only_net_ann_return"]}})
    (out / "cost_sensitivity.md").write_text(
        pd.DataFrame(cost_rows).to_markdown(index=False, floatfmt=".3f") + "\n")

    # stability: is the walk-forward IC positive most years, or driven by one lucky year?
    ic_reb = wf_ic.reindex(wf_bt.index).dropna()
    by_year = ic_reb.groupby(ic_reb.index.year).agg(["mean", "count"])
    by_year["ls_net_return"] = wf_bt["long_short_net"].groupby(wf_bt.index.year).apply(
        lambda r: (1 + r).prod() - 1)
    by_year.index.name = "year"
    by_year.columns = ["mean_ic", "rebalances", "long_short_net_return"]
    by_year["rebalances"] = by_year["rebalances"].astype(str)
    (out / "ic_by_year.md").write_text(by_year.to_markdown(floatfmt=".3f") + "\n")

    imp = pd.Series(np.mean(importances, axis=0), index=FEATURES).sort_values(ascending=False)
    (out / "feature_importance.md").write_text(
        imp.rename("gain_share").to_frame().to_markdown(floatfmt=".3f") + "\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    styles = {"naive_kfold": ("#c0392b", "naive shuffled K-fold"),
              "purged_kfold": ("#2471a3", "purged + embargoed K-fold"),
              "walk_forward": ("#1e8449", "walk-forward")}
    for scheme, (color, label) in styles.items():
        c = curves[(scheme, "lightgbm")]
        ax.plot(c.index, c.values, color=color, label=label, lw=1.8)
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"LightGBM long-short quintiles, net of {COST_BPS:.0f} bps costs")
    ax.set_ylabel("growth of 1 (log scale)")
    ax.legend(frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "cumulative_returns.png", dpi=130)
    return summary


if __name__ == "__main__":
    main()
