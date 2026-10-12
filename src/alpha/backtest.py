"""Turn predictions into portfolios and measure them.

Rebalancing: every 21 trading days (the label horizon), so holding periods do not
overlap and each period's return is exactly the label we predicted.

Portfolios on each rebalance date:
- long_short: buy the top quintile by prediction, short the bottom quintile, equal
  weights (+1 gross on each side). In India this needs stock futures, which most
  Nifty 100 names have.
- long_only: buy the top quintile, compared against an equal-weight portfolio of all
  stocks (the "market" for this universe).

Costs: `cost_bps` per unit of one-way turnover. 15 bps is a rough all-in figure for
large-cap NSE delivery trades (STT, exchange charges, stamp duty, brokerage,
slippage). Sensitivity to this number is part of the results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252 / 21


def rebalance_dates(dates: pd.Series, every: int = 21) -> pd.DatetimeIndex:
    uniq = np.sort(pd.Series(dates).unique())
    return pd.DatetimeIndex(uniq[::every])


def information_coefficient(df: pd.DataFrame, pred_col: str = "pred") -> pd.Series:
    """Spearman rank correlation between prediction and realised return, per date."""
    d = df.dropna(subset=[pred_col])
    # skip stray dates where only a few stocks traded; no rebalance happens on them anyway
    d = d[d.groupby("date")["ticker"].transform("size") >= 10]
    return d.groupby("date").apply(
        lambda g: g[pred_col].rank().corr(g["fwd_ret"].rank()), include_groups=False
    )


def _weights(g: pd.DataFrame, pred_col: str, q: float) -> pd.DataFrame:
    n = len(g)
    k = max(1, int(round(n * q)))
    order = g[pred_col].rank(method="first")
    top, bottom = order > n - k, order <= k
    w = pd.DataFrame(index=g["ticker"].values)
    w["long_short"] = np.where(top, 1 / k, 0) - np.where(bottom, 1 / k, 0)
    w["long_only"] = np.where(top, 1 / k, 0)
    w["equal_weight"] = 1 / n
    return w


def run_backtest(df: pd.DataFrame, pred_col: str = "pred", q: float = 0.2,
                 cost_bps: float = 15.0, every: int = 21, min_stocks: int = 10) -> pd.DataFrame:
    """Per-period gross and net returns for each portfolio."""
    d = df.dropna(subset=[pred_col, "fwd_ret"])
    rows, prev = [], None
    for date in rebalance_dates(d["date"], every):
        g = d[d["date"] == date]
        if len(g) < min_stocks:
            continue
        w = _weights(g, pred_col, q)
        r = pd.Series(g["fwd_ret"].values, index=g["ticker"].values)
        if prev is None:
            turnover = w.abs().sum()
        else:
            aligned = w.join(prev, how="outer", rsuffix="_prev").fillna(0)
            turnover = pd.Series({c: (aligned[c] - aligned[f"{c}_prev"]).abs().sum() for c in w})
        gross = w.mul(r, axis=0).sum()
        net = gross - turnover * cost_bps / 1e4
        row = {"date": date}
        for c in w:
            row[f"{c}_gross"], row[f"{c}_net"], row[f"{c}_turnover"] = gross[c], net[c], turnover[c]
        rows.append(row)
        prev = w
    return pd.DataFrame(rows).set_index("date")


def performance(returns: pd.Series) -> dict:
    r = returns.dropna()
    ann_ret = (1 + r).prod() ** (PERIODS_PER_YEAR / len(r)) - 1
    ann_vol = r.std() * np.sqrt(PERIODS_PER_YEAR)
    wealth = (1 + r).cumprod()
    max_dd = (wealth / wealth.cummax() - 1).min()
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR),
        "max_drawdown": max_dd,
        "hit_rate": (r > 0).mean(),
        "periods": len(r),
    }
