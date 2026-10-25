"""Nifty ML Alpha: interactive web app.

    streamlit run app/streamlit_app.py

Reads the out-of-sample predictions saved by app/precompute.py and rebuilds the
long-short portfolios live with src/alpha/backtest.py, so costs and quintile size can be
changed. No model is trained here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
sys.path.insert(0, str(ROOT / "src"))

from alpha.backtest import performance, run_backtest  # noqa: E402

REPO = "https://github.com/Subham1709/Nifty_ML_Alpha"
SCHEMES = {"naive_kfold": "Naive shuffled K-fold", "purged_kfold": "Purged + embargoed K-fold",
           "walk_forward": "Walk-forward"}
MODELS = {"lightgbm": "LightGBM", "ridge": "Ridge"}
COLORS = {"naive_kfold": "#d62728", "purged_kfold": "#1f77b4", "walk_forward": "#2ca02c"}

st.set_page_config(page_title="Nifty ML Alpha", page_icon="📈", layout="wide")


@st.cache_data
def load(tag):
    return (pd.read_parquet(APP / "assets" / f"{tag}_rebal.parquet"),
            pd.read_parquet(APP / "assets" / f"{tag}_ic.parquet"))


@st.cache_data
def backtest(tag, combo, q, cost):
    d, _ = load(tag)
    g = d[d[f"rebal|{combo}"]]
    return run_backtest(g, pred_col=combo, q=q, cost_bps=cost, every=1)


def compare(tag, model, q, cost, key):
    """Equity curves and a metrics table for the three validation schemes."""
    _, ic = load(tag)
    fig, rows = go.Figure(), {}
    for scheme, label in SCHEMES.items():
        c = f"{scheme}|{model}"
        bt = backtest(tag, c, q, cost)
        net = bt["long_short_net"]
        p = performance(net)
        rows[label] = {"mean IC": ic[c].reindex(bt.index).mean(), "net Sharpe": p["sharpe"],
                       "net annual return": p["ann_return"], "max drawdown": p["max_drawdown"],
                       "turnover / rebalance": bt["long_short_turnover"].mean(),
                       "rebalances": p["periods"]}
        fig.add_trace(go.Scatter(x=bt.index, y=(1 + net).cumprod(), name=label,
                                 line=dict(color=COLORS[scheme], width=2.5)))
    fig.add_hline(y=1, line_color="gray")
    fig.update_layout(title=f"{MODELS[model]} long-short, net of {cost:g} bps, growth of 1",
                      yaxis_type="log", yaxis_title="wealth (log scale)", height=460,
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch", key=key)
    df = pd.DataFrame(rows).T
    st.dataframe(df.style.format({"mean IC": "{:+.3f}", "net Sharpe": "{:+.2f}",
                                  "net annual return": "{:+.1%}", "max drawdown": "{:.1%}",
                                  "turnover / rebalance": "{:.2f}", "rebalances": "{:.0f}"}),
                 width="stretch")
    return df


def controls(key):
    c1, c2, c3 = st.columns(3)
    model = c1.radio("Model", list(MODELS), format_func=MODELS.get, horizontal=True,
                     key=f"model_{key}")
    cost = c2.slider("Trading cost (bps per unit turnover)", 0.0, 50.0, 15.0, 2.5,
                     key=f"cost_{key}", help="15 bps: rough all-in cost for large-cap NSE trades")
    q = c3.select_slider("Long and short each", [0.1, 0.2, 0.3], 0.2, key=f"q_{key}",
                         format_func=lambda x: f"top / bottom {x:.0%}")
    return model, cost, q


st.title("Nifty ML Alpha: how do you know your backtest isn't leaking?")
st.markdown(
    "LightGBM and ridge rank 98 Nifty 100 stocks by next-month (21 trading day) return from "
    "15 price and volume features, 2010 to 2026. The same predictions are validated three "
    "ways and turned into a long-short quintile portfolio rebalanced every 21 days. "
    f"Code: [GitHub]({REPO}).")

tab_leak, tab_placebo, tab_signal, tab_data = st.tabs(
    ["The leak", "Placebo test", "Is the signal real?", "Data and caveats"])

with tab_leak:
    st.markdown(
        "Labels on consecutive days share 20 of their 21 days. A **shuffled** split puts "
        "neighbours of every test day in the training set, so the model memorises them. "
        "**Purging** drops training rows whose label window overlaps the test fold, and the "
        "**embargo** drops 21 more days after it. **Walk-forward** retrains every January on "
        "labels already known and tests on that year.")
    model, cost, q = controls("leak")
    df = compare("real", model, q, cost, "leak")
    s = df["net Sharpe"]
    st.info(f"Naive K-fold reports a Sharpe of **{s.iloc[0]:.2f}**; walk-forward, the honest "
            f"number, is **{s.iloc[2]:.2f}**. Purged K-fold ({s.iloc[1]:.2f}) agrees with "
            "walk-forward, so purging and embargo remove the leak. Try ridge: the simpler "
            "model leaks much less.")

with tab_placebo:
    st.markdown(
        "The same pipeline on **40 fake stocks whose prices are pure random walks**. Nothing "
        "is predictable, so the true IC and Sharpe are exactly zero. A validation method "
        "that finds profit here is broken.")
    model, cost, q = controls("placebo")
    df = compare("placebo", model, q, cost, "placebo")
    st.info(f"Naive K-fold finds a Sharpe of **{df['net Sharpe'].iloc[0]:.2f}** in pure noise. "
            "The leak-free methods correctly find nothing.")

with tab_signal:
    model = st.radio("Model", list(MODELS), format_func=MODELS.get, horizontal=True,
                     key="model_signal")
    _, ic = load("real")
    c = f"walk_forward|{model}"
    bt = backtest("real", c, 0.2, 15.0)
    yearly = ic[c].reindex(bt.index).groupby(bt.index.year).mean()
    fig = go.Figure(go.Bar(x=yearly.index, y=yearly.values,
                           marker_color=np.where(yearly.values > 0, "#2ca02c", "#d62728")))
    fig.update_layout(title=f"Walk-forward {MODELS[model]}: mean IC by year (rank correlation "
                            "of prediction and next-month return)", yaxis_title="mean IC",
                      height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Positive in {(yearly > 0).sum()} of {len(yearly)} years. An IC of 0.03 to 0.05 "
               "is a small edge that comes and goes.")
    rows = {}
    for cost in (0, 5, 10, 15, 20, 30):
        bt_c = backtest("real", c, 0.2, float(cost))
        rows[f"{cost} bps"] = performance(bt_c["long_short_net"])["sharpe"]
    st.markdown(f"**Walk-forward {MODELS[model]} long-short Sharpe as costs rise**")
    st.dataframe(pd.DataFrame(rows, index=["net Sharpe"]).style.format("{:+.2f}"),
                 width="stretch")
    st.markdown("**LightGBM feature importance** (share of total gain, walk-forward models)")
    st.markdown((ROOT / "results" / "feature_importance.md").read_text())

with tab_data:
    st.markdown(f"""
- Daily OHLCV from Yahoo Finance for **today's** Nifty 100 constituents, 2010-01 to 2026-09.
  The exact snapshot is in the repo, so every number here can be reproduced.
- A jump screen flags every one-day move over 35%. Three unadjusted demergers (Adani
  Enterprises 2015, Tata Motors 2025, Vedanta 2026) are neutralised; real moves are kept.
- **Survivorship bias is measured, not ignored.** Equal-weighting today's constituents grew
  19.2% a year since 2011 against 9.1% for the actual Nifty 100 index (price only). That is
  why only long-short results are shown: both legs come from the same list.
- Short legs need stock futures, which most Nifty 100 names have. Costs are a flat
  bps charge on turnover; no market impact model.

Full write-up: [README]({REPO}#readme), [data report]({REPO}/blob/main/results/data_report.md).
""")
