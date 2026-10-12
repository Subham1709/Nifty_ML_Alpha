import numpy as np
import pandas as pd

from alpha.backtest import performance, run_backtest
from alpha.labels import build_labels


def test_perfect_foresight_beats_random(fake_prices):
    lab = build_labels(fake_prices, horizon=21)
    lab["pred"] = lab["fwd_ret"]                     # cheating predictor
    good = run_backtest(lab, q=0.2, cost_bps=0, min_stocks=5)
    lab["pred"] = np.random.default_rng(1).normal(size=len(lab))
    rand = run_backtest(lab, q=0.2, cost_bps=0, min_stocks=5)
    assert performance(good["long_short_gross"])["sharpe"] > 3
    assert abs(performance(rand["long_short_gross"])["sharpe"]) < 2


def test_costs_reduce_returns(fake_prices):
    lab = build_labels(fake_prices, horizon=21)
    lab["pred"] = np.random.default_rng(2).normal(size=len(lab))
    bt = run_backtest(lab, cost_bps=50, min_stocks=5)
    assert (bt["long_short_net"] <= bt["long_short_gross"]).all()
