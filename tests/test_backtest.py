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


def test_clean_prices_removes_bad_rows(fake_prices):
    from alpha.clean import clean_prices
    bad = fake_prices.copy()
    bad.loc[5, "volume"] = 0
    bad = pd.concat([bad, bad.iloc[[10]]])
    out = clean_prices(bad)
    assert len(out) == len(fake_prices) - 1
    assert not out.duplicated(["ticker", "date"]).any()


def test_neutralise_jumps_removes_fake_crash(fake_prices):
    from alpha.clean import neutralise_jumps
    df = fake_prices.copy()
    day = df["date"].unique()[300]
    hit = (df["ticker"] == "AAA") & (df["date"] >= day)
    df.loc[hit, ["close", "adj_close"]] *= 0.4          # fake 60% "crash" from a demerger
    fixed = neutralise_jumps(df, {("AAA", str(pd.Timestamp(day).date())): "test"})
    a = fixed[fixed.ticker == "AAA"].set_index("date")["adj_close"]
    orig = fake_prices[fake_prices.ticker == "AAA"].set_index("date")["adj_close"]
    # every daily return except the event day is unchanged, and the event day is 0
    r_fixed, r_orig = a.pct_change(), orig.pct_change()
    assert abs(r_fixed.loc[day]) < 1e-12
    np.testing.assert_allclose(r_fixed.drop(day).dropna(), r_orig.drop(day).dropna())
