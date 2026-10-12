"""Features on day t must not change when prices after day t change."""
import numpy as np

from alpha.features import FEATURES, compute_raw_features
from alpha.labels import build_labels


def test_features_ignore_future(fake_prices):
    cut = fake_prices["date"].unique()[400]
    before = compute_raw_features(fake_prices)
    shocked = fake_prices.copy()
    future = shocked["date"] > cut
    shocked.loc[future, ["close", "adj_close", "high", "low"]] *= 3.0
    shocked.loc[future, "volume"] *= 0.1
    after = compute_raw_features(shocked)
    past = before["date"] <= cut
    np.testing.assert_allclose(before.loc[past, FEATURES].values,
                               after.loc[past, FEATURES].values, equal_nan=True)


def test_label_is_forward_return(fake_prices):
    lab = build_labels(fake_prices, horizon=21)
    px = fake_prices[fake_prices.ticker == "AAA"].set_index("date")["adj_close"]
    row = lab[lab.ticker == "AAA"].iloc[10]
    expected = px.iloc[31] / px.iloc[10] - 1
    assert np.isclose(row["fwd_ret"], expected)
    assert row["label_end"] == px.index[31]
