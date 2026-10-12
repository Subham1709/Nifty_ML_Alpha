"""Models. LightGBM is the main one; ridge regression is the linear baseline.

The baseline answers an interview question before it is asked: "does the tree model
add anything over a linear combination of the same features?"
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,          # shallow trees: financial signals are weak and noisy
    min_child_samples=500,  # each leaf must average many stock-days
    subsample=0.7, subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    verbose=-1,
)


def make_model(name: str, seed: int = 0):
    if name == "lightgbm":
        return lgb.LGBMRegressor(random_state=seed, **LGB_PARAMS)
    if name == "ridge":
        return Ridge(alpha=10.0)
    raise ValueError(name)


def oos_predictions(df, features, splits, model_name: str) -> np.ndarray:
    """Out-of-sample predictions: each row is predicted by a model that never trained on it.

    Rows that no split tests (e.g. the first years in walk-forward) stay NaN.
    """
    pred = np.full(len(df), np.nan)
    X, y = df[features].to_numpy(), df["target"].to_numpy()
    for train_idx, test_idx in splits:
        m = make_model(model_name)
        m.fit(np.nan_to_num(X[train_idx]), y[train_idx])
        pred[test_idx] = m.predict(np.nan_to_num(X[test_idx]))
    return pred
