"""Step 0: placebo test on fake prices that contain NO signal at all.

Every stock is a pure random walk, so the true best possible IC is 0 and the true
Sharpe of any strategy is 0. A validation method that reports anything clearly above
zero here is measuring leakage, not skill.

    python scripts/00_placebo_test.py      -> results/placebo/
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment", ROOT / "scripts" / "02_run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def random_walk_prices(n_stocks: int = 40, start: str = "2011-01-01", end: str = "2018-12-31",
                       seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    rows = []
    for i in range(n_stocks):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": f"FAKE{i:02d}", "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "adj_close": close,
            "volume": rng.integers(100_000, 1_000_000, len(dates)).astype(float),
        }))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    experiment.FIRST_TEST_YEAR = 2014
    summary = experiment.main(random_walk_prices(), out=ROOT / "results" / "placebo")
    print(summary[["validation", "model", "mean_ic", "long_short_net_sharpe"]].round(3))
