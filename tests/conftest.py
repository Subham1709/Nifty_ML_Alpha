import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def fake_prices() -> pd.DataFrame:
    """Random-walk prices for a few tickers. Used only to test code mechanics."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=600)
    rows = []
    for t in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(dates))))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "adj_close": close,
            "volume": rng.integers(1e5, 1e6, len(dates)).astype(float),
        }))
    return pd.concat(rows, ignore_index=True)
