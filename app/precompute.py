"""Save the out-of-sample predictions the web app needs, for real and placebo data.

    python app/precompute.py        # ~ the same time as the two experiment scripts

Training 6 (validation, model) combinations takes minutes, too slow for a web page, so
this runs them once with the same code as scripts/02_run_experiment.py and keeps:
- app/assets/{real,placebo}_rebal.parquet  one row per stock per rebalance date:
  date, ticker, fwd_ret, one prediction column per combination ("<validation>|<model>")
  and a flag per combination marking its own rebalance dates. The app rebuilds
  the portfolios from these, so costs and quintile size can be changed live.
- app/assets/{real,placebo}_ic.parquet     daily IC per combination (all dates).
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


experiment = _load("experiment", "02_run_experiment.py")
placebo = _load("placebo", "00_placebo_test.py")

from alpha.backtest import information_coefficient, rebalance_dates  # noqa: E402
from alpha.clean import clean_prices  # noqa: E402
from alpha.cv import naive_kfold, purged_kfold, walk_forward  # noqa: E402
from alpha.data import load_prices  # noqa: E402
from alpha.features import FEATURES  # noqa: E402
from alpha.labels import HORIZON  # noqa: E402
from alpha.models import oos_predictions  # noqa: E402

OUT = ROOT / "app" / "assets"


def run(prices, first_test_year, tag):
    df = experiment.load_dataset(clean_prices(prices))  # as in experiment.main
    schemes = {
        "naive_kfold": lambda: naive_kfold(df["date"], 5),
        "purged_kfold": lambda: purged_kfold(df["date"], df["label_end"], 5, embargo_days=HORIZON),
        "walk_forward": lambda: walk_forward(df["date"], df["label_end"], first_test_year),
    }
    ics = {}
    for scheme, splits in schemes.items():
        for model in ["ridge", "lightgbm"]:
            col = f"{scheme}|{model}"
            df[col] = oos_predictions(df, FEATURES, splits(), model)
            ics[col] = information_coefficient(df, col)
            print(tag, col, f"IC {ics[col].mean():+.4f}", flush=True)
    cols = [c for c in df if "|" in c]
    # rebalance dates are taken per combination in run_backtest (rows with a prediction);
    # keep every date that is a rebalance date for any of them
    keep, flags = set(), {}
    for c in cols:
        reb = rebalance_dates(df.loc[df[c].notna() & df["fwd_ret"].notna(), "date"])
        flags[f"rebal|{c}"] = df["date"].isin(reb)
        keep |= set(reb)
    df = df.assign(**flags)
    small = df[df["date"].isin(keep)][["date", "ticker", "fwd_ret", *cols, *flags]]
    OUT.mkdir(parents=True, exist_ok=True)
    small.to_parquet(OUT / f"{tag}_rebal.parquet", index=False)
    pd.DataFrame(ics).to_parquet(OUT / f"{tag}_ic.parquet")


if __name__ == "__main__":
    run(load_prices(), experiment.FIRST_TEST_YEAR, "real")
    run(placebo.random_walk_prices(), 2014, "placebo")
