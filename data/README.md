# Data

The snapshot used for every number in `results/`. Source: Yahoo Finance, downloaded by
`scripts/01_download.py` for 2010-01-01 to 2026-09-30. Prices belong to their providers;
they are included here for research and education so the results can be reproduced
exactly.

| File | Rows | Columns |
|---|---|---|
| `prices.parquet` | one per stock per trading day (98 Nifty 100 stocks) | `date, ticker, open, high, low, close, adj_close, volume` |
| `indices.parquet` | one per index per trading day (Nifty 100, Nifty 50) | `date, ticker, close` |

`close` is the raw closing price; `adj_close` is adjusted for splits, bonuses and
dividends. The raw file is exactly what Yahoo returned. Cleaning (duplicates, zero-volume
days and three unadjusted demergers) happens in code, in `src/alpha/clean.py`, so it is
visible and testable.

```python
import pandas as pd
prices = pd.read_parquet("data/prices.parquet")
prices[prices.ticker == "RELIANCE"].tail()
prices.to_csv("prices.csv", index=False)   # if you prefer CSV
```

To refresh the data with newer dates, edit `START, END` in `scripts/01_download.py` and
run it (or the **download-data** workflow in the Actions tab). Results will then change.
