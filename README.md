# nifty-ml-alpha

Machine-learning return prediction on Nifty 100 stocks, with an honest answer to the
question every quant interviewer asks: *"how do you know your backtest isn't leaking?"*

The project trains a LightGBM model to rank stocks by their next-month return, then
evaluates it two ways:

1. **Naive (shuffled) K-fold cross-validation**, the default in most ML tutorials.
2. **Purged and embargoed K-fold** plus **walk-forward** validation (López de Prado, 2018),
   which remove the overlap between training labels and test labels.

The gap between the two is the point of the project. It is the finance version of the
random-split vs spatial-block result in my B.Tech project on landslide susceptibility,
where a random split made the model look far better than it really was.

> Work in progress. See [`docs/LEARNING_LOG.md`](docs/LEARNING_LOG.md) for the
> step-by-step notes on what was done and why.

## Layout

```
src/alpha/      library code (universe, data, features, labels, CV, backtest)
scripts/        runnable steps (01_download.py, ...)
tests/          checks for leakage and correctness
docs/           learning log
data/           downloaded prices (not committed)
results/        tables and figures
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/01_download.py      # daily OHLCV for the Nifty 100, 2010 onwards
```
