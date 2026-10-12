# Learning log

Notes written while building the project, in the order things were done. Each step has
**what**, **why**, and **what to say in an interview**. Read this before talking about
the project; you should be able to explain every line of it.

---

## Step 0. The big picture

**The question.** Can price and volume history tell us which Nifty 100 stocks will do
better than the others over the next month? And, more importantly, can we measure the
answer *without fooling ourselves*?

**The pipeline.**

```
prices -> features (known on day t) -> label (return t..t+21) -> model -> predictions
       -> validation (naive vs purged vs walk-forward) -> portfolio -> costs -> Sharpe
```

**Why this is a quant project and not just an ML project.** In ML the target is usually
accuracy. In quant research the target is *money after costs, out of sample*, and the
enemy is overfitting the past. Real equity signals are weak: an information coefficient
(IC, defined in step 6) of 0.02 to 0.05 is considered good. A model showing 0.25 is
almost certainly leaking.

**Link to my BTP.** In landslide mapping, a random train/test split put neighbouring
pixels on both sides, and nearby pixels look alike, so the model looked better than it
was (XGBoost: AUC 0.967 random split vs 0.851 on unseen regions). Spatial-block CV fixed
that. In finance, neighbouring *days* look alike for the same reason. Purged CV is the
time-series version of spatial-block CV.

---

## Step 1. Universe and data (`src/alpha/universe.py`, `src/alpha/data.py`)

**What.** Daily open/high/low/close/adjusted close/volume for the Nifty 100 from Yahoo
Finance, 2010 onwards, stored as one long table (`date, ticker, ...`).

**Why these choices.**
- *Nifty 100* rather than Nifty 500: large, liquid stocks, so trading costs are realistic
  and the data is clean. It is also the universe a real fund could actually trade.
- *Adjusted close* for returns: it corrects for splits, bonuses and dividends. A 1:1 bonus
  issue halves the raw price; without adjustment that looks like a 50% crash.
- *Long table*: "rank all stocks on each date" becomes `groupby("date")`.

**The known flaw: survivorship bias.** We use *today's* Nifty 100 list. Stocks that were
in the index in 2012 but later fell out (often because they performed badly) are missing,
and recent joiners are there *because* they did well. That biases backtests upwards,
especially for momentum. The proper fix is point-in-time index membership from NSE
archives. Saying this unprompted in an interview is a strong signal.

**Interview lines.**
- "What's survivorship bias and how did you handle it?" Explain the above, say you used
  current constituents as a first pass, and how you'd fix it.
- "Why adjusted prices?" Corporate actions.

---

## Step 2. Features (`src/alpha/features.py`)

**What.** 15 features per stock per day, all computed from data up to day t's close:

| Group | Features | Idea behind it |
|---|---|---|
| Momentum | `ret_21d`, `mom_63d`, `mom_126d`, `mom_12_1` | Winners keep winning for 3-12 months (Jegadeesh & Titman, 1993) |
| Reversal | `rev_5d` | Very short-term moves partly reverse (liquidity, overreaction) |
| Volatility | `vol_21d`, `vol_63d`, `vol_ratio`, `hl_range_21d` | Low-volatility stocks have historically had better risk-adjusted returns |
| Trend / price level | `dist_52w_high`, `ma_50_ratio`, `ma_200_ratio`, `rsi_14` | Anchoring near 52-week highs; trend-following |
| Liquidity / size | `log_dollar_vol`, `volume_surge` | Size effect; unusual volume often precedes news |

`mom_12_1` skips the latest month on purpose: the last month tends to reverse (that's
`rev_5d`'s territory), which would muddy the 12-month momentum signal.

**Cross-sectional ranking.** On every date, each feature is replaced by its percentile
rank among all stocks, minus 0.5. Why:
- We want *relative* information ("RELIANCE has higher momentum than 80% of stocks
  today"), because we will buy the best and short the worst on the same day.
- Raw levels drift with the market (in 2020 every stock's volatility jumped). Ranks don't.
- Ranks kill outliers, which matter a lot with noisy financial data.

**No look-ahead.** `tests/test_no_lookahead.py` triples all prices after a cut-off date
and checks that no feature before the cut-off changes. If a feature used `shift(-1)` or
a centred rolling window by mistake, this test would fail.

**Interview lines.**
- "What is look-ahead bias?" Using information at time t that was only available later.
  Explain how the test catches it.
- "Why rank features?" Stationarity across market regimes and robustness to outliers.

---

## Step 3. Labels (`src/alpha/labels.py`)

**What.** For stock i on day t: `fwd_ret = price[t+21] / price[t] - 1`, the next month's
return. The model's target is that return *ranked across stocks on day t*.

**Why rank the target too.** The whole market going up 5% in a month isn't something a
long-short portfolio earns or loses. Ranking removes it, so the model only learns which
stocks beat the others.

**Why 21 days.** Long enough that trading costs don't eat everything (monthly rebalancing
means about 12 rebalances a year), short enough to have many independent periods.

**`label_end`.** The date the label is finally known (t+21). This one column is what makes
purged CV possible (step 4).

**The crucial consequence.** Samples on consecutive days have labels that share 20 of
their 21 days. They are nearly the same number. Keep that in mind for step 4.

---

## Step 4. Validation: why naive K-fold leaks (`src/alpha/cv.py`)

**Naive K-fold** shuffles rows into 5 folds. The sample (RELIANCE, 3 March) lands in
training and (RELIANCE, 4 March) in testing. Their features are almost identical and
their labels share 20 of 21 days. A flexible model like LightGBM can memorise the
training sample and "predict" the test one. The test score measures memory.

**Purged K-fold** (López de Prado, *Advances in Financial Machine Learning*, ch. 7):
1. Split *time* into 5 contiguous blocks; each block is the test set once.
2. **Purge**: remove any training sample whose label window `[t, label_end]` overlaps the
   test block. Those samples have literally seen part of the test answers.
3. **Embargo**: also remove 21 trading days of training data right after the test block.
   Features like 12-month momentum change slowly, so samples just after the test block
   are still strongly correlated with it.

```
time ->   [ train ][purge][   TEST   ][embargo][ train ]
```

Purged K-fold still trains on data from the *future* of the test block (the blocks after
it). That's acceptable for measuring whether a signal exists, but not for a realistic
backtest. So we also run:

**Walk-forward** (`walk_forward`): to test year Y, train only on samples whose labels were
fully known before 1 January of year Y. Retrain every year. This is exactly how the model
would have been used in real life, so it's the number to quote for "what would this have
made".

**Tests.** `tests/test_cv.py` checks that no training label overlaps any test label, that
the embargo gap is empty, and that walk-forward never trains on an unrealised label.

**Interview lines.**
- "Why can't you use normal cross-validation on time series?" Overlapping labels plus
  serial correlation leak test information into training. Name purging and embargo.
- "Purged K-fold vs walk-forward?" Purged K-fold uses more data and is fine for signal
  detection; walk-forward is the honest simulation of live trading.
- Connect it to the BTP: "same problem, spatial instead of temporal autocorrelation."

---

## Step 5. The placebo test (`scripts/00_placebo_test.py`)

**What.** Before touching real data, run the whole pipeline on 40 fake stocks whose prices
are pure random walks (2011-2018). There is *nothing* to predict: the true IC is 0 and
the true Sharpe of every strategy is 0.

**Result** (LightGBM, long-short quintiles, net of 15 bps costs; full table in
`results/placebo/summary.md`):

| Validation | Mean IC | IC t-stat | Net Sharpe |
|---|---|---|---|
| Naive K-fold | **0.272** | 16.8 | **5.48** |
| Purged K-fold | -0.006 | -0.3 | -0.47 |
| Walk-forward | 0.001 | 0.1 | -0.14 |

Naive K-fold "finds" a Sharpe of 5.5 in pure noise. Purged K-fold and walk-forward
correctly find nothing. That one table is the whole argument for this project's
methodology, and it was produced on data where we *know* the right answer.

**A second lesson in the same table.** Ridge under walk-forward shows IC 0.037 with a
t-stat of 2.2 on the same noise (see the full table). It isn't leakage (walk-forward is
leak-free by construction and tested); it's luck. We ran six model/validation
combinations, and with that many tries one t-stat above 2 by chance is unsurprising.
This is the *multiple-testing* problem, and it is why López de Prado's Deflated Sharpe
Ratio exists: the more things you try, the higher the bar a result must clear.

**Interview lines.**
- "How do you know your backtest isn't overfit?" Describe the placebo test. Very few
  candidates have one.
- "What's a t-stat of 2 worth after trying many strategies?" Less than it looks:
  multiple testing.

---

## Step 6. Backtest and metrics (`src/alpha/backtest.py`)

**Portfolio.** Every 21 trading days, rank stocks by prediction. Long-short: buy the top
20% and short the bottom 20%, equal weights. Long-only: buy the top 20%, compared with an
equal-weight portfolio of all stocks. In India, shorting a stock beyond a day needs stock
futures; most Nifty 100 names have them.

**Why rebalance every 21 days.** Holding periods then don't overlap, and each period's
return equals the label we predicted.

**Costs.** 15 bps per unit of one-way turnover (STT, exchange charges, stamp duty,
brokerage, slippage, roughly). Turnover is the sum of absolute weight changes. A
long-short book that fully replaces both legs has turnover 4 (sell 1, buy 1 on each side),
so costs are 4 × 15 bps = 0.6% a month, about 7% a year. Many "great" signals die here.

**Metrics.**
- *IC*: Spearman rank correlation between predictions and realised returns across stocks
  on a date, averaged over rebalance dates. Measured only on non-overlapping dates so the
  t-stat isn't inflated.
- *Sharpe*: mean / std of period returns × √12 (12 periods a year).
- *Max drawdown*: worst peak-to-trough fall of the equity curve.
- *Turnover*: how much trading the strategy needs; drives costs.

**Interview lines.**
- "What IC is good?" 0.02-0.05 for monthly equity signals.
- "Your gross Sharpe is X, net is Y. Why the gap?" Turnover × cost per trade.
