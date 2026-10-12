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
`results/placebo/summary.md`, produced by the GitHub Actions run):

| Validation | Mean IC | IC t-stat | Net Sharpe |
|---|---|---|---|
| Naive K-fold | **0.270** | 16.7 | **5.05** |
| Purged K-fold | -0.007 | -0.4 | -0.32 |
| Walk-forward | -0.001 | -0.0 | -0.24 |

Naive K-fold "finds" a Sharpe of 5 in pure noise. Purged K-fold and walk-forward
correctly find nothing. That one table is the whole argument for this project's
methodology, and it was produced on data where we *know* the right answer.

**A second lesson in the same table.** Ridge under walk-forward shows IC 0.038 with a
t-stat of 2.3 on the same noise (see the full table). It isn't leakage (walk-forward is
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

---

## Step 7. Cleaning real data (`src/alpha/clean.py`, `scripts/01b_data_report.py`)

**What.** Every run writes `results/data_report.md`: rows removed, stocks per year, late
listings, every one-day move over 35%, and a survivorship check.

**What the report found.** 6 one-day moves over 35%. Each was checked by hand:

| Stock | Date | Move | Verdict |
|---|---|---|---|
| ADANIENT | 2015-06-03 | -38.7% | demerger of Adani Transmission: fixed |
| CANBK | 2017-10-25 | +38.7% | real: government bank recapitalisation plan, kept |
| PNB | 2017-10-25 | +46.2% | real: same news, kept |
| INDUSINDBK | 2020-03-26 | +44.7% | real: rebound after the COVID crash, kept |
| TATAMOTORS | 2025-10-14 | -40.2% | demerger of the commercial-vehicle business: fixed |
| VEDL | 2026-04-30 | -64.9% | demerger of Vedanta's businesses (inferred): fixed |

A demerger isn't a loss for shareholders: they receive shares of the new company. Yahoo
didn't adjust these three, so the data showed fake crashes. `neutralise_jumps` rescales
the adjusted price before each event so that day's return is zero. The bank moves are
real and stay in. *Never delete a big move just because it is big;* removing real crashes
makes a strategy look safer than it is.

**Missing symbols.** LTIM, PFC and TATAMOTORS failed in the first batch download. The
downloader now retries each failed ticker on its own, then under its new symbol if it was
renamed (`ALIASES` in `universe.py`, e.g. Tata Motors became TMPV after its demerger).
All 98 stocks now have data.

**Survivorship bias, measured.** From 2011:

| Series | Annual growth |
|---|---|
| Equal-weight, today's 98 constituents (total return) | 19.2% |
| Nifty 100 index (price only) | 9.1% |
| Nifty 50 index (price only) | 8.6% |

Owning *today's* Nifty 100 equally since 2011 grew about 10 percentage points a
year faster than the actual index. Part of that is equal weighting and dividends (the
index here is price-only), but much of it is hindsight: the list already contains the
winners. That's why the long-short result is the one to trust here. Both legs come from
the same biased list, so much of the bias cancels, whereas the long-only result is
inflated by it.

**Interview lines.**
- "How did you clean the data?" Duplicates, zero-volume days, unadjusted corporate
  actions found by a jump screen and verified by hand.
- "How big is survivorship bias in your data?" Give the number above.

---

## Step 8. Real results (`results/summary.md`)

98 Nifty 100 stocks, daily data 2010-01 to 2026-09, 336,189 stock-days. Long-short
quintiles rebalanced every 21 trading days, 15 bps costs.

| Validation | Model | Mean IC | Net Sharpe |
|---|---|---|---|
| Naive shuffled K-fold | ridge | 0.069 | 1.31 |
| Naive shuffled K-fold | lightgbm | 0.173 | 3.68 |
| Purged + embargoed K-fold | ridge | 0.047 | 0.89 |
| Purged + embargoed K-fold | lightgbm | 0.042 | 0.66 |
| Walk-forward | ridge | 0.048 | 1.07 |
| Walk-forward | lightgbm | 0.031 | 0.53 |

**Finding 1: naive K-fold inflates results, and far more for the flexible model.**
LightGBM's net Sharpe falls from 3.68 (shuffled K-fold) to
0.53 (walk-forward), 7x lower; its IC falls from
0.173 to 0.031. Ridge only falls from 1.31 to
1.07. A model with more capacity can memorise more of the leaked
neighbouring samples, so leakage hurts it most. This is the same pattern as the BTP,
where XGBoost looked best on a random split and worst on unseen regions.

**Finding 2: purged K-fold and walk-forward agree.** Purged LightGBM IC 0.042
vs walk-forward 0.031. Purging and embargo remove the leak; the remaining gap is
that purged K-fold also trains on future years.

**Finding 3: the simple model wins out of sample.** Walk-forward ridge: IC
0.048, net Sharpe 1.07. LightGBM: IC 0.031, net
Sharpe 0.53. With signals this weak and noisy, a linear model on ranked
features is hard to beat, and LightGBM also trades more (turnover 2.12 vs
1.68 per rebalance), so it pays more in costs. Reporting this, instead of
tuning LightGBM until it wins on the test years, is the honest result. Tuning on the
test period would be another form of leakage.

**Finding 4: the signal is small and unstable.** Walk-forward LightGBM IC was
positive in 8 of 12 years (`results/ic_by_year.md`). Costs matter:
the long-short Sharpe goes 0.85 → 0.53 → 0.21 at 0, 15 and
30 bps (`results/cost_sensitivity.md`).

**A lesson about reproducibility.** The first real run downloaded its own prices; the
final run used the snapshot now committed in `data/`, on a different GitHub runner.
Ridge barely moved (fourth decimal), but naive LightGBM's Sharpe went from 3.95 to 3.68.
The cause is either a few values Yahoo revised between downloads or floating-point
differences between machines; we can't tell which, because the first download wasn't
kept. Either way, trees split on thresholds, so a tiny change can flip a split, and a
leaky model that memorises noise is the most fragile of all. That's why the exact data
snapshot is now committed: every number in `results/` can be reproduced from it.

**What the model uses** (`results/feature_importance.md`): liquidity/size
(`log_dollar_vol`), 12-1 momentum and 3-month volatility carry the most weight, which
matches the size, momentum and low-volatility effects in the factor literature.

---

## Step 9. How to talk about this project

**Resume line.**
> Built a leak-free ML stock-selection pipeline on Nifty 100 (2010-2026, 98 stocks,
> LightGBM vs ridge); showed shuffled K-fold inflated LightGBM's net Sharpe 7x
> (3.7 vs 0.5) and validated purged/embargoed CV with a
> random-walk placebo test; a ridge baseline reached walk-forward IC 0.048 and
> net Sharpe 1.1 after costs.

**Two-minute version.** "I predicted which Nifty 100 stocks would beat the others over
the next month. The labels overlap, so shuffled cross-validation leaks: on pure random
walks it found a Sharpe of 5. I used purged and embargoed K-fold plus walk-forward,
checked them with that placebo test, and found the real out-of-sample Sharpe was 7
times lower than the naive one for LightGBM. A ridge baseline actually did better, which
taught me that with weak signals, validation and costs matter more than model
complexity. It's the same lesson as my BTP, where spatial-block CV exposed a random
split."

**Questions to be ready for.**
- Why is the IC so low? Monthly equity returns are mostly noise; 0.03-0.05 is normal,
  and it compounds across ~100 stocks and 12 rebalances a year.
- Why did LightGBM lose? Weak signal, more variance, more turnover. You'd need
  regularisation tuned with an *inner* purged CV, or ensembling with ridge.
- What would you do next? Point-in-time index membership (survivorship), the Deflated
  Sharpe Ratio for multiple testing, sector neutralisation, and fundamental features.

---

## Step 10. Limitations and next steps

- **Survivorship bias**: current constituents only (measured in step 7). Fix with NSE's
  historical index files.
- **Multiple testing**: six model/validation combinations were run. The Deflated Sharpe
  Ratio would adjust the headline numbers for that.
- **Costs** are a flat 15 bps; real costs depend on liquidity and trade size, and a
  long-short book via stock futures adds margin and roll costs.
- **No sector neutralisation**: part of the signal may be sector bets.
- **Price and volume only**: no fundamentals, news or ownership data.
