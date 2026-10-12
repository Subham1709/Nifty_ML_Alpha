# Data report

- Raw rows: 363,373; after cleaning: 362,943 (430 duplicate, zero-volume or bad-price rows removed)
- Stocks with data: 98 of 98
- Missing from Yahoo: none
- Date range: 2010-01-04 to 2026-09-30

## Stocks with data, by year

|        |   2010 |   2011 |   2012 |   2013 |   2014 |   2015 |   2016 |   2017 |   2018 |   2019 |   2020 |   2021 |   2022 |   2023 |   2024 |   2025 |   2026 |
|:-------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| stocks |     80 |     80 |     80 |     80 |     80 |     82 |     85 |     89 |     91 |     91 |     91 |     94 |     95 |     96 |     98 |     98 |     98 |

## Late listings (first date after 2011)

| ticker     | first               | last                |   rows |
|:-----------|:--------------------|:--------------------|-------:|
| ADANIENSOL | 2015-07-31 00:00:00 | 2026-09-30 00:00:00 |   2754 |
| INDIGO     | 2015-11-10 00:00:00 | 2026-09-30 00:00:00 |   2687 |
| LTIM       | 2016-07-21 00:00:00 | 2026-09-30 00:00:00 |   2514 |
| ICICIPRULI | 2016-09-29 00:00:00 | 2026-09-30 00:00:00 |   2470 |
| VBL        | 2016-11-08 00:00:00 | 2026-09-30 00:00:00 |   2445 |
| DMART      | 2017-03-21 00:00:00 | 2026-09-30 00:00:00 |   2354 |
| ICICIGI    | 2017-09-27 00:00:00 | 2026-09-30 00:00:00 |   2224 |
| SBILIFE    | 2017-10-03 00:00:00 | 2026-09-30 00:00:00 |   2221 |
| HDFCLIFE   | 2017-11-17 00:00:00 | 2026-09-30 00:00:00 |   2189 |
| HAL        | 2018-04-02 00:00:00 | 2026-09-30 00:00:00 |   2099 |
| ADANIGREEN | 2018-06-18 00:00:00 | 2026-09-30 00:00:00 |   2045 |
| IRFC       | 2021-01-29 00:00:00 | 2026-09-30 00:00:00 |   1399 |
| LODHA      | 2021-04-19 00:00:00 | 2026-09-30 00:00:00 |   1347 |
| ETERNAL    | 2021-07-23 00:00:00 | 2026-09-30 00:00:00 |   1279 |
| LICI       | 2022-05-17 00:00:00 | 2026-09-30 00:00:00 |   1080 |
| JIOFIN     | 2023-08-21 00:00:00 | 2026-09-30 00:00:00 |    766 |
| HYUNDAI    | 2024-10-22 00:00:00 | 2026-09-30 00:00:00 |    480 |
| SWIGGY     | 2024-11-13 00:00:00 | 2026-09-30 00:00:00 |    464 |

## One-day moves larger than 35% in adjusted close (6)

`ret` is the adjusted return, `raw_ret` the unadjusted one. A big `raw_ret` with an
equally big `ret` on a known split/bonus date means Yahoo failed to adjust.

| ticker     | date       |   close |   adj_close |    ret |   raw_ret | status                                                                           |
|:-----------|:-----------|--------:|------------:|-------:|----------:|:---------------------------------------------------------------------------------|
| ADANIENT   | 2015-06-03 |  59.803 |      56.319 | -0.387 |    -0.387 | fixed: demerger of Adani Transmission (2015)                                     |
| CANBK      | 2017-10-25 |  87.980 |      74.463 |  0.387 |     0.387 | kept, real: government's public-sector bank recapitalisation plan                |
| INDUSINDBK | 2020-03-26 | 435.900 |     420.270 |  0.447 |     0.447 | kept, real: rebound after the COVID crash                                        |
| PNB        | 2017-10-25 | 202.050 |     182.175 |  0.462 |     0.462 | kept, real: government's public-sector bank recapitalisation plan                |
| TATAMOTORS | 2025-10-14 | 395.450 |     392.197 | -0.402 |    -0.402 | fixed: demerger of the commercial-vehicle business (2025)                        |
| VEDL       | 2026-04-30 | 271.550 |     271.550 | -0.649 |    -0.649 | fixed: demerger of Vedanta's businesses (inferred from the -65% unadjusted drop) |

Unreviewed jumps: 0

## Survivorship check

Annual growth of an equal-weight portfolio of *today's* Nifty 100 stocks against the
actual Nifty 100 index over the same dates. Part of the gap is equal weighting and
dividends (the index here is price-only, about 1-1.5% a year), and the rest is
survivorship bias: the stock list already knows which companies would succeed.

| series                                            |   annual_growth |
|:--------------------------------------------------|----------------:|
| equal-weight, today's constituents (total return) |           0.192 |
| NIFTY100 index (price only)                       |           0.091 |
| NIFTY50 index (price only)                        |           0.086 |
