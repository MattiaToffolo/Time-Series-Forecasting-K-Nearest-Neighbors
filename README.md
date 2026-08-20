# The KNN Algorithm for Financial Time Series: a New Python Implementation

## Overview

This project presents a custom Python implementation of the **Time Series Forecasting K-Nearest Neighbours (TSFKNN)** algorithm, built from scratch as a Master's dissertation at the University of Milano-Bicocca. The algorithm is applied to forecast the adjusted closing price of the **TEZNY** stock (Terna S.p.A.) and to support investment decision-making through Pareto-optimal sell timing.

## Problem Statement

Predicting future stock prices is a well-known challenge in quantitative finance. This project addresses the forecasting problem by assuming that historical patterns tend to repeat — leveraging the KNN paradigm adapted to time series data. The algorithm does not rely on any existing TSFKNN library: the entire procedure is implemented from scratch in Python.

## Algorithm: TSFKNN

The **TSFKNN** (Time Series Forecasting K-Nearest Neighbours) works as follows:

1. Extract the most recent subsequence of length `p` from the time series (the *query*).
2. Scan the historical window and find the `k` subsequences most similar to the query, using **Euclidean distance** on **min-max scaled** subsequences.
3. For each matching subsequence, retrieve the `h` contiguous values that follow it (*candidate futures*).
4. Aggregate the candidate futures to produce a point forecast (mean) and an uncertainty estimate (standard deviation).

Key parameters:

| Parameter | Description |
|-----------|-------------|
| `k` | Number of nearest neighbours |
| `p` | Length of the query subsequence |
| `h` | Forecast horizon |
| `t` | Length of the training window |

## Data

- **Source**: [Yahoo Finance](https://finance.yahoo.com/)
- **Stock**: TEZNY — Terna S.p.A., Italian electricity transmission grid operator
- **Period**: August 3, 2012 – January 31, 2023 (daily frequency)
- **Target variable**: Adjusted Close price (accounts for dividends and non-trading distortions)
- **Format**: CSV — 2,641 rows × 7 columns (`Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`)

## Methodology

### Train / Validation / Test Split

The last 3 years of data (`t = 756` trading days) were used for modelling. The forecast target is **January 2023** (`h = 20` trading days). A validation set of 21 days was used for hyperparameter selection.

### Hyperparameter Grid Search

A grid search over `k ∈ {3, 5, 7}` and `p ∈ {10, 20, 40, 60}` (12 configurations total) was evaluated on the validation set using **MSE** as the selection criterion.

### Ensemble Learning — Two Approaches

| Approach | Description |
|----------|-------------|
| **Approach 1** (filtered) | Ensemble over the best-performing hyperparameter configurations on the validation set |
| **Approach 2** (full) | Ensemble over all 12 configurations — no configuration is discarded |

### Pareto Frontier

For the winning approach, the **Pareto frontier** on the (expected return, standard deviation) plane is drawn across forecast horizons. Each point on the frontier represents an efficient sell moment: risk-averse investors should prefer early days; risk-seeking investors can wait for higher expected returns.

### Adaptive Re-Forecasting (Real-World Simulation)

To simulate a realistic investment scenario where only partial ground truth is progressively revealed, the ensemble is updated after observing the first 7 actual test values: configurations producing systematically low forecasts are dropped, and TSFKNN is re-run on the remaining horizon.

## Results

| Approach | MSE |
|----------|-----|
| Approach 1 — filtered ensemble | 1.43 |
| Approach 2 — full ensemble | 0.80 |
| Approach 2 + adaptive re-forecast (days 8–20) | **0.06** |

- **Pareto-optimal sell moments** (full ensemble, days 1–20): 1, 8, 9, 10, 14, 17
- **Pareto-optimal sell moments** after adaptive re-forecast: 8, 9, 11, 12, 13, 16

The full ensemble consistently outperforms the validation-filtered approach. The adaptive re-forecasting step dramatically improves accuracy once early observations are incorporated.

## Tech Stack

- **Language**: Python 3
- **Libraries**: `numpy`, `pandas`, `matplotlib`

## Possible Extensions

- Replace standard deviation with asymmetric risk measures (VaR, CVaR, EVaR) — more appropriate for financial loss quantification
- Extend the nearest-neighbour search to multiple correlated stocks for cross-company pattern matching
- Benchmark against classical time-series models (ARIMA) or deep learning approaches (LSTM, Transformer)

## Academic Context

Master's dissertation — *Scienze Statistiche ed Economiche* (Statistics and Data Science track), University of Milano-Bicocca.

Full title: *"The KNN Algorithm for Financial Time Series: a New Python Implementation"*
