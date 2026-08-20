# The KNN Algorithm for Financial Time Series: a New Python Implementation

## What This Project Does

An investor holding a stock needs to answer one question: *when is the right moment to sell?*
Selling too early means leaving money on the table. Selling too late means losing gains already accumulated.

This project builds a tool to answer that question. By learning from the historical behaviour of a stock price, it forecasts the next 20 trading days and identifies which specific days offer the best return relative to the uncertainty of the prediction. The output is not just a forecast line, it is a ranked list of efficient sell moments, tailored to how much risk the investor is willing to accept.

## How It Works

The algorithm is called **TSFKNN** (Time Series Forecasting K-Nearest Neighbours) and is implemented entirely from scratch in Python, with no dependency on existing TSFKNN libraries.

The core idea is that financial markets tend to repeat patterns. Given the most recent price sequence of length `p` (the *query*), the algorithm scans the historical record and finds the `k` past periods that look most similar to today, using Euclidean distance on min-max scaled subsequences. For each match, the algorithm already knows what happened next, because that future is now part of the past. These candidate futures are then aggregated via ensemble learning to produce:

- a **point forecast** (expected price) for each of the next `h` days
- an **uncertainty band** (standard deviation) around each forecast

Key parameters:

| Parameter | Meaning |
|-----------|---------|
| `k` | Number of similar past patterns to retrieve |
| `p` | Length of the comparison window (how much history to match) |
| `h` | Forecast horizon in trading days |
| `t` | Total historical window used for search |

## Data

- **Source**: Yahoo Finance
- **Stock**: TEZNY (Terna S.p.A., Italian electricity transmission operator)
- **Period**: August 3, 2012 to January 31, 2023, daily frequency
- **Target variable**: Adjusted Close price, which corrects for dividends and other non-trading distortions
- **Size**: 2,641 observations

## Methodology

### Data Split

The model uses the last 3 years of data (`t = 756` trading days). January 2023 (20 trading days) is the test period. A 21-day validation set sits between training and test.

### Hyperparameter Grid Search

All combinations of `k` in {3, 5, 7} and `p` in {10, 20, 40, 60} are evaluated on the validation set, giving 12 configurations ranked by MSE.

### Two Ensemble Strategies

The same TSFKNN function is combined in two different ways:

**Strategy 1 (filtered):** Only the configurations that performed best on the validation set enter the ensemble. This is the classical approach.

**Strategy 2 (full):** All 12 configurations are included, without discarding any. This captures a wider range of possible futures and turns out to work better.

### From Forecast to Sell Decision: the Pareto Frontier

Each forecast day has an expected return and a standard deviation. These two quantities trade off against each other: days with higher expected return often come with higher uncertainty.

The Pareto frontier selects only the days where no other day is simultaneously better on both dimensions. These are the *efficient* sell moments: for a risk-averse investor, the earlier Pareto days are preferable; for a risk-seeking investor, the later ones offer higher potential upside. This transforms a statistical output into a concrete, investor-specific decision tool.

### Adaptive Re-Forecasting (Real-World Simulation)

In practice, an investor does not wait passively. After the first 7 actual prices become available, the model is updated: configurations whose forecasts were systematically too low are removed, and the ensemble is re-run on the remaining 13 days. This simulates how a real decision-maker would adjust the strategy as new information arrives.

## Results

| Approach | MSE |
|----------|-----|
| Strategy 1, filtered ensemble | 1.43 |
| Strategy 2, full ensemble | 0.80 |
| Strategy 2 + adaptive re-forecast (days 8-20) | **0.06** |

The full ensemble consistently outperforms the filtered one, showing that including more diverse forecasts improves accuracy even when some configurations performed poorly on validation.

Pareto-optimal sell days from the full ensemble: **1, 8, 9, 10, 14, 17**

After adaptive re-forecasting: **8, 9, 11, 12, 13, 16**

The re-forecast not only reduces error dramatically but also shifts the efficient frontier, changing which days are recommended. A risk-averse investor should target the earlier days; a risk-tolerant one can wait for the later ones.

## Tech Stack

- **Language**: Python 3
- **Libraries**: `numpy`, `pandas`, `matplotlib`

## Possible Extensions

- Replace standard deviation with asymmetric risk measures such as VaR, CVaR or EVaR, which penalise downside risk rather than symmetric deviation
- Extend the nearest-neighbour search across multiple correlated stocks, to find historical analogues that may not exist in a single company's record
- Benchmark against ARIMA models or recurrent neural networks such as LSTM

## Academic Context

Master's dissertation in Statistics and Data Science, University of Milano-Bicocca.

Full title: *"The KNN Algorithm for Financial Time Series: a New Python Implementation"*
