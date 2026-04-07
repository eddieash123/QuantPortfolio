# Project Overview

This project is a quantitative portfolio management system that combines:
- Factor-based stock selection
- Portfolio optimization
- Monte Carlo simulation

to construct and analyze diversified equity portfolios from the S&P 500 universe.

# Installation Instructions

## Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

## Install dependencies (from the root)
```bash
pip install -r py_libraries.txt
```

## Run the app (from the root)
```bash
streamlit run streamlit/app.py
```

# Architecture

## Core Components

1. Hybrid Selection Engine (hybrid_selection.py)
   - Loads S&P 500 price data via data_loader
   - Calculates multi-factor scores for each stock
   - Uses correlation-based clustering for diversification
   - Selects top performers from each cluster
   - Liquidity filter is disabled — all S&P 500 constituents are sufficiently liquid

2. Portfolio Optimizer (optomize_p.py)
   - Maximizes Sharpe ratio using scipy SLSQP optimization
   - Enforces position bounds (1-25% per stock)
   - Accounts for risk-free rate (4.13% US 10-year Treasury)

3. Monte Carlo Simulator (monte_carlo_p.py)
   - Simulates 1000 portfolio paths over 252 trading days
   - Uses log returns with Cholesky decomposition for correlated shocks
   - Applies Student-T scaling (df=5) for fat tails
   - Shrinks mu 50/50 toward S&P 500 long-run daily mean (0.04%/day)
   - Returns average cumulative return path and full simulation matrix

4. Data Management (data_loader.py)
   - Downloads S&P 500 constituents from Wikipedia
   - Downloads 2015-2025 adjusted close prices via yfinance
   - Caches data permanently to streamlit/data/sp500_prices.parquet
   - Delete the parquet file to force a fresh download
   - Provides log_returns() utility for Monte Carlo input

## Factor Model

### Three-Factor Scoring System

1. Momentum (40% weight)
   - 252-day price change, lagged 21 days
   - Captures trend-following behavior
   - Avoids short-term noise

2. Low Volatility (30% weight)
   - 60-day rolling standard deviation
   - Inverted (lower vol = higher score)
   - Reduces portfolio drawdowns

3. Quality (30% weight)
   - 126-day downside volatility
   - Measures worst-case risk
   - Filters out unstable stocks

### Why These Factors Work Together
- Momentum identifies what is trending and working right now
- Low volatility filters out the noisy speculative names from that momentum list, keeping only stocks that are trending steadily rather than spiking erratically
- Quality ensures the remaining stocks have no hidden blow-up risk by penalizing left-tail volatility specifically
- The result is a selection of steady, institutionally-owned, trending stocks — exactly the kind that compound well over time
- Without low volatility and quality, momentum alone would pick high-flying speculative stocks with large drawdown risk
- Without momentum, low volatility and quality alone would pick safe but stagnant stocks with limited upside

### Scoring Process
- Calculate raw factor values for all S&P 500 stocks
- Rank each factor from 0-1 (percentile ranking)
- Compute weighted composite alpha score
- Filter out stocks with insufficient data (NaN handling)

## Diversification Strategy

### Clustering Approach
1. Calculate correlation matrix from returns
2. Convert to distance matrix (1 - correlation)
3. Apply Agglomerative Clustering with precomputed distances
4. Create 10 clusters
5. Select top 2 stocks per cluster
6. Result: ~20 stocks across uncorrelated sectors

### Why This Works
- Prevents concentration in single sectors (e.g., all tech stocks)
- Maintains factor quality within each cluster
- Balances diversification with performance
- More robust than pure factor ranking

## Data Pipeline

### Historical Data
- Single continuous window: 2015-2025 (10 years)
- Covers multiple full market cycles: 2015-2016 correction, 2017-2019 bull, 2020 crash, 2021 recovery, 2022 bear, 2023-2024 bull
- Stored permanently in streamlit/data/sp500_prices.parquet

### Filtering Steps
1. Remove delisted/merged tickers (dropna on close prices)
2. Exclude stocks with insufficient history for factor calculations
3. Filter NaN factor scores before clustering

## Optimization Constraints

### Position Limits
- Minimum weight: 1% per stock
- Maximum weight: 25% per stock
- Total portfolio: 100% (fully invested)

### Objective Function
- Maximize: (Portfolio Return - Risk Free Rate) / Portfolio Volatility
- Method: SLSQP (Sequential Least Squares Programming)
- Risk-free rate: 4.13% (US 10-year Treasury)

## Performance Metrics

### Sharpe Ratio
- Annualized return / Annualized volatility
- Adjusted for 4.13% risk-free rate
- Compares optimized vs equal-weight portfolios

### Monte Carlo Outputs
- Expected 1-year return (mean path at day 252)
- Full simulation matrix (1000 x 252)
- Visualization: all paths in light grey, mean path in blue

## Technical Decisions

### Why a Continuous 2015-2025 Window?
- Covers a full decade including multiple bull and bear cycles
- 2022 bear market stress-tests momentum and volatility factors
- No splicing artifacts from combining non-contiguous date ranges
- Produces honest volatility and covariance estimates

### Why Permanent Cache?
- S&P 500 data (500 tickers, 10 years) is slow to download and subject to Yahoo Finance rate limits
- The 2015-2025 dataset is fixed and does not need periodic refresh
- Delete streamlit/data/sp500_prices.parquet to force a re-download

### Why Clustering Before Selection?
- Pure factor ranking often picks correlated stocks
- Clustering ensures sector diversification
- Reduces portfolio correlation risk
- More stable during market rotations

### Why Min/Max Weight Constraints?
- Unconstrained optimization concentrates in 2-3 stocks
- Forces diversification across all selected stocks
- Reduces single-stock risk
- More practical for real portfolios

### Why Shrink mu in Monte Carlo?
- Raw historical mean returns are the noisiest input in any portfolio model
- Factor-selected, optimizer-tilted portfolios produce inflated daily return estimates that compound to unrealistic annual projections
- mu is blended 50/50 between the raw historical mean and the S&P 500 long-run daily mean (~0.04%/day)
- Standard practice known as shrinkage estimation (related to James-Stein and Black-Litterman)
- Result: Monte Carlo expected returns land in a credible 15-20% range

### Why Disable the Liquidity Filter?
- All S&P 500 constituents already meet any reasonable daily dollar volume threshold
- The filter caused column misalignment between price and volume DataFrames, silently dropping most tickers before clustering
- Liquidity filtering adds value when expanding beyond S&P 500 to mid/small caps

## Expected Results

### Typical Portfolio
- ~20 stocks across 10 clusters
- Sharpe ratio: 1.0-1.5 optimized (vs 0.5-1.0 equal-weight)
- Expected return: 15-20% annually (Monte Carlo mean, shrinkage-adjusted)
- Max drawdown: 15-25%

### Diversification
- Tech: 2-3 stocks
- Healthcare: 1-2 stocks
- Financials: 1-2 stocks
- Consumer: 1-2 stocks
- Industrials: 1-2 stocks
- Other sectors: 2-3 stocks

## Dependencies

- yfinance — market data
- pandas / numpy — data manipulation
- scipy — portfolio optimization
- scikit-learn — agglomerative clustering
- matplotlib — Monte Carlo visualization
- streamlit — web UI
- requests / lxml — S&P 500 constituent scraping

## Future Enhancements

- Interactive Angular dashboard
- Backtesting engine with transaction costs
- Rebalancing scheduler
