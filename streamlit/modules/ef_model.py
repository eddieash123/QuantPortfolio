import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import requests

# Wikipedia URL for S&P 500 companies
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Pretend to be a browser to avoid 403 errors
headers = {"User-Agent": "Mozilla/5.0"}

# Get HTML content
html = requests.get(url, headers=headers).text

# Parse the first table on the page (S&P 500 constituents)
sp500_table = pd.read_html(html)[0]

# Get tickers as a list and fix any dots for Yahoo Finance
tickers = [t.replace(".", "-") for t in sp500_table["Symbol"].tolist()]

print(len(tickers), tickers[:10])  # Verify

#example universe, replace with s&p 500 once working
# tickers = [
#     "AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","JNJ","XOM","PG",
#     "HD","V","MA","LLY","AVGO","COST","MRK","PEP","KO","WMT"
# ]

prices = yf.download(tickers, start="2023-01-01", end="2025-01-01", auto_adjust=True, progress=False)
prices = prices.dropna(axis=1, how='all')  # remove tickers with no data


#download data
# prices = yf.download(tickers, start="2023-01-01", end="2025-01-01", auto_adjust=True, progress=False)
returns = prices.pct_change().dropna()

#build factors
#momentum - has the stock been in an upward or downward trend for the past year
momentum = prices.pct_change(252).shift(21).iloc[-1] #getting the most recent results shifted 21 trading days earlier (1 month)
#volatility - is this stock steady or rapidly changes prices
volatility = returns.rolling(60).std().iloc[-1] #getting most recent results of 60 day volatility
#downslide volatility - how bad does it get when it's at a low
downside_vol = returns.where(returns < 0).rolling(126, min_periods=1).std().iloc[-1] #checks only 126 days of neg returns, if theres at least 1 neg day

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(downside_vol)


#normalize factor scores
factors = pd.DataFrame({
    "Momentum": momentum,
    "LowVol": -volatility,        # negative because lower vol is better
    "Quality": -downside_vol      # lower downside risk = higher quality
})

factor_scores = factors.rank(pct=True) #rank factors from 0-1 to make the comparisons easier

#the weights for each factor
factor_weights = {
    "Momentum": 0.4,
    "LowVol": 0.3,
    "Quality": 0.3
}

#for each stock, multiply the weight by the score and add it together to get a final 0-1 Alpha score to use for comparison
alpha_score = sum(
    factor_scores[f] * w for f, w in factor_weights.items()
)

# print(alpha_score.head(30))

top_10 = alpha_score.sort_values(ascending=False).head(10).index.tolist()

portfolio_factors = factor_scores.loc[top_10].mean()
sp500_factors = factor_scores.mean()

#getting optimal weights
mu = returns[top_10].mean() * 252
Sigma = returns[top_10].cov() * 252

def objective(w):
    return -(w @ mu - 5 * (w @ Sigma @ w))

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds = [(0, 0.1)] * len(top_10)

w0 = np.ones(len(top_10)) / len(top_10)

res = minimize(objective, w0, bounds=bounds, constraints=constraints)
weights = pd.Series(res.x, index=top_10)

#backtest
portfolio_simple_returns = returns[top_10] @ weights
portfolio_returns = np.log(1 + portfolio_simple_returns)
sp500 = yf.download("^GSPC", start="2023-01-01", end="2025-01-01", auto_adjust=True, progress=False)
sp500_log_returns = np.log(sp500['Close'] / sp500['Close'].shift(1)).dropna()



#performance comparison
def performance_stats(r):
    return {
        "CAGR": (1 + r.mean())**252 - 1,
        "Volatility": r.std() * np.sqrt(252),
        "Sharpe": (r.mean() / r.std()) * np.sqrt(252),
        "Max Drawdown": (1 + r).cumprod().div((1 + r).cumprod().cummax()).min() - 1
    }

stats = pd.DataFrame({
    "Portfolio": performance_stats(portfolio_returns),
    "S&P 500": performance_stats(sp500_log_returns)
})

#plot results

# Align dates
# ===== PERFORMANCE COMPARISON: FACTOR PORTFOLIO VS S&P 500 =====

# Align dates
common_dates = portfolio_returns.index.intersection(sp500_log_returns.index)
portfolio_returns = portfolio_returns.loc[common_dates]
sp500_log_returns = sp500_log_returns.loc[common_dates]

# Convert log returns to cumulative growth of $1
portfolio_cum = np.exp(portfolio_returns.cumsum())
sp500_cum = np.exp(sp500_log_returns.cumsum())

# Plot
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum, label="Factor Portfolio", linewidth=2)
plt.plot(sp500_cum, label="S&P 500", linestyle="--")

plt.title("Factor Portfolio vs S&P 500 (Growth of $1)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.show()

# Sanity check
print("Final Portfolio Value:", portfolio_cum.iloc[-1])
print("Final S&P 500 Value:", sp500_cum.iloc[-1])
