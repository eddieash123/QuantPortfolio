# optomize_portfolio.py
import numpy as np
from scipy.optimize import minimize

RISK_FREE_RATE = 0.0413  # US 10-year Treasury yield at time of development

# -----------------------------
# Core functions
# -----------------------------
def neg_sharpe(weights, simple_returns, risk_free_rate=RISK_FREE_RATE):
    """
    Calculate the negative Sharpe ratio for a given set of weights.
    """
    weights = np.array(weights)
    portfolio_return = np.sum(simple_returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(simple_returns.cov() * 252, weights)))
    sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe  # negative because we will minimize


def optimize_portfolio(df, risk_free_rate=RISK_FREE_RATE, max_weight=0.3, min_weight=0.05):
    """
    Optimize the portfolio weights to maximize Sharpe ratio.
    Returns:
        optimum_weights: array of optimal weights
        sharpe_opt: Sharpe ratio of optimized portfolio
        sharpe_initial: Sharpe ratio of equal-weight portfolio
    """
    simple_returns = df.pct_change().dropna()
    num_assets = simple_returns.shape[1]
    initial_weights = [1 / num_assets] * num_assets
    bounds = [(min_weight, max_weight) for _ in range(num_assets)]
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    opt_results = minimize(
        neg_sharpe,
        initial_weights,
        args=(simple_returns, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimum_weights = opt_results.x
    sharpe_opt = -neg_sharpe(optimum_weights, simple_returns, risk_free_rate)
    sharpe_initial = -neg_sharpe(initial_weights, simple_returns, risk_free_rate)

    return optimum_weights, sharpe_opt, sharpe_initial
