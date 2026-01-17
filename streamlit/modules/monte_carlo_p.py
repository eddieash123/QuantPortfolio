import numpy as np

def monte_carlo_portfolio(df, weights, num_simulations=10000, horizon=252, df_t=5):
    """
    Monte Carlo simulation using log returns + fat tails (Student-T distribution).

    Args:
        df: DataFrame of daily log returns
        weights: np.array of portfolio weights
        num_simulations: number of Monte Carlo runs
        horizon: number of days to simulate
        df_t: degrees of freedom for Student-T (fat tails)
    
    Returns:
        avg_cumulative_returns: average cumulative returns across simulations
        all_simulations: matrix of cumulative portfolio cumulative returns
    """

    # LOG RETURN STATS
    log_returns = df.values
    mu = log_returns.mean(axis=0)            # daily log-return mean
    cov = np.cov(log_returns.T)              # covariance of log-returns

    num_assets = log_returns.shape[1]
    all_simulations = np.zeros((num_simulations, horizon))

    # Cholesky for correlation structure
    L = np.linalg.cholesky(cov)

    for i in range(num_simulations):

        # ---- Generate correlated normal shocks ----
        z = np.random.normal(size=(horizon, num_assets))
        correlated_shocks = z @ L.T
        
        # ---- Apply Student-T scaling for fat tails ----
        chi2 = np.random.chisquare(df_t, horizon)
        t_scale = np.sqrt(df_t / chi2)[:, None]
        correlated_shocks = correlated_shocks * t_scale

        # ---- Simulated log returns ----
        simulated_log_returns = mu + correlated_shocks

        # ---- Portfolio log returns (stay in log space) ----
        portfolio_log_returns = simulated_log_returns @ weights

        # ---- Cumulative performance (from log returns) ----
        all_simulations[i] = np.exp(np.cumsum(portfolio_log_returns)) - 1

    avg_cumulative_returns = all_simulations.mean(axis=0)

    return avg_cumulative_returns, all_simulations
