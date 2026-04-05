import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from modules.data_loader import download_data2

def select_hybrid_tickers(n_clusters, per_cluster, min_dollar_vol=2e6):
    """
    Hybrid selection: cluster S&P 500, then pick top N from each cluster using factor scores.
    Returns: list of selected tickers and their price data
    """

    prices = download_data2()

    returns = prices.pct_change().dropna()
    
    # Calculate factor scores
    momentum = prices.pct_change(252).shift(21).iloc[-1]
    volatility = returns.rolling(60).std().iloc[-1]
    downside_vol = returns.where(returns < 0).rolling(126, min_periods=1).std().iloc[-1]
    
    factors = pd.DataFrame({
        "Momentum": momentum,
        "LowVol": -volatility,
        "Quality": -downside_vol
    }).dropna()
    
    # Filter to only valid tickers
    valid_tickers = factors.index
    prices = prices[valid_tickers]
    returns = returns[valid_tickers]
    
    # Liquidity filter on valid tickers (for non s&p 500 universe, can skip for s&p 500)
    # if volumes is not None:
    #     avg_dollar_vol = (prices * volumes[valid_tickers]).tail(20).mean()
    #     liquid_tickers = avg_dollar_vol[avg_dollar_vol >= min_dollar_vol].index
    #     prices = prices[liquid_tickers]
    #     returns = returns[liquid_tickers]
    #     factors = factors.loc[liquid_tickers]
    
    # Cluster on filtered data
    corr = returns.corr().fillna(0)
    dist = 1 - corr.values
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
    labels = clustering.fit_predict(dist)
    
    # Calculate final scores
    factor_scores = factors.rank(pct=True)
    factor_weights = {"Momentum": 0.4, "LowVol": 0.3, "Quality": 0.3}
    alpha_score = sum(factor_scores[f] * w for f, w in factor_weights.items())
    
    # Select top N from each cluster
    tickers = []
    for cluster_id in range(n_clusters):
        cluster_tickers = [t for t, l in zip(prices.columns, labels) if l == cluster_id]
        if cluster_tickers:
            cluster_scores = alpha_score[cluster_tickers].sort_values(ascending=False)
            tickers.extend(cluster_scores.head(per_cluster).index.tolist())
    
    return tickers, prices[tickers]
    # final_selected = [t for t in selected if t in prices.columns]
    # return final_selected, prices[final_selected]
