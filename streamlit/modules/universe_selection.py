# universe_selection.py
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import zscore

def winsorize_series(s, lower=0.01, upper=0.99):
    low = s.quantile(lower)
    high = s.quantile(upper)
    return s.clip(lower=low, upper=high)

def compute_scores(prices, volumes, market_caps, lookbacks=None):
    # prices: DataFrame date x tickers (adj close)
    # volumes: DataFrame date x tickers (volume)
    # market_caps: Series ticker->marketcap (float)

    if lookbacks is None:
        lookbacks = {"mom": 252, "vol": 90, "liq": 20}

    returns = prices.pct_change().dropna()

    # Momentum (1y)
    mom = prices.iloc[-1] / prices.shift(lookbacks["mom"]).iloc[-1] - 1
    # Volatility (annualized)
    vol = returns.tail(lookbacks["vol"]).std() * np.sqrt(252)
    # Liquidity: average dollar volume (20d)
    avg_dollar_vol = (prices * volumes).tail(lookbacks["liq"]).mean()

    df = pd.DataFrame({
        "momentum": mom,
        "volatility": vol,
        "liquidity": avg_dollar_vol,
        "market_cap": market_caps
    }).dropna()

    # Hard filters
    min_mktcap = 2e9
    min_dollar_vol = 2e6
    df = df[(df["market_cap"] >= min_mktcap) & (df["liquidity"] >= min_dollar_vol)]

    # Winsorize
    for col in ["momentum", "volatility", "liquidity"]:
        df[col] = winsorize_series(df[col])

    # Z-score (higher better)
    df["z_mom"] = zscore(df["momentum"])
    df["z_vol"] = -zscore(df["volatility"])     # smaller vol -> better
    df["z_liq"] = zscore(df["liquidity"])

    # Composite score - weights can be tuned
    w_mom, w_vol, w_liq = 0.5, 0.2, 0.3
    df["score"] = w_mom * df["z_mom"] + w_vol * df["z_vol"] + w_liq * df["z_liq"]

    return df, returns

def select_top50_diversified(prices, volumes, market_caps, returns, n=50):
    # Compute raw scores
    feats, returns_full = compute_scores(prices, volumes, market_caps)
    candidates = feats.sort_values("score", ascending=False).copy()

    tickers = candidates.index.tolist()

    # Clustering on correlation distance
    # Build correlation on returns for tickers in feats
    ret_mat = returns_full[candidates.index].dropna(how='all').fillna(0)
    corr = ret_mat.corr().fillna(0)
    dist = 1 - corr.values

    # Use AgglomerativeClustering to form n clusters
    n_clusters = min(len(tickers), n)
    if n_clusters <= 1:
        return tickers[:n]
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
    labels = clustering.fit_predict(dist)

    selected = []
    for lab in range(n_clusters):
        cluster_tickers = [t for (t,l) in zip(candidates.index, labels) if l == lab]
        if not cluster_tickers:
            continue
        # pick top scorer in this cluster
        top = candidates.loc[cluster_tickers].sort_values("score", ascending=False).index[0]
        selected.append(top)

    # If clusters < n (unlikely), fill with next best
    if len(selected) < n:
        remaining = [t for t in candidates.index if t not in selected]
        selected += remaining[: (n - len(selected))]

    return selected[:n]
