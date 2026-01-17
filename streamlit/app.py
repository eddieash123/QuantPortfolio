from matplotlib import pyplot as plt
import streamlit as st
from modules.data_loader import load_data, log_returns
import pandas as pd
from modules.optomize_p import optimize_portfolio
from modules.monte_carlo_p import monte_carlo_portfolio
from modules.hybrid_selection import select_hybrid_tickers

st.title("Stock Price Data")

# Use hybrid selection: clustering + factor scoring
st.write("Selecting tickers using hybrid method (clustering + factor scoring)...")
tickers, df = select_hybrid_tickers(n_clusters=7, per_cluster=2)
st.write(f"Selected tickers: {tickers}")
df_log = log_returns(df)

st.write("Portfolio adj close data preview:")
st.dataframe(df.head())
print(tickers)

st.write("Portfolio Optomizing:")
optimum_weights, sharpe_opt, sharpe_initial = optimize_portfolio(df)
st.write("Optimum weights:")
st.dataframe(pd.DataFrame({"Ticker": tickers, "Weight": optimum_weights}))
st.write("Sharpe ratio of optimized portfolio:", sharpe_opt)
st.write("Sharpe ratio of equal-weight portfolio:", sharpe_initial)

# Monte Carlo simulation
avg_returns, sims = monte_carlo_portfolio(df_log, optimum_weights, num_simulations=1000, horizon=252)

# Example: show expected 1-year return
st.write(f"Monte Carlo expected 1-year return: {avg_returns[-1]*100:.2f}%")
# Plot Monte Carlo simulations
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sims.T, color='lightgrey', alpha=0.1)  # all simulations
ax.plot(avg_returns, color='blue', lw=2, label='Average')
ax.set_xlabel('Days')
ax.set_ylabel('Cumulative Return')
ax.set_title('Monte Carlo Portfolio Simulation')
ax.legend()

# Display in Streamlit
st.pyplot(fig)