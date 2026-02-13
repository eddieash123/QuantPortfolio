from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import os
import numpy as np
import requests

DATA_DIR = "streamlit/data"
DATA_FILE = f"{DATA_DIR}/data.csv"
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 1, 1)


def load_data(tickers):
    """Load CSV or download fresh data if outdated."""
    if is_data_outdated():
        download_data(tickers)
    return load_csv()


def load_csv():
    """Load CSV, set Date as index, and convert all columns to numeric."""
    df = pd.read_csv(DATA_FILE, parse_dates=["Date"], index_col="Date")

    # Convert everything to float (important)
    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def download_data(tickers):
    """Download 10 years of adjusted close prices and save locally."""
    end = datetime.today()
    start = end - timedelta(days=10 * 365)

    # auto_adjust=True means "Close" is actually adjusted close
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True, #Close column becomes Adj Close for more accurate modeling
        progress=False
    )

    # Keep only the adjusted close column
    df = df[["Close"]]

    # Rename columns from a MultiIndex style to single tickers
    df.columns = tickers if len(tickers) > 1 else [tickers[0]]

    df.index.name = "Date"

    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save CSV WITH the index (important)
    df.to_csv(DATA_FILE, index=True)

    return df


def download_data2():
    #download S&P500
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers).text
    sp500_table = pd.read_html(html)[0]
    tickers = [t.replace(".", "-") for t in sp500_table["Symbol"].tolist()]
    
    # Download price data
    # prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Download pre-COVID period (2017-2019)
    pre_covid = yf.download(tickers, start="2017-01-01", end="2020-01-01", auto_adjust=True, progress=False)
    
    # Download post-COVID period (2022-2025) 
    post_covid = yf.download(tickers, start="2022-01-01", end="2025-01-01", auto_adjust=True, progress=False)

    prices = pd.concat(pre_covid,post_covid)
    
    if isinstance(prices.columns, pd.MultiIndex):
        volumes = prices['Volume'].dropna(axis=1, how='all')
        prices = prices['Close'].dropna(axis=1, how='all')
    else:
        prices = prices.dropna(axis=1, how='all')
        volumes = None
    
    return prices, volumes


def is_data_outdated():
    """Return True if CSV doesn’t exist or is older than 7 days."""
    if not os.path.exists(DATA_FILE):
        return True

    modified_time = datetime.fromtimestamp(os.path.getmtime(DATA_FILE))
    return (datetime.now() - modified_time).days >= 7


def log_returns(df):
    """Compute log returns from price data."""
    return np.log(1 + df.pct_change()).dropna()
