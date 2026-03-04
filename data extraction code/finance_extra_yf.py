# Downloading extra daily "finance" predictors from Yahoo Finance via yfinance,
# convert to DAILY calendar frequency, forward-fill non-trading days, and compute log returns.

import pandas as pd
import numpy as np
import yfinance as yf

start_date = "2021-01-01"
end_date = "2025-12-31"

# Adding tickers here (Yahoo Finance symbols)
# These are common daily risk/macro proxies used in volatility papers.
tickers = {
    "DXY": "DX-Y.NYB",     # US Dollar Index
    "GOLD": "GC=F",        # Gold futures
    "SILVER": "SI=F",      # Silver futures
    "TLT": "TLT",          # 20y+ Treasury ETF (yield proxy)
    "IEF": "IEF",          # 7-10y Treasury ETF (yield proxy)
    "HYG": "HYG",          # High-yield credit ETF (credit risk proxy)
    "LQD": "LQD",          # Investment-grade credit ETF
    "EEM": "EEM",          # Emerging markets equity ETF (optional)
    # "USO": "USO",        # Oil ETF (optional, since we already have WTI from FRED)
    # "EURUSD": "EURUSD=X" # FX rate (optional, can see later)
}

# Downloading daily OHLCV
raw = yf.download(
    tickers=list(tickers.values()),
    start=start_date,
    end=pd.to_datetime(end_date) + pd.Timedelta(days=1),  # makes end inclusive
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    progress=False,
    threads=True,
)

# Building wide adjusted-close dataframe (one column per series)
frames = []
for name, tkr in tickers.items():
    if isinstance(raw.columns, pd.MultiIndex):
        d = raw[tkr].copy()
    else:
        d = raw.copy()

    d = d.reset_index().rename(columns={"Date": "date"})
    d["date"] = pd.to_datetime(d["date"])

    # Prefer Adj Close; if missing, fallback to Close
    if "Adj Close" in d.columns:
        s = d[["date", "Adj Close"]].rename(columns={"Adj Close": name})
    else:
        s = d[["date", "Close"]].rename(columns={"Close": name})

    frames.append(s.set_index("date"))

levels = pd.concat(frames, axis=1).sort_index()

#  DAILY calendar frequency + forward fill non-trading days (weekends/holidays)
full_idx = pd.date_range(levels.index.min(), levels.index.max(), freq="D")
levels_daily = levels.reindex(full_idx).ffill()

# Compute daily log returns
logret = np.log(levels_daily).diff().dropna()

# Quick sanity checks
# print("Levels (daily, ffilled) head:")
print(levels_daily.head(5)) # levels_daily head
print(logret.head(5)) # log returns head
print(logret.isna().sum()) # missing values per col in log returns

# Saves
# levels_daily.to_csv("../data/finance_extra_levels_daily_2021_2025.csv", index_label="date")
# logret.to_csv("../data/finance_extra_logret_2021_2025.csv", index_label="date")
