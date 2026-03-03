import pandas as pd
import yfinance as yf 
import numpy as np

start_date = "2021-01-01"
end_date = "2025-12-31"  # yfinance treats end as inclusive for daily data in practice

tickers = {
    "FTSE100": "^FTSE",
    "DOW30": "^DJI",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "SSE": "000001.SS",  # Shanghai Composite
}

# 1) Download daily OHLCV for all tickers
raw = yf.download(
    tickers=list(tickers.values()),
    start=start_date,
    end=pd.to_datetime(end_date) + pd.Timedelta(days=1),  # make end inclusive
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    progress=False,
    threads=True,
)

# 2) Convert to a tidy "long" dataframe: date, series, ohlcv, adjclose
frames = []
for name, tkr in tickers.items():
    # Some tickers return single-index columns; others return multi-index
    if isinstance(raw.columns, pd.MultiIndex):
        d = raw[tkr].copy()
    else:
        d = raw.copy()

    d = d.reset_index().rename(columns={"Date": "date"})
    d["series"] = name
    d["ticker"] = tkr

    # Standardize column names
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    d = d.rename(columns=rename_map)

    # Keep consistent columns
    keep_cols = ["date", "series", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    d = d[[c for c in keep_cols if c in d.columns]]

    frames.append(d)

prices_long = pd.concat(frames, ignore_index=True).sort_values(["series", "date"])

# 3) Also create a "wide" dataframe of adjusted close (one column per series)
prices_wide_adj = (
    prices_long.pivot(index="date", columns="series", values="adj_close")
    .sort_index()
)

#print(prices_long.head())
#print(prices_wide_adj.head())

# 4) Optional: save
#prices_long.to_csv("../../data/macro_indices_prices_long_2021_2025.csv", index=False)
#prices_wide_adj.to_csv("../../data/macro_indices_adjclose_wide_2021_2025.csv")

# use prices wide as the csv 

# dealing with the empty rows due to no trade on sundays for some indices, use forward fill. 

macro_levels = prices_wide_adj.copy()      
macro_levels.index = pd.to_datetime(macro_levels.index)
macro_levels = macro_levels.sort_index()

full_idx = pd.date_range(macro_levels.index.min(), macro_levels.index.max(), freq="D")

macro_levels_ffill = macro_levels.reindex(full_idx).ffill()

macro_logret = np.log(macro_levels_ffill).diff().dropna()

# sanity checks
# print(macro_levels_ffill.head(10))
# print(macro_logret.head(10))

# Macro variable with log returns 
#macro_logret.to_csv("../../data/macro_logret_2021_2025.csv", index_label="date")

# combine the crude oil with the macro-dataset
oil_csv = pd.read_csv("../../data/DCOILWTICO.csv")
oil = oil_csv.copy()

oil = oil.rename(columns={"observation_date": "date", "DCOILWTICO": "oil_price"})
oil["date"] = pd.to_datetime(oil["date"])
oil["oil_price"] = pd.to_numeric(oil["oil_price"], errors="coerce")
oil = oil.sort_values("date")

# create missing non-trading days and fill them
full_idx = pd.date_range(oil["date"].min(), oil["date"].max(), freq="D")
oil_daily = oil.set_index("date").reindex(full_idx)
oil_daily.index.name = "date"
oil_daily["oil_price"] = oil_daily["oil_price"].ffill()
print(oil_daily.head(15))

# Convert oil daily price levels to daily log returns 
oil_logret = np.log(oil_daily["oil_price"]).diff().dropna().to_frame("OIL")
print(oil_logret.head(5))


final_macro_df = macro_logret.join(oil_logret, how = "left")
# Save merged predictors
final_macro_df.to_csv("../../data/final_macro_df.csv", index_label="date")

print(final_macro_df.head(10))