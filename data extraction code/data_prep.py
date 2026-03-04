import pandas as pd
import numpy as np

binance = pd.read_csv("../data/btc_5min_binance_2021_2025.csv")
# print(binance.head(10))
# print(binance.columns)

df = binance.copy()

# Choose the timestamp and price columns we will use
ts_col = "close_time_utc"   # end-of-5min-bar time (UTC)
px_col = "close"            # 5-min bar close price

df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
df[px_col] = pd.to_numeric(df[px_col], errors="coerce")


# Define the day (UTC)
df["date"] = df[ts_col].dt.floor("D")
# Log prices 
df["log_close"] = np.log(df[px_col].astype(float))

# Compute 5 minute log returns within each day
df["r_5m"] = df.groupby("date")["log_close"].diff()
 
# Daily realised variance: sum of squared intraday returns
daily_rv = (
    df.dropna(subset=["r_5m"])
      .groupby("date")["r_5m"]
      .apply(lambda x: np.sum(x.values ** 2))
      .reset_index(name="RV")
)

print(daily_rv.head(5))

# Do log transformation to ensure positive forecasts of realised volatility 
 
eps = 1e-12   

daily_rv["log_RV"] = np.log(daily_rv["RV"] + eps)

print(daily_rv[["date", "RV", "log_RV"]].head())