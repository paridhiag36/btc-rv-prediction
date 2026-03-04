# Build daily realised variance (RV) from 5-min BTC prices, log-transform it, and create step-ahead targets (h = 1, 3, 5, 7 days) for forecasting.
import pandas as pd
import numpy as np

# Load 5-min Binance BTC data
binance = pd.read_csv("../data/btc_5min_binance_2021_2025.csv")
df = binance.copy()

# Choose the timestamp and price columns we will use
ts_col = "close_time_utc"   # end-of-5min-bar time (UTC)
px_col = "close"            # 5-min bar close price

# Parse + clean types
df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
df[px_col] = pd.to_numeric(df[px_col], errors="coerce")

# Drop rows with bad timestamps or prices (prevents silent issues later)
df = df.dropna(subset=[ts_col, px_col]).copy() # there were non actually
# Ensure time-sorted (important for diff within each day)
df = df.sort_values(ts_col)

# Define the day (UTC)
df["date"] = df[ts_col].dt.floor("D")
# Log prices
df["log_close"] = np.log(df[px_col].astype(float))
# Compute 5-min log returns WITHIN each day (no cross-midnight returns)
df["r_5m"] = df.groupby("date")["log_close"].diff()


# Daily realised variance (RV): sum of squared intraday returns
daily_rv = (
    df.dropna(subset=["r_5m"])
      .groupby("date")["r_5m"]
      .apply(lambda x: np.sum(x.values ** 2))
      .reset_index(name="RV")
)

# Make sure daily_rv is date-sorted before shifting horizons
daily_rv = daily_rv.sort_values("date").reset_index(drop=True)
print(daily_rv.head(5))


# Log transform RV (stabilizes variance; avoids negative volatility forecasts)
eps = 1e-12
daily_rv["log_RV"] = np.log(daily_rv["RV"] + eps)
print(daily_rv[["date", "RV", "log_RV"]].head())

# Create step-ahead forecasting targets
# At date t, y_h{h} = log_RV at date t+h (future realised volatility)
for h in [1, 3, 5, 7]:
    daily_rv[f"y_h{h}"] = daily_rv["log_RV"].shift(-h)

# Drop last few rows where future targets don't exist
daily_rv = daily_rv.dropna(subset=["y_h1", "y_h3", "y_h5", "y_h7"]).reset_index(drop=True)

print(daily_rv.head(5)) # Quick check
print(daily_rv.tail(5)) # Quick check
print(len(daily_rv)) # Length check 

# Save 
daily_rv.to_csv("../data/btc_daily_rv_targets_2021_2025.csv", index=False)