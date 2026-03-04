import pandas as pd
import numpy as np

# Load Binance 5-min BTC data
df = pd.read_csv("../data/btc_5min_binance_2021_2025.csv")
print(df.columns) #checking colnames for technical indicators creation
# Now we will compute technical indicators from the 5-min OHLCV data

df["date"] = pd.to_datetime(df["open_time_utc"])
df = df.sort_values("date")

# keep only needed columns
df = df[["date", "open", "high", "low", "close", "volume"]]

# EMA indicators
df["EMA10"] = df["close"].ewm(span=10).mean()
df["EMA30"] = df["close"].ewm(span=30).mean()
df["EMA200"] = df["close"].ewm(span=200).mean()

# RSI (14)
delta = df["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI14"] = 100 - (100 / (1 + rs))

# Momentum
df["MOM10"] = df["close"] - df["close"].shift(10)

# Rate of Change
df["ROC10"] = df["close"].pct_change(10)

# ATR (volatility)
high_low = df["high"] - df["low"]
high_close = (df["high"] - df["close"].shift()).abs()
low_close = (df["low"] - df["close"].shift()).abs()

tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

df["ATR14"] = tr.rolling(14).mean()

# Bollinger Band width
ma20 = df["close"].rolling(20).mean()
std20 = df["close"].rolling(20).std()

upper = ma20 + 2 * std20
lower = ma20 - 2 * std20

df["BB_WIDTH"] = (upper - lower) / ma20

# OBV (volume flow)
df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

# Stochastic Oscillator
low14 = df["low"].rolling(14).min()
high14 = df["high"].rolling(14).max()

df["STOCH_K"] = 100 * (df["close"] - low14) / (high14 - low14)


# Convert to daily frequency
df["day"] = df["date"].dt.date
daily = df.groupby("day").last()
daily.index = pd.to_datetime(daily.index)


# keep only indicators (drop the OHLCV)
features = daily[
[
"EMA10",
"EMA30",
"EMA200",
"RSI14",
"MOM10",
"ROC10",
"ATR14",
"BB_WIDTH",
"OBV",
"STOCH_K"
]
]

# Lag indicators (to avoid look-ahead bias). Just lagging by 1 day for simplicity, but could experiment with more lags.
features = features.shift(1) # # important # #

# Save output
features.to_csv("../data/btc_technical_indicators_daily_2021_2025.csv")

# sanity checks
print(features.head()) # Features head 
print(features.isna().sum()) # Missing values check
print(len(features)) # Number of days check
