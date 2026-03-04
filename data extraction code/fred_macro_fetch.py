# Pulls daily macro series from FRED and aligns it to DAILY calendar frequency,
# forward-fills non-trading days, and computes changes + yield slope (as per variable)

import pandas as pd
import numpy as np
from pandas_datareader import data as pdr

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

# Core FRED daily series (high value for BTC volatility)
FRED_SERIES = {
    "DGS2": "DGS2",                 # 2Y Treasury yield (%)
    "DGS10": "DGS10",               # 10Y Treasury yield (%)
    "EFFR": "EFFR",                 # Effective Federal Funds Rate (%)
    "HY_OAS": "BAMLH0A0HYM2",        # ICE BofA US High Yield Credit SpreadOAS (%)
    # "IG_OAS": "BAMLC0A0CM",        # ICE BofA US Corporate OAS (%) is optional alternative
}

OUT_LEVELS = "../data/fred_macro_levels_daily_2021_2025.csv"
OUT_FEATURES = "../data/fred_macro_features_daily_2021_2025.csv"


def fetch_fred(series_map: dict, start: str, end: str) -> pd.DataFrame:
    "Download FRED series into a wide dataframe indexed by date."
    frames = []
    for col, fred_id in series_map.items():
        s = pdr.DataReader(fred_id, "fred", start, end)
        s = s.rename(columns={fred_id: col})
        frames.append(s)
    df = pd.concat(frames, axis=1).sort_index()
    return df


def to_daily_ffill(df: pd.DataFrame) -> pd.DataFrame:
    "Reindex to daily calendar frequency and forward-fill missing days."
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_idx)
    df.index.name = "date"
    df = df.ffill()
    return df


def build_features(levels_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build FRED features:
    - Δ yields/rates/spreads (today - yesterday)
    - yield slope (10Y - 2Y) as a level instead of difference (common macro proxy)
    """
    feat = pd.DataFrame(index=levels_daily.index)

    # Yield slope as level (common macro proxy)
    if {"DGS10", "DGS2"}.issubset(levels_daily.columns):
        feat["YC_SLOPE_10Y_2Y"] = levels_daily["DGS10"] - levels_daily["DGS2"]

    # Daily changes for other raw series
    for c in levels_daily.columns:
        feat[f"D_{c}"] = levels_daily[c].diff()

    # Drop first day (since diff creates NaNs)
    feat = feat.dropna()

    return feat


if __name__ == "__main__":
    # print("Fetching FRED series...")
    fred_raw = fetch_fred(FRED_SERIES, START_DATE, END_DATE)

    # Convert to numeric (FRED sometimes has '.' missing values)
    fred_raw = fred_raw.apply(pd.to_numeric, errors="coerce")

    print("Aligning to daily calendar + forward fill...")
    fred_levels_daily = to_daily_ffill(fred_raw)

    print("Building features (Δ + slope)...")
    fred_features = build_features(fred_levels_daily)

    # Quick sanity checks
    print(fred_levels_daily.head(5)) # Daily levels head
    print(fred_features.head(5)) # Fred features head
    print(fred_features.isna().sum()) # Missing values check

    # Save
    fred_levels_daily.to_csv(OUT_LEVELS, index_label="date")
    fred_features.to_csv(OUT_FEATURES, index_label="date")
