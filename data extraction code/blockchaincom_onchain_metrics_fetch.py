# Fetch daily BTC on-chain metrics from Blockchain.com Charts API
# Align to a daily calendar, forward-fill gaps, create log-levels + daily log changes, and save CSVs.

import pandas as pd
import numpy as np
import requests

START_DATE = "2021-01-01"
END_DATE   = "2025-12-31"

CHARTS = {
    "HASH_RATE":        "hash-rate",
    "DIFFICULTY":       "difficulty",
    "TX_COUNT":         "n-transactions",
    "UNIQUE_ADDR":      "n-unique-addresses",
    "FEES_USD":         "transaction-fees-usd",
    "MINERS_REV_USD":   "miners-revenue",
}

OUT_LEVELS   = "../data/onchain_levels_blockchaincom_2021_2025.csv"
OUT_FEATURES = "../data/onchain_features_blockchaincom_2021_2025.csv"


def fetch_chart_daily(chart_name: str) -> pd.Series:
    """
    Download a Blockchain.com chart as *daily* series (UTC), returned as a pandas Series indexed by date.
    """
    url = f"https://api.blockchain.info/charts/{chart_name}"
    params = {
        "timespan": "all",                 # fetch full history, we'll clip later
        "samplingInterval": "24hours",     # force daily points
        "format": "json",
    }
    print("GET", url, params)

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()

    vals = js.get("values", [])
    if not vals:
        raise ValueError(f"No data returned for chart={chart_name}")

    df = pd.DataFrame(vals)
    df["date"] = pd.to_datetime(df["x"], unit="s", utc=True).dt.tz_convert(None)

    s = df.set_index("date")["y"].sort_index()
    return s


def to_daily_calendar(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Reindex to a full daily calendar and forward-fill gaps.
    (Some charts may still have missing days; ffill keeps continuity.)
    """
    idx = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    out = df.reindex(idx)
    out.index.name = "date"
    return out.ffill()


if __name__ == "__main__":
    # Download each daily metric series and combine into one wide dataframe
    series_list = []
    for col, chart in CHARTS.items():
        s = fetch_chart_daily(chart).rename(col)
        series_list.append(s)

    levels = pd.concat(series_list, axis=1)

    # Clip to study window + align to full daily calendar
    levels = levels.loc[pd.to_datetime(START_DATE): pd.to_datetime(END_DATE)]
    levels_daily = to_daily_calendar(levels, START_DATE, END_DATE)

    # Simple transformations:
    #    - log levels 
    #    - daily log changes (approx growth rates)
    eps = 1e-12  # avoids log(0) issues for series that can hit 0
    log_levels  = np.log(levels_daily + eps)
    log_changes = log_levels.diff()

    features = pd.DataFrame({
        "LOG_HASH_RATE":        log_levels["HASH_RATE"], # Mining activity / network security
        "LOG_DIFFICULTY":       log_levels["DIFFICULTY"], # Mining competition
        "DLOG_HASH_RATE":       log_changes["HASH_RATE"], # Short-run changes in mining activity
        "DLOG_DIFFICULTY":      log_changes["DIFFICULTY"], # Changes in mining competition
        "DLOG_TX_COUNT":        log_changes["TX_COUNT"], # Blockchain demand proxy
        "DLOG_UNIQUE_ADDR":     log_changes["UNIQUE_ADDR"],  # Network participation
        "DLOG_FEES_USD":        log_changes["FEES_USD"],  # Network congestion
        "DLOG_MINERS_REV_USD":  log_changes["MINERS_REV_USD"], # Miner economics. Miner revenue reflects both block rewards and transaction fees.
    }).dropna()

    # Quick sanity checks
    print(levels_daily.head()) #On-chain levsl head
    print(features.head()) # On-chain features head
    print(levels_daily.isna().sum()) # Missing values check

    # Save
    levels_daily.to_csv(OUT_LEVELS, index_label="date")
    features.to_csv(OUT_FEATURES, index_label="date")