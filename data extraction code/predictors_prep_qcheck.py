import pandas as pd

# Helpers
def load_with_date(path: str, name: str) -> pd.DataFrame:
    """
    Loads a CSV and standardizes date handling:
    - If 'date' exists as a column -> parse to datetime
    - Else assume first column is date-like (common when saved with index_label="date")
    Returns df with a 'date' column (datetime64[ns]) and sorted by date.
    """
    df = pd.read_csv(path)

    # If date is not a column, assume first column is the date index saved to CSV
    if "date" not in df.columns:
        first = df.columns[0]
        df = df.rename(columns={first: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"\n=== {name} ===")
    print("Path:", path)
    print("Shape:", df.shape)
    print("Date range:", df["date"].min(), "→", df["date"].max())
    print("Columns:", list(df.columns))
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values (top 15):")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    return df


# Load predictor datasets
PATHS = {
    "macro_core": "../data/final_macro_df.csv",
    "finance_extra": "../data/finance_extra_logret_2021_2025.csv",
    "fred_features": "../data/fred_macro_features_daily_2021_2025.csv",
    "onchain_features": "../data/onchain_features_blockchaincom_2021_2025.csv",
    "btc_tech_daily": "../data/btc_technical_indicators_daily_2021_2025.csv",
    # Later we can also add target file here:
    # "target_rv": "../data/btc_daily_rv_targets_2021_2025.csv",
}

dfs = {}
for k, p in PATHS.items():
    dfs[k] = load_with_date(p, k)


# Quick cross-check: do dates overlap?
common_min = max(df["date"].min() for df in dfs.values())
common_max = min(df["date"].max() for df in dfs.values())
print("\n=== OVERLAP WINDOW ACROSS ALL PREDICTORS ===")
print("Common date range:", common_min, "→", common_max)

# Print how many rows each dataset has inside the common window
print("\nRows within common window:")
for name, df in dfs.items():
    n = df[(df["date"] >= common_min) & (df["date"] <= common_max)].shape[0]
    print(f"{name}: {n}")