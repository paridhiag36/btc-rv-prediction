import pandas as pd


# Load predictor datasets
def load_df(path):
    df = pd.read_csv(path)
    # if date saved as index column
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # remove timezone if present
    df["date"] = df["date"].dt.tz_localize(None)
    return df

macro_core = load_df("../data/final_macro_df.csv")
finance_extra = load_df("../data/finance_extra_logret_2021_2025.csv")
fred = load_df("../data/fred_macro_features_daily_2021_2025.csv")
onchain = load_df("../data/onchain_features_blockchaincom_2021_2025.csv")
technical = load_df("../data/btc_technical_indicators_daily_2021_2025.csv")

# Merge predictors
predictors = (
    macro_core
    .merge(finance_extra, on="date", how="inner")
    .merge(fred, on="date", how="inner")
    .merge(onchain, on="date", how="inner")
    .merge(technical, on="date", how="inner")
)

print("Predictors shape:", predictors.shape)
print(predictors.head())

# Load target dataset
target = load_df("../data/btc_daily_rv_targets_2021_2025.csv")
print("Target shape:", target.shape)


# Merge predictors + target
final_df = predictors.merge(target, on="date", how="inner")
print("Final dataset shape:", final_df.shape)

# Sort by date
final_df = final_df.sort_values("date").reset_index(drop=True)


# Save modelling dataset
final_df.to_csv("../data/btc_volatility_model_dataset_2021_2025.csv", index=False)

print("\nFinal dataset saved.")
print(final_df.head())