import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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

# Part 1: Merging the dataset 
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


# Save modelling dataset without any changes
final_df.to_csv("../data/btc_volatility_model_dataset_2021_2025.csv", index=False)

print("\nFinal dataset saved.")
print(final_df.head())

# Part 2: Lagging the X variables by lags 1-3 
target_cols = ["y_h1", 
               "y_h3",
               "y_h5",
               "y_h7"]

#  Identify feature columns = all non-target, non-date columns 
exclude_cols = ["date"] + ["RV"] + target_cols
feature_cols = [c for c in final_df.columns if c not in exclude_cols]

# Create lagged features (lags 1,2,3) 
lags = [1, 2, 3]

full_df = final_df.copy()

for lag in lags:
    lagged_block = full_df[feature_cols].shift(lag)
    lagged_block.columns = [f"{c}_lag{lag}" for c in feature_cols]
    full_df = pd.concat([full_df, lagged_block], axis=1)

# After lag, missing values for 5 jan, 6 jan, 7 jan. Data can only start from 8 jan 
full_df = full_df.dropna(subset=[f"{feature_cols[0]}_lag3"]).reset_index(drop=True)

# Drop raw RV since not relevant 
full_df = full_df.drop(columns = ["RV"])
print(full_df.head(5))
full_df.to_csv("../data/full_df.csv", index=False)

# Part 3: Create train test sets 

# Sort data again just in case
full_df = full_df.sort_values("date").reset_index(drop=True)

# defining lagged feature cols
lagged_feature_cols = [c for c in full_df.columns if "_lag" in c]

# drop rows with any missing lagged X or target
#full_df = full_df.dropna(subset=lagged_feature_cols + target_cols).reset_index(drop=True)
full_df = full_df.dropna().reset_index(drop=True)
# 70-30 split by time
cut = int(len(full_df) * 0.7)
train_df = full_df.iloc[:cut].copy() # correct logic but still contains present day values which wouldnt be observed on training/tetsing day so use X_train
test_df  = full_df.iloc[cut:].copy() # same as above

# Part 4: Split into X/y (for various prediction horizons: 1-day, 3-days, 5-days, and 7-days ahead))
X_train = train_df[lagged_feature_cols] 
X_test  = test_df[lagged_feature_cols] 

y_train_h1 = train_df["y_h1"]
y_test_h1  = test_df["y_h1"]

y_train_h3 = train_df["y_h3"]
y_test_h3  = test_df["y_h3"]

y_train_h5 = train_df["y_h5"]
y_test_h5  = test_df["y_h5"]

y_train_h7 = train_df["y_h7"]
y_test_h7  = test_df["y_h7"]

# optional: include date with X too, for tracking/alignment
X_train_with_date = train_df[["date"] + lagged_feature_cols].copy()
X_test_with_date  = test_df[["date"] + lagged_feature_cols].copy()

# Part 5: Standardize features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# print("Train date range:", train_df["date"].min(), "to", train_df["date"].max(), "rows:", len(train_df))
# Train date range: 2021-01-08 00:00:00 to 2024-06-28 00:00:00 rows: 1268

# print("Test  date range:", test_df["date"].min(),  "to", test_df["date"].max(),  "rows:", len(test_df))
# Test  date range: 2024-06-29 00:00:00 to 2025-12-24 00:00:00 rows: 544

# create train and test csv
train_df.to_csv("../data/train_dataset.csv", index=False)
test_df.to_csv("../data/test_dataset.csv", index=False)

X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)

X_train_with_date.to_csv("../data/X_train_with_date.csv", index=False)
X_test_with_date.to_csv("../data/X_test_with_date.csv", index=False)

y_train_h1.to_csv("../data/y_train_h1.csv", index=False)
y_test_h1.to_csv("../data/y_test_h1.csv", index=False)

y_train_h3.to_csv("../data/y_train_h3.csv", index=False)
y_test_h3.to_csv("../data/y_test_h3.csv", index=False)

y_train_h5.to_csv("../data/y_train_h5.csv", index=False)
y_test_h5.to_csv("../data/y_test_h5.csv", index=False)

y_train_h7.to_csv("../data/y_train_h7.csv", index=False)
y_test_h7.to_csv("../data/y_test_h7.csv", index=False)
