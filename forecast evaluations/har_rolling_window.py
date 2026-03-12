import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

# --- File Loading ---
FULL_DATA_PATH = "../data/full_df.csv"

OUT_DIR = "../forecast evaluations/har_outputs"
OUT_PREDS_PATH = os.path.join(OUT_DIR, "har_family_rolling_oos_predictions.csv")
OUT_RMSE_PATH  = os.path.join(OUT_DIR, "har_family_rolling_oos_rmse.csv")

DATE_COL = "date"
LOG_RV_COL = "log_RV"
HORIZONS = [1, 3, 5, 7]
TARGET_COLS = ["y_h1", "y_h3", "y_h5", "y_h7"]
WINDOW_SIZE = 1095
REFIT_EVERY = 7
EVAL_START_DATE = pd.to_datetime("2024-06-29")

# Constructing feature columns for HAR
# Build full HAR feature frame
def build_har_family_frame(base_df):
    df = base_df[[DATE_COL, LOG_RV_COL] + TARGET_COLS].copy()

    # --- HAR-RV pieces (from log_RV); shifted to info up to t-1 ---
    df["har_d"] = df["log_RV"].shift(1)
    df["har_w"] = df["log_RV"].rolling(window=5).mean().shift(1)
    df["har_m"] = df["log_RV"].rolling(window=22).mean().shift(1)

    # --- Jump proxy + jump components ---
    # Estimating the jump proxy
    df["jump_proxy"] = np.maximum(df["log_RV"] - df["log_RV"].rolling(window=5).mean(), 0.0)
    # Jump regressors 
    df["jump_d"] = df["jump_proxy"].shift(1)
    df["jump_w"] = df["jump_proxy"].rolling(window=5).mean().shift(1)
    df["jump_m"] = df["jump_proxy"].rolling(window=22).mean().shift(1)

    # --- Target columns ---
    target_cols = ["y_h1", "y_h3", "y_h5", "y_h7"]

    # --- Drop rows missing any of the conditions ---
    required = [
        DATE_COL, LOG_RV_COL, "jump_proxy",
        "har_d", "har_w", "har_m",
        "jump_d", "jump_w", "jump_m",
    ] + TARGET_COLS

    df = df.dropna(subset=required).reset_index(drop=True)
    return df

# Rolling window helper function 
def compute_start_t(df):
    idxs = df.index[df[DATE_COL] >= EVAL_START_DATE]
    if len(idxs) == 0:
        raise ValueError("EVAL_START_DATE is after the last date in df.")
    eval_start_idx = int(idxs[0])
    return max(WINDOW_SIZE, eval_start_idx)

# Rolling window function
def rolling_eval_linear(df, model_name, feature_cols, h, start_t):
    target_col = f"y_h{h}"
    end_t = len(df) - 1

    rows = []
    model = None

    for t in range(start_t, end_t + 1):
        train_slice = df.iloc[t - WINDOW_SIZE : t]
        X_train = train_slice[feature_cols].to_numpy()
        y_train = train_slice[target_col].to_numpy()

        # Weekly refit 
        if model is None or ((t - start_t) % REFIT_EVERY == 0):
            model = LinearRegression()
            model.fit(X_train, y_train)

        X_test = df.iloc[t : t + 1][feature_cols].to_numpy()
        yhat = float(model.predict(X_test).reshape(-1)[0])
        ytrue = float(df.iloc[t][target_col])

        rows.append({
            "date": df.iloc[t][DATE_COL],
            "model": model_name,
            "h": h,
            "y_true": ytrue,
            "y_pred": yhat,
            "error": ytrue - yhat
        })

    out = pd.DataFrame(rows)
    rmse = float(np.sqrt(np.mean(out["error"] ** 2)))
    return out, rmse

# --- MAIN ---
os.makedirs(OUT_DIR, exist_ok=True)

full_df = pd.read_csv(FULL_DATA_PATH)
full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL], errors="raise")
full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)

if LOG_RV_COL not in full_df.columns:
    raise ValueError(f"Column '{LOG_RV_COL}' not found in full_df. Update LOG_RV_COL.")

for c in TARGET_COLS:
    if c not in full_df.columns:
        raise ValueError(f"Target column '{c}' not found in full_df.")

har_df = build_har_family_frame(full_df)
start_t = compute_start_t(har_df)

model_specs = {
    "HAR-RV":       ["har_d", "har_w", "har_m"],
    "HAR-RV-J":     ["har_d", "har_w", "har_m", "jump_d"],
    "HAR-RV-J-H":   ["har_d", "har_w", "har_m", "jump_d", "jump_w", "jump_m"],
}

all_preds = []
rmse_rows = []

for model_name, feat_cols in model_specs.items():
    for h in HORIZONS:
        pred_df, rmse_h = rolling_eval_linear(har_df, model_name, feat_cols, h, start_t)
        all_preds.append(pred_df)
        rmse_rows.append({"model": model_name, "h": h, "rmse": rmse_h})
        print(f"{model_name} | h={h} | RMSE={rmse_h:.6f}")

preds_all = pd.concat(all_preds, ignore_index=True)
rmse_df = pd.DataFrame(rmse_rows).sort_values(["model", "h"]).reset_index(drop=True)

preds_all.to_csv(OUT_PREDS_PATH, index=False)
rmse_df.to_csv(OUT_RMSE_PATH, index=False)

print("\nSaved predictions to:", OUT_PREDS_PATH)
print("Saved RMSE table to:", OUT_RMSE_PATH)
print("\nRMSE summary:")
print(rmse_df)

# RMSE summary:
#          model  h   rmse
# 0       HAR-RV  1 0.8359
# 1       HAR-RV  3 0.8098
# 2       HAR-RV  5 0.8387
# 3       HAR-RV  7 0.8500
# 4     HAR-RV-J  1 0.8138
# 5     HAR-RV-J  3 0.8075
# 6     HAR-RV-J  5 0.8367
# 7     HAR-RV-J  7 0.8497
# 8   HAR-RV-J-H  1 0.8004
# 9   HAR-RV-J-H  3 0.8047
# 10  HAR-RV-J-H  5 0.8275
# 11  HAR-RV-J-H  7 0.8556
