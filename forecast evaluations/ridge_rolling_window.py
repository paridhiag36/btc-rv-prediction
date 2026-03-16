# Load 
import os
import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ---------- CONFIG ----------
FULL_DATA_PATH = "../data/full_df.csv"
RIDGE_PARAM_PATH = "../models/ridge_tuned_hyperparams.csv"

OUT_DIR = "../forecast evaluations/ridge_outputs"
OUT_PREDS_PATH = os.path.join(OUT_DIR, "ridge_rolling_oos_predictions.csv")
OUT_RMSE_PATH = os.path.join(OUT_DIR, "ridge_rolling_oos_summary.csv")

DATE_COL = "date"
TARGET_COLS = ["y_h1", "y_h3", "y_h5", "y_h7"]
HORIZONS = [1, 3, 5, 7]

WINDOW_SIZE = 1095
REFIT_EVERY = 7
EVAL_START_DATE = pd.to_datetime("2024-06-29")

# ---------- HELPERS ----------
def get_feature_cols(df):
    return [c for c in df.columns if c not in ([DATE_COL] + TARGET_COLS)]

def compute_start_t(df):
    idxs = df.index[df[DATE_COL] >= EVAL_START_DATE]
    if len(idxs) == 0:
        raise ValueError("EVAL_START_DATE is after the last date in df.")
    eval_start_idx = int(idxs[0])
    return max(WINDOW_SIZE, eval_start_idx)

def load_ridge_alphas(csv_path):
    df = pd.read_csv(csv_path).copy()

    required_cols = ["horizon", "ridge_alpha"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    alpha_map = {}
    for _, row in df.iterrows():
        target = row["horizon"]
        alpha_map[target] = float(row["ridge_alpha"])

    return alpha_map

def fit_ridge_on_window(df_window, feature_cols, target_col, alpha):
    X_train = df_window[feature_cols]
    y_train = df_window[target_col].to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = Ridge(alpha=alpha)
    model.fit(X_train_s, y_train)

    return model, scaler

def rolling_eval_ridge_one_h(df, feature_cols, h, alpha, start_t):
    target_col = f"y_h{h}"
    end_t = len(df) - 1

    current_model = None
    current_scaler = None
    rows = []

    for t in range(start_t, end_t + 1):
        # training window = [t-WINDOW_SIZE, ..., t-1]
        train_slice = df.iloc[t - WINDOW_SIZE : t].copy()

        # weekly refit
        if current_model is None or ((t - start_t) % REFIT_EVERY == 0):
            current_model, current_scaler = fit_ridge_on_window(
                df_window=train_slice,
                feature_cols=feature_cols,
                target_col=target_col,
                alpha=alpha
            )

        # predict row t
        X_test = df.iloc[t : t + 1][feature_cols]
        X_test_s = current_scaler.transform(X_test)

        yhat = float(current_model.predict(X_test_s)[0])
        ytrue = float(df.iloc[t][target_col])

        rows.append({
            "date": df.iloc[t][DATE_COL],
            "model": "RIDGE",
            "h": h,
            "y_true": ytrue,
            "y_pred": yhat,
            "error": ytrue - yhat
        })

    out = pd.DataFrame(rows)
    rmse = float(np.sqrt(np.mean(out["error"] ** 2)))
    r2 = float(r2_score(out["y_true"], out["y_pred"]))
    return out, rmse, r2

# ---------- MAIN ----------
os.makedirs(OUT_DIR, exist_ok=True)

full_df = pd.read_csv(FULL_DATA_PATH)
full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL], errors="raise")
full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)

for c in TARGET_COLS:
    if c not in full_df.columns:
        raise ValueError(f"Target column '{c}' not found in full_df.")

feature_cols = get_feature_cols(full_df)
start_t = compute_start_t(full_df)

ridge_alphas = load_ridge_alphas(RIDGE_PARAM_PATH)

all_preds = []
summary_rows = []

for h in HORIZONS:
    target_col = f"y_h{h}"
    if target_col not in ridge_alphas:
        raise ValueError(f"Missing tuned alpha for {target_col} in {RIDGE_PARAM_PATH}")

    alpha = ridge_alphas[target_col]

    print(f"Running RIDGE rolling eval for h={h} with alpha={alpha:.6f}")
    pred_df, rmse_h, r2_h = rolling_eval_ridge_one_h(
        df=full_df,
        feature_cols=feature_cols,
        h=h,
        alpha=alpha,
        start_t=start_t
    )

    all_preds.append(pred_df)
    summary_rows.append({
        "model": "RIDGE",
        "h": h,
        "rmse": rmse_h,
        "r2": r2_h,
        "alpha": alpha
    })

    print(f"h={h} | RMSE={rmse_h:.6f} | R2={r2_h:.6f}")

preds_all = pd.concat(all_preds, ignore_index=True)
summary_df = pd.DataFrame(summary_rows).sort_values(["model", "h"]).reset_index(drop=True)

preds_all.to_csv(OUT_PREDS_PATH, index=False)
summary_df.to_csv(OUT_RMSE_PATH, index=False)

print("\nSaved predictions to:", OUT_PREDS_PATH)
print("Saved summary table to:", OUT_RMSE_PATH)
print("\nSummary:")
print(summary_df)

# Summary:
#    model  h      rmse        r2        alpha
# 0  RIDGE  1  0.735430  0.325161   123.284674
# 1  RIDGE  3  0.857864  0.093168  7220.809018
# 2  RIDGE  5  0.835938  0.146160  4430.621458
# 3  RIDGE  7  0.794073  0.223809   236.448941