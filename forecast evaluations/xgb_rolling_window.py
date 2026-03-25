import os
import numpy as np 
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

# Configuration 
FULL_DATA_PATH = "../data/full_df.csv"
XGB_PARAM_PATH = "../models/xgb_tuned_hyperparams.csv"

OUT_DIR = "../forecast evaluations/xgb_outputs"
OUT_PREDS_PATH = os.path.join(OUT_DIR, "xgb_rolling_oos_predictions.csv")
OUT_RMSE_PATH  = os.path.join(OUT_DIR, "xgb_rolling_oos_rmse.csv")

DATE_COL = "date"
TARGET_COLS = ["y_h1", "y_h3", "y_h5", "y_h7"]
HORIZONS = [1, 3, 5, 7]

WINDOW_SIZE = 1095
REFIT_EVERY = 7
EVAL_START_DATE = pd.to_datetime("2024-06-29")
SEED = 0

# Helpers - more specific for xgboost 
def get_xt_cols(df):
    exclude = set([DATE_COL] + TARGET_COLS)
    return [
        c for c in df.columns
        if (c not in exclude
            and not c.endswith("_lag1")
            and not c.endswith("_lag2")
            and not c.endswith("_lag3"))
    ]

def compute_start_t(df):
    idxs = df.index[df[DATE_COL] >= EVAL_START_DATE]
    if len(idxs) == 0:
        raise ValueError("EVAL_START_DATE is after the last date in df.")
    eval_start_idx = int(idxs[0])
    return max(WINDOW_SIZE, eval_start_idx)

def make_flat_features(df, base_cols, y_col):
    X_t  = df[base_cols].to_numpy()
    X_l1 = df[[f"{c}_lag1" for c in base_cols]].to_numpy()
    X_l2 = df[[f"{c}_lag2" for c in base_cols]].to_numpy()
    X_l3 = df[[f"{c}_lag3" for c in base_cols]].to_numpy()

    X_flat = np.concatenate([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)
    y = df[y_col].to_numpy().astype(np.float32)
    return X_flat, y

def make_X_flat_row(df_one_row, xt_cols):
    X_t  = df_one_row[xt_cols].to_numpy()
    X_l1 = df_one_row[[f"{c}_lag1" for c in xt_cols]].to_numpy()
    X_l2 = df_one_row[[f"{c}_lag2" for c in xt_cols]].to_numpy()
    X_l3 = df_one_row[[f"{c}_lag3" for c in xt_cols]].to_numpy()

    X_flat = np.concatenate([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)
    return X_flat

def load_xgb_params(csv_path):
    df = pd.read_csv(csv_path).copy()

    required_cols = [
        "h",
        "best_n_estimators",
        "best_max_depth",
        "best_lr",
        "best_subsample",
        "best_colsample"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    param_map = {}
    for _, row in df.iterrows():
        h = int(row["h"])
        param_map[h] = {
            "n_estimators": int(row["best_n_estimators"]),
            "max_depth": int(row["best_max_depth"]),
            "learning_rate": float(row["best_lr"]),
            "subsample": float(row["best_subsample"]),
            "colsample_bytree": float(row["best_colsample"])
        }

    return param_map

def fit_xgb_on_window(df_window, xt_cols, target_col, params):
    df_window = df_window.sort_values(DATE_COL).reset_index(drop=True)

    X_all, y_all = make_flat_features(df_window, xt_cols, target_col)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        **params
    )
    model.fit(X_all, y_all)
    return model

def predict_one(model, X_flat_one):
    yhat = model.predict(X_flat_one).reshape(-1)[0]
    return float(yhat)

def rolling_eval_xgb_one_h(full_df, xt_cols, h, params, start_t):
    target_col = f"y_h{h}"
    df = full_df.sort_values(DATE_COL).reset_index(drop=True)

    end_t = len(df) - 1
    current_model = None
    rows = []

    for t in range(start_t, end_t + 1):
        df_window = df.iloc[t - WINDOW_SIZE : t].copy()

        if current_model is None or ((t - start_t) % REFIT_EVERY == 0):
            current_model = fit_xgb_on_window(
                df_window=df_window,
                xt_cols=xt_cols,
                target_col=target_col,
                params=params
            )

        df_row = df.iloc[t : t + 1].copy()
        X_flat_one = make_X_flat_row(df_row, xt_cols)
        yhat = predict_one(current_model, X_flat_one)
        ytrue = float(df_row[target_col].iloc[0])

        rows.append({
            "date": df_row[DATE_COL].iloc[0],
            "model": "XGBoost",
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

xt_cols = get_xt_cols(full_df)

for lag in [1, 2, 3]:
    missing = [c for c in xt_cols if f"{c}_lag{lag}" not in full_df.columns]
    if missing:
        raise ValueError(f"Missing lag{lag} columns for: {missing[:10]}")

start_t = compute_start_t(full_df)
xgb_params = load_xgb_params(XGB_PARAM_PATH)

all_preds = []
summary_rows = []

for h in HORIZONS:
    if h not in xgb_params:
        raise ValueError(f"Missing tuned params for h={h} in {XGB_PARAM_PATH}")

    params = xgb_params[h]

    print(f"Running XGBoost rolling eval for h={h} with params={params}")
    pred_df, rmse_h, r2_h = rolling_eval_xgb_one_h(
        full_df=full_df,
        xt_cols=xt_cols,
        h=h,
        params=params,
        start_t=start_t
    )

    all_preds.append(pred_df)
    summary_rows.append({
        "model": "XGBoost",
        "h": h,
        "rmse": rmse_h,
        "r2": r2_h,
        **params
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

# NEW Summary:
#   model  h     rmse       r2  n_estimators  max_depth  learning_rate  subsample  colsample_bytree
# XGBoost  1 0.690832 0.404527           174          2            0.1        0.7               0.5
# XGBoost  3 0.769212 0.270908            73          2            0.1        0.5               0.5
# XGBoost  5 0.732333 0.344692            40          2            0.1        0.5               0.5
# XGBoost  7 0.730228 0.343605            84          2            0.1        0.5               0.5

# OLD Summary:
#      model  h      rmse        r2  n_estimators  max_depth  learning_rate  subsample  colsample_bytree
# 0  XGBoost  1  0.680152  0.422796           186          3           0.05        0.7               1.0
# 1  XGBoost  3  0.760595  0.287151            57          3           0.10        0.7               0.7
# 2  XGBoost  5  0.736309  0.337556            19          3           0.10        1.0               1.0
# 3  XGBoost  7  0.741313  0.323525            17          3           0.10        0.7               1.0