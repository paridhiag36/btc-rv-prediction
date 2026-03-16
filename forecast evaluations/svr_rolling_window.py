# Load library 
import os 
import pandas as pd
import numpy as np 
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Configuration 
FULL_DATA_PATH = "../data/full_df.csv"
SVR_PARAM_PATH = "../models/svr_final_summary.csv"

OUT_DIR = "../forecast evaluations/svr_outputs"
OUT_PREDS_PATH = os.path.join(OUT_DIR, "svr_rolling_oos_predictions.csv")
OUT_RMSE_PATH  = os.path.join(OUT_DIR, "svr_rolling_oos_rmse.csv")

DATE_COL = "date"
TARGET_COLS = ["y_h1", "y_h3", "y_h5", "y_h7"]
HORIZONS = [1, 3, 5, 7]

WINDOW_SIZE = 1095
REFIT_EVERY = 7
EVAL_START_DATE = pd.to_datetime("2024-06-29")

# Helper functions
# ---------- HELPERS ----------
def get_feature_cols(df):
    return [c for c in df.columns if c not in ([DATE_COL] + TARGET_COLS)]

def compute_start_t(df):
    idxs = df.index[df[DATE_COL] >= EVAL_START_DATE]
    if len(idxs) == 0:
        raise ValueError("EVAL_START_DATE is after the last date in df.")
    eval_start_idx = int(idxs[0])
    return max(WINDOW_SIZE, eval_start_idx)

def load_svr_params(csv_path):
    df = pd.read_csv(csv_path).copy()

    required_cols = ["target", "svr__C", "svr__epsilon", "svr__kernel", "svr__gamma"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    param_map = {}

    for _, row in df.iterrows():
        target = row["target"]

        params = {
            "kernel": row["svr__kernel"],
            "C": float(row["svr__C"]),
            "epsilon": float(row["svr__epsilon"])
        }

        # gamma handling
        if pd.notna(row["svr__gamma"]):
            gamma_val = row["svr__gamma"]

            # if it's numeric, cast to float; otherwise keep as string (e.g. "scale")
            try:
                gamma_val = float(gamma_val)
            except:
                gamma_val = str(gamma_val)

            params["gamma"] = gamma_val

        param_map[target] = params

    return param_map

def build_svr_pipeline(params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(**params))
    ])

def fit_svr_on_window(df_window, feature_cols, target_col, params):
    X_train = df_window[feature_cols]
    y_train = df_window[target_col].to_numpy()

    model = build_svr_pipeline(params)
    model.fit(X_train, y_train)
    return model

def rolling_eval_svr_one_h(df, feature_cols, h, params, start_t):
    target_col = f"y_h{h}"
    end_t = len(df) - 1

    current_model = None
    rows = []

    for t in range(start_t, end_t + 1):
        # rolling training window = [t-WINDOW_SIZE, ..., t-1]
        train_slice = df.iloc[t - WINDOW_SIZE : t].copy()

        # weekly refit
        if current_model is None or ((t - start_t) % REFIT_EVERY == 0):
            current_model = fit_svr_on_window(
                df_window=train_slice,
                feature_cols=feature_cols,
                target_col=target_col,
                params=params
            )

        # predict row t
        X_test = df.iloc[t : t + 1][feature_cols]
        yhat = float(current_model.predict(X_test)[0])
        ytrue = float(df.iloc[t][target_col])

        rows.append({
            "date": df.iloc[t][DATE_COL],
            "model": "SVR",
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

final_svr_params = load_svr_params(SVR_PARAM_PATH)

all_preds = []
summary_rows = []

for h in HORIZONS:
    target_col = f"y_h{h}"
    if target_col not in final_svr_params:
        raise ValueError(f"Missing tuned params for {target_col} in {SVR_PARAM_PATH}")

    params = final_svr_params[target_col]

    print(f"Running SVR rolling eval for h={h} with params={params}")
    pred_df, rmse_h, r2_h = rolling_eval_svr_one_h(
        df=full_df,
        feature_cols=feature_cols,
        h=h,
        params=params,
        start_t=start_t
    )

    all_preds.append(pred_df)
    summary_rows.append({
        "model": "SVR",
        "h": h,
        "rmse": rmse_h,
        "r2": r2_h,
        "kernel": params["kernel"],
        "C": params["C"],
        "epsilon": params["epsilon"],
        "gamma": params.get("gamma", np.nan)
    })

    print(f"h={h} | RMSE={rmse_h:.6f}| R2={r2_h:.6f}")

preds_all = pd.concat(all_preds, ignore_index=True)
summary_df = pd.DataFrame(summary_rows).sort_values(["model", "h"]).reset_index(drop=True)

preds_all.to_csv(OUT_PREDS_PATH, index=False)
summary_df.to_csv(OUT_RMSE_PATH, index=False)

print("\nSaved predictions to:", OUT_PREDS_PATH)
print("Saved summary table to:", OUT_RMSE_PATH)
print("\nSummary:")
print(summary_df)

# Summary:
#   model  h      rmse        r2  kernel      C  epsilon  gamma
#    SVR  1  0.734987  0.325975  linear  0.011    0.190    NaN
#    SVR  3  0.822534  0.166323     rbf  0.500    0.100  scale
#    SVR  5  0.786841  0.243511     rbf  0.500    0.050  scale
#    SVR  7  0.784694  0.242035     rbf  1.000    0.005  scale