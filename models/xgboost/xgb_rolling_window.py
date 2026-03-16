# import libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# --------- CONFIG ----------
TRAIN_DATA_PATH  = "../../data/train_dataset.csv"
TEST_DATA_PATH   = "../../data/test_dataset.csv"          
HYPERPARAMS_PATH = "../xgb_tuned_hyperparams.csv"        # CSV from xgb_tuning_final.ipynb
OUT_PREDS_PATH   = "../../forecast evaluations/xgb_outputs/xgb_rolling_oos_predictions.csv"
OUT_RMSE_PATH    = "../../forecast evaluations/xgb_outputs/xgb_rolling_oos_rmse.csv"

DATE_FORMAT = "%Y-%m-%d"
DATE_COL    = "date"
HORIZONS    = [1, 3, 5, 7]

WINDOW_SIZE  = 1095   # fixed rolling training window (~3 years daily) (same as LSTM)
REFIT_EVERY  = 7      # weekly refit (same as LSTM)

SEED = 0

# --------- HELPERS ----------
def get_xt_cols(df):
    target_cols = ["y_h1", "y_h3", "y_h5", "y_h7"]
    exclude = set([DATE_COL] + target_cols)
    return [
        c for c in df.columns
        if (c not in exclude
            and not c.endswith("_lag1")
            and not c.endswith("_lag2")
            and not c.endswith("_lag3"))

    ]

def make_flat_features(df, base_cols, y_col):
    """
    Build flat [X_lag3 | X_lag2 | X_lag1 | X_t] feature matrix.
    Captures the same temporal information as the 4-step sequence earlier.
    No standardisation needed — XGBoost is scale-invariant.
    """
    X_t  = df[base_cols].to_numpy()
    X_l1 = df[[f"{c}_lag1" for c in base_cols]].to_numpy()
    X_l2 = df[[f"{c}_lag2" for c in base_cols]].to_numpy()
    X_l3 = df[[f"{c}_lag3" for c in base_cols]].to_numpy()
    X_flat = np.concatenate([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)
    y = df[y_col].to_numpy().astype(np.float32)
    return X_flat, y

def make_flat_row(df_one_row, base_cols):
    """Build a single-row flat feature vector for prediction at time t."""
    X_t  = df_one_row[base_cols].to_numpy()
    X_l1 = df_one_row[[f"{c}_lag1" for c in base_cols]].to_numpy()
    X_l2 = df_one_row[[f"{c}_lag2" for c in base_cols]].to_numpy()
    X_l3 = df_one_row[[f"{c}_lag3" for c in base_cols]].to_numpy()
    return np.concatenate([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)  # (1, 4*p)

def fit_xgb_on_window(df_window, xt_cols, target_col, hp):
    """
    Fit XGBoost on a rolling training window using tuned hyperparameters.
    n_estimators is fixed to the tuned value (no eval set needed during rolling refit).
    """
    df_window = df_window.sort_values(DATE_COL).reset_index(drop=True)
    X, y = make_flat_features(df_window, xt_cols, target_col)

    # Drop rows with NaN target (safety check)
    mask = ~np.isnan(y)
    X, y = X[mask], y[mask]

    model = xgb.XGBRegressor(
        objective          = "reg:squarederror",
        n_estimators       = hp["n_estimators"],
        max_depth          = hp["max_depth"],
        learning_rate      = hp["learning_rate"],
        subsample          = hp["subsample"],
        colsample_bytree   = hp["colsample"],
        eval_metric        = "rmse",
        random_state       = SEED,
        n_jobs             = -1,
        verbosity          = 0,
    )
    model.fit(X, y, verbose=False)
    return model


def rolling_eval_one_h(full_df, xt_cols, h, hp, start_t):
    """
    Rolling OOS evaluation for one forecast horizon.
    Refits every REFIT_EVERY steps on the most recent WINDOW_SIZE observations,
    then predicts the next step.
    """
    target_col = f"y_h{h}"
    df = full_df.sort_values(DATE_COL).reset_index(drop=True)
    end_t = len(df) - 1

    current_model = None
    rows = []

    for t in range(start_t, end_t + 1):
        # Rolling window: rows [t - WINDOW_SIZE, ..., t-1]
        df_window = df.iloc[t - WINDOW_SIZE : t].copy()

        # Refit on schedule
        if current_model is None or ((t - start_t) % REFIT_EVERY == 0):
            current_model = fit_xgb_on_window(df_window, xt_cols, target_col, hp)

        # Predict for row t
        df_row  = df.iloc[t : t + 1].copy()
        X_row   = make_flat_row(df_row, xt_cols)
        yhat    = float(current_model.predict(X_row)[0])
        ytrue   = df_row[target_col].iloc[0]

        rows.append({
            "date"  : df_row[DATE_COL].iloc[0],
            "h"     : h,
            "y_true": float(ytrue) if pd.notna(ytrue) else np.nan,
            "y_pred": yhat,
        })

    out = pd.DataFrame(rows)
    out["error"] = out["y_true"] - out["y_pred"]
    rmse = float(np.sqrt(np.nanmean(out["error"] ** 2)))
    return out, rmse

# --------- MAIN ----------
# Load train + test datasets
train_df = pd.read_csv(TRAIN_DATA_PATH)
train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL], format="%d/%m/%y", errors="raise")
test_df = pd.read_csv(TEST_DATA_PATH)
test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL], format=DATE_FORMAT, errors="raise")

full_df = pd.concat([train_df, test_df], ignore_index=True)
full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)
# print (train_df.shape, test_df.shape, full_df.shape)
# print(train_df[DATE_COL].min(), train_df[DATE_COL].max()) # 2021-01-08 00:00:00 2024-06-28 00:00:00
# print(test_df[DATE_COL].min(), test_df[DATE_COL].max()) # 2024-06-29 00:00:00 2025-12-24 00:00:00

# Evaluation start date — consistent with test dataset as above
EVAL_START_DATE = pd.to_datetime("2024-06-29")
eval_start_idx  = int(full_df.index[full_df[DATE_COL] >= EVAL_START_DATE][0])
start_t = max(WINDOW_SIZE, eval_start_idx)

xt_cols = get_xt_cols(full_df)

# Load tuned hyperparameters
hp_df  = pd.read_csv(HYPERPARAMS_PATH)
hp_map = {
    int(r["h"]): {
        "n_estimators": int(r["best_n_estimators"]),
        "max_depth"   : int(r["best_max_depth"]),
        "learning_rate": float(r["best_lr"]),
        "subsample"   : float(r["best_subsample"]),
        "colsample"   : float(r["best_colsample"]),
    }
    for _, r in hp_df.iterrows()
}
# print(hp_map)

all_preds = []
rmse_rows = []

for h in HORIZONS:
    hp = hp_map[h]
    print(f"Running rolling eval for h={h} | "
          f"n_est={hp['n_estimators']}, max_depth={hp['max_depth']}, "
          f"lr={hp['learning_rate']}, subsample={hp['subsample']}, "
          f"colsample={hp['colsample']} ...")

    pred_df, rmse_h = rolling_eval_one_h(full_df, xt_cols, h, hp, start_t)

    all_preds.append(pred_df)
    rmse_rows.append({"h": h, "rmse": rmse_h, **hp})

    print(f"  h={h} RMSE = {rmse_h:.6f}")

preds_all = pd.concat(all_preds, ignore_index=True)
rmse_df   = pd.DataFrame(rmse_rows).sort_values("h")

# Save outputs
os.makedirs(os.path.dirname(OUT_PREDS_PATH), exist_ok=True)
preds_all.to_csv(OUT_PREDS_PATH, index=False)
rmse_df.to_csv(OUT_RMSE_PATH, index=False)

print("\nSaved predictions to:", OUT_PREDS_PATH)
print("Saved RMSE table to: ", OUT_RMSE_PATH)
print("\nRMSE summary:")
print(rmse_df[["h", "rmse"]].to_string(index=False))
