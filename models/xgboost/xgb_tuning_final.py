# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from itertools import product as iproduct
import xgboost as xgb
import os

# Load dataset
train_df = pd.read_csv("../../data/train_dataset.csv")
train_df["date"] = pd.to_datetime(train_df["date"], format="%d/%m/%y", errors="raise")
train_df = train_df.sort_values("date").reset_index(drop=True)
# print (train_df.head())
# print(train_df.columns)

# Define feature columns
# XGBoost takes a flat 2D feature matrix — each row is the full information set at time t:
#   [X_t | X_lag1 | X_lag2 | X_lag3]  (176 features = 44 base × 4 timesteps)
# No standardisation is applied — XGBoost is a tree-based model and is scale-invariant.

all_target_cols = ["y_h1", "y_h3", "y_h5", "y_h7"]
exclude = set(["date"] + all_target_cols)

xt_cols = [
    c for c in train_df.columns
    if (c not in exclude
        and not c.endswith("_lag1")
        and not c.endswith("_lag2")
        and not c.endswith("_lag3"))
]
# Verify lag columns exist
for lag in [1, 2, 3]:
    missing = [c for c in xt_cols if f"{c}_lag{lag}" not in train_df.columns]
    if missing:
        raise ValueError(f"Missing lag{lag} columns for: {missing[:10]}")

print(f"Base features: {len(xt_cols)}")
print(f"Total flat features per row: {len(xt_cols) * 4}")


# Build flat feature matrix
def make_flat_features(df, base_cols, y_col):
    """Concatenate [X_lag3 | X_lag2 | X_lag1 | X_t] into a single flat row per observation."""
    X_t  = df[base_cols].to_numpy()
    X_l1 = df[[f"{c}_lag1" for c in base_cols]].to_numpy()
    X_l2 = df[[f"{c}_lag2" for c in base_cols]].to_numpy()
    X_l3 = df[[f"{c}_lag3" for c in base_cols]].to_numpy()
    X_flat = np.concatenate([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)
    y = df[y_col].to_numpy().astype(np.float32)
    return X_flat, y

# Quick Checks
X_check, y_check = make_flat_features(train_df, xt_cols, "y_h1")
print(f"X shape: {X_check.shape}")          # expect (1268, 176)
print(f"y shape: {y_check.shape}")          # expect (1268,)
print(f"NaNs in X: {np.isnan(X_check).sum()}")  # expect 0
print(f"NaNs in y: {np.isnan(y_check).sum()}")  # expect 0

print(f"\ny_h1 stats:\n{train_df['y_h1'].describe()}")

print(f"\nDate range: {train_df['date'].min()} → {train_df['date'].max()}") # 2021-01-08 00:00:00 → 2024-06-28 00:00:00
print(f"Total rows: {len(train_df)}") # 1268 ofc



# Hyperparameter tuning via TimeSeriesSplit CV
# -----------------------------------------------------------------------
# TimeSeriesSplit is used instead of a single 80/20 split because:
# - it is more robust for time-series. 
# - it evaluates across multiple expanding windows and avoids over-fitting the hyperparameters to one particular validation period. 
# - This is the standard approach for tree-based models.
#
# Parameters tuned:
#   max_depth        : tree depth — primary lever for complexity / overfitting
#   learning_rate    : shrinkage; smaller = more trees needed but better generalisation
#   subsample        : row subsampling per tree (variance reduction)
#   colsample_bytree : feature subsampling per tree (further regularisation)
# n_estimators is determined via early stopping on the last CV fold's
# eval set, which avoids over-specifying this parameter in the grid.
# -----------------------------------------------------------------------

SEED = 0
N_SPLITS = 5   # TimeSeriesSplit folds

PARAM_GRID = {
    "max_depth"        : [3, 5, 7],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "subsample"        : [0.7, 1.0],
    "colsample_bytree" : [0.7, 1.0],
}


def tune_xgb_for_h(train_df, xt_cols, target_col, h,
                   param_grid=PARAM_GRID,
                   n_splits=N_SPLITS,
                   seed=SEED):
    """
    Tune XGBoost hyperparameters for a single forecast horizon.
    Uses TimeSeriesSplit CV; best combo is the one with lowest mean RMSE across folds.
    n_estimators is determined by early stopping on the final fold.
    """
    df = train_df.dropna(subset=[target_col]).copy()
    X, y = make_flat_features(df, xt_cols, target_col)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    keys   = list(param_grid.keys())
    values = list(param_grid.values())

    best_rmse   = np.inf
    best_params = None
    best_n_est  = None

    for combo in iproduct(*values):
        params = dict(zip(keys, combo))

        fold_rmses = []
        last_fold_n_trees = None

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr   = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = xgb.XGBRegressor(
                objective          = "reg:squarederror",
                n_estimators       = 1000,
                early_stopping_rounds = 20,
                eval_metric        = "rmse",
                random_state       = seed,
                n_jobs             = -1,
                verbosity          = 0,
                **params
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_hat = model.predict(X_val)
            rmse  = float(np.sqrt(np.mean((y_val - y_hat) ** 2)))
            fold_rmses.append(rmse)

            # Capture n_estimators from the last (most recent) fold
            if fold_idx == n_splits - 1:
                last_fold_n_trees = model.best_iteration + 1

        mean_rmse = float(np.mean(fold_rmses))

        if mean_rmse < best_rmse:
            best_rmse   = mean_rmse
            best_params = params.copy()
            best_n_est  = last_fold_n_trees

    return {
        "h"                : h,
        "best_n_estimators": best_n_est,
        "best_max_depth"   : best_params["max_depth"],
        "best_lr"          : best_params["learning_rate"],
        "best_subsample"   : best_params["subsample"],
        "best_colsample"   : best_params["colsample_bytree"],
        "best_cv_rmse"     : best_rmse,
    }

# Run tuning for all horizons
horizons = [1, 3, 5, 7]
infos = []

for h in horizons:
    target_col = f"y_h{h}"
    print(f"Tuning XGBoost for h={h} ...")
    info = tune_xgb_for_h(train_df, xt_cols, target_col, h)
    infos.append(info)
    print(f"  best_cv_rmse={info['best_cv_rmse']:.6f} | "
          f"n_est={info['best_n_estimators']} | "
          f"max_depth={info['best_max_depth']} | "
          f"lr={info['best_lr']} | "
          f"subsample={info['best_subsample']} | "
          f"colsample={info['best_colsample']}")

summary = pd.DataFrame(infos)
print("\nXGBoost tuning summary:")
print(summary.sort_values("h").to_string(index=False))

# Save hyperparameters to disk
os.makedirs("../models/xgboost", exist_ok=True)
summary.to_csv("../models/xgboost/xgb_tuned_hyperparams.csv", index=False)
print("\nSaved to ../models/xgboost/xgb_tuned_hyperparams.csv")