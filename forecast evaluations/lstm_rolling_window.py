import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os 
from sklearn.metrics import r2_score


# --------- CONFIG ----------
FULL_DATA_PATH = "../data/full_df.csv"  # CHANGE if your full dataset file name differs
HYPERPARAMS_PATH = "../models/lstm_tuned_hyperparams.csv" #csv saved from lstm_tuning.py
OUT_PREDS_PATH = "../forecast evaluations/lstm_rolling_oos_predictions.csv" # saving predictions 
OUT_RMSE_PATH = "../forecast evaluations/lstm_rolling_oos_rmse.csv" # save rmse results 

DATE_FORMAT = "%Y-%m-%d"
DATE_COL = "date"
HORIZONS = [1, 3, 5, 7]

WINDOW_SIZE = 1095     # fixed rolling training window length (~3 years daily)
REFIT_EVERY = 7        # set to 7 = weekly refit (compute-friendly)

DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 64
SEED = 0
VERBOSE_FIT = 0        # set to 1 if you want to see training progress

# --------- HELPERS ----------
def get_xt_cols(df):
    target_cols = ["y_h1", "y_h3", "y_h5", "y_h7"]
    exclude = set([DATE_COL] + target_cols)
    return [c for c in df.columns if (c not in exclude and not c.endswith("_lag1") and not c.endswith("_lag2") and not c.endswith("_lag3"))]

def make_lag_sequences(df, base_cols, y_col):
    X_t  = df[base_cols].to_numpy()
    X_l1 = df[[f"{c}_lag1" for c in base_cols]].to_numpy()
    X_l2 = df[[f"{c}_lag2" for c in base_cols]].to_numpy()
    X_l3 = df[[f"{c}_lag3" for c in base_cols]].to_numpy()
    X_seq = np.stack([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)
    y = df[y_col].to_numpy().astype(np.float32)
    return X_seq, y

def scale_fit_on_train(X_train_3d):
    T = X_train_3d.shape[1]
    p = X_train_3d.shape[2]
    scaler = StandardScaler()
    X_flat = X_train_3d.reshape(X_train_3d.shape[0] * T, p)
    X_scaled = scaler.fit_transform(X_flat).reshape(X_train_3d.shape[0], T, p)
    return scaler, X_scaled

def scale_transform(scaler, X_3d):
    n, T, p = X_3d.shape
    X_flat = X_3d.reshape(n * T, p)
    X_scaled = scaler.transform(X_flat).reshape(n, T, p)
    return X_scaled

def build_lstm(T, p, units, dropout=DROPOUT, lr=LR):
    model = tf.keras.Sequential([
        layers.Input(shape=(T, p)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

def make_X_seq_row(df_one_row, xt_cols):
    X_t  = df_one_row[xt_cols].to_numpy()
    X_l1 = df_one_row[[f"{c}_lag1" for c in xt_cols]].to_numpy()
    X_l2 = df_one_row[[f"{c}_lag2" for c in xt_cols]].to_numpy()
    X_l3 = df_one_row[[f"{c}_lag3" for c in xt_cols]].to_numpy()
    X_seq = np.stack([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)  # (1,4,p)
    return X_seq

# Train (refit) LSTM on rolling training window
def fit_lstm_on_window(df_window, xt_cols, target_col, units, epochs):
    # ensure date is sorted 
    df_window = df_window.sort_values(DATE_COL).reset_index(drop=True)

    X_all, y_all = make_lag_sequences(df_window, xt_cols, target_col)
    mask = ~np.isnan(y_all)
    X_all = X_all[mask]
    y_all = y_all[mask]

    scaler, X_scaled = scale_fit_on_train(X_all)
    T = X_scaled.shape[1]
    p = X_scaled.shape[2]

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    model = build_lstm(T, p, units)
    model.fit(
        X_scaled, y_all,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=VERBOSE_FIT
    )
    return model, scaler

def predict_one(model, scaler, X_seq_one):
    X_scaled = scale_transform(scaler, X_seq_one)
    return float(model.predict(X_scaled, verbose=0).reshape(-1)[0])

def rolling_eval_one_h(full_df, xt_cols, h, units, epochs, start_t):
    target_col = f"y_h{h}"
    df = full_df.sort_values(DATE_COL).reset_index(drop=True)

    end_t = len(df) - 1

    current_model, current_scaler = None, None
    rows = []

    for t in range(start_t, end_t + 1):
        # train window uses [t-WINDOW_SIZE, ..., t-1]
        df_window = df.iloc[t - WINDOW_SIZE : t].copy()

        # refit schedule
        if current_model is None or ((t - start_t) % REFIT_EVERY == 0):
            current_model, current_scaler = fit_lstm_on_window(
                df_window=df_window,
                xt_cols=xt_cols,
                target_col=target_col,
                units=units,
                epochs=epochs
            )

        # predict for row t
        df_row = df.iloc[t : t + 1].copy()
        X_seq_one = make_X_seq_row(df_row, xt_cols)
        yhat = predict_one(current_model, current_scaler, X_seq_one)
        ytrue = df_row[target_col].iloc[0]

        rows.append({
            "date": df_row[DATE_COL].iloc[0],
            "h": h,
            "y_true": float(ytrue) if pd.notna(ytrue) else np.nan,
            "y_pred": yhat
        })

    out = pd.DataFrame(rows)
    out["error"] = out["y_true"] - out["y_pred"]
    rmse = float(np.sqrt(np.nanmean(out["error"] ** 2)))
    r2 = float(r2_score(out["y_true"], out["y_pred"]))
    return out, rmse, r2

# --------- MAIN ----------
# Load full dataset
full_df = pd.read_csv(FULL_DATA_PATH)
full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL], format=DATE_FORMAT, errors="raise")
full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)

# Explicit evaluation start date (2024-06-29, consistent with the test dataset)
EVAL_START_DATE = pd.to_datetime("2024-06-29")

# first index in full_df where date >= eval_start 
eval_start_idx = int(full_df.index[full_df[DATE_COL] >= EVAL_START_DATE][0])

start_t = max(WINDOW_SIZE, eval_start_idx)

# Identify xt_cols (must match how you built features)
xt_cols = get_xt_cols(full_df)

# Load tuned hyperparams
hp = pd.read_csv(HYPERPARAMS_PATH)
hp_map = {int(r["h"]): {"units": int(r["best_units"]), "epochs": int(r["best_epoch"])} for _, r in hp.iterrows()}

all_preds = []
rmse_rows = []

for h in HORIZONS:
    units = hp_map[h]["units"]
    epochs = hp_map[h]["epochs"]

    print(f"Running rolling eval for h={h} with units={units}, epochs={epochs} ...")
    pred_df, rmse_h, r2_h = rolling_eval_one_h(full_df, xt_cols, h, units, epochs, start_t)

    all_preds.append(pred_df)
    rmse_rows.append({"h": h, "rmse": rmse_h, "r2": r2_h, "units": units, "epochs": epochs})

    print(f"h={h} RMSE={rmse_h:.6f} r2= {r2_h: 6f}")

preds_all = pd.concat(all_preds, ignore_index=True)
rmse_df = pd.DataFrame(rmse_rows).sort_values("h")

# Save outputs
os.makedirs(os.path.dirname(OUT_PREDS_PATH), exist_ok=True)
preds_all.to_csv(OUT_PREDS_PATH, index=False)
rmse_df.to_csv(OUT_RMSE_PATH, index=False)

print("\nSaved predictions to:", OUT_PREDS_PATH)
print("Saved RMSE table to:", OUT_RMSE_PATH)
print("\nRMSE summary:")
print(rmse_df)

# RMSE summary:
#    h      rmse        r2  units  epochs
# 0  1  0.785121  0.230887    128      66
# 1  3  0.946351 -0.103556     16      61
# 2  5  0.944629 -0.090314     16      58
# 3  7  0.959048 -0.132215     16      56