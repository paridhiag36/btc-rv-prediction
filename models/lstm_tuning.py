import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load dataset
train_df = pd.read_csv("../data/train_dataset.csv")
train_df["date"] = pd.to_datetime(train_df["date"], format="%d/%m/%y", errors="raise")
train_df = train_df.sort_values("date").reset_index(drop=True)

# Specify the target
target_col = "y_h1"
all_target_cols = ["y_h1", "y_h3", "y_h5", "y_h7"]

# Identify the Xt feature columns to form sequences using Xt and the lag columns
# Xt columns = not date/target columns and not (_lag1/2/3)
exclude = set(["date"] + all_target_cols)
xt_cols = [
    c for c in train_df.columns
    if (c not in exclude and not c.endswith("_lag1") and not c.endswith("_lag2") and not c.endswith("_lag3"))
]

# Check that lag columns exist for each Xt feature
for lag in [1, 2, 3]:
    missing = [c for c in xt_cols if f"{c}_lag{lag}" not in train_df.columns]
    if missing:
        raise ValueError(f"Missing lag{lag} columns for: {missing[:10]}")

# -----------------------------
# D) Create sequences of length 4: [t-3, t-2, t-1, t]
# X shape: (n_samples, timesteps=4, n_features=len(xt_cols))
# -----------------------------
def make_lag_sequences(df, base_cols, y_col):
    X_t  = df[base_cols].to_numpy()
    X_l1 = df[[f"{c}_lag1" for c in base_cols]].to_numpy()
    X_l2 = df[[f"{c}_lag2" for c in base_cols]].to_numpy()
    X_l3 = df[[f"{c}_lag3" for c in base_cols]].to_numpy()

    X_seq = np.stack([X_l3, X_l2, X_l1, X_t], axis=1).astype(np.float32)  # (n,4,p)
    y = df[y_col].to_numpy().astype(np.float32)
    return X_seq, y

X_all, y_all = make_lag_sequences(train_df, xt_cols, target_col)
# -----------------------------
# E) Split train into sub-train / validation (time-ordered-split)
# Use last 20% as validation (no shuffle).
# -----------------------------
val_frac = 0.2
n = X_all.shape[0]
cut = int(n * (1 - val_frac))

# Training model to fit the model paramaters 
X_sub, y_sub = X_all[:cut], y_all[:cut]

# Validation set used for hyperparameter tuning 
X_val, y_val = X_all[cut:], y_all[cut:]

# -----------------------------
# F) Standardize (fit on sub-train only; transform sub-train + val)
# -----------------------------
def scale_fit_on_train(X_train_3d, X_other_3d=None):
    """
    Fit StandardScaler on X_train_3d only (flattened across timesteps as standardscalar cannot
    handle 3d),
    transform X_train_3d and optionally transform X_other_3d.
    """
    T = X_train_3d.shape[1]
    p = X_train_3d.shape[2]

    scaler = StandardScaler()

    X_train_flat = X_train_3d.reshape(X_train_3d.shape[0] * T, p)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train_3d.shape[0], T, p)

    if X_other_3d is None:
        return scaler, X_train_scaled, None

    X_other_flat = X_other_3d.reshape(X_other_3d.shape[0] * T, p)
    X_other_scaled = scaler.transform(X_other_flat).reshape(X_other_3d.shape[0], T, p)
    return scaler, X_train_scaled, X_other_scaled

scaler_sub, X_sub_scaled, X_val_scaled = scale_fit_on_train(X_sub, X_val)

T = X_sub_scaled.shape[1]   # 4
p = X_sub_scaled.shape[2]   # number of features

# -----------------------------
# G) LSTM Implementation
# -----------------------------
def build_lstm(T, p, units, dropout=0.2, lr=1e-3):
    model = tf.keras.Sequential([
        layers.Input(shape=(T, p)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

# -----------------------------
# Tune hyperparameters for LSTM
# -----------------------------
def tune_lstm_for_h(train_df, xt_cols, target_col, h,
                                 units_grid=(16,32,64,128),
                                 val_frac=0.2,
                                 dropout=0.2, lr=1e-3,
                                 batch_size=64, epochs=100, patience=8,
                                 seed=0):
    # Drop NA target variables - should be 0
    train_df = train_df.dropna(subset=[target_col]).copy()

    # 1) sequences
    X_all, y_all = make_lag_sequences(train_df, xt_cols, target_col)

    # 2) sub-train / val split
    n = X_all.shape[0]
    cut = int(n * (1 - val_frac))
    X_sub, y_sub = X_all[:cut], y_all[:cut]
    X_val, y_val = X_all[cut:], y_all[cut:]

    # 3) scale (fit on sub-train only)
    _, X_sub_scaled, X_val_scaled = scale_fit_on_train(X_sub, X_val)

    T = X_sub_scaled.shape[1]
    p = X_sub_scaled.shape[2]

    # 4) tune units
    tf.random.set_seed(seed)
    np.random.seed(seed)

    best_units, best_epoch, best_val_loss = None, None, np.inf

    for units in units_grid:
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = build_lstm(T, p, units, dropout=dropout, lr=lr)
        early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

        hist = model.fit(
            X_sub_scaled, y_sub,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

        this_best_val = float(np.min(hist.history["val_loss"]))
        this_best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)

        if this_best_val < best_val_loss:
            best_val_loss = this_best_val
            best_units = int(units)
            best_epoch = this_best_epoch

    return {
        "h": h,
        "target_col": target_col,
        "best_units": best_units,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size
    }   

# --------------------------
# Run tuning for all horizons, save ONLY hyperparameters
# --------------------------
horizons = [1, 3, 5, 7]
infos = []

for h in horizons:
    target_col = f"y_h{h}"
    info = tune_lstm_for_h(train_df, xt_cols, target_col, h)
    infos.append(info)

summary = pd.DataFrame(infos)[["h", "best_units", "best_epoch", "best_val_loss", "dropout", "lr", "batch_size"]]
print("\nLSTM tuning summary (hyperparameters only):")
print(summary.sort_values("h"))

# Save hyperparameters to disk
os.makedirs("../models", exist_ok=True)
summary.to_csv("../models/lstm_tuned_hyperparams.csv", index=False)
print("\nSaved to ../models/lstm_tuned_hyperparams.csv")

# LSTM tuning summary (hyperparameters only):
#    h  best_units  best_epoch  best_val_loss  dropout     lr  batch_size
# 0  1          64          75       0.984143      0.2  0.001          64
# 1  3          16          66       1.774219      0.2  0.001          64
# 2  5          16          47       1.277434      0.2  0.001          64
# 3  7          64           6       1.400542      0.2  0.001          64