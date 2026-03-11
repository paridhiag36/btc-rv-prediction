import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

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
# 5) Hyperparameter tuning: 
# Find optimal best m on validation step
# -----------------------------
tf.random.set_seed(0)
np.random.seed(0)

# canadidate values of units (m) i want to test to find the optimal hyperparameter
units_grid = [16, 32, 64, 128]

# fixed hyperparameters 
dropout = 0.2
lr = 1e-3
batch_size = 64
epochs = 100
patience = 8

results = []

# Hyperparameter loop over candidate m
for units in units_grid:
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    np.random.seed(0)

    # Building LSTM with candidate chosen m
    model = build_lstm(T,p,units, dropout=dropout, lr= lr)

    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    # Train the model and evaluates each epoch on validation set to compute val_loss
    hist = model.fit(
        X_sub_scaled, y_sub,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False, # keep time order 
        callbacks=[early_stop],
        verbose=0
    )

    # Extract best validation performance 
    best_val_loss = float(np.min(hist.history["val_loss"]))
    best_epoch = int(np.argmin(hist.history["val_loss"]) + 1)

    results.append({"units": units, "best_val_loss": best_val_loss, "best_epoch": best_epoch})
    print(f"units={units:>3} | best_val_loss={best_val_loss:.6f} | best_epoch={best_epoch}")

results_df = pd.DataFrame(results).sort_values("best_val_loss").reset_index(drop=True)
best_units = int(results_df.loc[0, "units"])
best_epoch = int(results_df.loc[0, "best_epoch"])

print("\nTuning results (sorted):")
print(results_df)
print("\nSelected units (m):", best_units, "| best_epoch:", best_epoch)

# --------------- Results
# units= 16 | best_val_loss=1.098961 | best_epoch=46
# units= 32 | best_val_loss=1.049508 | best_epoch=51
# units= 64 | best_val_loss=1.124638 | best_epoch=86
# units=128 | best_val_loss=0.889370 | best_epoch=44

# Choose the best m overall 
# Selected units (m): 128  

# --------------------
# Retrain final trained model 
# Fit scaler on ALL training data now
# -------------------
scaler_full, X_all_scaled, _ = scale_fit_on_train(X_all, None)

tf.keras.backend.clear_session()
tf.random.set_seed(0)
np.random.seed(0)

final_model_h1 = build_lstm(T, p, best_units, dropout=dropout, lr=lr)

final_model_h1.fit(
    X_all_scaled, y_all,
    epochs=best_epoch,
    batch_size=batch_size,
    shuffle=False,
    verbose=1
)

print("\nFinal model trained on ALL training data.")
print("Final hyperparameters -> units (m):", best_units, "epochs:", best_epoch)

# Save the final trained model
os.makedirs("../models", exist_ok=True)
# Save the trained Keras model
final_model_h1.save("../models/final_model_h1.keras")

# Save the scaler used for this model (needed for consistent future predictions)
joblib.dump(scaler_full, "../models/scaler_h1.pkl")

print("Saved model + scaler to ../models/")

# ----- Final LSTM Prediction model -----
def predict_lstm(model, scaler, X_seq):
    X_seq = np.asarray(X_seq, dtype=np.float32)
    n, T, p = X_seq.shape

    X_flat = X_seq.reshape(n * T, p)
    X_scaled = scaler.transform(X_flat).reshape(n, T, p)

    return model.predict(X_scaled, verbose=0).reshape(-1)

