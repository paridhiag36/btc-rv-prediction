import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load your saved rolling OOS predictions
preds = pd.read_csv("../data/lstm_rolling_oos_predictions.csv")
preds["date"] = pd.to_datetime(preds["date"], format="%d/%m/%y", errors="raise")

print("Unique h:", sorted(preds["h"].unique()))

os.makedirs("../figs", exist_ok=True)

for h in sorted(preds["h"].unique()):
    df_h = preds[preds["h"] == h].sort_values("date")

    plt.figure(figsize=(12, 4))
    plt.plot(df_h["date"], df_h["y_true"], label="Actual")
    plt.plot(df_h["date"], df_h["y_pred"], label="Predicted")
    plt.title(f"LSTM Rolling OOS: Actual vs Predicted (h={h})")
    plt.xlabel("Date")
    plt.ylabel("Target (log RV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../figs/lstm_actual_vs_pred_h{h}.png", dpi=200)
    plt.close()

print("Saved plots to ../figs/")