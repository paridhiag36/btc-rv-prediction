import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Load rolling OOS predictions
preds = pd.read_csv("../forecast evaluations/xgboost/xgb_rolling_oos_predictions.csv")
preds["date"] = pd.to_datetime(preds["date"])
preds = preds.sort_values(["h", "date"])

print("Unique h:", sorted(preds["h"].unique()))

os.makedirs("../figs", exist_ok=True)

for h in sorted(preds["h"].unique()):
    df_h = preds[preds["h"] == h].dropna(subset=["y_true"])

    plt.figure(figsize=(12, 4))
    plt.plot(df_h["date"], df_h["y_true"], label="Actual",    linewidth=1.2)
    plt.plot(df_h["date"], df_h["y_pred"], label="Predicted", linewidth=1.0, alpha=0.85)
    plt.title(f"XGBoost Rolling OOS: Actual vs Predicted (h={h})")
    plt.xlabel("Date")
    plt.ylabel("Target (log RV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../figs/xgb_actual_vs_pred_h{h}.png", dpi=200)
    plt.close()

print("Saved plots to ../figs/")