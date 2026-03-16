import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Load rolling OOS predictions
preds = pd.read_csv("../../forecast evaluations/xgb_outputs/xgb_rolling_oos_predictions.csv")
preds["date"] = pd.to_datetime(preds["date"])
preds = preds.sort_values(["h", "date"]) # Ensure sorting by h and date
# print(preds.head())
# print (preds.tail())

print("Unique h:", sorted(preds["h"].unique()))

os.makedirs("../../figs", exist_ok=True)

horizons = sorted(preds["h"].unique())
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

for ax, h in zip(axes.flatten(), horizons):
    df_h = preds[preds["h"] == h].dropna(subset=["y_true"])
    ax.plot(df_h["date"], df_h["y_true"], label="Actual", linewidth=1.2)
    ax.plot(df_h["date"], df_h["y_pred"], label="Predicted", linewidth=1.0, alpha=0.85)
    ax.set_title(f"h={h}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log RV")
    ax.legend()

fig.suptitle("XGBoost Rolling OOS: Actual vs Predicted", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("../../figs/all_models/xgb_actual_vs_pred_grid.png", dpi=200, bbox_inches="tight")
plt.close()

print("Saved plot to ../../figs/all_models/xgb_actual_vs_pred_grid.png")