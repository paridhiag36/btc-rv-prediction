# Load Library
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
from matplotlib.patches import Patch

# Configuration 
FULL_DATA_PATH = "../data/full_df.csv"
RF_PARAM_PATH = "../models/rf_tuned_hyperparams.csv"
OUT_DIR = "../shap_outputs/rf_shap_analysis"

DATE_COL = "date"
TARGET_COLS = ["y_h1", "y_h3", "y_h5", "y_h7"]
HORIZONS = [1, 3, 5, 7]

WINDOW_SIZE = 1095
EVAL_START_DATE = pd.to_datetime("2024-06-29")

N_SHAP_STEP = 7 
SEED = 42

# -----------------------------------------------------------------------
# Feature group mapping
# Maps each base feature to one of five groups defined in our proposal.
# Used for group-level importance and H2 analysis.
# -----------------------------------------------------------------------
FEATURE_GROUPS = {
    # Macroeconomic
    "DOW30"            : "Macro",
    "FTSE100"          : "Macro",
    "NASDAQ"           : "Macro",
    "SP500"            : "Macro",
    "SSE"              : "Macro",
    "VIX"              : "Macro",
    "OIL"              : "Macro",
    "DXY"              : "Macro",
    # Asset-based
    "GOLD"             : "Asset",
    "SILVER"           : "Asset",
    "TLT"              : "Asset",
    "IEF"              : "Asset",
    "HYG"              : "Asset",
    "LQD"              : "Asset",
    "EEM"              : "Asset",
    "YC_SLOPE_10Y_2Y"  : "Asset",
    "D_DGS2"           : "Asset",
    "D_DGS10"          : "Asset",
    "D_EFFR"           : "Asset",
    "D_HY_OAS"         : "Asset",
    # Blockchain
    "LOG_HASH_RATE"        : "Blockchain",
    "LOG_DIFFICULTY"       : "Blockchain",
    "DLOG_HASH_RATE"       : "Blockchain",
    "DLOG_DIFFICULTY"      : "Blockchain",
    "DLOG_MINERS_REV_USD"  : "Blockchain",
    "DLOG_TX_COUNT"        : "Blockchain",
    "DLOG_UNIQUE_ADDR"     : "Blockchain",
    "DLOG_EST_TX_VOL_USD"  : "Blockchain",
    "DLOG_FEES_USD"        : "Blockchain",
    "DLOG_FEES_USD_PER_TX" : "Blockchain",
    "DLOG_AVG_BLOCK_SIZE"  : "Blockchain",
    "DLOG_MED_CONFIRM_TIME": "Blockchain",
    "DLOG_AVG_TX_VALUE_USD": "Blockchain",
    # Technical
    "EMA10"    : "Technical",
    "EMA30"    : "Technical",
    "EMA200"   : "Technical",
    "RSI14"    : "Technical",
    "MOM10"    : "Technical",
    "ROC10"    : "Technical",
    "ATR14"    : "Technical",
    "BB_WIDTH" : "Technical",
    "OBV"      : "Technical",
    "STOCH_K"  : "Technical",
    "log_RV"   : "Technical",   # lagged RV is the HAR backbone — treat as technical
}

GROUP_COLORS = {
    "Macro"      : "#4C72B0",
    "Asset"      : "#DD8452",
    "Blockchain" : "#55A868",
    "Technical"  : "#C44E52",
    "Sentiment"  : "#8172B2",
}

    
# ---------- HELPERS ----------
def get_feature_cols(df):
    return [c for c in df.columns if c not in ([DATE_COL] + TARGET_COLS)]

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

def load_rf_params(csv_path):
    df = pd.read_csv(csv_path).copy()

    required_cols = ["horizon", "rf_val_max_features", "n_estimators", "n_predictors"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    param_map = {}
    for _, row in df.iterrows():
        target = row["horizon"]   # e.g. y_h1
        param_map[target] = {
            "n_estimators": int(row["n_estimators"]),
            "max_features": float(row["rf_val_max_features"]),
            "n_predictors": int(row["n_predictors"])
        }

    return param_map

def fit_rf_on_window(df_window, feature_cols, target_col, params):
    X_train = df_window[feature_cols]
    y_train = df_window[target_col].to_numpy()

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        max_depth=None,
        bootstrap=True,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, X_train

# -----------------------------------------------------------------------
# Flat feature names (with lag suffixes) → used as column labels for SHAP
# -----------------------------------------------------------------------
def make_flat_feature_names(base_cols):
    """
    Returns the 176 flat feature names in the same order as make_flat_features:
        [lag3_feat, ..., lag2_feat, ..., lag1_feat, ..., feat_t, ...]
    """
    names = []
    for lag in [3, 2, 1, 0]:
        for c in base_cols:
            names.append(c if lag == 0 else f"{c}_lag{lag}")
    return names

# -----------------------------------------------------------------------
# Core: collapse lag dimensions → base feature SHAP importance
# -----------------------------------------------------------------------
def collapse_shap_to_base(shap_vals_2d, flat_names, base_cols):
    """
    shap_vals_2d : (n_samples, 176) array of |SHAP| values
    flat_names   : list of 176 feature names (with lag suffixes)
    base_cols    : list of 44 base feature names

    For each base feature, sum |SHAP| contributions across all 4 timesteps
    (t, lag1, lag2, lag3). This gives a single importance score per feature
    that is comparable across horizons.

    Returns a Series indexed by base feature name, values = mean |SHAP|.
    """
    df_shap   = pd.DataFrame(np.abs(shap_vals_2d), columns=flat_names)
    collapsed = {}
    for c in base_cols:
        lag_cols = [c] + [f"{c}_lag{l}" for l in [1, 2, 3]]
        existing = [col for col in lag_cols if col in df_shap.columns]
        collapsed[c] = df_shap[existing].values.sum(axis=1).mean()
    return pd.Series(collapsed)


# -----------------------------------------------------------------------
# SHAP computation across rolling OOS window
# -----------------------------------------------------------------------
def compute_shap_for_h(full_df, all_features, xt_cols, h, params):
    """
    Fits RF on WINDOW_SIZE rolling windows at N_SHAP_STEP intervals
    across the OOS period, computes TreeSHAP on the window rows, and
    accumulates mean |SHAP| per base feature.

    Returns:
        base_shap : Series(44,) — mean |SHAP| per base feature, averaged
                    across all sampled refit windows
    """
    target_col = f"y_h{h}"
    df         = full_df.sort_values(DATE_COL).reset_index(drop=True)
    flat_names = make_flat_feature_names(xt_cols)
    all_features = flat_names

    # OOS start index
    idxs    = df.index[df[DATE_COL] >= EVAL_START_DATE]
    start_t = max(WINDOW_SIZE, int(idxs[0]))
    end_t   = len(df) - 1

    # Sample every N_SHAP_STEP refit points across OOS window
    refit_points = list(range(start_t, end_t + 1, N_SHAP_STEP))
    print(f"  h={h}: {len(refit_points)} refit windows sampled for SHAP")

    accumulated = []
    for i, t in enumerate(refit_points):
        if i % 5 == 0: # Print every 5th refit
            print(f"    - Processing refit {i+1}/{len(refit_points)} (Date: {df.iloc[t][DATE_COL].date()})")
        df_window      = df.iloc[t - WINDOW_SIZE : t].copy()
        # train on ALL features (including lags)
        model, X_train = fit_rf_on_window(df_window, all_features, target_col, params)

        # TreeSHAP on the training window rows (exact, not approximate)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_train)   # (WINDOW_SIZE, 176)

        base_imp  = collapse_shap_to_base(shap_vals, flat_names, xt_cols)
        accumulated.append(base_imp)

    # Average across all sampled windows
    base_shap = pd.concat(accumulated, axis=1).mean(axis=1)
    return base_shap
    

# -----------------------------------------------------------------------
# Plot 1: 2×2 grid of top-20 feature bar charts (one panel per horizon)
# -----------------------------------------------------------------------
def plot_top20_grid(shap_by_h, out_dir):
    """
    Single figure with 2×2 subplots — one panel per horizon.
    Each panel shows top 20 features coloured by group.
    Saved as xgb_shap_top20_grid.png
    """

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()   # order: H=1, H=3, H=5, H=7

    # Collect all groups present across all horizons for shared legend
    all_feats_seen = set()
    for base_shap in shap_by_h.values():
        all_feats_seen.update(base_shap.sort_values(ascending=False).head(20).index)
    legend_handles = [
        Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
        if g in {FEATURE_GROUPS.get(f, "") for f in all_feats_seen}
    ]

    for ax, h in zip(axes, HORIZONS):
        base_shap = shap_by_h[h]
        top20     = base_shap.sort_values(ascending=False).head(20)
        colors    = [GROUP_COLORS.get(FEATURE_GROUPS.get(f, "Technical"), "#999999")
                     for f in top20.index]

        ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
        ax.set_title(f"H = {h}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |SHAP value|", fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="y", labelsize=9)

    # Shared legend anchored to bottom right of the figure
    fig.legend(handles=legend_handles, loc="lower right",
               fontsize=10, frameon=True, title="Feature Group",
               bbox_to_anchor=(0.98, 0.02))

    fig.suptitle("Top 20 Features by Mean |SHAP|  —  Random Forest",
                 fontsize=15, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = os.path.join(out_dir, "rf_shap_top20_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# -----------------------------------------------------------------------
# Plot 2: Group-level importance per horizon (tests H2)
# -----------------------------------------------------------------------
def plot_group_importance(shap_by_h, out_dir):
    """
    Stacked bar chart: each bar = one horizon, segments = feature groups.
    Directly addresses H2 — do nonlinear/blockchain features gain importance
    at longer horizons relative to technical/HAR features?
    Saved as rf_shap_group_importance_by_horizon.png
    """
    group_rows = []
    for h, base_shap in shap_by_h.items():
        row = {"h": h}
        for feat, imp in base_shap.items():
            grp      = FEATURE_GROUPS.get(feat, "Technical")
            row[grp] = row.get(grp, 0.0) + imp
        group_rows.append(row)

    gdf     = pd.DataFrame(group_rows).set_index("h").sort_index()
    gdf_pct = gdf.div(gdf.sum(axis=1), axis=0) * 100   # normalise to % share

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom  = np.zeros(len(gdf_pct))

    for grp, color in GROUP_COLORS.items():
        if grp not in gdf_pct.columns:
            continue
        vals = gdf_pct[grp].values
        ax.bar(gdf_pct.index.astype(str), vals, bottom=bottom,
               color=color, label=grp, width=0.5)
        bottom += vals

    ax.set_xlabel("Forecast Horizon (H)", fontsize=11)
    ax.set_ylabel("% Share of Total |SHAP|", fontsize=11)
    ax.set_title("Feature Group Importance by Horizon  —  Random Forest",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    path = os.path.join(out_dir, "rf_shap_group_importance_by_horizon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    
# -----------------------------------------------------------------------
# Plot 3: Feature importance decay across horizons (horizon-robustness)
# -----------------------------------------------------------------------
def plot_importance_decay(shap_by_h, out_dir, top_n=10):
    """
    Line chart: x = horizon, y = mean |SHAP|, one line per top feature.
    Features that stay flat = horizon-robust.
    Features that drop sharply = short-horizon specialists.
    Directly answers the core RQ on forecast deterioration across horizons.
    Saved as rf_shap_importance_decay.png
    """
    all_shap      = pd.concat(shap_by_h.values(), axis=1)
    all_shap.columns = shap_by_h.keys()
    top_feats     = all_shap.mean(axis=1).sort_values(ascending=False).head(top_n).index

    fig, ax = plt.subplots(figsize=(9, 5))
    for feat in top_feats:
        vals  = [shap_by_h[h][feat] for h in HORIZONS]
        color = GROUP_COLORS.get(FEATURE_GROUPS.get(feat, "Technical"), "#999999")
        ax.plot(HORIZONS, vals, marker="o", label=feat, color=color, linewidth=1.8)

    ax.set_xlabel("Forecast Horizon (H)", fontsize=11)
    ax.set_ylabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Feature Importance Decay Across Horizons  —  Random Forest (Top {top_n})",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    plt.tight_layout()
    path = os.path.join(out_dir, "rf_shap_importance_decay.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

full_df = pd.read_csv(FULL_DATA_PATH)
full_df[DATE_COL] = pd.to_datetime(full_df[DATE_COL], errors="raise")
full_df = full_df.sort_values(DATE_COL).reset_index(drop=True)

all_features = get_feature_cols(full_df)
xt_cols    = get_xt_cols(full_df)   # 44 columns (everything except date, targets, lags)
rf_params = load_rf_params(RF_PARAM_PATH)

shap_by_h = {}   # h → Series(44,) of mean |SHAP| per base feature

for h in HORIZONS:
    print(f"\nComputing SHAP for h={h} ...")
    base_shap    = compute_shap_for_h(full_df, all_features, xt_cols, h, rf_params[f"y_h{h}"])
    shap_by_h[h] = base_shap

    # Save raw SHAP importance to CSV — prefixed with 
    out_csv = os.path.join(OUT_DIR, f"rf_shap_importance_h{h}.csv")
    base_shap.sort_values(ascending=False).to_csv(out_csv, header=["mean_abs_shap"])
    print(f"  Saved: {out_csv}")

# Plot 1: 2×2 grid of top-20 bar charts (all horizons in one figure)
print("\nPlotting top-20 grid ...")
plot_top20_grid(shap_by_h, OUT_DIR)

# Plot 2: group-level importance stacked bar (H2 analysis)
print("Plotting group-level importance ...")
plot_group_importance(shap_by_h, OUT_DIR)

# Plot 3: importance decay line chart (core RQ)
print("Plotting importance decay ...")
plot_importance_decay(shap_by_h, OUT_DIR, top_n=10)

print("\nAll SHAP outputs saved to:", OUT_DIR)
