import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
from matplotlib.patches import Patch

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
FULL_DATA_PATH   = "../data/full_df.csv"
LASSO_PARAM_PATH = "../models/lasso_tuned_hyperparams.csv"
OUT_DIR          = "../shap_outputs/lasso_feature_analysis"

DATE_COL         = "date"
TARGET_COLS      = ["y_h1", "y_h3", "y_h5", "y_h7"]
HORIZONS         = [1, 3, 5, 7]

WINDOW_SIZE      = 1095
REFIT_EVERY      = 7
EVAL_START_DATE  = pd.to_datetime("2024-06-29")

# -----------------------------------------------------------------------
# Feature group mapping
# -----------------------------------------------------------------------
FEATURE_GROUPS = {
    # Macroeconomic
    "DOW30"                : "Macro",
    "FTSE100"              : "Macro",
    "NASDAQ"               : "Macro",
    "SP500"                : "Macro",
    "SSE"                  : "Macro",
    "VIX"                  : "Macro",
    "OIL"                  : "Macro",
    "DXY"                  : "Macro",
    # Asset-based
    "GOLD"                 : "Asset",
    "SILVER"               : "Asset",
    "TLT"                  : "Asset",
    "IEF"                  : "Asset",
    "HYG"                  : "Asset",
    "LQD"                  : "Asset",
    "EEM"                  : "Asset",
    "YC_SLOPE_10Y_2Y"      : "Asset",
    "D_DGS2"               : "Asset",
    "D_DGS10"              : "Asset",
    "D_EFFR"               : "Asset",
    "D_HY_OAS"             : "Asset",
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
    "log_RV"   : "Technical",
}
GROUP_COLORS = {
    "Macro"      : "#4C72B0",
    "Asset"      : "#DD8452",
    "Blockchain" : "#55A868",
    "Technical"  : "#C44E52",
    "Sentiment"  : "#8172B2",
}

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def get_feature_cols(df):
    # All columns including lags — used as LASSO training input
    return [c for c in df.columns if c not in ([DATE_COL] + TARGET_COLS)]

def get_xt_cols(df):
    # Base feature columns only (no lag suffixes) — used for collapsing
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
    return max(WINDOW_SIZE, int(idxs[0]))

def load_lasso_alphas(csv_path):
    df = pd.read_csv(csv_path).copy()
    required_cols = ["horizon", "lasso_alpha"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
    alpha_map = {}
    for _, row in df.iterrows():
        alpha_map[row["horizon"]] = float(row["lasso_alpha"])
    return alpha_map

def fit_lasso_on_window(df_window, feature_cols, target_col, alpha):
    X_train = df_window[feature_cols].to_numpy()
    y_train = df_window[target_col].to_numpy()
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_s, y_train)
    return model, scaler

def make_flat_feature_names(base_cols):
    """
    Returns 176 flat feature names in the same order as make_flat_features:
        [lag3_feat, ..., lag2_feat, ..., lag1_feat, ..., feat_t, ...]
    Mirrors make_flat_feature_names in XGBoost SHAP script.
    """
    names = []
    for lag in [3, 2, 1, 0]:
        for c in base_cols:
            names.append(c if lag == 0 else f"{c}_lag{lag}")
    return names

# -----------------------------------------------------------------------
# Core: collapse lag dimensions → base feature importance
# Mirrors collapse_shap_to_base from XGBoost SHAP script
# -----------------------------------------------------------------------
def collapse_coefs_to_base(coef_df, flat_names, xt_cols):
    """
    coef_df    : DataFrame with columns = flat_names (176 lag features) + metadata
    flat_names : list of 176 feature names with lag suffixes
    xt_cols    : list of 44 base feature names (no lag suffix)

    For each base feature, sums |coefficient| across t, lag1, lag2, lag3,
    then takes the mean across all OOS rows.
    Returns a Series indexed by base feature name — directly comparable
    to the mean |SHAP| Series from XGBoost.
    """
    df_coef = coef_df[flat_names].abs()
    collapsed = {}
    for c in xt_cols:
        lag_cols = [c] + [f"{c}_lag{l}" for l in [1, 2, 3]]
        existing = [col for col in lag_cols if col in df_coef.columns]
        collapsed[c] = df_coef[existing].values.sum(axis=1).mean()
    return pd.Series(collapsed)

# -----------------------------------------------------------------------
# Coefficient extraction across rolling OOS window
# -----------------------------------------------------------------------
def get_lasso_coefs_only(df, feature_cols, base_cols, h, alpha, start_t):
    """
    Runs the rolling OOS loop for one horizon, refitting every REFIT_EVERY
    days. Records the full coefficient vector at every OOS time step.

    feature_cols : all 176 columns LASSO trains on (including lags)
    base_cols    : 44 base feature names — used to build flat_names
    Returns a DataFrame of shape (n_oos_days, 176 + metadata cols).
    """
    target_col = f"y_h{h}"
    flat_names  = make_flat_feature_names(base_cols)

    # Verify alignment between feature_cols and flat_names
    assert set(flat_names).issubset(set(feature_cols)), \
        f"Mismatch: some flat_names are missing from feature_cols for h={h}"

    end_t         = len(df) - 1
    current_model = None
    coef_history  = []

    for t in range(start_t, end_t + 1):
        train_slice = df.iloc[t - WINDOW_SIZE : t].copy()

        # Refit every REFIT_EVERY steps
        if current_model is None or ((t - start_t) % REFIT_EVERY == 0):
            current_model, current_scaler = fit_lasso_on_window(
                df_window=train_slice,
                feature_cols=feature_cols,
                target_col=target_col,
                alpha=alpha
            )

        # Record coefficients — keyed by flat_names to ensure correct alignment
        coef_row = dict(zip(flat_names, current_model.coef_))
        coef_row["date"]      = df.iloc[t][DATE_COL]
        coef_row["h"]         = h
        coef_row["n_nonzero"] = np.count_nonzero(current_model.coef_)
        coef_history.append(coef_row)

    return pd.DataFrame(coef_history)


# -----------------------------------------------------------------------
# Plot 1: Stable features — selected >80% of OOS windows (LASSO-specific)
# -----------------------------------------------------------------------
def plot_lasso_stable_features(freq_stats, out_dir):
    stable_df = freq_stats[freq_stats["Frequency"] >= 80].copy()
    if stable_df.empty:
        print("\n[NOTICE] No features met the 80% stability threshold. Skipping plot.")
        return

    plot_data      = stable_df.groupby(["h", "Group"])["Feature_Lag"].nunique().unstack(fill_value=0)
    current_groups = plot_data.columns
    colors         = [GROUP_COLORS.get(g, "#999999") for g in current_groups]

    ax = plot_data.plot(kind="bar", stacked=True, figsize=(12, 7),
                        color=colors, edgecolor="white", linewidth=0.5)
    plt.title("Stable Predictors (Selected >80% of OOS Refits)",
              fontsize=14, fontweight="bold")
    plt.ylabel("Number of Stable Features", fontsize=12)
    plt.xlabel("Forecast Horizon (H)", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Variable Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2, p.get_y() + height / 2,
                    int(height), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, "lasso_stable_features.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# -----------------------------------------------------------------------
# Plot 2: Absolute magnitude importance by group (LASSO-specific)
# -----------------------------------------------------------------------
def plot_lasso_absolute_magnitude(df_long, out_dir):
    df_long = df_long.copy()
    df_long["Abs_Coef"] = df_long["Coef"].abs()

    daily_cat_imp = (df_long.groupby(["date", "h", "Group"])["Abs_Coef"]
                     .sum().reset_index())
    final_abs_imp = (daily_cat_imp.groupby(["h", "Group"])["Abs_Coef"]
                     .mean().unstack(fill_value=0))

    # Use GROUP_COLORS instead of viridis — matches all other plots
    colors = [GROUP_COLORS.get(g, "#999999") for g in final_abs_imp.columns]

    ax = final_abs_imp.plot(kind="bar", stacked=True, figsize=(12, 7),
                            color=colors, width=0.8, edgecolor="white")
    plt.title("Out-of-Sample Category Importance: Mean Sum of Absolute Coefficients",
              fontsize=14, fontweight="bold")
    plt.ylabel("Total Absolute Magnitude", fontsize=12)
    plt.xlabel("Forecast Horizon (H)", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Variable Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for p in ax.patches:
        height = p.get_height()
        if height > 0.005:
            ax.text(p.get_x() + p.get_width() / 2, p.get_y() + height / 2,
                    f"{height:.3f}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "lasso_absolute_magnitude_importance.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# -----------------------------------------------------------------------
# Plot 3: 2×2 grid of top-20 base features (mirrors XGBoost top-20 grid)
# -----------------------------------------------------------------------
def plot_lasso_top20_grid(coef_by_h, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()

    all_feats_seen = set()
    for base_imp in coef_by_h.values():
        all_feats_seen.update(base_imp.sort_values(ascending=False).head(20).index)
    legend_handles = [
        Patch(color=c, label=g) for g, c in GROUP_COLORS.items()
        if g in {FEATURE_GROUPS.get(f, "") for f in all_feats_seen}
    ]

    for ax, h in zip(axes, HORIZONS):
        base_imp = coef_by_h[h]
        top20    = base_imp.sort_values(ascending=False).head(20)
        colors   = [GROUP_COLORS.get(FEATURE_GROUPS.get(f, "Technical"), "#999999")
                    for f in top20.index]
        ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])
        ax.set_title(f"H = {h}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |Coefficient| (summed across lags)", fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="y", labelsize=9)

    fig.legend(handles=legend_handles, loc="lower right",
               fontsize=10, frameon=True, title="Feature Group",
               bbox_to_anchor=(0.98, 0.02))
    fig.suptitle("Top 20 Features by Mean |Coefficient|  —  LASSO",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "lasso_top20_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# -----------------------------------------------------------------------
# Plot 4: Group-level importance per horizon (mirrors XGBoost group plot)
# -----------------------------------------------------------------------
def plot_lasso_group_importance(coef_by_h, out_dir):
    group_rows = []
    for h, base_imp in coef_by_h.items():
        row = {"h": h}
        for feat, imp in base_imp.items():
            grp      = FEATURE_GROUPS.get(feat, "Technical")
            row[grp] = row.get(grp, 0.0) + imp
        group_rows.append(row)

    gdf     = pd.DataFrame(group_rows).set_index("h").sort_index()
    gdf_pct = gdf.div(gdf.sum(axis=1), axis=0) * 100

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
    ax.set_ylabel("% Share of Total |Coefficient|", fontsize=11)
    ax.set_title("Feature Group Importance by Horizon  —  LASSO",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    path = os.path.join(out_dir, "lasso_group_importance_by_horizon.png")
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

feature_cols = get_feature_cols(full_df)   # 176 columns including lags
base_cols    = get_xt_cols(full_df)        # 44 base features
flat_names   = make_flat_feature_names(base_cols)
start_t      = compute_start_t(full_df)
lasso_alphas = load_lasso_alphas(LASSO_PARAM_PATH)

# -----------------------------------------------------------------------
# Extract rolling coefficients for all horizons
# -----------------------------------------------------------------------
all_h_coefs = []
for h in HORIZONS:
    print(f"Extracting LASSO coefficients for h={h}...")
    alpha    = lasso_alphas[f"y_h{h}"]
    h_coef_df = get_lasso_coefs_only(full_df, feature_cols, base_cols, h, alpha, start_t)
    all_h_coefs.append(h_coef_df)
    out_csv = os.path.join(OUT_DIR, f"lasso_coefs_h{h}.csv")
    h_coef_df.to_csv(out_csv, index=False)
    print(f"  Saved raw coefficients: {out_csv}")

master_coef_df = pd.concat(all_h_coefs, ignore_index=True)

# -----------------------------------------------------------------------
# Melt to long format for frequency/magnitude analyses
# -----------------------------------------------------------------------
df_long = master_coef_df.melt(
    id_vars=["date", "h", "n_nonzero"],
    value_vars=flat_names,
    var_name="Feature_Lag",
    value_name="Coef"
)
df_long["Base"]  = df_long["Feature_Lag"].str.replace(r"_lag[123]$", "", regex=True)
df_long["Group"] = df_long["Base"].map(FEATURE_GROUPS)

# -----------------------------------------------------------------------
# Selection frequency stats (LASSO-specific sparsity analysis)
# -----------------------------------------------------------------------
active_df    = df_long[df_long["Coef"].abs() > 1e-10].copy()
total_refits = master_coef_df.groupby("h")["date"].nunique()

freq_stats = active_df.groupby(["h", "Group", "Feature_Lag"]).size().reset_index(name="Times_Selected")
freq_stats["Frequency"] = freq_stats.apply(
    lambda x: (x["Times_Selected"] / total_refits[x["h"]]) * 100, axis=1
)
feature_detail_report = freq_stats.sort_values(["h", "Frequency"], ascending=[True, False])
detail_csv = os.path.join(OUT_DIR, "lasso_feature_selection_details.csv")
feature_detail_report.to_csv(detail_csv, index=False)
print(f"\nDetailed selection frequency saved to: {detail_csv}")

# -----------------------------------------------------------------------
# Collapse to base features (mirrors SHAP collapse)
# -----------------------------------------------------------------------
coef_by_h = {}
for h in HORIZONS:
    h_df         = master_coef_df[master_coef_df["h"] == h].copy()
    base_imp     = collapse_coefs_to_base(h_df, flat_names, base_cols)
    coef_by_h[h] = base_imp
    out_csv = os.path.join(OUT_DIR, f"lasso_collapsed_importance_h{h}.csv")
    base_imp.sort_values(ascending=False).to_csv(out_csv, header=["mean_abs_coef"])
    print(f"  Saved collapsed importance: {out_csv}")

# -----------------------------------------------------------------------
# Print summaries
# -----------------------------------------------------------------------
unique_ever = freq_stats.groupby(["h", "Group"])["Feature_Lag"].nunique().unstack(fill_value=0)
unique_ever["Total_Unique"] = unique_ever.sum(axis=1)
print("\nTotal Unique Lags Selected (At least once):")
print(unique_ever)

print("\n" + "=" * 50)
for h in HORIZONS:
    top_h = freq_stats[freq_stats["h"] == h].sort_values("Frequency", ascending=False).head(5)
    print(f"\n[Horizon H={h}] Top 5 Most Frequent Features")
    if not top_h.empty:
        for _, row in top_h.iterrows():
            print(f"  {row['Frequency']:>6.1f}% | {row['Group']:<15} | {row['Feature_Lag']}")
    else:
        print("  No features selected for this horizon.")
print("\n" + "=" * 50)

# -----------------------------------------------------------------------
# All plots
# -----------------------------------------------------------------------
print("Plotting stable features ...")
plot_lasso_stable_features(freq_stats, OUT_DIR)

print("Plotting absolute magnitude importance ...")
plot_lasso_absolute_magnitude(df_long, OUT_DIR)

print("\nPlotting top-20 grid ...")
plot_lasso_top20_grid(coef_by_h, OUT_DIR)

print("Plotting group-level importance ...")
plot_lasso_group_importance(coef_by_h, OUT_DIR)

print("\nAll LASSO feature analysis outputs saved to:", OUT_DIR)
