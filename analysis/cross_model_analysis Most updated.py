"""
Cross-Model Comparison
==========================================================
HOW TO RUN:
  1. Clone the repo so the folder structure is intact:
       btc-rv-prediction/
       ├── analysis/                  ← this script lives here
       └── forecast evaluations/
           ├── har_outputs/
           ├── lasso_outputs/
           ├── lstm_outputs/
           ├── rf_outputs/
           ├── ridge_outputs/
           ├── svr_outputs/
           └── xgb_outputs/
  2. Run:  python "cross_model_analysis Most updated.py"
  3. Outputs (PNGs + printed analysis) go to analysis/cross_model_outputs/


"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings, sys
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# PATHS — everything is relative to wherever this script lives
# ═══════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
FORE_DIR   = (SCRIPT_DIR / '../forecast evaluations').resolve()
OUT_DIR    = (SCRIPT_DIR / 'cross_model_outputs').resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not FORE_DIR.exists():
    print(f"\n  ERROR: Could not find '{FORE_DIR}'")
    print(f"  Make sure 'forecast evaluations' folder exists at the repo root.\n")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# 1. Load all prediction CSVs
# ═══════════════════════════════════════════════════════════════════

FILES = {
    'LASSO':   FORE_DIR / 'lasso_outputs'  / 'lasso_rolling_oos_predictions.csv',
    'LSTM':    FORE_DIR / 'lstm_outputs'   / 'lstm_rolling_oos_predictions.csv',
    'RF':      FORE_DIR / 'rf_outputs'     / 'rf_rolling_oos_predictions.csv',
    'Ridge':   FORE_DIR / 'ridge_outputs'  / 'ridge_rolling_oos_predictions.csv',
    'SVR':     FORE_DIR / 'svr_outputs'    / 'svr_rolling_oos_predictions.csv',
    'XGBoost': FORE_DIR / 'xgb_outputs'   / 'xgb_rolling_oos_predictions.csv',
    '_HAR_':   FORE_DIR / 'har_outputs'   / 'har_family_rolling_oos_predictions.csv',
}

missing = [name for name, path in FILES.items() if not path.exists()]
if missing:
    print("\n  ERROR: Missing prediction CSVs:")
    for m in missing:
        print(f"    - {FILES[m]}")
    sys.exit(1)

frames = []
for key, fpath in FILES.items():
    df = pd.read_csv(fpath, parse_dates=['date'])
    df = df.rename(columns={'y_true': 'actual', 'y_pred': 'predicted'})
    if key != '_HAR_':
        df['model'] = key
    df['horizon'] = 'y_h' + df['h'].astype(str)
    frames.append(df[['date', 'model', 'horizon', 'actual', 'predicted']])

all_preds = pd.concat(frames, ignore_index=True)
all_preds['sq_error'] = (all_preds['actual'] - all_preds['predicted']) ** 2
all_preds['abs_error'] = np.abs(all_preds['actual'] - all_preds['predicted'])
all_preds['year_month'] = all_preds['date'].dt.to_period('M')

print(f"Loaded {len(all_preds)} predictions")
print(f"Models: {sorted(all_preds['model'].unique())} ({all_preds['model'].nunique()} total)")
print(f"Horizons: {sorted(all_preds['horizon'].unique())}")
print(f"Date range: {all_preds['date'].min().date()} -> {all_preds['date'].max().date()}")
n_per = len(all_preds) // (all_preds['model'].nunique() * 4)
print(f"Predictions per (model, horizon): ~{n_per}")

# ═══════════════════════════════════════════════════════════════════
# 2. RMSE & R² tables
# ═══════════════════════════════════════════════════════════════════

rmse_rows = []
for model_name in sorted(all_preds['model'].unique()):
    for h in [1, 3, 5, 7]:
        sub = all_preds[(all_preds['model'] == model_name) & (all_preds['horizon'] == f'y_h{h}')]
        if len(sub) == 0:
            continue
        rmse = np.sqrt(np.mean(sub['sq_error']))
        ss_res = np.sum(sub['sq_error'])
        ss_tot = np.sum((sub['actual'] - sub['actual'].mean())**2)
        r2 = 1 - ss_res / ss_tot
        rmse_rows.append({'model': model_name, 'h': h, 'rmse': rmse, 'r2': r2})

rmse_df = pd.DataFrame(rmse_rows)

print("\n" + "="*80)
print("  COMPLETE MODEL COMPARISON — RMSE (lower is better)")
print("="*80)
rmse_pivot = rmse_df.pivot(index='model', columns='h', values='rmse').round(4)
rmse_pivot.columns = [f'h={h}' for h in rmse_pivot.columns]
rmse_pivot = rmse_pivot.sort_values('h=1')
print(rmse_pivot.to_string())

print("\n" + "="*80)
print("  COMPLETE MODEL COMPARISON — R² (higher is better)")
print("="*80)
r2_pivot = rmse_df.pivot(index='model', columns='h', values='r2').round(4)
r2_pivot.columns = [f'h={h}' for h in r2_pivot.columns]
r2_pivot = r2_pivot.sort_values('h=1', ascending=False)
print(r2_pivot.to_string())

# ═══════════════════════════════════════════════════════════════════
# 3. Monthly MSE analysis
# ═══════════════════════════════════════════════════════════════════

TOP_N = 5

def monthly_analysis(all_preds, horizon, top_n=TOP_N):
    sub = all_preds[all_preds['horizon'] == horizon].copy()
    monthly = (sub.groupby(['year_month', 'model'])['sq_error']
               .mean().reset_index().rename(columns={'sq_error': 'mse'}))
    pivot = monthly.pivot(index='year_month', columns='model', values='mse').sort_index()
    worst = {}
    for m in pivot.columns:
        col = pivot[m].dropna().sort_values(ascending=False)
        worst[m] = set(col.head(top_n).index)
    month_counts = {}
    for m, months in worst.items():
        for mo in months:
            month_counts[mo] = month_counts.get(mo, []) + [m]
    shared = {mo: models for mo, models in month_counts.items() if len(models) >= 2}
    n_models = len(pivot.columns)
    universal = {mo: models for mo, models in month_counts.items() if len(models) == n_models}
    return pivot, worst, shared, universal


for horizon in ['y_h1', 'y_h3', 'y_h5', 'y_h7']:
    pivot, worst, shared, universal = monthly_analysis(all_preds, horizon)
    n_models = len(pivot.columns)
    print(f"\n{'='*70}")
    print(f"  MONTHLY MSE ANALYSIS — {horizon}  ({n_models} models)")
    print(f"{'='*70}")
    for m in sorted(pivot.columns):
        ranked = pivot[m].dropna().sort_values(ascending=False).head(TOP_N)
        print(f"\n  {m} — Top {TOP_N} worst months:")
        for mo, mse in ranked.items():
            flag = " ← SHARED" if mo in shared else ""
            print(f"    {mo}  MSE = {mse:.4f}{flag}")
    print(f"\n  {'─'*50}")
    if universal:
        print(f"  UNIVERSAL bad months (ALL {n_models} models struggled):")
        for mo in sorted(universal.keys()):
            avg_mse = pivot.loc[mo].mean()
            print(f"    → {mo}  (avg MSE = {avg_mse:.4f}) — exogenous shock")
    if shared:
        print(f"\n  Months shared by 2+ models:")
        for mo in sorted(shared.keys(), key=lambda x: -len(shared[x])):
            who = ', '.join(sorted(shared[mo]))
            n = len(shared[mo])
            print(f"    → {mo}  ({n}/{n_models} models: {who})")
    unique_bad = {}
    for m in pivot.columns:
        unique = worst[m] - set(shared.keys())
        if unique:
            unique_bad[m] = unique
    if unique_bad:
        print(f"\n  Model-specific bad months (only that model struggled):")
        for m, months in sorted(unique_bad.items()):
            for mo in sorted(months):
                print(f"    → {mo}  ({m} only)")

# ═══════════════════════════════════════════════════════════════════
# 4. Statistical vs ML split
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  KEY ANALYSIS: Statistical (HAR) vs ML models — overlap in worst months")
print("="*70)

stat_models = ['HAR-RV', 'HAR-RV-J', 'HAR-RV-J-H']
ml_models = ['LASSO', 'LSTM', 'RF', 'Ridge', 'SVR', 'XGBoost']

for horizon in ['y_h1', 'y_h7']:
    pivot, worst, _, _ = monthly_analysis(all_preds, horizon)
    stat_worst = set()
    for m in stat_models:
        if m in worst: stat_worst |= worst[m]
    ml_worst = set()
    for m in ml_models:
        if m in worst: ml_worst |= worst[m]
    both = stat_worst & ml_worst
    stat_only = stat_worst - ml_worst
    ml_only = ml_worst - stat_worst
    print(f"\n  {horizon}:")
    print(f"    Bad for BOTH stat & ML:  {', '.join(str(m) for m in sorted(both)) or '—'}")
    print(f"    Bad for stat only:       {', '.join(str(m) for m in sorted(stat_only)) or '—'}")
    print(f"    Bad for ML only:         {', '.join(str(m) for m in sorted(ml_only)) or '—'}")

# ═══════════════════════════════════════════════════════════════════
# 5. Visualizations
# ═══════════════════════════════════════════════════════════════════

colors_map = {
    'XGBoost': '#2ca02c', 'RF': '#e377c2', 'LASSO': '#ff7f0e',
    'SVR': '#8c564b', 'Ridge': '#9467bd', 'LSTM': '#d62728',
    'HAR-RV': '#1f77b4', 'HAR-RV-J': '#17becf', 'HAR-RV-J-H': '#aec7e8'
}
model_order = ['XGBoost', 'RF', 'LASSO', 'SVR', 'Ridge', 'LSTM',
               'HAR-RV-J-H', 'HAR-RV-J', 'HAR-RV']

# Plot A: RMSE bar chart
fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
fig.suptitle('OOS RMSE — All Models × All Horizons', fontsize=14, fontweight='bold')
for idx, h in enumerate([1, 3, 5, 7]):
    ax = axes[idx]
    h_data = rmse_df[rmse_df['h'] == h].set_index('model').reindex(model_order)
    colors = [colors_map.get(m, '#333') for m in model_order]
    bars = ax.barh(model_order, h_data['rmse'], color=colors, alpha=0.85)
    ax.set_title(f'h = {h}', fontsize=12, fontweight='bold')
    ax.set_xlim(0.6, 1.0)
    ax.axvline(x=h_data['rmse'].min(), color='green', linestyle='--', alpha=0.5, linewidth=0.8)
    for bar, val in zip(bars, h_data['rmse']):
        if not np.isnan(val):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=7)
    if idx == 0: ax.set_xlabel('RMSE')
plt.tight_layout()
fig.savefig(OUT_DIR / 'rmse_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: outputs/rmse_all_models.png")

# Plot B: Degradation chart
fig, ax = plt.subplots(figsize=(11, 7))
har_style = {'linestyle': '--', 'marker': 's', 'linewidth': 1.5, 'markersize': 5, 'alpha': 0.7}
ml_style  = {'linestyle': '-', 'marker': 'o', 'linewidth': 2, 'markersize': 6}
for model_name in sorted(all_preds['model'].unique()):
    rmses = []
    for h in [1, 3, 5, 7]:
        sub = all_preds[(all_preds['model'] == model_name) & (all_preds['horizon'] == f'y_h{h}')]
        rmses.append(np.sqrt(np.mean(sub['sq_error'])))
    style = har_style if 'HAR' in model_name else ml_style
    ax.plot([1, 3, 5, 7], rmses, label=model_name, color=colors_map.get(model_name, '#333'), **style)
ax.set_xlabel('Forecast Horizon (h)', fontsize=12)
ax.set_ylabel('OOS RMSE', fontsize=12)
ax.set_title('RMSE Degradation Across Horizons — All Models\n(dashed = HAR family, solid = ML)', fontsize=13, fontweight='bold')
ax.set_xticks([1, 3, 5, 7])
ax.legend(fontsize=9, ncol=2, loc='upper left')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / 'degradation_all_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/degradation_all_models.png")

# Plot C: Monthly MSE bars
for horizon in ['y_h1', 'y_h7']:
    pivot, _, shared, _ = monthly_analysis(all_preds, horizon)
    fig, ax = plt.subplots(figsize=(20, 7))
    models = sorted(pivot.columns)
    n_models = len(models)
    x = np.arange(len(pivot))
    width = 0.8 / n_models
    palette = ['#1f77b4', '#aec7e8', '#17becf', '#ff7f0e', '#d62728',
               '#e377c2', '#9467bd', '#8c564b', '#2ca02c']
    for i, m in enumerate(models):
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, pivot[m].fillna(0), width, label=m, color=palette[i % len(palette)], alpha=0.85)
    for mo in shared.keys():
        if mo in pivot.index:
            idx_pos = list(pivot.index).index(mo)
            ax.axvspan(idx_pos - 0.45, idx_pos + 0.45, alpha=0.12, color='red')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in pivot.index], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean Squared Error')
    ax.set_title(f'Monthly MSE — {horizon} | {n_models} models (red = bad for 2+ models)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fname = f'monthly_mse_{horizon}.png'
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: outputs/{fname}")

# Plot D: Zoomed worst months
def plot_zoomed_month(all_preds, horizon, year_month_str, out_dir):
    sub = all_preds[all_preds['horizon'] == horizon].copy()
    sub['ym_str'] = sub['date'].dt.strftime('%Y-%m')
    month_data = sub[sub['ym_str'] == year_month_str]
    if month_data.empty: return
    models = sorted(month_data['model'].unique())
    fig, ax = plt.subplots(figsize=(15, 5))
    first = month_data[month_data['model'] == models[0]].sort_values('date')
    ax.plot(first['date'], first['actual'], 'k-', linewidth=2.5, label='Actual', zorder=10)
    for m in models:
        m_data = month_data[month_data['model'] == m].sort_values('date')
        c = colors_map.get(m, '#333')
        ls = '--' if 'HAR' in m else '-'
        lw = 1.0 if 'HAR' in m else 1.3
        ax.plot(m_data['date'], m_data['predicted'], linewidth=lw, label=m, color=c, alpha=0.8, linestyle=ls)
    ax.set_title(f'Actual vs Predicted | {horizon} | {year_month_str}', fontsize=12, fontweight='bold')
    ax.set_ylabel('log(RV)')
    ax.legend(loc='upper right', ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = f'zoom_{year_month_str.replace("-","_")}_{horizon}.png'
    fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: outputs/{fname}")

for horizon in ['y_h1', 'y_h7']:
    pivot, _, _, _ = monthly_analysis(all_preds, horizon)
    avg_mse = pivot.mean(axis=1).sort_values(ascending=False)
    for mo in avg_mse.head(3).index:
        plot_zoomed_month(all_preds, horizon, str(mo), OUT_DIR)

# ═══════════════════════════════════════════════════════════════════
# 6. Final summary
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  CROSS-MODEL MONTHLY OVERLAP SUMMARY")
print("="*70)
for horizon in ['y_h1', 'y_h3', 'y_h5', 'y_h7']:
    pivot, worst, shared, universal = monthly_analysis(all_preds, horizon)
    avg_mse = pivot.mean(axis=1).sort_values(ascending=False)
    n_models = len(pivot.columns)
    print(f"\n  {horizon}:")
    print(f"    Worst month (avg across {n_models} models): {avg_mse.index[0]} (MSE = {avg_mse.iloc[0]:.4f})")
    if universal:
        print(f"    Universal ({n_models}/{n_models}): {', '.join(str(m) for m in sorted(universal.keys()))}")
    shared_only = {mo: m for mo, m in shared.items() if mo not in universal}
    if shared_only:
        shared_str = ', '.join(f"{mo} ({len(m)}/{n_models})" for mo, m in sorted(shared_only.items(), key=lambda x: -len(x[1])))
        print(f"    Partial overlap: {shared_str}")

print(f"\n\nAll outputs saved to: {OUT_DIR}/")
print("Done.")
