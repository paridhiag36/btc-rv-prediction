# Bitcoin Realised Volatility Forecasting

This repository contains a comparative forecasting pipeline for **Bitcoin realised volatility (RV)** using high-frequency Binance data together with macro, financial, on-chain, and technical predictors. The project builds daily RV targets from 5-minute BTC prices, engineers multi-source predictors, tunes several forecasting models, and evaluates them using a **rolling out-of-sample framework**.

## Project objective

The goal is to forecast future Bitcoin realised volatility at multiple horizons:

- **1 day ahead** (`y_h1`)
- **3 days ahead** (`y_h3`)
- **5 days ahead** (`y_h5`)
- **7 days ahead** (`y_h7`)

The target is based on **log realised variance**, constructed from intraday 5-minute BTC returns.

## Methods included

The repository contains both benchmark and machine-learning models:

- **HAR family**
  - HAR-RV
  - HAR-RV-J
  - HAR-RV-J-H
- **Ridge regression**
- **LASSO**
- **Random Forest**
- **Support Vector Regression (SVR)**
- **XGBoost**
- **LSTM**

It also includes:

- **Rolling out-of-sample forecast evaluation**
- **Diebold-Mariano testing**
- **Model comparison plots**
- **Feature-importance analysis**
  - SHAP for tree-based models
  - Permutation feature importance for SVR
  - coefficient/selection-style analysis for linear models
  - SHAP for LSTM

## Repository structure

```text
btc-rv-prediction/
├── data extraction code/
├── data/
├── exploratory data analysis/
├── models/
├── forecast evaluations/
├── model feature importance/
├── figs/
│   └── all_models/
```

### `data extraction code/`
Scripts used to fetch, clean, and assemble the modelling dataset.

Key files:
- `target_data_prep.py` — builds daily realised variance and multi-horizon targets from 5-minute BTC prices
- `btc_technical_indicators.py` — creates daily technical indicators from Binance OHLCV data
- `macro_yf_fetch.py` — downloads macro market series from Yahoo Finance and computes daily log returns
- `finance_extra_yf.py` — downloads additional financial predictors from Yahoo Finance
- `fred_macro_fetch.py` — fetches macro series from FRED and constructs daily features
- `blockchaincom_onchain_metrics_fetch.py` — fetches on-chain metrics from Blockchain.com
- `merge_model_dataset.py` — merges predictors and targets, creates lagged features, and prepares train/test datasets

### `data/`
Contains raw/intermediate/final CSV files used in the project.

Important datasets include:
- `btc_5min_binance_2021_2025.csv` — raw 5-minute BTC data
- `btc_daily_rv_targets_2021_2025.csv` — daily RV and forecasting targets
- `btc_technical_indicators_daily_2021_2025.csv` — daily technical features
- `final_macro_df.csv` — merged macro predictor set
- `finance_extra_logret_2021_2025.csv` — additional financial returns/features
- `fred_macro_features_daily_2021_2025.csv` — FRED-based daily macro features
- `onchain_features_blockchaincom_2021_2025.csv` — on-chain feature set
- `btc_volatility_model_dataset_2021_2025.csv` — merged modelling dataset before lag expansion
- `full_df.csv` — final modelling dataset with lagged predictors
- `train_dataset.csv` / `test_dataset.csv` — chronological split used for tuning/evaluation

### `exploratory data analysis/`
EDA notebooks for the raw 5-minute and hourly BTC datasets.

### `models/`
Model tuning scripts and notebooks.

Examples:
- `lasso.py`
- `ridge.py`
- `rf.py`
- `xgb_tuning_final.py`
- `svr_tuning.ipynb`
- `lstm_tuning_final.ipynb`
- `HAR.ipynb`
- `HAR_Analysis_Notebook.ipynb`

This folder also stores tuned hyperparameter summaries such as:
- `lasso_tuned_hyperparams.csv`
- `ridge_tuned_hyperparams.csv`
- `rf_tuned_hyperparams.csv`
- `xgb_tuned_hyperparams.csv`
- `svr_final_summary.csv`
- `lstm_tuned_hyperparams.csv`

### `forecast evaluations/`
Rolling out-of-sample evaluation scripts and saved outputs.

Examples:
- `har_rolling_window.py`
- `lasso_rolling_window.py`
- `ridge_rolling_window.py`
- `rf_rolling_window.py`
- `svr_rolling_window.py`
- `xgb_rolling_window.py`
- `lstm_rolling_window.py`
- `dm_test.ipynb`
- `plots.ipynb`

### `model feature importance/`
Post-estimation interpretation workflows.

Examples:
- `lasso_feature_analysis.py`
- `rf_shap_analysis.py`
- `svr_pfi_analysis.ipynb`
- `xgb_shap_analysis.py`

### `figs/all_models/`
Saved prediction-comparison and rolling-RMSE figures.

### `shap_outputs/`
Saved SHAP importance tables and visualisations.

## Data and feature construction

### 1. Realised volatility target
`target_data_prep.py`:
- reads 5-minute BTC close prices
- computes intraday log returns within each day
- aggregates squared returns to daily **realised variance (RV)**
- applies a log transform to obtain `log_RV`
- creates future targets `y_h1`, `y_h3`, `y_h5`, and `y_h7`

### 2. Predictor groups
The modelling dataset combines several predictor blocks:

- **Macro market indices** from Yahoo Finance
- **Extra financial proxies** such as DXY, gold, silver, and bond/credit ETFs
- **FRED macro series** such as yields, policy rates, and credit spread measures
- **On-chain metrics** from Blockchain.com
- **Technical indicators** built from intraday BTC OHLCV data

### 3. Lag structure
`merge_model_dataset.py` expands the predictor matrix with **lags 1, 2, and 3** for all non-target predictors. The final dataset therefore contains:

- current predictors at time `t`
- lagged predictors at `t-1`, `t-2`, and `t-3`
- forecast targets at multiple horizons

## Forecast design

The project uses a **rolling-window out-of-sample evaluation**.

Common settings across the rolling evaluation scripts include:
- **window size:** 1095 observations (approximately 3 years of daily data)
- **refit frequency:** every 7 days
- **evaluation start date:** `2024-06-29`
- **forecast horizons:** 1, 3, 5, and 7 days

This setup is designed to mimic real forecasting conditions more closely than a single static train/test split.

## Suggested reproduction workflow

Because most scripts use **relative paths**, it is best to run them **from the folder where the script is stored**.

### Option A: use the processed data already included in the repo
If you only want to reproduce the forecasting results, start from model tuning or directly from rolling evaluation.

### Option B: rebuild the pipeline from raw data

#### Step 1 — build targets and predictors
From `data extraction code/`:

```bash
cd "data extraction code"
python target_data_prep.py
python btc_technical_indicators.py
python macro_yf_fetch.py
python finance_extra_yf.py
python fred_macro_fetch.py
python blockchaincom_onchain_metrics_fetch.py
python merge_model_dataset.py
```

#### Step 2 — tune model hyperparameters
From `models/`:

```bash
cd ../models
python lasso.py
python ridge.py
python rf.py
python xgb_tuning_final.py
```

Then run the notebook-based tuning workflows as needed:
- `svr_tuning.ipynb`
- `lstm_tuning_final.ipynb`
- HAR notebooks if you want to reproduce the HAR specifications step by step

#### Step 3 — run rolling out-of-sample evaluations
From `forecast evaluations/`:

```bash
cd ../forecast\ evaluations
python har_rolling_window.py
python lasso_rolling_window.py
python ridge_rolling_window.py
python rf_rolling_window.py
python svr_rolling_window.py
python xgb_rolling_window.py
python lstm_rolling_window.py
```

Then open:
- `plots.ipynb` for visual comparisons
- `dm_test.ipynb` for Diebold-Mariano tests

#### Step 4 — run feature-importance analysis
From `model feature importance/`:

```bash
cd ../model\ feature\ importance
python lasso_feature_analysis.py
python rf_shap_analysis.py
python xgb_shap_analysis.py
```

For SVR interpretation, run:
- `svr_pfi_analysis.ipynb`

## Installation

A formal environment file is not currently included, so package installation is manual.
A likely starting point is:

```bash
pip install pandas numpy scikit-learn scipy matplotlib jupyter \
            xgboost tensorflow shap yfinance pandas-datareader requests
```

Depending on your environment and notebook usage, you may also need additional Jupyter-related packages.

## Notes and caveats

1. **Relative paths matter.**
   Many scripts reference files using paths like `../data/...`, so running them from the wrong working directory may break the pipeline.

2. **Some save lines are commented out.**
   In parts of the data-prep workflow, some `.to_csv(...)` lines are commented. If you want to regenerate datasets from scratch, review the scripts and uncomment the relevant save statements.

3. **Folder names contain spaces.**
   This is manageable, but it makes command-line execution more error-prone. Renaming folders to snake_case would improve usability.

4. **The repo mixes scripts, notebooks, and generated outputs.**
   This is useful for a class project, but a future cleanup could separate source code from generated artefacts more clearly.

## Recommended future cleanup

To make the repository easier for collaborators, markers, and future users to navigate, the following improvements would help:

- add a root-level `requirements.txt` or `environment.yml`
- add a simple `run_pipeline.sh` or `Makefile`
- rename folders to avoid spaces
- separate raw data, processed data, and generated outputs more systematically
- add a small sample dataset if the full data is too large for lightweight reproduction
- document expected runtime for tuning and rolling evaluation scripts

## Outputs available in the repo

The repository already includes:
- processed datasets
- tuned hyperparameter tables
- rolling prediction files
- RMSE summary files
- comparison plots
- SHAP and feature-importance outputs

This makes the repo useful both as a **research archive** and as a **reproducible project handover** for the realised-volatility forecasting study.

## Acknowledgement

This repository appears to have been developed as part of a comparative realised-volatility forecasting project on Bitcoin, with emphasis on:
- multi-horizon forecasting
- benchmark vs machine-learning model comparison
- rolling out-of-sample performance
- interpretability through feature-importance analysis

If you use this repository for extension work, it is a good idea to cite both the modelling approach and the data sources used in the scripts.
