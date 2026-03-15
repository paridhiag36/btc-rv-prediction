import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, PredefinedSplit
from scipy.stats import uniform

def get_rf_params_val(Y, target_col):
    y_train = Y[target_col].values
    X_train = Y.drop(columns=[target_col])

    # 80/20 Validation Split
    split_idx = int(len(Y) * 0.8)
    test_fold = np.concatenate([-1 * np.ones(split_idx), 0 * np.ones(len(Y) - split_idx)])
    ps = PredefinedSplit(test_fold)
    
    # Range: 0.05 to 1.0 (5% to 100% of features)
    param_dist = {'max_features': uniform(0.05, 0.95)}

    rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=50, cv=ps, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42
    )
    rs.fit(X_train, y_train)
    return rs.best_params_

def get_rf_params_cv(Y, target_col):
    y_train = Y[target_col].values
    X_train = Y.drop(columns=[target_col])

    # 5-fold Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {'max_features': uniform(0.05, 0.95)}

    rs = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=50, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42
    )
    rs.fit(X_train, y_train)
    return rs.best_params_

def run_rf(Y_slice, target_col, fixed_params):
    train_df = Y_slice.iloc[:-1]
    forecast_row = Y_slice.iloc[[-1]]
    
    y_train = train_df[target_col].values
    X_train = train_df.drop(columns=[target_col])
    X_out = forecast_row.drop(columns=[target_col])

    model = RandomForestRegressor(**fixed_params, random_state=42)
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_out)[0]
    # Count of features drawn at each split
    num_features = int(fixed_params['max_features'] * X_train.shape[1])    
    return prediction, num_features

train_df = pd.read_csv("../data/train_dataset.csv") # 1246 obs
train_df["date"] = pd.to_datetime(train_df["date"], format="%d/%m/%y")
train_df = train_df.sort_values("date").reset_index(drop=True)

test_df = pd.read_csv("../data/test_dataset.csv") # 544 obs
test_df["date"] = pd.to_datetime(test_df["date"])

full_df = pd.concat([train_df, test_df], axis=0).sort_values("date").reset_index(drop=True)

W = len(train_df)
horizons = ['y_h1', 'y_h3', 'y_h5', 'y_h7']

global_params = {h: {} for h in horizons}
tuning_records = []

print("Tuning RF (CV and Val)")
for h in horizons:
    others = [col for col in horizons if col != h]
    initial_window = full_df.iloc[:W].drop(columns=others + ['date'], errors='ignore')
    num_preds = initial_window.shape[1] - 1 # Total X variables

    print(f"Tuning for {h}...")
    params_val = get_rf_params_val(initial_window, h)
    params_cv  = get_rf_params_cv(initial_window, h)
    
    global_params[h] = {'val': params_val, 'cv': params_cv}

    # Save details for the hyperparameter file
    tuning_records.append({
        'horizon': h,
        'rf_val_max_features': params_val['max_features'],
        'rf_val_m_count': int(params_val['max_features'] * num_preds),
        'rf_cv_max_features': params_cv['max_features'],
        'rf_cv_m_count': int(params_cv['max_features'] * num_preds)
    })

pd.DataFrame(tuning_records).to_csv("rf_hyperparameters.csv", index=False)
print("Saved rf_hyperparameters.csv")

horizon_data = {h: [] for h in horizons}

print("Rolling Forecast")
for t in range(W, len(full_df)):
    current_date = full_df.iloc[t]['date']
    if (t - W) % 50 == 0:
        print(f"Processing Day {t-W}...")

    for h in horizons:
        row_entry = {'date': current_date}
        others = [col for col in horizons if col != h]
        Y_slice = full_df.iloc[t - W : t + 1].drop(columns=others + ['date'], errors='ignore')
        row_entry[f"actual_{h}"] = full_df.iloc[t][h]
        
        pred_val, count_val = run_rf(Y_slice, h, global_params[h]['val'])
        row_entry[f"rf_val_{h}_pred"] = pred_val
        row_entry[f"rf_val_{h}_m"] = count_val 

        pred_cv, count_cv = run_rf(Y_slice, h, global_params[h]['cv'])
        row_entry[f"rf_cv_{h}_pred"] = pred_cv
        row_entry[f"rf_cv_{h}_m"] = count_cv 

        horizon_data[h].append(row_entry)

for h in horizons:
    pd.DataFrame(horizon_data[h]).to_csv(f"rf_forecast_results_{h}.csv", index=False)
    print(f"Saved results for {h}")
