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
    param_dist = {'max_features': uniform(0.10, 0.90)}

    rs = RandomizedSearchCV(
        estimator=RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        param_distributions=param_dist,
        n_iter=50,
        cv=ps,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )

    rs.fit(X_train, y_train)
    return rs.best_params_

train_df = pd.read_csv("../data/train_dataset.csv") # 1246 obs
train_df["date"] = pd.to_datetime(train_df["date"], format="%d/%m/%y")
train_df = train_df.sort_values("date").reset_index(drop=True)

test_df = pd.read_csv("../data/test_dataset.csv") # 544 obs
test_df["date"] = pd.to_datetime(test_df["date"])

full_df = pd.concat([train_df, test_df], axis=0).sort_values("date").reset_index(drop=True)

W = len(train_df)
horizons = ["y_h1", "y_h3", "y_h5", "y_h7"]

global_params = {h: {} for h in horizons}
tuning_records = []

print("Tuning RF")
for h in horizons:
    others = [col for col in horizons if col != h]
    initial_window = full_df.iloc[:W].drop(columns=others + ['date'], errors='ignore')
    num_preds = initial_window.shape[1] - 1 # Total X variables

    print(f"Tuning for {h}...")
    params_val = get_rf_params_val(initial_window, h)
    
    global_params[h] = {'val': params_val}

    # Save details for the hyperparameter file
    tuning_records.append({
        'horizon': h,
        'rf_val_max_features': params_val['max_features'],
        'n_predictors': num_preds,
        'n_estimators': 500
    })

pd.DataFrame(tuning_records).to_csv("rf_tuned_hyperparams.csv", index=False)
print("Saved rf_tuned_hyperparams.csv")
        
