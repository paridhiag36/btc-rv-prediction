import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LassoCV, RidgeCV, LassoLarsIC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def get_lasso_cv_alpha(Y, target_col):
    scaler = StandardScaler()
    X = scaler.fit_transform(Y.drop(columns=[target_col]))
    y = Y[target_col].values
    tscv = TimeSeriesSplit(n_splits=5)
    # Finds alpha automatically via Cross-Validation
    model = LassoCV(cv=tscv, max_iter=10000).fit(X, y)
    return model.alpha_

def get_lasso_ic_alpha(Y, target_col, ic='bic'):
    scaler = StandardScaler()
    X = scaler.fit_transform(Y.drop(columns=[target_col]))
    y = Y[target_col].values
    # Finds alpha automatically via Information Criterion (BIC or AIC)
    model = LassoLarsIC(criterion=ic).fit(X, y)
    return model.alpha_

def get_ridge_cv_alpha(Y, target_col):
    scaler = StandardScaler()
    X = scaler.fit_transform(Y.drop(columns=[target_col]))
    y = Y[target_col].values
    # Searches across a log-scale grid for the best Ridge alpha
    model = RidgeCV(alphas=np.logspace(-3, 3, 100)).fit(X, y)
    return model.alpha_

def get_best_alpha_val(Y, target_col, model_type='lasso'):
    # 80/20 Split of the initial training window
    split_idx = int(len(Y) * 0.8)
    train_sub = Y.iloc[:split_idx]
    val_sub = Y.iloc[split_idx:]
    
    y_train = train_sub[target_col].values
    X_train = train_sub.drop(columns=[target_col])
    
    y_val = val_sub[target_col].values
    X_val = val_sub.drop(columns=[target_col])

    # Scale based on the 80% (to avoid leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Search across a standard log-space
    alphas = np.logspace(-4, 4, 100)
    best_mse = np.inf
    best_alpha = 1.0
    
    for a in alphas:
        if model_type == 'lasso':
            m = Lasso(alpha=a, max_iter=10000).fit(X_train_s, y_train)
        else:
            m = Ridge(alpha=a).fit(X_train_s, y_train)
            
        preds = m.predict(X_val_s)
        mse = mean_squared_error(y_val, preds)
        
        if mse < best_mse:
            best_mse = mse
            best_alpha = a
            
    return best_alpha


def run_model(Y_slice, target_col, model_type, alpha):
    train_df = Y_slice.iloc[:-1]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_out = Y_slice.iloc[[-1]].drop(columns=[target_col])
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_out_s = scaler.transform(X_out)

    if model_type == 'lasso':
        model = Lasso(alpha=alpha, max_iter=10000).fit(X_train_s, y_train)
    else: # ridge
        model = Ridge(alpha=alpha).fit(X_train_s, y_train)
        
    return model.predict(X_out_s)[0], np.sum(model.coef_ != 0)

train_df = pd.read_csv("../data/train_dataset.csv") # 1246 obs
train_df["date"] = pd.to_datetime(train_df["date"], format="%d/%m/%y")
train_df = train_df.sort_values("date").reset_index(drop=True)

test_df = pd.read_csv("../data/test_dataset.csv") # 544 obs
test_df["date"] = pd.to_datetime(test_df["date"])

full_df = pd.concat([train_df, test_df], axis=0).sort_values("date").reset_index(drop=True)

W = len(train_df)
horizons = ['y_h1', 'y_h3', 'y_h5', 'y_h7']

fixed_alphas = {h: {} for h in horizons}
tuning_records = []

print("--- STAGE 1: Tuning Alphas ---")
for h in horizons:
    others = [col for col in horizons if col != h]
    # Use only the training block for tuning
    Y_init = full_df.iloc[:W].drop(columns=others + ['date'], errors='ignore')

    a_lasso_bic = get_lasso_ic_alpha(Y_init, h, 'bic')
    a_lasso_aic = get_lasso_ic_alpha(Y_init, h, 'aic')
    a_lasso_cv  = get_lasso_cv_alpha(Y_init, h)
    a_lasso_val = get_best_alpha_val(Y_init, h, model_type='lasso')
    
    a_ridge_cv  = get_ridge_cv_alpha(Y_init, h)
    a_ridge_val = get_best_alpha_val(Y_init, h, model_type='ridge')

    # Store for use in Stage 2
    fixed_alphas[h] = {
        'bic': a_lasso_bic, 
        'aic': a_lasso_aic,
        'cv': a_lasso_cv, 
        'l_val': a_lasso_val,
        'r_cv': a_ridge_cv,
        'r_val': a_ridge_val
    }
    
    # Save to the CSV record
    tuning_records.append({
        'horizon': h, 
        'lasso_bic_alpha': a_lasso_bic, 
        'lasso_aic_alpha': a_lasso_aic,
        'lasso_cv_alpha': a_lasso_cv, 
        'lasso_val_alpha': a_lasso_val,
        'ridge_cv_alpha': a_ridge_cv,
        'ridge_val_alpha': a_ridge_val
    })

pd.DataFrame(tuning_records).to_csv("lasso_lambda.csv", index=False)


horizon_results = {h: [] for h in horizons}

print("--- STAGE 2: Running Rolling Forecast ---")
for t in range(W, len(full_df)):
    current_date = full_df.iloc[t]['date']
    if (t - W) % 100 == 0:
        print(f"Processing day {t-W}...")

    for h in horizons:
        row_entry = {'date': current_date}
        others = [col for col in horizons if col != h]
        
        # Slice current window
        Y_slice = full_df.iloc[t - W : t + 1].drop(columns=others + ['date'], errors='ignore')
        row_entry[f"actual_{h}"] = full_df.iloc[t][h]

        # Lasso
        pred_bic, count_bic = run_model(Y_slice, h, 'lasso', fixed_alphas[h]['bic'])
        row_entry[f"lasso_bic_{h}"] = pred_bic
        row_entry[f"lasso_bic_var_count_{h}"] = count_bic
        
        pred_aic, count_aic = run_model(Y_slice, h, 'lasso', fixed_alphas[h]['aic'])
        row_entry[f"lasso_aic_{h}"] = pred_aic
        row_entry[f"lasso_aic_var_count_{h}"] = count_aic
            
        pred_cv, count_cv = run_model(Y_slice, h, 'lasso', fixed_alphas[h]['cv'])
        row_entry[f"lasso_cv_{h}"] = pred_cv
        row_entry[f"lasso_cv_var_count_{h}"] = count_cv

        pred_val, count_val = run_model(Y_slice, h, 'lasso', fixed_alphas[h]['l_val'])
        row_entry[f"lasso_val_{h}"] = pred_val
        row_entry[f"lasso_val_var_count_{h}"] = count_val

        # Ridge CV
        row_entry[f"ridge_cv_{h}"], _ = run_model(Y_slice, h, 'ridge', fixed_alphas[h]['r_cv'])
        
        # Ridge 80/20 Validation
        row_entry[f"ridge_val_{h}"], _ = run_model(Y_slice, h, 'ridge', fixed_alphas[h]['r_val'])

        horizon_results[h].append(row_entry)

# Save Final Results
for h in horizons:
    pd.DataFrame(horizon_results[h]).to_csv(f"lasso_forecast_{h}.csv", index=False)
    print(f"Done for {h}")
