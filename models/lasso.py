import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def get_lasso_val_alpha(Y, target_col):
    # 80/20 Split of the training window
    split_idx = int(len(Y) * 0.8)
    train_sub = Y.iloc[:split_idx] # Training set
    val_sub = Y.iloc[split_idx:] # Validation set 
    
    X_train = train_sub.drop(columns=[target_col])
    y_train = train_sub[target_col].values
    X_val = val_sub.drop(columns=[target_col])
    y_val = val_sub[target_col].values

    # Scale strictly on the 80% to avoid leakage
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    alphas = np.logspace(-4, 2, 100)
    best_mse = np.inf
    best_alpha = 1.0
    
    for a in alphas:
        m = Lasso(alpha=a, max_iter=10000).fit(X_train_s, y_train)
        mse = mean_squared_error(y_val, m.predict(X_val_s))
        if mse < best_mse:
            best_mse = mse
            best_alpha = a
            
    return best_alpha

train_df = pd.read_csv("../data/train_dataset.csv") # 1246 obs
train_df["date"] = pd.to_datetime(train_df["date"], format="%d/%m/%y")
train_df = train_df.sort_values("date").reset_index(drop=True)

test_df = pd.read_csv("../data/test_dataset.csv") # 544 obs
test_df["date"] = pd.to_datetime(test_df["date"])

full_df = pd.concat([train_df, test_df], axis=0).sort_values("date").reset_index(drop=True)

W = len(train_df)
horizons = ['y_h1', 'y_h3', 'y_h5', 'y_h7']

global_alphas = {}
alpha_records = []

for h in horizons:
    others = [col for col in horizons if col != h]
    initial_window = full_df.iloc[:W].drop(columns=others + ['date'], errors='ignore')
    
    best_a = get_lasso_val_alpha(initial_window, h)
    global_alphas[h] = best_a
    
    alpha_records.append({'horizon': h, 'lasso_alpha': best_a})

pd.DataFrame(alpha_records).to_csv("lasso_tuned_hyperparams.csv", index=False)


