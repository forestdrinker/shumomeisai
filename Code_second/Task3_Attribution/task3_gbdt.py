
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import glob
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import json

# Paths
DATASET_PATH = r'd:\shumomeisai\Code_second\Results\task3_data\task3_weekly_dataset.parquet'
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\task3_analysis'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_task3_gbdt():
    print("--- Running Task 3 GBDT Analysis ---")
    
    # 1. Load Data
    print("Loading Weekly Dataset...")
    df = pd.read_parquet(DATASET_PATH)
    
    # Preprocessing
    # Encode Categoricals
    le_industry = LabelEncoder()
    df['industry_enc'] = le_industry.fit_transform(df['industry'].astype(str))
    
    le_partner = LabelEncoder()
    df['partner_enc'] = le_partner.fit_transform(df['ballroom_partner'].astype(str))
    
    feature_cols = ['age', 'industry_enc', 'partner_enc', 'week', 
                    'rolling_avg_pJ', 'rolling_std_pJ',
                    'partner_network_degree', 'partner_network_pagerank', 'partner_network_closeness',
                    'partner_embedding_0', 'partner_embedding_1', 'partner_embedding_2', 'partner_embedding_3']
    
    X = df[feature_cols]
    y_judge = df['pJ_it']
    # y_fan_point = df['v_mean'] # Use point estimate for main CV
    # Note: v_mean might be NaN if no summary loaded. Check.
    if 'v_mean' not in df.columns or df['v_mean'].isna().all():
        print("Warning: v_mean missing. Skipping Fan GBDT.")
        RUN_FAN = False
    else:
        y_fan_point = df['v_mean'].fillna(0) # Handle missing if any (shouldn't be for active)
        RUN_FAN = True

    groups = df['season']
    
    # 2. Main Model CV (Judge)
    print("Training Judge GBDT (CV)...")
    clf_j = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, 
                             enable_categorical=False, n_jobs=4)
    
    # Simple GroupKFold CV
    gkf = GroupKFold(n_splits=5)
    rmse_scores = []
    
    for train_idx, test_idx in gkf.split(X, y_judge, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_judge.iloc[train_idx], y_judge.iloc[test_idx]
        
        clf_j.fit(X_train, y_train)
        pred = clf_j.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - pred)**2))
        rmse_scores.append(rmse)
        
    print(f"Judge CV RMSE: {np.mean(rmse_scores):.4f}")
    
    # Save CV Metrics
    cv_metrics = {
        "judge_cv_rmse_mean": np.mean(rmse_scores),
        "judge_cv_rmse_std": np.std(rmse_scores),
        "n_folds": 5
    }
    with open(os.path.join(OUTPUT_DIR, 'task3_gbdt_cv_metrics.json'), 'w') as f:
        json.dump(cv_metrics, f, indent=2)
    
    # 3. Bootstrap SHAP CI (Fan)
    # Only if Fan data exists
    if RUN_FAN:
        print("Running Fan GBDT Bootstrap SHAP CI...")
        
        # We need posterior samples for "proper" CI (method 1 in doc)
        # Default: Train/Tune on point estimate, but in bootstrap sample, 
        # replace y_fan with a posterior draw y_fan^(r).
        
        # Load posterior metadata/pointers
        # Similar logic to LMM script to load draws on the fly?
        # Since bootstrap loop B is small (e.g. 50-100), we can load on fly.
        
        season_files = glob.glob(os.path.join(POSTERIOR_DIR, "season_*.npz"))
        valid_files = [f for f in season_files if 'summary' not in f]
        
        # To avoid re-loading/mapping every time in bootstrap, let's pre-load *some* draws into memory?
        # Or just use the point estimate if posterior is too heavy?
        # Doc says "Default: ... random draw ... generate labels".
        
        # Let's perform the mapping logic once for ALL draws? No, memory.
        # Let's clean mapping logic: 
        # Map (season, week, pair_id) -> row_index in df.
        df['row_id'] = range(len(df))
        lookup = df.set_index(['season', 'week', 'pair_id'])['row_id'].to_dict()
        
        # We will create a matrix Y_draws (N_rows, N_draws) roughly?
        # If N=2400, Draws=50, Matrix is small (2400x50 float).
        # We can pre-load 50 draws!
        N_DRAWS = 50
        Y_draws = np.zeros((len(df), N_DRAWS)) * np.nan
        
        print(f"Pre-loading {N_DRAWS} posterior draws for bootstrap...")
        
        for fpath in valid_files:
            try:
                data = np.load(fpath)
                s_id = int(os.path.basename(fpath).split('_')[1].split('.')[0])
                
                if 'v' not in data: continue
                v_all = data['v'] # (R, T, N)
                
                # Pick 50 random indices from R
                R_avail = v_all.shape[0]
                total_draws = min(N_DRAWS, R_avail)
                idx_draws = np.linspace(0, R_avail-1, N_DRAWS, dtype=int)
                
                c_ids = data.get('pair_ids')
                if c_ids is None: c_ids = data.get('couple_ids')
                if c_ids is None: continue
                
                # Iterate time and couples
                n_weeks = v_all.shape[1]
                for t in range(n_weeks):
                    wk = t + 1
                    for n, pid in enumerate(c_ids):
                         # Get row index
                        rid = lookup.get((s_id, wk, pid))
                        if rid is not None:
                            # Fill Y_draws
                            # v_all[idx_draws, t, n] -> shape (N_DRAWS,)
                            Y_draws[rid, :] = v_all[idx_draws, t, n]
            except:
                pass
                
        # Fill missing with point estimate? Or leave NaN (and drop during training)?
        # For simplicity, if missing draws, fill with point estimate.
        # Check NaNs
        mask_nan = np.isnan(Y_draws)
        if mask_nan.any():
            # Broadcast mean
            means = df['v_mean'].values[:, None]
            Y_draws[mask_nan] = np.broadcast_to(means, Y_draws.shape)[mask_nan]

        # Bootstrap Loop
        B = 50 
        shap_importances_fan = []
        shap_importances_judge = []
        
        for b in range(B):
            if b % 10 == 0: print(f"  Bootstrap {b}/{B}...")
            
            # Block Bootstrap by Season
            unique_seasons = df['season'].unique()
            season_sample = np.random.choice(unique_seasons, size=len(unique_seasons), replace=True)
            
            # Construct dataset indices
            indices = []
            for s in season_sample:
                s_indices = df.index[df['season'] == s].tolist()
                indices.extend(s_indices)
            
            X_b = X.iloc[indices]
            
            # --- Fan Target (Posterior Sampled) ---
            # Draw index for this bootstrap iter
            r_idx = np.random.randint(0, N_DRAWS)
            y_b_fan = Y_draws[indices, r_idx]
            
            # --- Judge Target (Fixed) ---
            y_b_judge = df['pJ_it'].iloc[indices].values
            
            # Clean NaNs (intersection of valid)
            # Usually y_judge is valid. y_fan might have NaNs (if loading failed).
            mask_valid = (~np.isnan(y_b_fan)) & (~np.isnan(y_b_judge))
            X_b = X_b[mask_valid]
            y_b_fan = y_b_fan[mask_valid]
            y_b_judge = y_b_judge[mask_valid]
            
            if len(X_b) == 0: continue
            
            # Explainer helper
            def get_mean_shap(X_train, y_train):
                model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1)
                model.fit(X_train, y_train)
                explainer = shap.TreeExplainer(model)
                X_shap = X_train.sample(200, replace=False) if len(X_train) > 200 else X_train
                shap_values = explainer.shap_values(X_shap)
                return np.mean(np.abs(shap_values), axis=0)

            # 1. Fan SHAP
            shap_importances_fan.append(get_mean_shap(X_b, y_b_fan))
            
            # 2. Judge SHAP
            shap_importances_judge.append(get_mean_shap(X_b, y_b_judge))
            
        # Function to save CI
        def save_shap_ci(shap_list, fname):
            shap_arr = np.array(shap_list)
            means = np.mean(shap_arr, axis=0)
            q025 = np.percentile(shap_arr, 2.5, axis=0)
            q975 = np.percentile(shap_arr, 97.5, axis=0)
            
            res_df = pd.DataFrame({
                'feature': feature_cols,
                'mean_shap': means,
                'q2.5': q025,
                'q97.5': q975
            })
            res_df.sort_values('mean_shap', ascending=False, inplace=True)
            res_path = os.path.join(OUTPUT_DIR, fname)
            res_df.to_csv(res_path, index=False)
            print(f"Saved {fname}")
            
        save_shap_ci(shap_importances_fan, 'task3_shap_ci_fan.csv')
        save_shap_ci(shap_importances_judge, 'task3_shap_ci_judge.csv')

    print("Task 3 GBDT Finished.")

if __name__ == "__main__":
    run_task3_gbdt()
