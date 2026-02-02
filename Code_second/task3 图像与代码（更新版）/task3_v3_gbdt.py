"""
Task 3 GBDT Analysis — V3 (Dual-Channel Attribution)
=====================================================
Track 2 (Non-linear complement):
  Judge GBDT + Fan GBDT with:
    - GroupKFold by season (cross-season generalisation)
    - Block bootstrap × posterior draw → SHAP CI
    - fan_base + partner_experience as new features

Key upgrades over V2:
  1. fan_base in feature set (tests if pre-season popularity matters)
  2. Partner experience (# prior seasons) replaces raw label encoding
  3. Cleaner SHAP CI with dual-layer uncertainty
  4. Outputs: CSV tables + JSON metrics for figure generation
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import os
import glob
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
DATASET_PATH   = r'd:\shumomeisai\Code_second\Results\task3_data\task3_weekly_dataset.parquet'
POSTERIOR_DIR  = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR     = r'd:\shumomeisai\Code_second\Results\task3_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_task3_gbdt():
    print("=" * 60)
    print("Task 3 GBDT Analysis (V3 — Dual Channel)")
    print("=" * 60)

    # ── 1. Load & prep ──
    print("\n[1] Loading data...")
    df = pd.read_parquet(DATASET_PATH)

    le_industry = LabelEncoder()
    df['industry_enc'] = le_industry.fit_transform(df['industry'].astype(str))
    le_partner = LabelEncoder()
    df['partner_enc'] = le_partner.fit_transform(df['ballroom_partner'].astype(str))

    # Feature set (V3: added fan_base, partner_experience)
    feature_cols = [
        'age', 'industry_enc', 'partner_enc', 'week_norm',
        'rolling_avg_pJ', 'rolling_std_pJ',
        'fan_base', 'partner_experience',
        'partner_net_degree', 'partner_net_pagerank', 'partner_net_norm_degree',
        'partner_emb_0', 'partner_emb_1', 'partner_emb_2', 'partner_emb_3'
    ]
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Human-readable names for figures
    feature_names = {
        'age': 'Age', 'industry_enc': 'Industry',
        'partner_enc': 'Partner Identity', 'week_norm': 'Week Progress',
        'rolling_avg_pJ': 'Rolling Avg Score', 'rolling_std_pJ': 'Rolling Std Score',
        'fan_base': 'Fan Base (Pre-season)', 'partner_experience': 'Partner Experience',
        'partner_net_degree': 'Partner Net Degree', 'partner_net_pagerank': 'Partner Net PageRank',
        'partner_net_norm_degree': 'Partner Net NormDeg',
        'partner_emb_0': 'Partner Emb-0', 'partner_emb_1': 'Partner Emb-1',
        'partner_emb_2': 'Partner Emb-2', 'partner_emb_3': 'Partner Emb-3'
    }

    X = df[feature_cols].fillna(0)
    y_judge = df['pJ_it']
    groups = df['season']

    # Check fan data availability
    RUN_FAN = 'v_mean' in df.columns and not df['v_mean'].isna().all()

    # ══════════════════════════════════════════════════════════
    # 2. GroupKFold CV — both channels
    # ══════════════════════════════════════════════════════════
    print("\n[2] GroupKFold CV (5-fold, grouped by season)...")

    xgb_params = dict(n_estimators=100, max_depth=4, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, n_jobs=4)

    def run_cv(X, y, groups, label):
        gkf = GroupKFold(n_splits=5)
        rmse_list, r2_list = [], []
        for train_idx, test_idx in gkf.split(X, y, groups):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(Xtr, ytr, verbose=False)
            pred = model.predict(Xte)
            rmse = np.sqrt(np.mean((yte - pred) ** 2))
            ss_res = np.sum((yte - pred) ** 2)
            ss_tot = np.sum((yte - yte.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse_list.append(rmse); r2_list.append(r2)
        print(f"    {label}: RMSE={np.mean(rmse_list):.4f}±{np.std(rmse_list):.4f}, "
              f"R²={np.mean(r2_list):.3f}±{np.std(r2_list):.3f}")
        return {'rmse_mean': np.mean(rmse_list), 'rmse_std': np.std(rmse_list),
                'r2_mean': np.mean(r2_list), 'r2_std': np.std(r2_list)}

    cv_j = run_cv(X, y_judge, groups, 'Judge')
    cv_f = None
    if RUN_FAN:
        y_fan = df['v_mean'].fillna(0)
        cv_f = run_cv(X, y_fan, groups, 'Fan')

    cv_metrics = {'judge': cv_j}
    if cv_f: cv_metrics['fan'] = cv_f
    with open(os.path.join(OUTPUT_DIR, 'task3_gbdt_cv_metrics.json'), 'w') as f:
        json.dump(cv_metrics, f, indent=2)

    # ══════════════════════════════════════════════════════════
    # 3. Bootstrap × Posterior SHAP CI
    # ══════════════════════════════════════════════════════════
    print("\n[3] Bootstrap SHAP CI (B=50)...")

    # Pre-load posterior draws for fan channel
    N_DRAWS = 50
    Y_draws = None
    if RUN_FAN:
        Y_draws = np.full((len(df), N_DRAWS), np.nan)
        df_lookup = {(int(r['season']), int(r['week']), r['pair_id']): i
                     for i, r in df[['season', 'week', 'pair_id']].iterrows()}

        season_files = [f for f in glob.glob(os.path.join(POSTERIOR_DIR, "season_*.npz"))
                        if 'summary' not in f and 'diagnostics' not in f]
        for fpath in season_files:
            try:
                data = np.load(fpath, allow_pickle=True)
                s_id = int(os.path.basename(fpath).split('_')[1].split('.')[0])
                if 'v' not in data: continue
                v_all = data['v']
                R_avail = v_all.shape[0]
                idx = np.linspace(0, R_avail - 1, N_DRAWS, dtype=int)
                c_ids = data.get('pair_ids', data.get('couple_ids'))
                if c_ids is None: continue
                for t in range(v_all.shape[1]):
                    for n, pid in enumerate(c_ids):
                        rid = df_lookup.get((s_id, t + 1, pid))
                        if rid is not None:
                            Y_draws[rid, :] = v_all[idx, t, n]
            except:
                pass

        # Fill NaN with point estimate
        mask = np.isnan(Y_draws)
        if mask.any():
            means = df['v_mean'].fillna(0).values[:, None]
            Y_draws[mask] = np.broadcast_to(means, Y_draws.shape)[mask]

    # Bootstrap loop
    B = 50
    shap_judge_list, shap_fan_list = [], []

    def get_shap(X_train, y_train, n_sample=250):
        model = xgb.XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=1)
        model.fit(X_train, y_train, verbose=False)
        explainer = shap.TreeExplainer(model)
        Xs = X_train.sample(min(n_sample, len(X_train)), replace=False)
        sv = explainer.shap_values(Xs)
        return np.mean(np.abs(sv), axis=0)

    unique_seasons = df['season'].unique()
    for b in range(B):
        if b % 10 == 0: print(f"    Bootstrap {b+1}/{B}...")

        # Block bootstrap by season
        s_sample = np.random.choice(unique_seasons, size=len(unique_seasons), replace=True)
        indices = np.concatenate([df.index[df['season'] == s].values for s in s_sample])
        X_b = X.iloc[indices]
        y_b_j = y_judge.iloc[indices].values

        mask_valid = ~np.isnan(y_b_j)
        if mask_valid.sum() < 30: continue

        # Judge SHAP
        shap_judge_list.append(get_shap(X_b[mask_valid], y_b_j[mask_valid]))

        # Fan SHAP (with posterior draw injection)
        if RUN_FAN and Y_draws is not None:
            r_idx = np.random.randint(0, N_DRAWS)
            y_b_f = Y_draws[indices, r_idx]
            mask_f = ~np.isnan(y_b_f)
            combined_mask = mask_valid & mask_f
            if combined_mask.sum() > 30:
                shap_fan_list.append(get_shap(X_b[combined_mask], y_b_f[combined_mask]))

    # ── Save SHAP results ──
    def save_shap(shap_list, fname):
        arr = np.array(shap_list)
        res = pd.DataFrame({
            'feature': feature_cols,
            'feature_name': [feature_names.get(f, f) for f in feature_cols],
            'mean_shap': arr.mean(axis=0),
            'ci_lo': np.percentile(arr, 2.5, axis=0),
            'ci_hi': np.percentile(arr, 97.5, axis=0),
            'std': arr.std(axis=0),
        })
        res.sort_values('mean_shap', ascending=False, inplace=True)
        res.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print(f"    Saved {fname} ({len(shap_list)} bootstraps)")
        return res

    if shap_judge_list:
        shap_j_df = save_shap(shap_judge_list, 'task3_shap_judge.csv')
    if shap_fan_list:
        shap_f_df = save_shap(shap_fan_list, 'task3_shap_fan.csv')

    print("\n✅ Task 3 GBDT Finished.")


if __name__ == "__main__":
    run_task3_gbdt()
