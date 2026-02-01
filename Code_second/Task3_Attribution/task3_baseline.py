import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
BASELINE_V_PATH = r'd:\shumomeisai\Code_second\Results\baseline_samples\task1_baseline_v_est.csv'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\baseline_results'

def run_task3_baseline():
    print("--- Running Task 3 Baseline (Simple LMM) ---")
    
    # Load
    if PANEL_PATH.endswith('.csv'):
        df = pd.read_csv(PANEL_PATH)
    else:
        df = pd.read_parquet(PANEL_PATH)
        
    v_df = pd.read_csv(BASELINE_V_PATH)
    
    # Merge V into Panel
    # Keys: season, week, pair_id
    df = df.merge(v_df[['season', 'week', 'pair_id', 'v_baseline']], 
                  on=['season', 'week', 'pair_id'], 
                  how='left')
    
    # Preprocessing
    # Normalize Targets
    # Judge Share: Estimate if not strictly present
    if 'pJ_it' not in df.columns:
        # Group by season, week to calc share
        sums = df.groupby(['season', 'week'])['S_it'].transform('sum')
        df['judge_share'] = df['S_it'] / sums
    else:
        df['judge_share'] = df['pJ_it']
        
    # Rename or map columns for consistency
    if 'ballroom_partner' in df.columns:
        df['partner_id'] = df['ballroom_partner']
    
    # Standardize Age usually helps convergence
    age_col = 'celebrity_age' if 'celebrity_age' in df.columns else 'celebrity_age_during_season'
    if age_col in df.columns:
        mean_age = df[age_col].mean()
        std_age = df[age_col].std()
        df['age_std'] = (df[age_col] - mean_age) / std_age
    else:
        df['age_std'] = 0
        
    # Ensure partner_id is efficient string or cat
    df['partner_str'] = df['partner_id'].astype(str)
    
    # Dropna
    model_df = df.dropna(subset=['judge_share', 'v_baseline', 'age_std', 'week'])
    
    results_txt = []
    
    # --- Model 1: Judge Preferences ---
    results_txt.append("=== Model 1: Judge Share LMM ===")
    # Formula: judge_share ~ age_std + week + (1|partner) + (1|season)
    # Using Variance Components
    
    try:
        md_j = smf.mixedlm("judge_share ~ age_std + week", model_df, 
                           groups=model_df["season"], 
                           re_formula="~1", # Random Intercept for Season (Group)
                           vc_formula={"partner": "0 + C(partner_str)"}) # Random Intercept for Partner
        mdf_j = md_j.fit()
        results_txt.append(mdf_j.summary().as_text())
    except Exception as e:
        results_txt.append(f"Model J Failed: {e}")
        
    # --- Model 2: Fan Preferences (Baseline V) ---
    results_txt.append("\n=== Model 2: Fan Share (Baseline) LMM ===")
    try:
        md_f = smf.mixedlm("v_baseline ~ age_std + week", model_df, 
                           groups=model_df["season"], 
                           re_formula="~1", 
                           vc_formula={"partner": "0 + C(partner_str)"})
        mdf_f = md_f.fit()
        results_txt.append(mdf_f.summary().as_text())
    except Exception as e:
        results_txt.append(f"Model F Failed: {e}")
        
    # Save
    out_path = os.path.join(OUTPUT_DIR, 'task3_baseline_lmm_summary.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results_txt))
        
    print(f"Task 3 Baseline Results saved to {out_path}")

if __name__ == "__main__":
    run_task3_baseline()
