
import pandas as pd
import numpy as np
import os
import json

# Paths
PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
RAW_DATA_PATH = r'd:\shumomeisai\Code_second\Data\2026_MCM_Problem_C_Data.csv'
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples' # For v samples
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\task3_data'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def prep_task3_data():
    print("--- Preparing Task 3 Dataset ---")
    
    # 1. Load Panel (Score Targets)
    print("Loading Panel...")
    panel = pd.read_parquet(PANEL_PATH)
    # panel keys: season, week, pair_id, celebrity_name, ballroom_partner, S_it, pJ_it
    
    # 2. Load Raw Data (Features: Age, Industry)
    print("Loading Raw Metadata...")
    # Handle BOM if present
    try:
        raw = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    except:
        raw = pd.read_csv(RAW_DATA_PATH, encoding='ISO-8859-1')
        
    # Standardize names for merge
    # raw columns usually: celebrity_name, ballroom_partner, celebrity_age_at_season_premiere, industry_category, season
    # Let's clean column names
    raw.columns = [c.strip().replace('ï»¿', '') for c in raw.columns]
    
    # Select feature columns
    # We need season + name -> age, industry
    # raw might have duplicate rows per celebrity? (wide format). Usually 1 row per celeb.
    feat_cols = ['season', 'celebrity_name', 'celebrity_age_at_season_premiere', 'industry_category']
    # Check if exist
    for c in feat_cols:
        if c not in raw.columns:
            print(f"Warning: Column {c} not found in raw data. Available: {raw.columns}")
            # Try fuzzy match?
            # Age often: 'celebrity_age_at_season_premiere'
            # Industry: 'industry_category'
    
    # Drop duplicates in raw (just in case)
    meta = raw[feat_cols].drop_duplicates()
    
    # Merge
    # We merge on [season, celebrity_name]
    # Panel name might differ slighly? 
    # Let's hope exact match works for now.
    
    # Normalize strings
    panel['celebrity_name_clean'] = panel['celebrity_name'].astype(str).str.lower().str.strip()
    meta['celebrity_name_clean'] = meta['celebrity_name'].astype(str).str.lower().str.strip()
    
    merged = pd.merge(panel, meta, how='left', on=['season', 'celebrity_name_clean'], suffixes=('', '_raw'))
    
    # Check match rate
    missing_age = merged['celebrity_age_at_season_premiere'].isna().sum()
    print(f"Merged Metadata. Missing Age: {missing_age} / {len(merged)}")
    
    if missing_age > 0:
        # Fill mean? or drop? LMM needs no NaNs.
        # Impute with mean age
        mean_age = merged['celebrity_age_at_season_premiere'].mean()
        merged['celebrity_age_at_season_premiere'].fillna(mean_age, inplace=True)
        merged['industry_category'].fillna('Unknown', inplace=True)
        
    # 3. Enhance Features
    # Rolling stats for pJ_it
    # Need to sort by season/week/pair
    merged.sort_values(['season', 'pair_id', 'week'], inplace=True)
    
    # Rolling Mean (window=3)
    merged['rolling_avg_pJ'] = merged.groupby(['season', 'pair_id'])['pJ_it'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    # Volatility
    merged['rolling_std_pJ'] = merged.groupby(['season', 'pair_id'])['pJ_it'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
    
    # 4. Integrate Posterior Samples (Fan Target yF)
    # We need to load v samples per season and attach to week-rows.
    # To avoid huge file, we can save:
    # Option A: Save point estimate (mean) + CI width as features
    # Option B: Save ALL replicates (Huge) -> Execution Doc says "Fan target... posterior samples... outer loop".
    # Ref Doc 3.1: "Construct weekly dataset... yF^(r)".
    # "yF^(r) = v^(r)".
    # This implies we need to save the samples or load them on the fly in LMM script.
    # Loading on the fly in LMM script is better to keep this prep file small.
    # So here in prep, we just prepare the Fixed Features X matrix.
    
    # Load Summary for v_mean (Point Estimate) and v_ci_width
    # Generated in Task 1 Metrics
    # season_{s}_summary.csv
    
    v_stats_list = []
    seasons = merged['season'].unique()
    for s in seasons:
        summ_path = os.path.join(POSTERIOR_DIR, f"season_{s}_summary.csv")
        if os.path.exists(summ_path):
            sdf = pd.read_csv(summ_path)
            # Keys: season, week, pair_id, v_mean, v_ci_width
            v_stats_list.append(sdf)
            
    if v_stats_list:
        v_stats = pd.concat(v_stats_list)
        # Merge onto merged
        # On [season, week, pair_id]
        # Ensure types (week might be int/float)
        merged = pd.merge(merged, v_stats[['season','week','pair_id','v_mean','v_ci_width']], 
                          on=['season','week','pair_id'], how='left')
                          
    # 5. Final Columns
    # Keep useful columns
    final_cols = [
        'season', 'week', 'pair_id', 'celebrity_name', 'ballroom_partner',
        'S_it', 'pJ_it', # Judge Targets
        'v_mean', 'v_ci_width', # Fan Features/Target Proxy
        'celebrity_age_at_season_premiere', 'industry_category',
        'rolling_avg_pJ', 'rolling_std_pJ'
    ]
    
    # Clean output
    out_df = merged[final_cols].copy()
    out_df.rename(columns={'celebrity_age_at_season_premiere': 'age', 'industry_category': 'industry'}, inplace=True)
    
    # One-Hot Industry? Or Category? LMM script can handle category encoding or formulae.
    # Keep raw category here.
    
    out_path = os.path.join(OUTPUT_DIR, 'task3_weekly_dataset.parquet')
    out_df.to_parquet(out_path)
    print(f"Saved Task 3 Base Dataset to {out_path}")
    print(out_df.head())

if __name__ == "__main__":
    prep_task3_data()
