
import pandas as pd
import numpy as np
import os
import glob
from itertools import combinations

# Paths
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv' # Updated to Partner's CSV
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples' # For v samples
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\task3_data'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_network_features(panel):
    print("Generating Network Features (Pro-Pro Co-occurrence) [Manual Impl]...")
    net_feats = []
    seasons = sorted(panel['season'].unique())
    season_pros = panel.groupby('season')['ballroom_partner'].unique().to_dict()
    
    # helper for pagerank
    def power_iteration(A, num_simulations=20):
        # M_ij = A_ij / deg_j
        deg = A.sum(axis=0)
        deg[deg==0] = 1
        M = A / deg
        n = M.shape[0]
        v = np.ones(n) / n
        d = 0.85
        for _ in range(num_simulations):
            v = d * M.dot(v) + (1-d)/n
        return v

    for s in seasons:
        # Build graph using seasons < s to avoid leakage
        past_seasons = [ps for ps in seasons if ps < s]
        
        # Get unique pros in history
        all_past = set()
        for ps in past_seasons:
            all_past.update(season_pros.get(ps, []))
        all_past = sorted(list(all_past))
        n = len(all_past)
        
        # Mapping
        pro2idx = {p: i for i, p in enumerate(all_past)}
        
        # Build Matrix
        w_deg, pr_vec, clo_vec, emb = [], [], [], []
        
        if n > 0:
            A = np.zeros((n, n))
            for ps in past_seasons:
                pros = season_pros.get(ps, [])
                idxs = [pro2idx[p] for p in pros if p in pro2idx]
                for i1, i2 in combinations(idxs, 2):
                    A[i1, i2] += 1
                    A[i2, i1] += 1 # undirected
            
            # Degree (Weighted)
            w_deg = A.sum(axis=1) # weighted degree
            
            # PageRank
            try:
                pr_vec = power_iteration(A)
            except:
                pr_vec = np.zeros(n)
            
            # Simple Closeness Proxy (Normalized Degree)
            clo_vec = w_deg / (w_deg.sum() + 1e-9)
            
            # Embedding (SVD)
            try:
                u, _, _ = np.linalg.svd(A) # SVD of Adjacency
                if u.shape[1] >= 4:
                    emb = u[:, :4]
                else:
                    emb = np.zeros((n, 4))
                    emb[:, :u.shape[1]] = u[:, :u.shape[1]]
            except:
                emb = np.zeros((n, 4))
        
        # Map to current rows
        cur_rows = panel[panel['season'] == s][['season', 'week', 'pair_id', 'ballroom_partner']].drop_duplicates(subset=['pair_id'])
        s_feats = []
        
        for p in cur_rows['ballroom_partner']:
            f = {}
            if p in pro2idx:
                idx = pro2idx[p]
                f['partner_network_degree'] = w_deg[idx]
                f['partner_network_pagerank'] = pr_vec[idx]
                f['partner_network_closeness'] = clo_vec[idx]
                for k in range(4):
                    f[f'partner_embedding_{k}'] = emb[idx, k]
            else:
                # new partner
                f['partner_network_degree'] = 0.0
                f['partner_network_pagerank'] = 0.0
                f['partner_network_closeness'] = 0.0
                for k in range(4):
                    f[f'partner_embedding_{k}'] = 0.0
            
            s_feats.append(f)
            
        s_df = pd.DataFrame(s_feats, index=cur_rows.index)
        temp = pd.concat([cur_rows, s_df], axis=1)
        net_feats.append(temp)
    
    if net_feats:
        net_df = pd.concat(net_feats)
        return net_df[['season', 'pair_id', 
                       'partner_network_degree', 'partner_network_pagerank', 'partner_network_closeness',
                       'partner_embedding_0', 'partner_embedding_1', 'partner_embedding_2', 'partner_embedding_3']]
    else:
        return pd.DataFrame()

def prep_task3_data():
    print("--- 正在准备 Task 3 数据集 ---")
    
    # 1. Load Panel (Score Targets & Features)
    print(f"正在加载面板数据自 {PANEL_PATH} ...")
    if PANEL_PATH.endswith('.csv'):
        panel = pd.read_csv(PANEL_PATH)
    else:
        panel = pd.read_parquet(PANEL_PATH)
        
    # Compatibility Fix for Partner Data
    if 'celebrity_age' in panel.columns and 'celebrity_age_during_season' not in panel.columns:
        print("Notice: Renaming 'celebrity_age' to 'celebrity_age_during_season' for compatibility.")
        panel.rename(columns={'celebrity_age': 'celebrity_age_during_season'}, inplace=True)

    # 2. Compatibility Fix: Create pJ_it (Judge Score) from S_it if missing
    if 'pJ_it' not in panel.columns and 'S_it' in panel.columns:
        print("Notice: Creating 'pJ_it' from 'S_it' (assuming S_it is the score metric).")
        panel['pJ_it'] = panel['S_it']

    # Expected columns: season, week, pair_id, celebrity_name, ballroom_partner, S_it, pJ_it, 
    #                   celebrity_age_during_season, celebrity_industry
    
    print(f"Panel Shape: {panel.shape}")
    
    # Check for required feature columns
    req_cols = ['celebrity_age_during_season', 'celebrity_industry']
    for c in req_cols:
        if c not in panel.columns:
            print(f"Error: Column {c} not found in panel. Available: {list(panel.columns[:10])}...")
            # Verify if maybe named differently?
            # Based on check_cols check, they should be there.
            return

    # 2. Integrate Posterior Comparisons (Fan Feature/Target Proxy)
    # Load Summary for v_mean (Point Estimate) and v_ci_width
    print("Loading Posterior Summaries...")
    v_stats_list = []
    
    # We iterate through seasons present in panel
    seasons = panel['season'].unique()
    
    for s in sorted(seasons):
        summ_path = os.path.join(POSTERIOR_DIR, f"season_{s}_summary.csv")
        if os.path.exists(summ_path):
            try:
                sdf = pd.read_csv(summ_path)
                # Keep only join keys and features of interest
                # Keys: season, week, pair_id
                cols_to_keep = ['season', 'week', 'pair_id', 'v_mean', 'v_ci_width']
                
                # Check if columns exist
                available_cols = [c for c in cols_to_keep if c in sdf.columns]
                
                if len(available_cols) < len(cols_to_keep):
                    print(f"  Warning: S{s} summary missing columns. Found: {sdf.columns}")
                    continue
                    
                v_stats_list.append(sdf[available_cols])
            except Exception as e:
                print(f"  Error loading S{s} summary: {e}")
        else:
            print(f"  Warning: Summary not found for S{s}")

    merged = panel.copy()
    
    if v_stats_list:
        v_stats = pd.concat(v_stats_list, ignore_index=True)
        # Merge onto panel
        # Ensure join keys type consistency usually good, but let's be safe
        print(f"Merging posterior stats (Shape: {v_stats.shape})...")
        
        merged = pd.merge(merged, v_stats, on=['season', 'week', 'pair_id'], how='left')
    else:
        print("Warning: No posterior summaries loaded. v_mean/v_ci_width will be missing.")
        merged['v_mean'] = np.nan
        merged['v_ci_width'] = np.nan

    # 3. Enhance Features
    print("Generating rolling features...")
    # Sort for rolling
    merged.sort_values(['season', 'pair_id', 'week'], inplace=True)
    
    # Rolling stats for pJ_it (Performance Momentum)
    # Window=3
    merged['rolling_avg_pJ'] = merged.groupby(['season', 'pair_id'])['pJ_it'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    merged['rolling_std_pJ'] = merged.groupby(['season', 'pair_id'])['pJ_it'].transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)
    
    # 3.5 Network Features
    net_df = generate_network_features(merged)
    if not net_df.empty:
        print("Merging network features...")
        merged = pd.merge(merged, net_df, on=['season', 'pair_id'], how='left')
        # Fill NaN for any rows that didn't match (shouldn't happen if logic correct)
        net_cols = [c for c in net_df.columns if c not in ['season', 'pair_id']]
        merged[net_cols] = merged[net_cols].fillna(0.0)
    
    # 4. Final Columns & Cleanup
    merged.rename(columns={
        'celebrity_age_during_season': 'age', 
        'celebrity_industry': 'industry'
    }, inplace=True)
    
    # Fill basic missing values for features if any
    # Age: fill with mean
    if merged['age'].isnull().any():
        mean_age = merged['age'].mean()
        merged['age'] = merged['age'].fillna(mean_age)
        print(f"Imputed {merged['age'].isnull().sum()} missing ages with mean {mean_age:.1f}")
        
    # Industry: fill Unknown and CLEAN UP
    if merged['industry'].isnull().any():
        merged['industry'] = merged['industry'].fillna('Unknown')
    
    # Audit Fix: Data Hygiene
    # 1. Fix Typos
    replace_map = {
        'Beauty Pagent': 'Beauty Pageant',
        'Con artist': 'Personality' # Fix the "Con artist" issue
    }
    merged['industry'] = merged['industry'].replace(replace_map)
    # 2. Normalize case
    merged['industry'] = merged['industry'].astype(str).str.strip().str.title()

    final_cols = [
        'season', 'week', 'pair_id', 'celebrity_name', 'ballroom_partner',
        'pJ_it', # Judge Target
        'v_mean', 'v_ci_width', # Fan Features / Proxy Target
        'age', 'industry',
        'rolling_avg_pJ', 'rolling_std_pJ',
        'partner_network_degree', 'partner_network_pagerank', 'partner_network_closeness',
        'partner_embedding_0', 'partner_embedding_1', 'partner_embedding_2', 'partner_embedding_3'
    ]
    
    # Filter only existing columns (e.g. if v_mean missing)
    out_cols = [c for c in final_cols if c in merged.columns]
    
    out_df = merged[out_cols].copy()
    
    # Drop rows where we have absolutely no target data? 
    # Usually we want to keep all active rows.
    
    out_path = os.path.join(OUTPUT_DIR, 'task3_weekly_dataset.parquet')
    out_df.to_parquet(out_path)
    print(f"Task 3 数据集已保存至 {out_path} (行数: {len(out_df)})")
    print(out_df.head())

if __name__ == "__main__":
    prep_task3_data()
