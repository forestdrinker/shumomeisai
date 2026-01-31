
import numpy as np
import pandas as pd
import argparse
import os
import glob
from scipy.stats import kendalltau, entropy, spearmanr

# Constants
PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
REPLAY_DIR = r'd:\shumomeisai\Code_second\Results\replay_results'
OUTPUT_FILE = r'd:\shumomeisai\Code_second\Results\task2_metrics.csv'
CONTROVERSY_FILE = r'd:\shumomeisai\Code_second\Results\controversy_cases.csv'

def compute_analysis():
    df_panel = pd.read_parquet(PANEL_PATH)
    # Create (Season, PairID) -> Name mapping
    # pair_id seems to be reused per season or we just want to be safe.
    # Group by [season, pair_id] and take first name.
    pair_name_map = df_panel.set_index(['season', 'pair_id'])['celebrity_name'].to_dict()
    
    files = glob.glob(os.path.join(REPLAY_DIR, "season_*_*.npz"))
    rows = []
    controversy_rows = []
    
    # Group files by season
    season_files = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.replace('.npz', '').split('_')
        season = int(parts[1])
        rule = "_".join(parts[2:])
        if season not in season_files:
            season_files[season] = {}
        season_files[season][rule] = fpath
        
    for season in sorted(season_files.keys()):
        rules_dict = season_files[season]
        print(f"Processing Season {season}...")
        
        # Load Baseline (rank) if available
        baseline_placements = None
        baseline_rule = 'rank'
        
        if baseline_rule in rules_dict:
            b_data = np.load(rules_dict[baseline_rule])
            baseline_placements = b_data['placements'] # (Samples, N)
            
        # Ground Truths
        post_path = os.path.join(POSTERIOR_DIR, f"season_{season}.npz")
        if not os.path.exists(post_path):
            print(f"  Warning: Posterior not found for S{season}")
            continue
            
        post_data = np.load(post_path)
        v_samples = post_data['v']
        week_values = post_data['week_values']
        
        # True Ranks (for bias)
        v_mean_t = np.mean(v_samples, axis=0) 
        v_overall = np.mean(v_mean_t, axis=0) 
        rank_v_true = pd.Series(v_overall).rank(ascending=False).values
        
        df_s = df_panel[df_panel['season'] == season]
        
        # Ensure pair_ids match
        # We need pair_ids from replay file
        first_key = list(rules_dict.keys())[0]
        data_0 = np.load(rules_dict[first_key])
        pair_ids = data_0['pair_ids']
        
        s_stats = df_s.groupby('pair_id')['pJ_it'].mean()
        s_overall = np.array([s_stats.get(pid, 0) for pid in pair_ids])
        rank_j_true = pd.Series(s_overall).rank(ascending=False).values
        
        # Iterate Rules
        for rule, fpath in rules_dict.items():
            data = np.load(fpath)
            placements = data['placements']
            conflict_hist = data['conflict_history']
            upset_counts = data['upset_counts']
            elim_weeks = data['elim_weeks']
            
            n_samples, n_pairs = placements.shape
            
            # 1. Bias Metrics (Changed to Spearman)
            rho_F_list, rho_J_list = [], []
            for i in range(n_samples):
                p_sim = placements[i]
                # P0 Requirement: Use Spearman
                rho_f, _ = spearmanr(p_sim, rank_v_true)
                rho_j, _ = spearmanr(p_sim, rank_j_true)
                
                # Handle NaNs from constant input
                if np.isnan(rho_f): rho_f = 0
                if np.isnan(rho_j): rho_j = 0
                
                rho_F_list.append(rho_f)
                rho_J_list.append(rho_j)
                
            # 2. Drama
            d_mean_sample = np.nanmean(conflict_hist, axis=1)
            d_bar = np.nanmean(d_mean_sample)
            
            n_weeks_replay = conflict_hist.shape[1]
            k_late = min(3, n_weeks_replay)
            d_late = np.nanmean(conflict_hist[:, -k_late:])
            
            upset_rate = np.mean(upset_counts) / max(1, n_weeks_replay - 1)
            
            # Suspense H
            entropy_list = []
            for t_idx, w_val in enumerate(week_values[:-1]):
                is_elim = (elim_weeks == w_val)
                elim_counts = np.sum(is_elim, axis=0)
                total = np.sum(elim_counts)
                if total > 0:
                    entropy_list.append(entropy(elim_counts/total))
            
            h_bar = np.mean(entropy_list) if entropy_list else 0
            h_late = np.mean(entropy_list[-k_late:]) if entropy_list else 0
            
            # 3. Change vs Baseline
            p_cham_change = np.nan
            p_top3_change = np.nan
            
            if baseline_placements is not None and rule != baseline_rule:
                if len(baseline_placements) == n_samples:
                    win_run = np.argmin(placements, axis=1)
                    win_base = np.argmin(baseline_placements, axis=1)
                    change = np.mean(win_run != win_base)
                    p_cham_change = change
                    
                    diff_count = 0
                    for i in range(n_samples):
                        top3_run = set(np.argsort(placements[i])[:3])
                        top3_base = set(np.argsort(baseline_placements[i])[:3])
                        if top3_run != top3_base:
                            diff_count += 1
                    p_top3_change = diff_count / n_samples
            elif rule == baseline_rule:
                p_cham_change = 0.0
                p_top3_change = 0.0
                    
            rows.append({
                'season': season,
                'rule': rule,
                'p_champion_change': p_cham_change,
                'p_top3_change': p_top3_change,
                'rho_F': np.mean(rho_F_list),
                'rho_J': np.mean(rho_J_list),
                'drama_D': d_bar,
                'drama_D_late': d_late,
                'upset_rate': upset_rate,
                'suspense_H': h_bar,
                'suspense_H_late': h_late
            })
            
            # --- Controversy / Per-Celebrity Metrics (P0) ---
            # Calculate metrics for each pair in this Season+Rule
            # p_win, p_top3, expected_rank, expected_survival_weeks
            
            # placements: (S, N). 1=Best.
            # elim_weeks: (S, N).
            
            p_win_arr = np.mean(placements == 1, axis=0) # (N,)
            p_top3_arr = np.mean(placements <= 3, axis=0) # (N,)
            exp_rank_arr = np.mean(placements, axis=0) # (N,)
            exp_weeks_arr = np.mean(elim_weeks, axis=0) # (N,)
            
            for p_idx, pid in enumerate(pair_ids):
                # Look up (season, pid)
                name = pair_name_map.get((season, pid), f"Pair_{pid}")
                
                controv_item = {
                    'season': season,
                    'rule': rule,
                    'pair_id': pid,
                    'celebrity_name': name,
                    'p_win': p_win_arr[p_idx],
                    'p_top3': p_top3_arr[p_idx],
                    'expected_rank': exp_rank_arr[p_idx],
                    'expected_survival_weeks': exp_weeks_arr[p_idx]
                }
                controversy_rows.append(controv_item)

        # --- Generate Weekly Diff CSV (Expanded P1) ---
        # Compare ALL available pairs of rules
        available_rules = list(rules_dict.keys())
        diff_rows = []
        
        # Iterate all unique pairs
        for i in range(len(available_rules)):
            for j in range(i + 1, len(available_rules)):
                r1 = available_rules[i]
                r2 = available_rules[j]
                
                d1 = np.load(rules_dict[r1])
                d2 = np.load(rules_dict[r2])
                
                # Check compatibility
                if d1['elim_weeks'].shape != d2['elim_weeks'].shape:
                    continue
                    
                for t_idx, w_val in enumerate(week_values[:-1]):
                    # Check who was eliminated at this week
                    mask1 = (d1['elim_weeks'] == w_val)
                    mask2 = (d2['elim_weeks'] == w_val)
                    
                    # Difference?
                    # Any bitwise XOR means difference in the SET of eliminated people
                    diff_bool = np.any(np.bitwise_xor(mask1, mask2), axis=1) # (S,)
                    p_diff = np.mean(diff_bool)
                    
                    diff_rows.append({
                        'season': season,
                        'week': w_val,
                        'rule_A': r1,
                        'rule_B': r2,
                        'p_elim_diff': p_diff
                    })
        
        if diff_rows:
            diff_df = pd.DataFrame(diff_rows)
            out_path = os.path.join(REPLAY_DIR, f"season_{season}_weekly_diff.csv")
            diff_df.to_csv(out_path, index=False)
            # print(f"  -> Generated Weekly Diff: {out_path}")
            
    # Save Metrics
    if rows:
        df_out = pd.DataFrame(rows)
        # Sort
        rule_order = {'rank':0, 'rank_save':1, 'percent':2, 'percent_save':3}
        df_out['rule_ord'] = df_out['rule'].map(rule_order).fillna(99)
        df_out.sort_values(['season', 'rule_ord'], inplace=True)
        df_out.drop(columns=['rule_ord'], inplace=True)
        
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"Metrics saved to {OUTPUT_FILE}")
        
    # Save Controversy
    if controversy_rows:
        df_controv = pd.DataFrame(controversy_rows)
        df_controv.sort_values(['season', 'pair_id', 'rule'], inplace=True)
        df_controv.to_csv(CONTROVERSY_FILE, index=False)
        print(f"Controversy cases saved to {CONTROVERSY_FILE}")
    else:
        print("No results found.")

if __name__ == "__main__":
    compute_analysis()
