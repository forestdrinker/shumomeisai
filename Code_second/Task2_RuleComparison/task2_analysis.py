
import numpy as np
import pandas as pd
import argparse
import os
import glob
from scipy.stats import kendalltau, entropy

# Constants
PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
REPLAY_DIR = r'd:\shumomeisai\Code_second\Results\replay_results'
OUTPUT_FILE = r'd:\shumomeisai\Code_second\Results\task2_metrics.csv'

def compute_analysis():
    df = pd.read_parquet(PANEL_PATH)
    
    files = glob.glob(os.path.join(REPLAY_DIR, "season_*_*.npz"))
    rows = []
    
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
        
        # Load Baseline (rank) if available
        baseline_placements = None
        baseline_top3 = None
        baseline_rule = 'rank'
        
        if baseline_rule in rules_dict:
            b_data = np.load(rules_dict[baseline_rule])
            baseline_placements = b_data['placements'] # (Samples, N)
            # Top 3 mask or sets?
            # We want P(Top 3 Change). 
            # Boolean: Does Set(Top3_Sim) == Set(Top3_Base)?
            # Or "Any change in set"?
            # Let's use strict set equality for "Top 3 Composition Change"
            # placements 1,2,3 are Top 3.
            pass
            
        # Ground Truths
        df = pd.read_parquet(PANEL_PATH)
        post_path = os.path.join(POSTERIOR_DIR, f"season_{season}.npz")
        post_data = np.load(post_path)
        v_samples = post_data['v']
        week_values = post_data['week_values']
        
        # True Ranks (for bias)
        v_mean_t = np.mean(v_samples, axis=0) 
        v_overall = np.mean(v_mean_t, axis=0) 
        rank_v_true = pd.Series(v_overall).rank(ascending=False).values
        
        df_s = df[df['season'] == season]
        # Ensure pair_ids match
        # We need pair_ids from replay file (they should be consistent across rules)
        # Load one to get pair_ids
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
            
            # 1. Bias Metrics
            rho_F_list, rho_J_list = [], []
            for i in range(n_samples):
                p_sim = placements[i]
                tau_f, _ = kendalltau(p_sim, rank_v_true)
                tau_j, _ = kendalltau(p_sim, rank_j_true)
                rho_F_list.append(tau_f)
                rho_J_list.append(tau_j)
                
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
            
            if baseline_placements is not None:
                # Compare placements[i] vs baseline_placements[i]
                # Assuming samples correspond 1-to-1 (same seed/order)
                # Ensure sizes match
                if len(baseline_placements) == n_samples:
                    # Winner Change: P(Rank1_Run != Rank1_Base)
                    # Indices of rank 1
                    win_run = np.argmin(placements, axis=1)
                    win_base = np.argmin(baseline_placements, axis=1)
                    change = np.mean(win_run != win_base)
                    p_cham_change = change
                    
                    # Top 3 Change
                    # Set comparison per sample
                    diff_count = 0
                    for i in range(n_samples):
                        # Top 3 pairs
                        top3_run = set(np.argsort(placements[i])[:3])
                        top3_base = set(np.argsort(baseline_placements[i])[:3])
                        if top3_run != top3_base:
                            diff_count += 1
                    p_top3_change = diff_count / n_samples
                    
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

        # --- Generate Weekly Diff CSV (Item 71 in Self-Check) ---
        # Need at least Rank and Percent rules for this season
        diff_rows = []
        if 'rank' in rules_dict and 'percent' in rules_dict:
            # Compare Eliminations Week by Week for Rank vs Percent
            d_rank = np.load(rules_dict['rank'])
            d_perc = np.load(rules_dict['percent'])
            
            # elim_weeks: (Samples, Pairs) -> Week Value when elim.
            # Convert to "Elim Set per Week"
            # Or just check "Who was eliminated at week t?"
            # For each sample r, week t: Set(Elim_Rank) vs Set(Elim_Perc).
            # Diff = 1 if Sets differ.
            
            # week_values comes from Posterior.
            # Note: week_values[-1] is Finals which has no elimination usually?
            # Or elim_weeks stores n_weeks for finals.
            # We iterate all weeks that HAD eliminations.
            # week_values usually includes Final week.
            
            for t_idx, w_val in enumerate(week_values[:-1]):
                # Indices eliminated at w_val
                # rank
                elim_mask_r = (d_rank['elim_weeks'] == w_val) # (R, N) boolean
                # percent
                elim_mask_p = (d_perc['elim_weeks'] == w_val) # (R, N) boolean
                
                # Compare per sample
                # Are masks identical?
                # XOR gives differences. Any difference means set not equal.
                diff = np.any(np.bitwise_xor(elim_mask_r, elim_mask_p), axis=1) # (R,)
                p_diff = np.mean(diff)
                
                # Check Save vs NoSave (if rank_save exists)
                p_diff_save = np.nan
                if 'rank_save' in rules_dict:
                     d_save = np.load(rules_dict['rank_save'])
                     elim_mask_s = (d_save['elim_weeks'] == w_val)
                     diff_s = np.any(np.bitwise_xor(elim_mask_r, elim_mask_s), axis=1)
                     p_diff_save = np.mean(diff_s)
                
                diff_rows.append({
                    'season': season,
                    'week': w_val,
                    'p_elim_diff_rank_vs_percent': p_diff,
                    'p_elim_diff_save_vs_nosave': p_diff_save
                })
        
        if diff_rows:
            diff_df = pd.DataFrame(diff_rows)
            out_path = os.path.join(REPLAY_DIR, f"season_{season}_weekly_diff.csv")
            diff_df.to_csv(out_path, index=False)
            print(f"  -> Generated Weekly Diff: {out_path}")
            
    if rows:
        df_out = pd.DataFrame(rows)
        # Sort
        rule_order = {'rank':0, 'rank_save':1, 'percent':2, 'percent_save':3}
        df_out['rule_ord'] = df_out['rule'].map(rule_order).fillna(99)
        df_out.sort_values(['season', 'rule_ord'], inplace=True)
        df_out.drop(columns=['rule_ord'], inplace=True)
        
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"Metrics saved to {OUTPUT_FILE}")
    else:
        print("No results found.")

if __name__ == "__main__":
    compute_analysis()
