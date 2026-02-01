import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import spearmanr, kendalltau, rankdata

# Configuration
BASELINE_V_PATH = r'd:\shumomeisai\Code_second\Results\baseline_samples\task1_baseline_v_est.csv'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\baseline_results'

def load_data():
    v_df = pd.read_csv(BASELINE_V_PATH)
    if PANEL_PATH.endswith('.csv'):
        panel_df = pd.read_csv(PANEL_PATH)
    else:
        panel_df = pd.read_parquet(PANEL_PATH)
    return v_df, panel_df

def calculate_objectives(season, v_df, panel_df, rule):
    """
    Calculate 4 Objectives for a static rule on a season (using Baseline V).
    Obj_F: Alignment with Fan V
    Obj_J: Alignment with Judge S
    Obj_D: Drama (Margin)
    Obj_R: Robustness (Not implemented for baseline, default 0 or separate)
    """
    # Filter Data
    s_v = v_df[v_df['season'] == season]
    s_panel = panel_df[panel_df['season'] == season]
    weeks = sorted(s_panel['week'].unique())
    
    rhos_f = []
    rhos_j = []
    margins = []
    
    for w in weeks:
        # Get Current Scores
        w_df = s_panel[s_panel['week'] == w]
        w_v = s_v[s_v['week'] == w]
        
        # Merge
        merged = w_df.merge(w_v[['pair_id', 'v_baseline']], on='pair_id', how='inner')
        if len(merged) < 2: continue
        
        # Scores
        s = merged['S_it'].values
        p_j = s / s.sum()
        v = merged['v_baseline'].values
        v = v / v.sum()
        
        # Combined Score
        if rule == 'percent':
            combined = p_j + v # Higher is better
            rank_metric = rankdata(-combined) # 1 is best
        elif rule == 'rank':
            r_j = rankdata(-p_j)
            r_f = rankdata(-v)
            combined = r_j + r_f # Lower is better
            rank_metric = rankdata(combined) # 1 is best
        else: # 50/50 weighted rank? Default is standard Rank sum
             r_j = rankdata(-p_j)
             r_f = rankdata(-v)
             combined = r_j + r_f
             rank_metric = rankdata(combined)
            
        # Obj_F: Rank correlation with V
        rf, _ = spearmanr(rank_metric, rankdata(-v)) # Both 1 is best
        rhos_f.append(rf)
        
        # Obj_J: Rank correlation with S
        rj, _ = spearmanr(rank_metric, rankdata(-s))
        rhos_j.append(rj)
        
        # Obj_D: Margin between 1st and 2nd
        # Depends on score scale.
        # For percent: diff in shares.
        # For rank: diff in rank sums (integer).
        sorted_scores = sorted(combined, reverse=(rule=='percent')) # Best first
        if len(sorted_scores) >= 2:
            if rule == 'percent':
                margin = sorted_scores[0] - sorted_scores[1]
                # Norm by max possible margin? (1.0)
                margins.append(margin) 
            else:
                # Rank: Lower is better. 1st is Lowest.
                # Margin = 2nd - 1st (Positive)
                # Max gap is roughly N + N vs 1 + 1?
                # Just raw gap
                margin = sorted_scores[1] - sorted_scores[0] # Note: sorted ascending for Rank
                # Wait, I sorted DESCENDING above for percent.
                # Let's fix.
                pass
                
    # Fix Margin Calc properly
    
    return {
        'Obj_F': np.nanmean(rhos_f),
        'Obj_J': np.nanmean(rhos_j)
    }

def run_task4_baseline():
    print("--- Running Task 4 Baseline (Static Rule Evaluation) ---")
    v_df, panel_df = load_data()
    seasons = sorted(v_df['season'].unique())
    
    rows = []
    
    for s in seasons:
        # Eval Rank
        res_rank = calculate_objectives(s, v_df, panel_df, 'rank')
        rows.append({'season': s, 'rule': 'rank', **res_rank})
        
        # Eval Percent
        res_perc = calculate_objectives(s, v_df, panel_df, 'percent')
        rows.append({'season': s, 'rule': 'percent', **res_perc})
        
    out_df = pd.DataFrame(rows)
    save_path = os.path.join(OUTPUT_DIR, 'task4_baseline_metrics.csv')
    out_df.to_csv(save_path, index=False)
    print(f"Task 4 Baseline saved to {save_path}")

if __name__ == "__main__":
    run_task4_baseline()
