import pandas as pd
import numpy as np
import json
import os
from scipy.optimize import minimize
from scipy.stats import rankdata
from tqdm import tqdm

# Configuration
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\baseline_samples'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Logic Parsers ---

def solve_percent_maxent(s_shares, elim_indices, active_indices):
    """
    MaxEnt solver for Percent Rule.
    Maximize -sum(v * log(v))
    Subject to:
      sum(v) = 1
      v >= 0
      For all surv in Survivors, all elim in Eliminated:
         0.5*s_surv + 0.5*v_surv >= 0.5*s_elim + 0.5*v_elim
         => v_surv - v_elim >= s_elim - s_surv
    """
    N = len(s_shares)
    
    # Survivors: Indices in active_indices NOT in elim_indices
    survivors = [i for i in range(N) if active_indices[i] not in elim_indices]
    eliminated = [i for i in range(N) if active_indices[i] in elim_indices]
    
    if not eliminated:
        # returns uniform
        return np.ones(N) / N
        
    def neg_entropy(v):
        # Avoid log(0)
        v_safe = np.clip(v, 1e-10, 1.0)
        return np.sum(v_safe * np.log(v_safe))
        
    constraints = [
        {'type': 'eq', 'fun': lambda v: np.sum(v) - 1.0}
    ]
    
    # Inequality constraints: v_surv - v_elim - (s_elim - s_surv) >= 0
    for surv_idx in survivors:
        for elim_idx in eliminated:
            delta_s = s_shares[elim_idx] - s_shares[surv_idx]
            # Capture values in closure
            def constr(v, si=surv_idx, ei=elim_idx, ds=delta_s):
                return v[si] - v[ei] - ds
            constraints.append({'type': 'ineq', 'fun': constr})
            
    # Initial guess: Uniform
    v0 = np.ones(N) / N
    bounds = [(0, 1) for _ in range(N)]
    
    res = minimize(neg_entropy, v0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    if res.success:
        return res.x
    else:
        # Fallback: return uniform or s_shares inverse?
        # print("  MaxEnt failed, returning uniform")
        return np.ones(N) / N

def solve_rank_feasible(s_ranks, elim_indices, active_indices, is_save_rule=False, n_samples=1000):
    """
    CSP Solver for Rank Rule via Monte Carlo.
    Find valid permutations of Fan Ranks (R_v).
    Condition:
       Strict Rank: Eliminated must have worst Rank Sum.
       Save Rule: Eliminated must be in Bottom 2 (at most 1 person worse).
    Returns: Average v derived from valid rank permutations.
    """
    N = len(s_ranks)
    valid_rank_sums = np.zeros(N)
    valid_count = 0
    
    survivors_rel = [i for i in range(N) if active_indices[i] not in elim_indices]
    eliminated_rel = [i for i in range(N) if active_indices[i] in elim_indices]
    
    if not eliminated_rel:
        return np.ones(N) / N
        
    attempts = 0
    max_attempts = n_samples * 50 # Increase limit
    
    while valid_count < n_samples and attempts < max_attempts:
        attempts += 1
        r_v = np.random.permutation(N) + 1 # 1-based ranks
        scores = s_ranks + r_v
        
        is_valid = True
        for elim_idx in eliminated_rel:
            p_score = scores[elim_idx]
            # Count strictly worse (Higher Score = Worse)
            worse_count = np.sum(scores > p_score)
            
            if is_save_rule:
                # Bottom 2 condition: Rank >= N-1.
                # Means at most 1 person is worse.
                if worse_count > 1:
                    is_valid = False
                    break
            else:
                # Bottom 1 condition: Rank = N.
                # No one can be worse.
                if worse_count > 0:
                    is_valid = False
                    break
        
        if is_valid:
            valid_rank_sums += r_v
            valid_count += 1
            
    if valid_count == 0:
        return np.ones(N) / N
        
    avg_ranks = valid_rank_sums / valid_count
    inv_ranks = (N + 1) - avg_ranks
    v_est = inv_ranks / np.sum(inv_ranks)
    return v_est

def run_baseline():
    print("--- Running Task 1 Baseline (MaxEnt/CSP) ---")
    
    # Load Data
    if PANEL_PATH.endswith('.csv'):
        df = pd.read_csv(PANEL_PATH)
    else:
        df = pd.read_parquet(PANEL_PATH)
        
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_json = json.load(f)
        
    seasons = sorted(df['season'].unique())
    results = [] 
    
    for s in tqdm(seasons, desc="Processing Seasons"):
        df_s = df[df['season'] == s]
        weeks = sorted(df_s['week'].unique())
        
        # Determine Rule
        if s <= 2:
            rule_type = 'rank'
        elif s >= 28:
            rule_type = 'rank_save'
        else:
            rule_type = 'percent'
            
        # Parse Names to IDs
        name_map = {}
        pair_data = df_s[['pair_id', 'celebrity_name']].drop_duplicates()
        for _, r in pair_data.iterrows():
            name_map[r['celebrity_name']] = r['pair_id']
            
        for w in weeks:
            df_w = df_s[df_s['week'] == w].sort_values('pair_id')
            active_ids = df_w['pair_id'].values
            s_scores = df_w['S_it'].values 
            
            if 'pJ_it' in df_w.columns:
                s_shares = df_w['pJ_it'].values
            else:
                s_shares = s_scores / s_scores.sum()
                
            elim_ids = []
            key = f"{s}_{w}"
            if key in elim_json:
                names = elim_json[key].get('eliminated_names', [])
                for n in names:
                    if n in name_map:
                        elim_ids.append(name_map[n])
            
            if not elim_ids:
                N = len(active_ids)
                v_sol = np.ones(N) / N
            else:
                if rule_type == 'percent':
                    v_sol = solve_percent_maxent(s_shares, elim_ids, active_ids)
                else:
                    s_ranks = rankdata(-s_scores, method='average')
                    is_save = (rule_type == 'rank_save')
                    v_sol = solve_rank_feasible(s_ranks, elim_ids, active_ids, is_save_rule=is_save)
            
            for i, pid in enumerate(active_ids):
                results.append({
                    'season': s,
                    'week': w,
                    'pair_id': pid,
                    'v_baseline': v_sol[i],
                    'rule_used': rule_type
                })
                
    out_df = pd.DataFrame(results)
    save_path = os.path.join(OUTPUT_DIR, 'task1_baseline_v_est.csv')
    out_df.to_csv(save_path, index=False)
    print(f"Baseline results saved to {save_path}")

if __name__ == "__main__":
    run_baseline()
