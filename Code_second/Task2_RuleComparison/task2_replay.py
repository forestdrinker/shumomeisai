
import numpy as np
import pandas as pd
import argparse
import os
import json
from tqdm import tqdm
from scipy.stats import rankdata, spearmanr

# Constants
PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\replay_results'

def get_elimination_count(season, week, elim_data):
    key = f"{season}_{week}"
    if key in elim_data:
        return len(elim_data[key].get('eliminated_names', []))
    return 0


def run_replay(season, rule, num_simulations=None, gamma=None):
    """
    Replay a season using posterior samples under a specific rule.
    rule: 'rank', 'percent', 'rank_save', 'percent_save'
    gamma: float, optional. parameter for probabilistic judges save. 
           If None, uses deterministic save (save better score).
    """
    print(f"--- Replaying Season {season} | Rule: {rule} | Gamma: {gamma} ---")
    
    # 1. Load Data
    post_file = os.path.join(POSTERIOR_DIR, f"season_{season}.npz")
    if not os.path.exists(post_file):
        print(f"Posterior file not found: {post_file}")
        return

    data = np.load(post_file, allow_pickle=True)
    v_samples = data['v'] # (samples, T, N)
    pair_ids = data['pair_ids']
    week_values = data['week_values']
    
    n_samples, n_weeks, n_pairs = v_samples.shape
    
    # Subset samples if requested
    if num_simulations and num_simulations < n_samples:
        indices = np.random.choice(n_samples, num_simulations, replace=False)
        v_samples = v_samples[indices]
        n_samples = num_simulations
        
    # Load Panel for Judges Scores (S_it)
    df = pd.read_parquet(PANEL_PATH)
    df_s = df[df['season'] == season].copy()
    
    # Build Judge Score Matrix S_mat (T, N)
    # We use raw S_it. 
    S_mat = np.zeros((n_weeks, n_pairs))
    
    week_map = {w: i for i, w in enumerate(week_values)}
    pid_map = {p: i for i, p in enumerate(pair_ids)}
    
    for _, row in df_s.iterrows():
        if row['week'] in week_map and row['pair_id'] in pid_map:
            t = week_map[row['week']]
            i = pid_map[row['pair_id']]
            S_mat[t, i] = row['S_it']
            
    # Load Elim Data
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_data = json.load(f)
        
    # Prepare outputs
    # Placements: 1=Best, N=Worst
    placements = np.zeros((n_samples, n_pairs), dtype=int)
    elim_weeks = np.zeros((n_samples, n_pairs), dtype=int)
    
    # Metrics Lists (List of arrays, one per sample)
    # D_t_history: Each element is (T,) array of conflict scores
    conflict_history = [] 
    upset_counts = []
    
    # Determine base rule type
    is_save = 'save' in rule
    base_mode = 'percent' if 'percent' in rule else 'rank'
    
    # Loop Samples
    for s_idx in tqdm(range(n_samples), desc="Simulating"):
        
        active_mask = np.ones(n_pairs, dtype=bool)
        current_rank = n_pairs
        
        sample_conflicts = np.full(n_weeks, np.nan)
        sample_upsets = 0
        
        # Loop Weeks
        for t in range(n_weeks):
            week_val = week_values[t]
            is_finals = (t == n_weeks - 1)
            
            # Identify active indices
            active_indices = np.where(active_mask)[0]
            n_active = len(active_indices)
            
            if n_active == 0:
                break
                
            # --- 1. Re-normalization Step ---
            # Get S and v for active subset
            S_active = S_mat[t, active_indices]
            v_active_raw = v_samples[s_idx, t, active_indices]
            
            # Normalize v to sum to 1
            v_sum = np.sum(v_active_raw)
            if v_sum > 0:
                v_active = v_active_raw / v_sum
            else:
                v_active = np.ones_like(v_active_raw) / n_active
                
            # Normalize S to percentages (p_J) logic
            S_sum = np.sum(S_active)
            if S_sum > 0:
                pJ_active = S_active / S_sum
            else:
                pJ_active = np.ones_like(S_active) / n_active
                
            # --- 2. Calculate Badness / Combined Score ---
            
            # Rank Data (Higher Score is Better -> Higher Rank Number is Better or Worse?)
            # rankdata: smallest=1. 
            # We assume S and v are "Higher is Better".
            # So rank(S): Smallest S gets 1 (Worst). Largest S gets N (Best).
            # Wait. In DWTS "Rank score" Usually: 1st Place = Best. 
            # The rule says: "lowest combined total of judges points and audience votes".
            # This implies "Rank 1" is Best. 
            # rankdata(-score): Largest score gets rank 1 (Best). Smallest gets rank N (Worst).
            
            rJ_active = rankdata(-S_active, method='min') # 1=Best
            rF_active = rankdata(-v_active, method='min') # 1=Best
            
            # Conflict Metric D_t
            # D_t = 1 - Spearman(Rank(S), Rank(v))
            # Note: Spearman on small N can be noisy.
            if n_active > 1:
                # We use the ranks we just computed
                corr, _ = spearmanr(rJ_active, rF_active)
                if np.isnan(corr): corr = 0 # Happens if constant variance
                sample_conflicts[t] = 1 - corr
            else:
                sample_conflicts[t] = 0.0
                
            # Combined Score / Badness
            if base_mode == 'rank':
                # Rank Rule: Sum of Ranks.
                # Combined = rJ + rF
                # Lower is Better. Higher is Worse.
                combined = rJ_active + rF_active
                
                # Tie-breaker: Vote Share (v_active). Higher v -> Better.
                # So we subtract a tiny fraction of v to break ties in favor of high v.
                # combined_final = combined - v * small
                # Sorting: Ascending (Smallest/Best first).
                
                sort_metric = combined - (v_active * 1e-6)
                
            else:
                 # Percent Rule: pJ + v
                 # Higher is Better.
                 combined = pJ_active + v_active
                 # We want to sort Best to Worst? Or Worst to Elim?
                 # Standard: Sort Best to Worst.
                 # Best has High Combined.
                 # So we sort by -Combined.
                 
                 sort_metric = -combined # Smallest (Most Negative) is Best.
                 
            # ArgSort: [Best_Idx, 2nd_Best_Idx, ... , Worst_Idx]
            sorted_local_args = np.argsort(sort_metric)
            
            # --- 3. Elimination Logic ---
            if is_finals:
                # Assign ranks to all remaining
                # sorted_local_args[0] is Winner (Rank 1)
                
                for rank_offset, local_idx in enumerate(sorted_local_args):
                    pidx = active_indices[local_idx]
                    placements[s_idx, pidx] = 1 + rank_offset
                    elim_weeks[s_idx, pidx] = n_weeks
                    
                break # End of season
            else:
                # Regular Week
                n_elim = get_elimination_count(season, week_val, elim_data)
                
                if n_elim == 0:
                    continue
                    
                # Identify potential eliminees (The worst n_elim)
                # "Bottom N" are the last N in sorted_local_args
                
                candidates_local = sorted_local_args[-n_elim:]
                
                should_save = False
                if is_save and n_elim == 1:
                    # Judges Save applies if Single Elimination (usually)
                    # "First pick Bottom 2"
                    # Bottom 2 are the last 2.
                    if n_active >= 2:
                        candidates_local = sorted_local_args[-2:] # Bottom 2
                        should_save = True
                
                # Upset Checking (Did we eliminate someone NOT in Judge Bottom 2?)
                # Judge Bottom 2 calculation (Handle Ties correctly)
                # Logic: Is the candidate's score <= the 2nd lowest score?
                
                distinct_scores = np.unique(S_active)
                if len(distinct_scores) >= 2:
                    bottom_2_thresh = distinct_scores[1] # 2nd LOWEST value
                else:
                    bottom_2_thresh = distinct_scores[0] if len(distinct_scores)>0 else 0
                    
                # A person is in Bottom 2 if their S <= bottom_2_thresh
                
                final_elim_local = []
                
                if should_save:
                    # Candidates are Bottom 2 (Combined).
                    # Judges save one. 
                    
                    c1_local = candidates_local[0] # Better of the two (3rd score / 2nd Worst)
                    c2_local = candidates_local[1] # Worst of the two (Last)
                    
                    r1 = rJ_active[c1_local]
                    r2 = rJ_active[c2_local]
                    
                    elim_c1 = False
                    
                    # Probabilistic Save
                    if gamma is not None and not np.isinf(gamma):
                        # P(elim i) propto exp(gamma * rank_i)
                        # rank 1=Best. Higher Rank = Worse.
                        # So worse rank should have higher Prob of Elim.
                        
                        score_1 = np.exp(gamma * r1)
                        score_2 = np.exp(gamma * r2)
                        prob_elim_c1 = score_1 / (score_1 + score_2)
                        
                        elim_c1 = np.random.rand() < prob_elim_c1
                        
                    else:
                        # Deterministic: Eliminate the one with WORSE Judge Rank (Higher rJ)
                        # rJ: 1=Best.
                        if r1 > r2:
                            elim_c1 = True # c1 is worse
                        elif r2 > r1:
                            elim_c1 = False # c2 is worse
                        else:
                            # Equal Judge Rank. Fallback to Combined Rank.
                            # c2 has worse Combined Rank (it is candidates_local[1])
                            elim_c1 = False
                            
                    if elim_c1:
                        final_elim_local = [c1_local]
                    else:
                        final_elim_local = [c2_local]
                        
                else:
                    # Standard Elim: All candidates go home
                    final_elim_local = candidates_local
                
                # Execute Elimination
                # final_elim_local contains local indices to eliminate
                # The WORST one gets the worst rank (current_rank)
                # If multiple, we start from current_rank and go down.
                # Assuming final_elim_local is ordered [Better ... Worst] (subset of sorted args)
                # We should assign ranks in reverse order.
                
                for k, local_idx in enumerate(reversed(final_elim_local)):
                    # k=0 -> Worst -> Rank N
                    pidx = active_indices[local_idx]
                    placements[s_idx, pidx] = current_rank - k
                    elim_weeks[s_idx, pidx] = week_val
                    active_mask[pidx] = False
                    
                    # Upset Check (Corrected Logic)
                    s_val = S_active[local_idx]
                    # If s_val > bottom_2_thresh, it is strictly NOT in the bottom 2 scores.
                    # This is an upset.
                    if s_val > bottom_2_thresh:
                         sample_upsets += 1
                         
                current_rank -= len(final_elim_local)

        # End Week Loop
        conflict_history.append(sample_conflicts)
        upset_counts.append(sample_upsets)
        
    # Save Results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    save_path = os.path.join(OUTPUT_DIR, f"season_{season}_{rule}.npz")
    
    # Pack Metrics
    conflict_history = np.array(conflict_history) # (samples, T)
    
    # Calculate Derived
    champion_ids = pair_ids[np.argmin(placements, axis=1)]
    # Top 3 (indexes)
    # argsort gives indices of ranks. strictly we want pair_ids.
    # placements (R, N). Sort axis 1.
    top3_indices = np.argsort(placements, axis=1)[:, :3]
    top3_ids = pair_ids[top3_indices]
    
    np.savez(save_path, 
             placements=placements, 
             elim_weeks=elim_weeks, 
             pair_ids=pair_ids,
             conflict_history=conflict_history,
             upset_counts=upset_counts,
             champion_ids=champion_ids,
             top3_ids=top3_ids,
             meta=np.array([str(season), rule], dtype=str))
             
    print(f"Saved replay to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int)
    parser.add_argument('--rule', type=str)
    parser.add_argument('--all', action='store_true', help="Run all seasons/rules")
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=None, help="Gamma parameter for probabilistic judges save")
    
    args = parser.parse_args()
    
    rules = ['rank', 'percent', 'rank_save', 'percent_save']
    
    if args.all:
        # Scan dir for seasons
        files = os.listdir(POSTERIOR_DIR)
        seasons = sorted([int(f.split('_')[1].split('.')[0]) for f in files if f.startswith('season_')])
        for s in seasons:
            for r in rules:
                run_replay(s, r, args.num, args.gamma)
    else:
        if args.season and args.rule:
            run_replay(args.season, args.rule, args.num, args.gamma)
        else:
            print("Please specify --season and --rule, or --all")

