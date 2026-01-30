
import numpy as np
import pandas as pd
import json
import os
import glob
from scipy.stats import entropy

# Metrics calculation
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
SAMPLES_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_FILE = r'd:\shumomeisai\Code_second\Results\task1_metrics.csv'

def compute_row_metrics(probs, true_elim_indices):
    """
    probs: (N_active,) probability of being eliminated for each active person
    true_elim_indices: list of indices (in local active array) that were eliminated
    """
    # 1. Coverage 90
    # Sort probs descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    
    # Cutoff where cumsum >= 0.9
    cutoff_idx = np.searchsorted(cumsum, 0.9)
    # The set is sorted_indices[:cutoff_idx+1]
    cred_set = sorted_indices[:cutoff_idx+1]
    
    hit_cov = any(idx in cred_set for idx in true_elim_indices)
    
    # 2. Acc (Top-K where K is number of eliminations)
    # If m eliminated, are they the top m predicted?
    # Or just "Is the true elim in the top X?" 
    # Logic: argmax matches?
    # If 1 eliminated: Top-1 acc.
    # If 2 eliminated: Top-2 acc?
    
    # Let's use: Is ANY true elim in the Top-K (where K=len(true))?
    # Or Strict Acc: Predicted Set == True Set?
    # Definition in doc: Acc: 1[argmax p = e_true]. (Implies 1 elim).
    # If multiple, maybe 1[argmax p \in e_true]?
    
    top_1 = sorted_indices[0]
    acc = 1 if top_1 in true_elim_indices else 0
    
    # Top-2
    top_2 = sorted_indices[:2]
    top2_hit = 1 if any(t in top_2 for t in true_elim_indices) else 0
    
    # 3. Brier
    # Sum (p_i - y_i)^2
    # y_i = 1 if i in true_elim, else 0
    # normalized by N? Doc says "Sum over i".
    # And "average over N events". 
    N = len(probs)
    y = np.zeros(N)
    for idx in true_elim_indices:
        y[idx] = 1.0
        
    brier = np.sum((probs - y)**2)
    
    return hit_cov, acc, top2_hit, brier, N
    
def compute_ci_width(v_samples, active_indices_local):
    # v_samples: (Samples, N_active)
    # Compute q025 and q975
    # Then width
    low = np.percentile(v_samples, 2.5, axis=0)
    high = np.percentile(v_samples, 97.5, axis=0)
    width = high - low
    return np.mean(width)

def compute_metrics(samples_dir=SAMPLES_DIR):
    # Load Elim Events
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_data = json.load(f)
        
    # Find results
    files = glob.glob(os.path.join(samples_dir, "season_*.npz"))
    print(f"Found {len(files)} sample files in {samples_dir}")
    
    metrics_rows = []
    
    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        season = int(data['season'])
        pair_ids = data['pair_ids'] # map index -> pair_id
        week_values = data['week_values'] # map t -> week_num
        
        # Posterior samples
        # u: (samples, time, N)
        # v: (samples, time, N)
        v_samples = data['v']
        u_samples = data['u']
        
        n_samples, n_weeks, n_pairs = v_samples.shape
        
        # We need to simulate Elimination Probability p_{s,t}(i)
        # This requires re-running the elimination likelihood logic (PL) on the samples.
        # Or simpler:
        # In Task 1 "Posterior Predictive", we estimate p(e=i).
        # We need 'b' (badness) samples. 
        # v and u are saved. 
        # We can reconstruct 'b' from v and S (observed).
        # S is in panel.
        
        # Load panel for S_it
        # (This is slightly inefficient to reload per season, but okay)
        pan = pd.read_parquet(r'd:\shumomeisai\Code_second\Data\panel.parquet')
        pan_s = pan[pan['season'] == season]
        
        # Reconstruct S matrix and Active Mask
        S_mat = np.zeros((n_weeks, n_pairs))
        p_mat = np.zeros((n_weeks, n_pairs)) # judge percent
        mask_mat = np.zeros((n_weeks, n_pairs), dtype=bool)
        
        # Needed for 'b' calculation
        # Map: week value -> t index
        # Map: pair_id -> i index
        # We assume pair_ids in npz matches our reconstruction logic, 
        # but better to rely on pair_ids stored in npz.
        
        pid_map = {pid: i for i, pid in enumerate(pair_ids)}
        w_map = {w: i for i, w in enumerate(week_values)}
        
        rule_segment = pan_s['rule_segment'].iloc[0]
        
        for _, row in pan_s.iterrows():
            if row['week'] in w_map and row['pair_id'] in pid_map:
                t = w_map[row['week']]
                i = pid_map[row['pair_id']]
                S_mat[t, i] = row['S_it']
                p_mat[t, i] = row['pJ_it']
                mask_mat[t, i] = True
        
        # Calculate PCR for v (Variance Ratio)
        # Prior Variance? 
        # Heuristic: Prior for u is RW. u_t ~ N(0, t*sigma^2). v is softmax.
        # Var(v) prior is roughly constant (Dirichlet-like) or slowly growing?
        # Without running Prior predictive, we can't get exact PCR. 
        # Let's skip PCR in this script for now, or use a placeholder based on Posterior Var.
        # Doc says "PCR = Var(prior)/Var(posterior)".
        # We will log Mean Posterior Variance as a proxy for information gain (lower is better).
        
        mean_post_var = np.mean(np.var(v_samples, axis=0))
        
        # Loop weeks for Coverage/Accuracy
        for t in range(n_weeks):
            week_num = week_values[t]
            key = f"{season}_{week_num}"
            
            if key not in elim_data:
                continue
                
            edata = elim_data[key]
            elim_names = edata.get('eliminated_names', [])
            
            if not elim_names:
                continue
                
            # Map names to Indices
            # We need Name -> PairID -> Index
            # Extract name map from panel
            # (Assuming unique names)
            current_week_df = pan_s[pan_s['week'] == week_num]
            name_to_idx = {}
            for _, r in current_week_df.iterrows():
                if r['pair_id'] in pid_map:
                    name_to_idx[r['celebrity_name']] = pid_map[r['pair_id']]
            
            true_elim_indices = [name_to_idx[n] for n in elim_names if n in name_to_idx]
            
            if not true_elim_indices:
                continue
                
            # Compute Predictive Probabilities P(e=i)
            # using samples of v (and S) -> b -> PL
            # Need to implement the 'b' calculation logic here (numpy version)
            
            # Kappa constants (match model)
            kappa_J = 0.5
            kappa_F = 0.05
            kappa_C = 0.05
            
            # Helper for SoftRank (numpy)
            def soft_rank_np(score_vec, kappa, mask):
                # score_vec: (N,)
                # mask: (N,)
                diff = score_vec[None, :] - score_vec[:, None] # (xi, xj) ?? No, (row - col)
                # s_row (1, N) - s_col (N, 1) = diff[i, k] = x_k - x_i
                
                s_row = score_vec.reshape(1, -1)
                s_col = score_vec.reshape(-1, 1)
                d = s_row - s_col
                
                sig = 1.0 / (1.0 + np.exp(-d / kappa))
                
                # Mask k
                valid_k = mask.reshape(1, -1)
                sig_masked = np.where(valid_k, sig, 0.0)
                
                # Sum
                r = 1.0 + np.sum(sig_masked, axis=1) - 0.5
                return np.where(mask, r, 0.0)

            # Compute b samples
            # v_samples[:, t, :] -> (S, N)
            # S_mat[t, :] -> (N) (constant)
            
            # This loop over samples is heavy if S is large. 
            # Vectorize over samples?
            # soft_rank is pointwise.
            
            # Rank Rule: b = SoftRank(S) + SoftRank(v)
            # Percent: b = SoftRank(p + v)
            
            # Precompute S Rank (Constant over samples)
            rJ = soft_rank_np(S_mat[t], kappa_J, mask_mat[t])
            
            # Compute b for each sample
            # We want P(e=i) = Mean_over_samples( P(e=i | b_sample) )
            
            # Calculate P(e=i | b) for PL single elim (assuming single for now approximation)
            # P(e=i) = exp(lam * b_i) / Sum exp
            
            lambda_pl = 10.0
            
            probs_sum = np.zeros(n_pairs)
            
            for s_idx in range(n_samples):
                v_t_s = v_samples[s_idx, t]
                
                if rule_segment == 'percent':
                    comb = p_mat[t] + v_t_s
                    b_val = soft_rank_np(comb, kappa_C, mask_mat[t])
                else:
                    rF = soft_rank_np(v_t_s, kappa_F, mask_mat[t])
                    b_val = rJ + rF
                
                # PL Prob
                logits = lambda_pl * b_val
                # Mask
                # exp(logits) where mask, else 0
                exps = np.where(mask_mat[t], np.exp(logits), 0.0)
                sum_exps = np.sum(exps)
                if sum_exps > 0:
                    probs = exps / sum_exps
                else:
                    probs = np.zeros_like(exps)
                    
                probs_sum += probs
                
            avg_probs = probs_sum / n_samples
            
            # Now compute metrics
            # Filter to active only for indices
            active_indices = np.where(mask_mat[t])[0]
            
            # We need to map global pair index to "active index" or just use global with masking?
            # compute_row_metrics works on active subset probabilities
            
            active_probs = avg_probs[active_indices]
            
            # CI Width for active only
            # v_samples[:, t, :] is (S, N). active_indices is (N_active,)
            v_active_samples = v_samples[:, t, :][:, active_indices]
            ci_width_val = compute_ci_width(v_active_samples, active_indices)
            # Need to map true_elim_indices to local active indices?
            # true_elim_indices are global indices.
            
            local_true_elim = []
            for glob_idx in true_elim_indices:
                # Find position in active_indices
                res = np.where(active_indices == glob_idx)[0]
                if len(res) > 0:
                    local_true_elim.append(res[0])
            
            if not local_true_elim:
                continue
                
            hit, acc, top2, brier, n_active = compute_row_metrics(active_probs, local_true_elim)
            
            metrics_rows.append({
                'season': season,
                'week': week_num,
                'n_active': n_active,
                'n_elim': len(local_true_elim),
                'coverage_90': int(hit),
                'accuracy': acc,
                'top2_acc': top2,
                'brier': brier,
                'top2_acc': top2,
                'brier': brier,
                'post_var_v_mean': mean_post_var,
                'avg_ci_width': ci_width_val
            })
            
        # --- Generate Posterior Summary (CSV) ---
        # season, week, pair_id, v_mean, v_q05, v_q50, v_q95, v_ci_width, S_it, pJ_it
        summary_rows = []
        
        # We assume t, i are mapped.
        # w_map: week dict, pid_map: pair dict
        # inverse maps
        inv_w_map = {i: w for w, i in w_map.items()}
        inv_pid_map = {i: p for p, i in pid_map.items()}
        
        for t in range(n_weeks):
            week_val = inv_w_map[t]
            for i in range(n_pairs):
                # Check active
                if not mask_mat[t, i]:
                    continue
                    
                v_dist = v_samples[:, t, i]
                mean_v = np.mean(v_dist)
                q05 = np.percentile(v_dist, 5)
                q50 = np.median(v_dist)
                q95 = np.percentile(v_dist, 95)
                # 95% CI Width for summary (usually q975-q025 or q95-q05?)
                # Doc says v_ci_width. Let's use 95% interval (2.5-97.5) to match metric
                q025 = np.percentile(v_dist, 2.5)
                q975 = np.percentile(v_dist, 97.5)
                width = q975 - q025
                
                pair_id = inv_pid_map[i]
                
                summary_rows.append({
                    'season': season,
                    'week': week_val,
                    'pair_id': pair_id,
                    'v_mean': mean_v,
                    'v_q05': q05,
                    'v_q50': q50,
                    'v_q95': q95,
                    'v_ci_width': width,
                    'S_it': S_mat[t, i],
                    'pJ_it': p_mat[t, i]
                })
        
        if summary_rows:
            df_summ = pd.DataFrame(summary_rows)
            # Save to specific summary file
            summ_path = os.path.join(samples_dir, f"season_{season}_summary.csv")
            df_summ.to_csv(summ_path, index=False)
            print(f"  -> Generated Summary: {summ_path}")
            
    # Save
    if metrics_rows:
        df_out = pd.DataFrame(metrics_rows)
        # Output file relative to input dir or fixed? 
        # Let's save to same dir as samples or fixed.
        out_path = os.path.join(os.path.dirname(samples_dir), 'task1_metrics.csv')
        df_out.to_csv(out_path, index=False)
        print(f"Metrics saved to {out_path}")
        print(df_out.groupby('season')[['accuracy', 'coverage_90', 'brier']].mean())
    else:
        print("No metrics calculated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=r'd:\shumomeisai\Code_second\Results\posterior_samples', help='Directory containing .npz samples')
    args = parser.parse_args()
    
    compute_metrics(args.dir)
