
import numpy as np
import pandas as pd
import json
import os
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import euclidean

# Pre-load data class to keep memory resident
class SeasonSimulatorFast:
    def __init__(self, panel_path, posteriors_dir, elim_path):
        """
        Initialize and pre-load all static data (Panel, Elim, Posteriors)
        to memory for fast Monte Carlo.
        """
        self.panel = pd.read_parquet(panel_path)
        self.posteriors_dir = posteriors_dir
        with open(elim_path, 'r', encoding='utf-8') as f:
            self.elim_data = json.load(f)
            
        # Cache for loaded seasons: {season_id: {'S':.., 'v_pool':.., 'elim_counts':..}}
        self.season_cache = {}
        
    def preload_season(self, season):
        if season in self.season_cache:
            return
            
        # 1. Load Posterior Pool
        # We load ALL draws or a subset? For Optuna, we might fix a subset of draws (e.g. 50 R)
        # to reduce variance during optimization (Standard SAA approach).
        # Or load full and sample on fly? SAA is better for optimization stability.
        
        post_path = os.path.join(self.posteriors_dir, f"season_{season}.npz")
        try:
            data = np.load(post_path, allow_pickle=True)
            v_all = data['v'] # (R_total, T, N)
            
            # Subsample 50 draws for stable evaluation
            np.random.seed(42) # Fixed seed for SAA
            if v_all.shape[0] > 50:
                idx = np.random.choice(v_all.shape[0], 50, replace=False)
                v_pool = v_all[idx]
            else:
                v_pool = v_all
                
            pair_ids = data['pair_ids']
            week_values = data['week_values'] # e.g. [1, 2, ..., 11]
            
        except Exception as e:
            print(f"Error loading S{season}: {e}")
            return

        # 2. Build S matrix (Judge Scores)
        df_s = self.panel[self.panel['season'] == season]
        n_weeks = len(week_values)
        n_pairs = len(pair_ids)
        
        S_mat = np.zeros((n_weeks, n_pairs))
        week_map = {w: i for i, w in enumerate(week_values)}
        pid_map = {p: i for i, p in enumerate(pair_ids)}
        
        for _, row in df_s.iterrows():
            if row['week'] in week_map and row['pair_id'] in pid_map:
                S_mat[week_map[row['week']], pid_map[row['pair_id']]] = row['S_it']
                
        # 3. Elim Counts per week index
        elim_schedule = []
        for w in week_values:
            key = f"{season}_{w}"
            count = 0
            if key in self.elim_data:
                count = len(self.elim_data[key].get('eliminated_names', []))
            elim_schedule.append(count)
            
        self.season_cache[season] = {
            'S_mat': S_mat, # (T, N)
            'v_pool': v_pool, # (R_fixed, T, N)
            'pair_ids': pair_ids,
            'week_values': week_values,
            'elim_schedule': elim_schedule,
            'n_pairs': n_pairs,
            'n_weeks': n_weeks
        }

    def simulate(self, theta, season, perturb_kappa=0.0):
        """
        Run simulation for a season with parameters theta.
        theta: dict {
            'a': float, 'b': float, (logistic weights)
            'eta': float, (vote compression)
            'gamma': float, (save prob)
            'save_flag': int (0 or 1)
        }
        perturb_kappa: float. noise scale for robustness check.
        
        Returns: 
            placements_matrix (R, N)
            metrics: dict
        """
        if season not in self.season_cache:
            self.preload_season(season)
            
        cache = self.season_cache[season]
        S_mat = cache['S_mat']
        v_pool = cache['v_pool'] # (R, T, N)
        n_weeks = cache['n_weeks']
        n_pairs = cache['n_pairs']
        elim_schedule = cache['elim_schedule']
        
        R = v_pool.shape[0]
        
        # Unpack theta
        a = theta['a']
        b = theta['b']
        eta = theta['eta']
        gamma = theta['gamma']
        use_save = (theta['save_flag'] > 0.5)
        
        # Precompute Weights w_t
        # w_t = sigma(a * (t - b))
        # t is week index 0..T-1? Or physical week?
        # Physical week usually better? Or normalized progress?
        # Let's use normalized progress t_norm in [0, 10]
        t_seq = np.arange(n_weeks)
        w_t_seq = 1.0 / (1.0 + np.exp(-a * (t_seq - b))) # Logistic
        
        # Prepare outputs
        placements = np.zeros((R, n_pairs), dtype=int)
        drama_list_r = [] # Initialize list for drama scores
        
        # Matrices for correlation calculation (Average over R)
        # We need per-week alignment logic? "Obj_F/J" definition:
        # rho(rank(mean_v), rank(P_theta)).
        # Actually Obj_F is final rank correlation.
        
        # Loop R samples
        for r in range(R):
            v_seq = v_pool[r] # (T, N)
            
            # Perturb votes if Robustness check
            if perturb_kappa > 1e-6:
                # Add noise: v_new ~ Dirichlet(v_old * kappa_inv?) 
                # or v_new = normalize(v_old + noise)
                # Simple: v_new = v * exp(N(0, kappa)). re-norm.
                noise = np.random.normal(0, perturb_kappa, size=v_seq.shape)
                v_seq = v_seq * np.exp(noise)
                # Re-norm row wise (active subset norm happens later loop, but good to keep scale)
                # v here is full matrix, normalization happens per active set.
            
            active_mask = np.ones(n_pairs, dtype=bool)
            current_rank = n_pairs
            
            for t in range(n_weeks):
                is_finals = (t == n_weeks - 1)
                active_idx = np.where(active_mask)[0]
                n_active = len(active_idx)
                if n_active == 0: break
                
                # Active Data
                S_act = S_mat[t, active_idx]
                v_act = v_seq[t, active_idx]
                
                # Norm v
                v_sum = np.sum(v_act)
                v_act = (v_act / v_sum) if v_sum > 0 else (np.ones(n_active)/n_active)
                
                # Norm S -> pJ
                S_sum = np.sum(S_act)
                pJ_act = (S_act / S_sum) if S_sum > 0 else (np.ones(n_active)/n_active)
                
                # Apply Power Eta to v
                # v_tilde = v^eta / sum(v^eta)
                v_pow = np.power(v_act, eta)
                v_pow_sum = np.sum(v_pow)
                v_tilde = (v_pow / v_pow_sum) if v_pow_sum > 0 else v_act
                
                # Combined Score
                # C = w * pJ + (1-w) * v_tilde
                wt = w_t_seq[t]
                scores = wt * pJ_act + (1.0 - wt) * v_tilde
                
                # Sort: Descending (Higher Score is Better)
                sorted_local_args = np.argsort(scores) 
                
                # Drama Metric Calculation (Margin)
                # Need 1st and 2nd highest scores for Active set
                # sorted_local_args[-1] is Best (Highest)
                # sorted_local_args[-2] is 2nd Best
                if n_active >= 2:
                    score_1st = scores[sorted_local_args[-1]]
                    score_2nd = scores[sorted_local_args[-2]]
                    margin = score_1st - score_2nd
                    
                    # Drama = 1 - (margin / max_margin). 
                    # Assuming practical max margin ~ 0.5 (usually much smaller).
                    # Let's normalize by something robust. 
                    # Or just return raw margin and let optimizer maximize (1-Margin).
                    # Implementation: Return (1 - margin).
                    # Since scores sum to 1, margin <= 1.
                    drama_t = 1.0 - margin
                    drama_list_r.append(drama_t)
                else:
                    drama_list_r.append(1.0) # Only 1 left? Max drama? Or 0? Undefined. Usually ignore.

                if is_finals:
                    # Best gets rank 1.
                    # sorted_local_args[-1] is Winner.
                    # [Worst ... Winner]
                    for offset, local_idx in enumerate(reversed(sorted_local_args)):
                        # offset 0 -> Winner -> Rank 1
                        pidx = active_idx[local_idx]
                        placements[r, pidx] = 1 + offset
                    break
                else:
                    # Elimination
                    n_elim = elim_schedule[t]
                    if n_elim == 0: continue
                    
                    # Candidates: The Worst n_elim
                    # Indices 0 to n_elim-1 in sorted_local_args
                    candidates_local = sorted_local_args[:n_elim] # Worst ones
                    
                    # Save Logic (Bottom 2 + Judges Choice)
                    should_save = False
                    if use_save and n_elim == 1 and n_active >= 2:
                        # Bottom 2 are indices 0 and 1
                        candidates_local = sorted_local_args[:2]
                        should_save = True
                        
                    final_elim_local = []
                    
                    if should_save:
                        # c1 (idx 0) is Worst Combined. c2 (idx 1) is 2nd Worst.
                        # Judge Save: Save the one with higher Judge Score?
                        # Usually Judges pick who to SAVE.
                        # They check S_act.
                        # Compare S_act of c1 and c2.
                        idx_worst = candidates_local[0]
                        idx_2nd = candidates_local[1]
                        
                        s1 = S_act[idx_worst]
                        s2 = S_act[idx_2nd]
                        
                        # Probabilistic Save Logic (Gammatized)
                        # P(save c1) ~ exp(gamma * s1) ??
                        # Judges prefer Higher S.
                        # Let's say P(pick c1) = exp(gamma * s1) / sum
                        # If picked, c1 is SAVED -> Elim c2.
                        
                        if gamma > 100: # Deterministic
                             # Save higher score
                             if s1 > s2: elim_target = idx_2nd
                             elif s2 > s1: elim_target = idx_worst
                             else: elim_target = idx_worst # Fallback (Worst combined goes)
                        else:
                            # Probabilistic
                            # Scale gamma? S is in range 0-30?
                            # exp(gamma * S). If gamma=1, exp(30) is huge.
                            # Standardize S or scale gamma small. 
                            # Let's assume gamma is small (0.1 - 1.0).
                            
                            # Prob of SAVING c1
                            # To avoid overflow, subtract max
                            s_max = max(s1, s2)
                            top = np.exp(gamma * (s1 - s_max))
                            bot = top + np.exp(gamma * (s2 - s_max))
                            p_save_c1 = top / bot
                            
                            if np.random.rand() < p_save_c1:
                                elim_target = idx_2nd # Given c1 saved
                            else:
                                elim_target = idx_worst
                        
                        final_elim_local = [elim_target]
                    else:
                        final_elim_local = list(candidates_local)
                        
                    # Execute Elim
                    # Eliminate them -> Assign current_rank
                    # If multiple, worst gets worst rank?
                    # final_elim_local are indices in active_idx array.
                    # We assign ranks.
                    for local_idx in final_elim_local:
                        pidx = active_idx[local_idx]
                        placements[r, pidx] = current_rank
                        # Decrease rank for next? 
                        # If multiple elims, they usually tie or ordered?
                        # Let's ordered by Combined score.
                        # In loop, candidates_local is [Worst, 2nd Worst].
                        # So first one gets current_rank (Worst).
                        
                        active_mask[pidx] = False
                        current_rank -= 1
        
        # Mean Drama for this simulation
        avg_drama = np.mean(drama_list_r) if drama_list_r else 0.0
        return placements, avg_drama

