
import os
import optuna
import pandas as pd
import numpy as np
# Fix 1: Reproducibility
np.random.seed(2026)
import json
from scipy.stats import spearmanr, rankdata, kendalltau
from task4_simulator import SeasonSimulatorFast

# Config
N_TRIALS = 30 # As per plan (Small budget for demo/speed)
N_SEASONS = 5 # Evaluate on first 5 seasons for speed
SEASON_IDS = [1, 2, 3, 4, 5] 
R_DRAWS = 5 # Reduced draws for optimization speed (Validation uses 30)
K_PERTURB = 5 # Perturbation runs for Robustness
PERTURB_KAPPA = 0.5 # Noise scale

# Paths
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
POSTERIORS_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize Simulator (Global to avoid reload)
sim = SeasonSimulatorFast(PANEL_PATH, POSTERIORS_DIR, ELIM_PATH)
# Preload seasons
for s in SEASON_IDS:
    sim.preload_season(s)

def objective(trial):
    # 1. Sample Theta
    # w_t = sigma(a * (t - b))
    # a > 0. Controls steepness. Range [0.1, 2.0]
    # b > 0. Controls shifts. Range [0, 10] (weeks)
    a = trial.suggest_float('a', 0.1, 2.0)
    b = trial.suggest_float('b', 0.0, 10.0)
    
    # eta: Vote compression. 0.0=Uniform, 1.0=Linear, 3.0=Strong.
    eta = trial.suggest_float('eta', 0.1, 3.0)
    
    # gamma: Save prob. 
    # If using save.
    save_flag = trial.suggest_categorical('save_flag', [0, 1])
    gamma = 1.0
    if save_flag == 1:
        gamma = trial.suggest_float('gamma', 0.1, 2.0)
        
    theta = {'a': a, 'b': b, 'eta': eta, 'gamma': gamma, 'save_flag': save_flag}
    
    # 2. Evaluate across Seasons
    obj_F_list = []
    obj_J_list = []
    obj_D_list = []
    obj_R_list = []
    
    for s in SEASON_IDS:
        # A. Base Simulation (No Perturbation)
        placements, avg_drama = sim.simulate(theta, s, perturb_kappa=0.0) # (R, N)
        
        # Calculate Obj F, J, D
        # Need Ground Truths: rank(mean_v), rank(S)
        # sim config has raw data.
        cache = sim.season_cache[s]
        v_pool = cache['v_pool'] 
        S_mat = cache['S_mat'] # (T, N)
        
        # Ground Truth Ranks (Final)
        v_mean_all = np.mean(v_pool, axis=0) # (T, N)
        v_total = np.sum(v_mean_all, axis=0)
        rank_v_true = rankdata(-v_total) # 1=Best
        
        S_total = np.sum(S_mat, axis=0)
        rank_j_true = rankdata(-S_total) # 1=Best
        
        # Simulated Rank (Mean Placement)
        mean_placement = np.mean(placements, axis=0)
        rank_sim = rankdata(mean_placement) # Small placement is Best. 1=1.
        
        # Metrics
        rho_F, _ = spearmanr(rank_sim, rank_v_true) # High is Better
        rho_J, _ = spearmanr(rank_sim, rank_j_true) # High is Better
        
        obj_F_list.append(rho_F)
        obj_J_list.append(rho_J)
        
        # Obj D: Drama (Margin)
        # avg_drama returned from sim is mean(1 - margin)
        obj_D_list.append(avg_drama) 
        
        # B. Robustness (Perturbation)
        # Run K times with noise
        kendall_dists = []
        for k in range(K_PERTURB):
            p_pert, _ = sim.simulate(theta, s, perturb_kappa=PERTURB_KAPPA)
            mp_pert = np.mean(p_pert, axis=0)
            tau, _ = kendalltau(mean_placement, mp_pert)
            kendall_dists.append(tau)
            
        obj_R_list.append(np.mean(kendall_dists))

    # Mean and SD across seasons
    
    # Fix 3: Robustness - Failure Rate Penalty
    # If too many seasons failed (NaN), penalize this rule heavily
    valid_count = np.sum(~np.isnan(obj_F_list))
    total_seasons = len(SEASON_IDS)
    if valid_count < (0.9 * total_seasons):
        # Return bad scores to prune this parameter set
        return -1.0, -1.0, -1.0, -1.0

    f_mean = np.nanmean(obj_F_list)
    j_mean = np.nanmean(obj_J_list)
    d_mean = np.nanmean(obj_D_list)
    r_mean = np.nanmean(obj_R_list)
    
    f_sd = np.nanstd(obj_F_list)
    j_sd = np.nanstd(obj_J_list)
    d_sd = np.nanstd(obj_D_list)
    r_sd = np.nanstd(obj_R_list)
    
    # Store auxiliary attributes (Mean and SD)
    trial.set_user_attr("obj_F_mean", f_mean)
    trial.set_user_attr("obj_J_mean", j_mean)
    trial.set_user_attr("obj_D_mean", d_mean)
    trial.set_user_attr("obj_R_mean", r_mean)
    
    trial.set_user_attr("obj_F_sd", f_sd)
    trial.set_user_attr("obj_J_sd", j_sd)
    trial.set_user_attr("obj_D_sd", d_sd)
    trial.set_user_attr("obj_R_sd", r_sd)
    
    # Optuna minimizes? Directions set later.
    return f_mean, j_mean, d_mean, r_mean

def run_optimizer():
    print("--- 正在运行 Task 4 Optuna 多目标优化 (MOBO) ---")
    
    # Directions: Maximize all (Correlation, Correlation, Drama, Similarity/Robustness)
    study = optuna.create_study(directions=["maximize", "maximize", "maximize", "maximize"])
    
    # Stage 1: LHS Warmup (simulated by RandomSampler or LatinHypercubeSampler if available)
    # Optuna NSGAII uses Random sampling for initialization.
    
    print(f"优化 {N_TRIALS} 次尝试 (N_SEASONS={len(SEASON_IDS)})...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # Save Results
    print("正在保存结果...")
    
    # 1. Theta Evaluations (All Trials)
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE: continue
        row = t.params.copy()
        # Get Attributes from user_attrs (More reliable than values for SD)
        for k, v in t.user_attrs.items():
            row[k] = v
            
        # Add N_seasons
        row['N_seasons'] = len(SEASON_IDS)
        rows.append(row)
        
    df_evals = pd.DataFrame(rows)
    # Audit Fix: Fill gamma NaN and Round
    if 'gamma' in df_evals.columns:
        df_evals['gamma'] = df_evals['gamma'].fillna(1.0)
    df_evals = df_evals.round(4)
    df_evals.to_parquet(os.path.join(OUTPUT_DIR, 'task4_theta_evaluations.parquet'))
    
    # 2. Pareto Front
    pareto_trials = study.best_trials
    p_rows = []
    for t in pareto_trials:
        row = t.params.copy()
        # Ensure we get all metrics including SD
        for k, v in t.user_attrs.items():
            row[k] = v
        p_rows.append(row)
        
    df_pareto = pd.DataFrame(p_rows)
    # Audit Fix: Fill gamma NaN and Round
    if 'gamma' in df_pareto.columns:
        df_pareto['gamma'] = df_pareto['gamma'].fillna(1.0)
    df_pareto = df_pareto.round(4)
    df_pareto.to_csv(os.path.join(OUTPUT_DIR, 'task4_pareto_front.csv'), index=False)
    
    print("优化完成。")
    print(f"Pareto 前沿大小: {len(df_pareto)}")

if __name__ == "__main__":
    run_optimizer()
