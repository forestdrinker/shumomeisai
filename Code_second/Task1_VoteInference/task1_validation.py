"""
Task 1 修正版验证脚本
====================
核心修改: 实现时序预测验证 (Temporal Predictive Validation)

【问题根源】
原代码问题：模型训练时使用淘汰事件作为似然约束，验证时测试同一组淘汰事件
这导致了循环验证 (Circular Validation)

【修正方案】
时序预测验证：
- 对于t周的淘汰预测，使用t-1周的后验信息 + 随机游走先验
- 不使用t周的后验v（因为它已被t周淘汰事件约束）
- 这模拟了"真正的预测"场景

【预期效果】
- Coverage_90: 98% → ~85-92% (更接近目标90%)
- Accuracy: 90% → ~35-50% (合理的预测难度)
"""

import numpy as np
import pandas as pd
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============== 模型超参数 ==============
MODEL_PARAMS = {
    'sigma_u': 0.1,      # 随机游走漂移
    'lambda_pl': 2.0,    # 淘汰harshness
    'kappa_J': 1.0,
    'kappa_F': 0.1,
    'kappa_C': 0.1,
}


def soft_rank_np(score_vec, kappa, mask):
    """Soft-rank函数"""
    s_row = score_vec.reshape(1, -1)
    s_col = score_vec.reshape(-1, 1)
    d = s_row - s_col
    d_clipped = np.clip(d / kappa, -20, 20)
    sig = 1.0 / (1.0 + np.exp(-d_clipped))
    valid_k = mask.reshape(1, -1)
    sig_masked = np.where(valid_k, sig, 0.0)
    r = 1.0 + np.sum(sig_masked, axis=1) - 0.5
    return np.where(mask, r, 0.0)


def predict_v_next_week(u_samples_prev, mask_next, sigma_u, n_monte_carlo=10):
    """
    【关键修正】使用t-1周的u后验预测t周的v分布
    
    u_t = u_{t-1} + eps, eps ~ N(0, sigma_u)
    v_t = softmax(u_t)
    """
    n_samples, n_pairs = u_samples_prev.shape
    
    # 生成随机游走噪声
    eps = np.random.randn(n_samples, n_monte_carlo, n_pairs) * sigma_u
    u_pred = u_samples_prev[:, np.newaxis, :] + eps
    u_pred_flat = u_pred.reshape(-1, n_pairs)
    
    # 应用mask并计算softmax
    huge_neg = -1e9
    u_masked = np.where(mask_next, u_pred_flat, huge_neg)
    u_max = np.max(u_masked, axis=1, keepdims=True)
    exp_u = np.exp(u_masked - u_max)
    exp_u = np.where(mask_next, exp_u, 0.0)
    v_pred = exp_u / (np.sum(exp_u, axis=1, keepdims=True) + 1e-10)
    
    return v_pred


def compute_temporal_predictive_metrics(
    u_samples, v_samples, S_mat, p_mat, mask_mat,
    elim_data, week_values, pair_ids,
    season, pan_s, rule_segment
):
    """
    【核心修正】计算时序预测指标
    
    关键区别：
    - 原代码：使用v_t的后验来预测t周淘汰 (循环验证!)
    - 修正后：使用u_{t-1}的后验 + 随机游走先验 (真正的预测!)
    """
    n_samples, n_weeks, n_pairs = v_samples.shape
    pid_map = {pid: i for i, pid in enumerate(pair_ids)}
    
    sigma_u = MODEL_PARAMS['sigma_u']
    lambda_pl = MODEL_PARAMS['lambda_pl']
    kappa_J = MODEL_PARAMS['kappa_J']
    kappa_F = MODEL_PARAMS['kappa_F']
    kappa_C = MODEL_PARAMS['kappa_C']
    
    metrics_rows = []
    
    for t in range(n_weeks):
        week_num = week_values[t]
        key = f"{season}_{week_num}"
        
        if key not in elim_data:
            continue
            
        edata = elim_data[key]
        elim_names = edata.get('eliminated_names', [])
        
        if not elim_names:
            continue
        
        current_week_df = pan_s[pan_s['week'] == week_num]
        name_to_idx = {}
        for _, r in current_week_df.iterrows():
            if r['pair_id'] in pid_map:
                name_to_idx[r['celebrity_name']] = pid_map[r['pair_id']]
        
        true_elim_indices = [name_to_idx[n] for n in elim_names if n in name_to_idx]
        
        if not true_elim_indices:
            continue
        
        # ========== 关键修正：时序预测 ==========
        if t == 0:
            # 第一周：无历史信息，使用均匀先验
            n_active = np.sum(mask_mat[t])
            v_pred = np.ones((n_samples * 10, n_pairs)) / n_active
            v_pred = np.where(mask_mat[t], v_pred, 0.0)
        else:
            # t > 0: 使用t-1周的u后验预测t周的v
            u_prev = u_samples[:, t-1, :]
            v_pred = predict_v_next_week(u_prev, mask_mat[t], sigma_u, n_monte_carlo=10)
        
        # 计算淘汰概率
        probs_sum = np.zeros(n_pairs)
        n_pred_samples = v_pred.shape[0]
        
        rJ = soft_rank_np(S_mat[t], kappa_J, mask_mat[t])
        
        for s_idx in range(n_pred_samples):
            v_t_s = v_pred[s_idx]
            
            if rule_segment == 'percent':
                comb = p_mat[t] + v_t_s
                b_val = soft_rank_np(comb, kappa_C, mask_mat[t])
            else:
                rF = soft_rank_np(v_t_s, kappa_F, mask_mat[t])
                b_val = rJ + rF
            
            logits = lambda_pl * b_val
            logits_masked = np.where(mask_mat[t], logits, -100)
            logits_shifted = logits_masked - np.max(logits_masked)
            exps = np.where(mask_mat[t], np.exp(logits_shifted), 0.0)
            sum_exps = np.sum(exps)
            
            probs = exps / sum_exps if sum_exps > 0 else np.zeros_like(exps)
            probs_sum += probs
        
        avg_probs = probs_sum / n_pred_samples
        
        # 计算指标
        active_indices = np.where(mask_mat[t])[0]
        active_probs = avg_probs[active_indices]
        
        local_true_elim = []
        for glob_idx in true_elim_indices:
            res = np.where(active_indices == glob_idx)[0]
            if len(res) > 0:
                local_true_elim.append(res[0])
        
        if not local_true_elim:
            continue
        
        # Coverage 90%
        sorted_indices = np.argsort(active_probs)[::-1]
        sorted_probs = active_probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, 0.9)
        cred_set = set(sorted_indices[:cutoff_idx+1])
        coverage_hit = any(idx in cred_set for idx in local_true_elim)
        
        # Accuracy
        top_1 = sorted_indices[0]
        accuracy = 1 if top_1 in local_true_elim else 0
        
        top_2 = set(sorted_indices[:2])
        top2_acc = 1 if any(t_idx in top_2 for t_idx in local_true_elim) else 0
        
        # Brier Score
        n_active = len(active_probs)
        y = np.zeros(n_active)
        for idx in local_true_elim:
            y[idx] = 1.0
        brier = np.sum((active_probs - y)**2) / n_active
        
        random_prob = 1.0 / n_active
        brier_random = (1 - random_prob)**2 + (n_active - 1) * random_prob**2
        
        entropy = -np.sum(active_probs * np.log(active_probs + 1e-10))
        
        metrics_rows.append({
            'season': season,
            'week': week_num,
            'week_idx': t,
            'n_active': n_active,
            'n_elim': len(local_true_elim),
            'coverage_90': int(coverage_hit),
            'accuracy': accuracy,
            'top2_acc': top2_acc,
            'brier': brier,
            'brier_random': brier_random,
            'cred_set_size': cutoff_idx + 1,
            'max_prob': np.max(active_probs),
            'entropy': entropy,
            'validation_type': 'temporal_predictive'
        })
    
    return metrics_rows


def compute_insample_metrics(
    v_samples, S_mat, p_mat, mask_mat,
    elim_data, week_values, pair_ids,
    season, pan_s, rule_segment
):
    """计算样本内拟合指标 (In-Sample, 仅供参考)"""
    n_samples, n_weeks, n_pairs = v_samples.shape
    pid_map = {pid: i for i, pid in enumerate(pair_ids)}
    
    lambda_pl = MODEL_PARAMS['lambda_pl']
    kappa_J = MODEL_PARAMS['kappa_J']
    kappa_F = MODEL_PARAMS['kappa_F']
    kappa_C = MODEL_PARAMS['kappa_C']
    
    metrics_rows = []
    
    for t in range(n_weeks):
        week_num = week_values[t]
        key = f"{season}_{week_num}"
        
        if key not in elim_data:
            continue
            
        edata = elim_data[key]
        elim_names = edata.get('eliminated_names', [])
        
        if not elim_names:
            continue
        
        current_week_df = pan_s[pan_s['week'] == week_num]
        name_to_idx = {}
        for _, r in current_week_df.iterrows():
            if r['pair_id'] in pid_map:
                name_to_idx[r['celebrity_name']] = pid_map[r['pair_id']]
        
        true_elim_indices = [name_to_idx[n] for n in elim_names if n in name_to_idx]
        
        if not true_elim_indices:
            continue
        
        probs_sum = np.zeros(n_pairs)
        rJ = soft_rank_np(S_mat[t], kappa_J, mask_mat[t])
        
        for s_idx in range(n_samples):
            v_t_s = v_samples[s_idx, t]
            
            if rule_segment == 'percent':
                comb = p_mat[t] + v_t_s
                b_val = soft_rank_np(comb, kappa_C, mask_mat[t])
            else:
                rF = soft_rank_np(v_t_s, kappa_F, mask_mat[t])
                b_val = rJ + rF
            
            logits = lambda_pl * b_val
            logits_masked = np.where(mask_mat[t], logits, -100)
            logits_shifted = logits_masked - np.max(logits_masked)
            exps = np.where(mask_mat[t], np.exp(logits_shifted), 0.0)
            sum_exps = np.sum(exps)
            
            probs = exps / sum_exps if sum_exps > 0 else np.zeros_like(exps)
            probs_sum += probs
        
        avg_probs = probs_sum / n_samples
        
        active_indices = np.where(mask_mat[t])[0]
        active_probs = avg_probs[active_indices]
        
        local_true_elim = []
        for glob_idx in true_elim_indices:
            res = np.where(active_indices == glob_idx)[0]
            if len(res) > 0:
                local_true_elim.append(res[0])
        
        if not local_true_elim:
            continue
        
        sorted_indices = np.argsort(active_probs)[::-1]
        sorted_probs = active_probs[sorted_indices]
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, 0.9)
        cred_set = set(sorted_indices[:cutoff_idx+1])
        coverage_hit = any(idx in cred_set for idx in local_true_elim)
        
        top_1 = sorted_indices[0]
        accuracy = 1 if top_1 in local_true_elim else 0
        
        n_active = len(active_probs)
        y = np.zeros(n_active)
        for idx in local_true_elim:
            y[idx] = 1.0
        brier = np.sum((active_probs - y)**2) / n_active
        
        metrics_rows.append({
            'season': season,
            'week': week_num,
            'coverage_90_insample': int(coverage_hit),
            'accuracy_insample': accuracy,
            'brier_insample': brier,
        })
    
    return metrics_rows


if __name__ == "__main__":
    PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
    ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
    SAMPLES_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
    VALIDATION_DIR = r'd:\shumomeisai\Code_second\Results\validation_results'
    
    if not os.path.exists(VALIDATION_DIR):
        os.makedirs(VALIDATION_DIR)
        
    print("Loading data...")
    if not os.path.exists(PANEL_PATH):
        print(f"Error: Panel data not found at {PANEL_PATH}")
        exit(1)
        
    panel_df = pd.read_parquet(PANEL_PATH)
    
    if not os.path.exists(ELIM_PATH):
        print(f"Error: Elim data not found at {ELIM_PATH}")
        exit(1)
        
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_data = json.load(f)
        
    npz_files = glob.glob(os.path.join(SAMPLES_DIR, "season_*.npz"))
    all_metrics = []
    
    print(f"Found {len(npz_files)} posterior files in {SAMPLES_DIR}.")
    
    if len(npz_files) == 0:
        print("No posterior samples found. Please run task1_runner.py first.")
        # Create dummy report to avoid empty file errors if this is a dry run
        with open(os.path.join(VALIDATION_DIR, 'validation_report.txt'), 'w') as f:
            f.write("TASK 1 VALIDATION REPORT (Fixed)\nNo samples found.\n")
        exit(0)
    
    for npz_path in sorted(npz_files):
        try:
            print(f"Loading {os.path.basename(npz_path)}...")
            data = np.load(npz_path, allow_pickle=True)
            if 'season' not in data:
                print(f"Skipping {npz_path}: 'season' key missing")
                continue
                
            season_scalar = data['season']
            # Handle 0-d array
            season = int(season_scalar) if np.ndim(season_scalar) == 0 else int(season_scalar[0])
            
            print(f"Validating Season {season}...")
            
            # Extract samples
            # Note: task1_runner saves dict keys as is from MCMC samples
            # task1_model defines 'u' and 'v'
            if 'u' not in data or 'v' not in data:
                print(f"Skipping Season {season}: 'u' or 'v' missing in npz")
                continue
                
            u_samples = data['u']
            v_samples = data['v']
            
            # Reconstruct Data Matrices from Panel
            df_s = panel_df[panel_df['season'] == season]
            if df_s.empty:
                print(f"Warning: No panel data for season {season}")
                continue
                
            # Alignment logic must match task1_runner exactly
            weeks = sorted(df_s['week'].unique())
            pair_ids = sorted(df_s['pair_id'].unique())
            
            pid_map = {p: i for i, p in enumerate(pair_ids)}
            week_map = {w: i for i, w in enumerate(weeks)}
            
            n_weeks = len(weeks)
            n_pairs = len(pair_ids)
            
            # Check dimensions
            if v_samples.shape[1] != n_weeks or v_samples.shape[2] != n_pairs:
                print(f"Dimension mismatch for S{season}: Sample T={v_samples.shape[1]}, Data T={n_weeks}")
                # Try to proceed if safe, or skip
                # Actually if mismatch, indexing will fail
                # Let's trust they match if generated correctly
                pass

            mask_mat = np.zeros((n_weeks, n_pairs), dtype=bool)
            S_mat = np.zeros((n_weeks, n_pairs))
            p_mat = np.zeros((n_weeks, n_pairs))
            
            for _, row in df_s.iterrows():
                t = week_map[row['week']]
                i = pid_map[row['pair_id']]
                mask_mat[t, i] = True
                S_mat[t, i] = row['S_it']
                p_mat[t, i] = row['pJ_it']
            
            # Rule segment
            if 'rule_segment' in df_s.columns:
                rule_segment = df_s['rule_segment'].iloc[0]
            else:
                rule_segment = 'percent' # default
            
            # Compute temporal predictive metrics
            rows = compute_temporal_predictive_metrics(
                u_samples, v_samples, S_mat, p_mat, mask_mat,
                elim_data, weeks, pair_ids,
                season, df_s, rule_segment
            )
            all_metrics.extend(rows)
            
        except Exception as e:
            print(f"Error validating season {season} from {npz_path}: {e}")
            # import traceback
            # traceback.print_exc()

    # Save outputs
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if all_metrics:
        res_df = pd.DataFrame(all_metrics)
        res_df.to_csv(os.path.join(VALIDATION_DIR, 'detailed_predictions.csv'), index=False)
        
        # Season Summary
        summary = res_df.groupby('season').agg({
            'accuracy': 'mean',
            'top2_acc': 'mean',
            'coverage_90': 'mean',
            'brier': 'mean',
            'n_elim': 'sum'
        }).reset_index()
        summary.to_csv(os.path.join(VALIDATION_DIR, 'season_summary.csv'), index=False)
        
        # Report
        mean_acc = res_df['accuracy'].mean()
        mean_top2 = res_df['top2_acc'].mean()
        mean_cov = res_df['coverage_90'].mean()
        mean_brier = res_df['brier'].mean()
        
        report_path = os.path.join(VALIDATION_DIR, 'validation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("======================================================================\n")
            f.write("TASK 1 MCMC VALIDATION REPORT (FIXED)\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("Method: Temporal Predictive Validation (t-1 posterior -> t prediction)\n")
            f.write("======================================================================\n\n")
            
            f.write("1. OVERALL SUMMARY\n")
            f.write("------------------\n")
            f.write(f"  Total Predictions:       {len(res_df)}\n")
            f.write(f"  Mean Accuracy (Top-1):   {mean_acc:.4f}\n")
            f.write(f"  Mean Top-2 Accuracy:     {mean_top2:.4f}\n")
            f.write(f"  Mean 90% Coverage:       {mean_cov:.4f}\n")
            f.write(f"  Mean Brier Score:        {mean_brier:.4f}\n\n")
            
            f.write("2. SEASON BY SEASON\n")
            f.write("-------------------\n")
            f.write(summary.to_string(index=False))
            f.write("\n")
            
        print(f"\nValidation Complete. Metrics saved to {VALIDATION_DIR}")
        print(f"Overall Accuracy: {mean_acc:.2%}")
        print(f"Overall Coverage: {mean_cov:.2%}")
        
    else:
        print("No metrics computed. Check input data or elim_events.")

