"""
Task 1 Metrics - 修正版
=======================
核心修改：实现时序预测验证 (Temporal Predictive Validation)

修正循环验证问题：
- 原方法：使用v_t后验预测t周淘汰（循环验证）
- 新方法：使用u_{t-1}后验 + 随机游走预测t周淘汰（时序预测）
"""

import numpy as np
import pandas as pd
import json
import os
import glob
from scipy.stats import entropy

# ============== 路径配置 ==============
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
SAMPLES_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
OUTPUT_FILE = r'd:\shumomeisai\Code_second\Results\task1_metrics_fixed.csv'

# ============== 模型超参数 ==============
MODEL_PARAMS = {
    'sigma_u': 0.1,      # 随机游走漂移 (需与task1_model.py一致)
    'lambda_pl': 2.0,    # 淘汰harshness (降低以增加不确定性)
    'kappa_J': 1.0,      # Judge score soft-rank scale
    'kappa_F': 0.1,      # Fan vote soft-rank scale  
    'kappa_C': 0.1,      # Combined percent soft-rank scale
}


def soft_rank_np(score_vec, kappa, mask):
    """Soft-rank函数 (NumPy版本)"""
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
    
    原理：
    u_t = u_{t-1} + eps, eps ~ N(0, sigma_u)
    v_t = softmax(u_t)
    
    Parameters:
    -----------
    u_samples_prev : (n_samples, n_pairs) - t-1周的u后验样本
    mask_next : (n_pairs,) - t周的活跃mask
    sigma_u : float - 随机游走标准差
    n_monte_carlo : int - 每个后验样本的蒙特卡洛采样数
    
    Returns:
    --------
    v_pred : (n_samples * n_monte_carlo, n_pairs) - 预测的v分布样本
    """
    n_samples, n_pairs = u_samples_prev.shape
    
    # 生成随机游走噪声
    eps = np.random.randn(n_samples, n_monte_carlo, n_pairs) * sigma_u
    u_pred = u_samples_prev[:, np.newaxis, :] + eps
    u_pred_flat = u_pred.reshape(-1, n_pairs)
    
    # Softmax (只在活跃选手上)
    huge_neg = -1e9
    u_masked = np.where(mask_next, u_pred_flat, huge_neg)
    u_max = np.max(u_masked, axis=1, keepdims=True)
    exp_u = np.exp(u_masked - u_max)
    exp_u = np.where(mask_next, exp_u, 0.0)
    v_pred = exp_u / (np.sum(exp_u, axis=1, keepdims=True) + 1e-10)
    
    return v_pred


def compute_row_metrics(probs, true_elim_indices):
    """计算单行指标"""
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)
    
    cutoff_idx_90 = np.searchsorted(cumsum, 0.9)
    cred_set_90 = sorted_indices[:cutoff_idx_90+1]
    hit_cov_90 = any(idx in cred_set_90 for idx in true_elim_indices)
    
    cutoff_idx_50 = np.searchsorted(cumsum, 0.5)
    cred_set_50 = sorted_indices[:cutoff_idx_50+1]
    hit_cov_50 = any(idx in cred_set_50 for idx in true_elim_indices)
    
    top_1 = sorted_indices[0]
    acc = 1 if top_1 in true_elim_indices else 0
    
    top_2 = sorted_indices[:2]
    top2_hit = 1 if any(t in top_2 for t in true_elim_indices) else 0
    
    N = len(probs)
    y = np.zeros(N)
    for idx in true_elim_indices:
        y[idx] = 1.0
    brier = np.sum((probs - y)**2)
    
    return hit_cov_90, hit_cov_50, acc, top2_hit, brier, N, len(cred_set_90), len(cred_set_50)


def compute_ci_width(v_samples):
    """计算95%置信区间宽度"""
    low = np.percentile(v_samples, 2.5, axis=0)
    high = np.percentile(v_samples, 97.5, axis=0)
    width = high - low
    return np.mean(width)


def compute_metrics_temporal(samples_dir=SAMPLES_DIR):
    """
    【主函数】计算时序预测指标
    
    同时输出：
    1. 时序预测指标 (temporal) - 用于评估真正的预测能力
    2. 样本内拟合指标 (insample) - 用于评估模型拟合质量
    """
    # Load Elim Events
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_data = json.load(f)
    
    # Load Panel
    panel = pd.read_parquet(PANEL_PATH)
    
    files = glob.glob(os.path.join(samples_dir, "season_*.npz"))
    print(f"Found {len(files)} sample files in {samples_dir}")
    
    metrics_rows = []
    
    sigma_u = MODEL_PARAMS['sigma_u']
    lambda_pl = MODEL_PARAMS['lambda_pl']
    kappa_J = MODEL_PARAMS['kappa_J']
    kappa_F = MODEL_PARAMS['kappa_F']
    kappa_C = MODEL_PARAMS['kappa_C']
    
    for fpath in sorted(files):
        data = np.load(fpath, allow_pickle=True)
        season = int(data['season'])
        pair_ids = data['pair_ids']
        week_values = data['week_values']
        
        v_samples = data['v']
        u_samples = data['u']
        
        n_samples, n_weeks, n_pairs = v_samples.shape
        
        # 重建数据矩阵
        pan_s = panel[panel['season'] == season]
        
        S_mat = np.zeros((n_weeks, n_pairs))
        p_mat = np.zeros((n_weeks, n_pairs))
        mask_mat = np.zeros((n_weeks, n_pairs), dtype=bool)
        
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
        
        mean_post_var = np.mean(np.var(v_samples, axis=0))
        
        # 遍历每周的淘汰事件
        for t in range(n_weeks):
            week_num = week_values[t]
            key = f"{season}_{week_num}"
            
            if key not in elim_data:
                continue
            
            edata = elim_data[key]
            elim_names = edata.get('eliminated_names', [])
            
            if not elim_names:
                continue
            
            # 名字到索引的映射
            current_week_df = pan_s[pan_s['week'] == week_num]
            name_to_idx = {}
            for _, r in current_week_df.iterrows():
                if r['pair_id'] in pid_map:
                    name_to_idx[r['celebrity_name']] = pid_map[r['pair_id']]
            
            true_elim_indices = [name_to_idx[n] for n in elim_names if n in name_to_idx]
            
            if not true_elim_indices:
                continue
            
            # ========== 1. 时序预测 ==========
            if t == 0:
                # 第一周：均匀先验
                n_active = np.sum(mask_mat[t])
                v_pred = np.ones((n_samples * 10, n_pairs)) / n_active
                v_pred = np.where(mask_mat[t], v_pred, 0.0)
            else:
                # t > 0: 使用t-1周的u后验
                u_prev = u_samples[:, t-1, :]
                v_pred = predict_v_next_week(u_prev, mask_mat[t], sigma_u, n_monte_carlo=10)
            
            # 计算时序预测的淘汰概率
            probs_temporal = np.zeros(n_pairs)
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
                
                if sum_exps > 0:
                    probs = exps / sum_exps
                else:
                    probs = np.zeros_like(exps)
                
                probs_temporal += probs
            
            probs_temporal /= n_pred_samples
            
            # ========== 2. 样本内拟合 ==========
            probs_insample = np.zeros(n_pairs)
            
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
                
                if sum_exps > 0:
                    probs = exps / sum_exps
                else:
                    probs = np.zeros_like(exps)
                
                probs_insample += probs
            
            probs_insample /= n_samples
            
            # ========== 计算指标 ==========
            active_indices = np.where(mask_mat[t])[0]
            n_active = len(active_indices)
            
            # 映射到局部索引
            local_true_elim = []
            for glob_idx in true_elim_indices:
                res = np.where(active_indices == glob_idx)[0]
                if len(res) > 0:
                    local_true_elim.append(res[0])
            
            if not local_true_elim:
                continue
            
            # 时序预测指标
            active_probs_temp = probs_temporal[active_indices]
            cov90_temp, cov50_temp, acc_temp, top2_temp, brier_temp, _, cz90_temp, cz50_temp = compute_row_metrics(
                active_probs_temp, local_true_elim)
            
            # 样本内指标
            active_probs_ins = probs_insample[active_indices]
            cov90_ins, cov50_ins, acc_ins, top2_ins, brier_ins, _, cz90_ins, cz50_ins = compute_row_metrics(
                active_probs_ins, local_true_elim)
            
            # CI宽度
            v_active = v_samples[:, t, :][:, active_indices]
            ci_width = compute_ci_width(v_active)
            
            # 熵
            entropy_temp = -np.sum(active_probs_temp * np.log(active_probs_temp + 1e-10))
            
            # Brier Random Correction
            random_prob = 1.0 / n_active
            brier_random_sum = (1 - random_prob)**2 + (n_active - 1) * random_prob**2
            brier_random = brier_random_sum # Validation script uses sum for brier_random? 
            # Wait, in validation script I normalized both. 
            # In compute_row_metrics (line 104), brier is SUM.
            # So here brier_random should also be SUM to be comparable?
            # User said: "brier... unit inconsistent... brier is average version (divide by n_active) in CURRENT CODE?"
            # Let's check compute_row_metrics: "brier = np.sum((probs - y)**2)" -> This is SUM.
            # User said "your code ... brier is average version". Is it?
            # Ah, maybe they were referring to 'task1_validation.py' where I divided by n_active.
            # In task1_metrics.py, compute_row_metrics returns SUM.
            # So I should keep brier_random as SUM.
            # OR, I should normalize BOTH.
            # User recommended: "Plan A: Divide brier_random by n_active (to match mean version)"
            # BUT my compute_row_metrics returns sum.
            # I should probably normalize my brier in compute_row_metrics to be mean as well?
            # Or just normalize both here.
            # Let's normalize BOTH here for the report to be consistent with validation script.
            
            brier_temp_norm = brier_temp / n_active
            brier_ins_norm = brier_ins / n_active
            brier_random_norm = brier_random_sum / n_active

            metrics_rows.append({
                'season': season,
                'week': week_num,
                'n_active': n_active,
                'n_elim': len(local_true_elim),
                
                # 时序预测指标 (主要报告)
                'coverage_90_temporal': int(cov90_temp),
                'coverage_50_temporal': int(cov50_temp),
                'accuracy_temporal': acc_temp,
                'top2_acc_temporal': top2_temp,
                'brier_temporal': brier_temp_norm, # Normalized
                'brier_random': brier_random_norm,
                'entropy_temporal': entropy_temp,
                'max_prob_temporal': np.max(active_probs_temp),
                'cred_set_size_90_temporal': cz90_temp,
                'cred_set_size_50_temporal': cz50_temp,
                'avg_credset90_ratio_temporal': cz90_temp / n_active,
                'avg_credset50_ratio_temporal': cz50_temp / n_active,
                
                # 样本内指标 (参考)
                'coverage_90_insample': int(cov90_ins),
                'coverage_50_insample': int(cov50_ins),
                'accuracy_insample': acc_ins,
                'brier_insample': brier_ins_norm, # Normalized
                
                # 后验不确定性
                'post_var_v_mean': mean_post_var,
                'avg_ci_width': ci_width,
            })
        
        print(f"Season {season}: processed {sum(1 for r in metrics_rows if r['season']==season)} events")
    
    # 保存结果
    if metrics_rows:
        df_out = pd.DataFrame(metrics_rows)
        df_out.to_csv(OUTPUT_FILE, index=False)
        print(f"\nMetrics saved to {OUTPUT_FILE}")
        
        # 打印汇总
        print("\n" + "="*60)
        print("SUMMARY: Temporal Predictive vs In-Sample")
        print("="*60)
        print(f"{'Metric':<25} {'Temporal':>12} {'In-Sample':>12}")
        print("-"*60)
        print(f"{'Mean Accuracy':<25} {df_out['accuracy_temporal'].mean():>12.3f} {df_out['accuracy_insample'].mean():>12.3f}")
        print(f"{'Mean Coverage_90':<25} {df_out['coverage_90_temporal'].mean():>12.3f} {df_out['coverage_90_insample'].mean():>12.3f}")
        print(f"{'Mean Coverage_50':<25} {df_out['coverage_50_temporal'].mean():>12.3f} {df_out['coverage_50_insample'].mean():>12.3f}")
        print(f"{'Mean Brier':<25} {df_out['brier_temporal'].mean():>12.3f} {df_out['brier_insample'].mean():>12.3f}")
        print(f"{'Mean Brier Random':<25} {df_out['brier_random'].mean():>12.3f} {'-':>12}")
        print("="*60)
        print("\n解读:")
        print("- Temporal指标反映真正的预测能力 (预期Acc~35-50%, Cov90~80-90%)")
        print("- In-Sample指标反映模型拟合质量 (预期Acc~85-95%)")
        print("- Brier Temporal < Brier Random? " + ("YES" if df_out['brier_temporal'].mean() < df_out['brier_random'].mean() else "NO"))
        
    else:
        print("No metrics calculated.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=SAMPLES_DIR, 
                        help='Directory containing .npz samples')
    args = parser.parse_args()
    
    compute_metrics_temporal(args.dir)
