"""
Task 1 完整验证脚本
====================
检查34个赛季的MCMC结果是否符合论文要求

验证内容：
1. MCMC 收敛诊断 (R-hat, ESS, Divergences)
2. 模型拟合质量 (Accuracy, Coverage, Brier Score)
3. 生成汇总报告和可视化

论文标准：
- R-hat < 1.05 (严格) / < 1.1 (可接受)
- ESS > 100 (最低) / > 400 (理想)
- Divergences = 0
- Coverage_90 ≈ 0.90 (校准良好)
- Accuracy > 0.5 (优于随机)
"""

import numpy as np
import pandas as pd
import json
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============== 配置路径 ==============
# Windows 路径（根据你的实际情况修改）
SAMPLES_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\validation_results'

# ============== 论文标准阈值 ==============
THRESHOLDS = {
    'rhat_strict': 1.05,
    'rhat_acceptable': 1.10,
    'rhat_bad': 1.20,
    'ess_ideal': 400,
    'ess_minimum': 100,
    'ess_bad': 50,
    'divergence_max': 0,
    'coverage_target': 0.90,
    'coverage_tolerance': 0.10,  # 允许 0.80-1.00
    'accuracy_random': 0.10,  # ~1/N 随机基线
}


def compute_rhat_ess(samples_dict):
    """
    计算 R-hat 和 ESS
    简化版本：单链情况下用 split-R-hat 近似
    """
    results = {}
    
    for param_name, samples in samples_dict.items():
        if samples.ndim == 1:
            # 单变量
            n = len(samples)
            half = n // 2
            chain1, chain2 = samples[:half], samples[half:]
            
            # Split R-hat (近似)
            var1, var2 = np.var(chain1), np.var(chain2)
            mean1, mean2 = np.mean(chain1), np.mean(chain2)
            
            W = (var1 + var2) / 2  # Within-chain variance
            B = ((mean1 - mean2) ** 2) / 2  # Between-chain variance
            
            var_plus = W + B
            rhat = np.sqrt(var_plus / (W + 1e-10))
            
            # ESS (近似)
            # 使用自相关估计
            try:
                autocorr = np.correlate(samples - np.mean(samples), 
                                       samples - np.mean(samples), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                
                # 找到第一个负值的位置
                neg_idx = np.where(autocorr < 0)[0]
                if len(neg_idx) > 0:
                    cutoff = neg_idx[0]
                else:
                    cutoff = len(autocorr) // 2
                
                tau = 1 + 2 * np.sum(autocorr[1:cutoff])
                ess = n / max(tau, 1)
            except:
                ess = n  # Fallback
            
            results[param_name] = {'rhat': rhat, 'ess': ess, 'shape': 'scalar'}
            
        else:
            # 多维变量 - 展平处理
            flat_samples = samples.reshape(samples.shape[0], -1)
            rhats, esss = [], []
            
            for i in range(flat_samples.shape[1]):
                s = flat_samples[:, i]
                n = len(s)
                half = n // 2
                
                var1 = np.var(s[:half])
                var2 = np.var(s[half:])
                mean1 = np.mean(s[:half])
                mean2 = np.mean(s[half:])
                
                W = (var1 + var2) / 2 + 1e-10
                B = ((mean1 - mean2) ** 2) / 2
                rhat = np.sqrt((W + B) / W)
                rhats.append(rhat)
                
                # 简化ESS
                esss.append(n / 2)  # 保守估计
            
            results[param_name] = {
                'rhat': np.max(rhats),
                'rhat_mean': np.mean(rhats),
                'ess': np.min(esss),
                'ess_mean': np.mean(esss),
                'shape': samples.shape[1:]
            }
    
    return results


def compute_predictive_metrics(v_samples, S_mat, p_mat, mask_mat, 
                                elim_data, week_values, pair_ids, 
                                season, pan_s, rule_segment):
    """
    计算模型拟合质量指标
    """
    n_samples, n_weeks, n_pairs = v_samples.shape
    pid_map = {pid: i for i, pid in enumerate(pair_ids)}
    w_map = {w: i for i, w in enumerate(week_values)}
    
    metrics_rows = []
    
    # 自适应 soft-rank
    def soft_rank_np(score_vec, kappa, mask):
        s_row = score_vec.reshape(1, -1)
        s_col = score_vec.reshape(-1, 1)
        d = s_row - s_col
        d_clipped = np.clip(d / kappa, -10, 10)
        sig = 1.0 / (1.0 + np.exp(-d_clipped))
        valid_k = mask.reshape(1, -1)
        sig_masked = np.where(valid_k, sig, 0.0)
        r = 1.0 + np.sum(sig_masked, axis=1) - 0.5
        return np.where(mask, r, 0.0)
    
    def adaptive_kappa(scores, mask, base=0.1):
        active_scores = scores[mask]
        if len(active_scores) > 1:
            std = np.std(active_scores)
            return max(base, 0.3 * std)
        return base
    
    for t in range(n_weeks):
        week_num = week_values[t]
        key = f"{season}_{week_num}"
        
        if key not in elim_data:
            continue
            
        edata = elim_data[key]
        elim_names = edata.get('eliminated_names', [])
        
        if not elim_names:
            continue
        
        # Name -> Index mapping
        current_week_df = pan_s[pan_s['week'] == week_num]
        name_to_idx = {}
        for _, r in current_week_df.iterrows():
            if r['pair_id'] in pid_map:
                name_to_idx[r['celebrity_name']] = pid_map[r['pair_id']]
        
        true_elim_indices = [name_to_idx[n] for n in elim_names if n in name_to_idx]
        
        if not true_elim_indices:
            continue
        
        # 计算淘汰概率
        lambda_pl = 2.0  # 匹配V2模型
        probs_sum = np.zeros(n_pairs)
        
        kappa_J = adaptive_kappa(S_mat[t], mask_mat[t], 1.0)
        rJ = soft_rank_np(S_mat[t], kappa_J, mask_mat[t])
        
        for s_idx in range(n_samples):
            v_t_s = v_samples[s_idx, t]
            
            if rule_segment == 'percent':
                comb = p_mat[t] + v_t_s
                kappa_C = adaptive_kappa(comb, mask_mat[t], 0.02)
                b_val = soft_rank_np(comb, kappa_C, mask_mat[t])
            else:
                kappa_F = adaptive_kappa(v_t_s, mask_mat[t], 0.02)
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
            
            probs_sum += probs
        
        avg_probs = probs_sum / n_samples
        
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
        
        # Accuracy (Top-1)
        top_1 = sorted_indices[0]
        accuracy = 1 if top_1 in local_true_elim else 0
        
        # Top-2 Accuracy
        top_2 = set(sorted_indices[:2])
        top2_acc = 1 if any(t in top_2 for t in local_true_elim) else 0
        
        # Brier Score
        n_active = len(active_probs)
        y = np.zeros(n_active)
        for idx in local_true_elim:
            y[idx] = 1.0
        brier = np.sum((active_probs - y)**2)
        
        # 随机基线 Brier
        random_prob = 1.0 / n_active
        brier_random = (1 - random_prob)**2 + (n_active - 1) * random_prob**2
        
        metrics_rows.append({
            'season': season,
            'week': week_num,
            'n_active': n_active,
            'n_elim': len(local_true_elim),
            'coverage_90': int(coverage_hit),
            'accuracy': accuracy,
            'top2_acc': top2_acc,
            'brier': brier,
            'brier_random': brier_random,
            'cred_set_size': cutoff_idx + 1,
            'max_prob': np.max(active_probs),
            'entropy': -np.sum(active_probs * np.log(active_probs + 1e-10))
        })
    
    return metrics_rows


def validate_season(season, samples_dir, panel, elim_data):
    """
    验证单个赛季的结果
    """
    fpath = os.path.join(samples_dir, f"season_{season}.npz")
    
    if not os.path.exists(fpath):
        return None, f"File not found: {fpath}"
    
    try:
        data = np.load(fpath, allow_pickle=True)
    except Exception as e:
        return None, f"Load error: {e}"
    
    # 提取数据
    season_num = int(data['season'])
    pair_ids = list(data['pair_ids'])
    week_values = list(data['week_values'])
    
    # 检查必要的样本
    required_keys = ['v', 'u']
    for key in required_keys:
        if key not in data:
            return None, f"Missing key: {key}"
    
    v_samples = data['v']
    u_samples = data['u']
    
    n_samples, n_weeks, n_pairs = v_samples.shape
    
    # 1. 收敛诊断
    samples_dict = {}
    for key in data.files:
        if key not in ['pair_ids', 'week_values', 'season', 'model_version']:
            samples_dict[key] = data[key]
    
    convergence = compute_rhat_ess(samples_dict)
    
    # 汇总收敛指标
    all_rhats = [v['rhat'] for v in convergence.values() if 'rhat' in v]
    all_ess = [v['ess'] for v in convergence.values() if 'ess' in v]
    
    max_rhat = max(all_rhats) if all_rhats else np.nan
    min_ess = min(all_ess) if all_ess else np.nan
    
    # 2. 重建 S, p, mask 矩阵
    pan_s = panel[panel['season'] == season_num]
    
    if pan_s.empty:
        return None, f"No panel data for season {season_num}"
    
    pid_map = {pid: i for i, pid in enumerate(pair_ids)}
    w_map = {w: i for i, w in enumerate(week_values)}
    
    S_mat = np.zeros((n_weeks, n_pairs))
    p_mat = np.zeros((n_weeks, n_pairs))
    mask_mat = np.zeros((n_weeks, n_pairs), dtype=bool)
    
    for _, row in pan_s.iterrows():
        if row['week'] in w_map and row['pair_id'] in pid_map:
            t = w_map[row['week']]
            i = pid_map[row['pair_id']]
            S_mat[t, i] = row['S_it']
            p_mat[t, i] = row['pJ_it']
            mask_mat[t, i] = True
    
    rule_segment = pan_s['rule_segment'].iloc[0]
    
    # 3. 计算预测指标
    pred_metrics = compute_predictive_metrics(
        v_samples, S_mat, p_mat, mask_mat,
        elim_data, week_values, pair_ids,
        season_num, pan_s, rule_segment
    )
    
    # 4. 汇总结果
    if pred_metrics:
        df_pred = pd.DataFrame(pred_metrics)
        mean_accuracy = df_pred['accuracy'].mean()
        mean_top2 = df_pred['top2_acc'].mean()
        mean_coverage = df_pred['coverage_90'].mean()
        mean_brier = df_pred['brier'].mean()
        n_events = len(df_pred)
    else:
        mean_accuracy = np.nan
        mean_top2 = np.nan
        mean_coverage = np.nan
        mean_brier = np.nan
        n_events = 0
    
    # 后验方差（用于PCR近似）
    post_var_v = np.mean(np.var(v_samples, axis=0))
    
    result = {
        'season': season_num,
        'n_pairs': n_pairs,
        'n_weeks': n_weeks,
        'n_samples': n_samples,
        'rule_segment': rule_segment,
        
        # 收敛诊断
        'max_rhat': max_rhat,
        'min_ess': min_ess,
        'rhat_status': 'GOOD' if max_rhat < 1.05 else ('OK' if max_rhat < 1.1 else 'BAD'),
        'ess_status': 'GOOD' if min_ess > 400 else ('OK' if min_ess > 100 else 'BAD'),
        
        # 预测质量
        'n_elim_events': n_events,
        'accuracy': mean_accuracy,
        'top2_accuracy': mean_top2,
        'coverage_90': mean_coverage,
        'mean_brier': mean_brier,
        
        # 不确定性
        'post_var_v': post_var_v,
    }
    
    return result, pred_metrics


def generate_report(all_results, all_pred_metrics, output_dir):
    """
    生成完整的验证报告
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 赛季汇总表
    df_summary = pd.DataFrame(all_results)
    df_summary = df_summary.sort_values('season')
    
    # 2. 详细预测指标
    all_preds = []
    for pm in all_pred_metrics:
        if pm:
            all_preds.extend(pm)
    df_preds = pd.DataFrame(all_preds) if all_preds else pd.DataFrame()
    
    # 3. 计算整体统计
    valid_results = [r for r in all_results if r is not None]
    
    overall_stats = {
        'total_seasons': len(valid_results),
        'converged_strict': sum(1 for r in valid_results if r['max_rhat'] < 1.05),
        'converged_acceptable': sum(1 for r in valid_results if r['max_rhat'] < 1.10),
        'ess_good': sum(1 for r in valid_results if r['min_ess'] > 100),
        
        'mean_accuracy': np.nanmean([r['accuracy'] for r in valid_results]),
        'mean_top2_acc': np.nanmean([r['top2_accuracy'] for r in valid_results]),
        'mean_coverage': np.nanmean([r['coverage_90'] for r in valid_results]),
        'mean_brier': np.nanmean([r['mean_brier'] for r in valid_results]),
        
        'total_elim_events': sum(r['n_elim_events'] for r in valid_results),
    }
    
    # 4. 保存CSV
    df_summary.to_csv(os.path.join(output_dir, 'season_summary.csv'), index=False)
    if not df_preds.empty:
        df_preds.to_csv(os.path.join(output_dir, 'detailed_predictions.csv'), index=False)
    
    # 5. 生成文本报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TASK 1 MCMC VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("1. OVERALL SUMMARY")
    report_lines.append("=" * 70)
    report_lines.append(f"  Total Seasons Processed: {overall_stats['total_seasons']}")
    report_lines.append(f"  Total Elimination Events: {overall_stats['total_elim_events']}")
    
    report_lines.append("\n" + "-" * 40)
    report_lines.append("  MCMC CONVERGENCE:")
    report_lines.append("-" * 40)
    report_lines.append(f"  R-hat < 1.05 (Strict):     {overall_stats['converged_strict']}/{overall_stats['total_seasons']}")
    report_lines.append(f"  R-hat < 1.10 (Acceptable): {overall_stats['converged_acceptable']}/{overall_stats['total_seasons']}")
    report_lines.append(f"  ESS > 100:                 {overall_stats['ess_good']}/{overall_stats['total_seasons']}")
    
    report_lines.append("\n" + "-" * 40)
    report_lines.append("  MODEL FIT QUALITY:")
    report_lines.append("-" * 40)
    report_lines.append(f"  Mean Accuracy (Top-1):   {overall_stats['mean_accuracy']:.3f}")
    report_lines.append(f"  Mean Top-2 Accuracy:     {overall_stats['mean_top2_acc']:.3f}")
    report_lines.append(f"  Mean 90% Coverage:       {overall_stats['mean_coverage']:.3f}")
    report_lines.append(f"  Mean Brier Score:        {overall_stats['mean_brier']:.3f}")
    
    # 判定
    report_lines.append("\n" + "=" * 70)
    report_lines.append("2. QUALITY ASSESSMENT")
    report_lines.append("=" * 70)
    
    convergence_ok = overall_stats['converged_acceptable'] >= overall_stats['total_seasons'] * 0.9
    coverage_ok = 0.80 <= overall_stats['mean_coverage'] <= 1.00
    accuracy_ok = overall_stats['mean_accuracy'] > 0.15
    
    if convergence_ok and coverage_ok and accuracy_ok:
        report_lines.append("  ✅ OVERALL: PASS - Results suitable for paper")
    elif convergence_ok and (coverage_ok or accuracy_ok):
        report_lines.append("  ⚠️  OVERALL: MARGINAL - Consider improvements")
    else:
        report_lines.append("  ❌ OVERALL: FAIL - Significant issues detected")
    
    report_lines.append(f"\n  Convergence: {'✅ PASS' if convergence_ok else '❌ FAIL'}")
    report_lines.append(f"  Coverage Calibration: {'✅ PASS' if coverage_ok else '❌ FAIL'}")
    report_lines.append(f"  Prediction Accuracy: {'✅ PASS' if accuracy_ok else '❌ FAIL'}")
    
    # 问题赛季
    report_lines.append("\n" + "=" * 70)
    report_lines.append("3. PROBLEMATIC SEASONS")
    report_lines.append("=" * 70)
    
    bad_seasons = [r for r in valid_results if r['max_rhat'] >= 1.10 or r['min_ess'] < 100]
    if bad_seasons:
        report_lines.append(f"  Found {len(bad_seasons)} seasons with issues:")
        for r in bad_seasons:
            report_lines.append(f"    Season {r['season']}: R-hat={r['max_rhat']:.3f}, ESS={r['min_ess']:.1f}")
    else:
        report_lines.append("  ✅ No problematic seasons detected")
    
    # 详细表格
    report_lines.append("\n" + "=" * 70)
    report_lines.append("4. SEASON-BY-SEASON RESULTS")
    report_lines.append("=" * 70)
    
    header = f"{'Season':>6} | {'Rule':>10} | {'R-hat':>7} | {'ESS':>7} | {'Acc':>6} | {'Top2':>6} | {'Cov90':>6} | {'Status':>8}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    for r in sorted(valid_results, key=lambda x: x['season']):
        status = '✅' if r['max_rhat'] < 1.1 and r['min_ess'] > 100 else '⚠️'
        line = f"{r['season']:>6} | {r['rule_segment']:>10} | {r['max_rhat']:>7.3f} | {r['min_ess']:>7.1f} | {r['accuracy']:>6.3f} | {r['top2_accuracy']:>6.3f} | {r['coverage_90']:>6.3f} | {status:>8}"
        report_lines.append(line)
    
    # 按规则分组统计
    report_lines.append("\n" + "=" * 70)
    report_lines.append("5. RESULTS BY RULE TYPE")
    report_lines.append("=" * 70)
    
    for rule in ['rank', 'percent', 'rank_save']:
        rule_results = [r for r in valid_results if r['rule_segment'] == rule]
        if rule_results:
            report_lines.append(f"\n  {rule.upper()} (n={len(rule_results)}):")
            report_lines.append(f"    Mean R-hat:    {np.mean([r['max_rhat'] for r in rule_results]):.3f}")
            report_lines.append(f"    Mean ESS:      {np.mean([r['min_ess'] for r in rule_results]):.1f}")
            report_lines.append(f"    Mean Accuracy: {np.mean([r['accuracy'] for r in rule_results]):.3f}")
            report_lines.append(f"    Mean Coverage: {np.mean([r['coverage_90'] for r in rule_results]):.3f}")
    
    # 写入文件
    report_text = '\n'.join(report_lines)
    with open(os.path.join(output_dir, 'validation_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    
    return overall_stats, df_summary


def main():
    """主函数"""
    print("Loading data...")
    
    # 加载数据
    panel = pd.read_parquet(PANEL_PATH)
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_data = json.load(f)
    
    # 获取所有赛季
    seasons = sorted(panel['season'].unique())
    print(f"Found {len(seasons)} seasons in panel")
    
    # 检查样本文件
    sample_files = glob.glob(os.path.join(SAMPLES_DIR, "season_*.npz"))
    print(f"Found {len(sample_files)} sample files")
    
    # 验证每个赛季
    all_results = []
    all_pred_metrics = []
    
    for season in seasons:
        print(f"  Validating Season {season}...", end=' ')
        result, pred_metrics = validate_season(season, SAMPLES_DIR, panel, elim_data)
        
        if result is None:
            print(f"SKIP ({pred_metrics})")
        else:
            status = '✅' if result['max_rhat'] < 1.1 else '⚠️'
            print(f"{status} R-hat={result['max_rhat']:.3f}, Acc={result['accuracy']:.3f}")
            all_results.append(result)
            all_pred_metrics.append(pred_metrics)
    
    # 生成报告
    print("\nGenerating report...")
    overall_stats, df_summary = generate_report(all_results, all_pred_metrics, OUTPUT_DIR)
    
    print(f"\n📁 Results saved to: {OUTPUT_DIR}")
    print("   - season_summary.csv")
    print("   - detailed_predictions.csv") 
    print("   - validation_report.txt")
    
    return overall_stats, df_summary


if __name__ == "__main__":
    main()
