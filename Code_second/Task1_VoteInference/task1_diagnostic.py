"""
Task 1 诊断脚本：检查模型是否存在过拟合/循环验证问题
======================================================

问题背景：
- Coverage = 100% 太完美
- Accuracy = 92.6% 不合理的高
- 需要验证模型是否真的在"预测"还是在"记忆"
"""

import numpy as np
import pandas as pd
import os
import glob

SAMPLES_DIR = r'd:\shumomeisai\Code_second\posterior_samples'
PANEL_PATH = r'd:\shumomeisai\Code_second\panel.parquet'

def diagnose_samples(samples_dir):
    """诊断后验样本的质量"""
    
    print("=" * 70)
    print("TASK 1 DIAGNOSTIC REPORT")
    print("=" * 70)
    
    files = glob.glob(os.path.join(samples_dir, "season_*.npz"))
    print(f"\nFound {len(files)} sample files\n")
    
    all_issues = []
    
    for fpath in sorted(files):
        data = np.load(fpath, allow_pickle=True)
        season = int(data['season'])
        
        # 检查 v 样本
        if 'v' not in data:
            print(f"Season {season}: Missing 'v' samples!")
            continue
            
        v = data['v']  # (n_samples, n_weeks, n_pairs)
        n_samples, n_weeks, n_pairs = v.shape
        
        issues = []
        
        # 诊断1：v 的方差是否为0（完全确定）
        v_var = np.var(v, axis=0)  # (n_weeks, n_pairs)
        zero_var_count = np.sum(v_var < 1e-10)
        if zero_var_count > 0:
            issues.append(f"⚠️ {zero_var_count} elements with zero variance")
        
        # 诊断2：v 的样本是否几乎相同
        v_range = np.max(v, axis=0) - np.min(v, axis=0)
        tiny_range = np.sum(v_range < 0.01)
        if tiny_range > n_weeks * n_pairs * 0.5:
            issues.append(f"⚠️ {tiny_range}/{n_weeks*n_pairs} elements with tiny range")
        
        # 诊断3：检查样本之间的相关性
        # 如果样本高度相关，ESS 应该很低，但报告显示 ESS=1000
        v_flat = v.reshape(n_samples, -1)
        
        # 计算连续样本的自相关
        autocorr_lag1 = []
        for i in range(v_flat.shape[1]):
            if np.std(v_flat[:, i]) > 1e-10:
                corr = np.corrcoef(v_flat[:-1, i], v_flat[1:, i])[0, 1]
                if not np.isnan(corr):
                    autocorr_lag1.append(corr)
        
        if autocorr_lag1:
            mean_autocorr = np.mean(autocorr_lag1)
            if mean_autocorr > 0.5:
                issues.append(f"⚠️ High autocorrelation: {mean_autocorr:.3f}")
            elif mean_autocorr < 0.01:
                issues.append(f"⚠️ Suspiciously low autocorr: {mean_autocorr:.3f} (independent samples?)")
        
        # 诊断4：v 的分布是否合理
        # 理论上 v 应该是 simplex 上的分布 (sum=1, all >= 0)
        v_sums = np.sum(v, axis=-1)  # (n_samples, n_weeks)
        sum_check = np.allclose(v_sums, 1.0, atol=0.01)
        if not sum_check:
            issues.append(f"⚠️ v doesn't sum to 1")
        
        # 诊断5：检查是否所有样本都相同（完全退化）
        unique_samples = len(np.unique(v.round(6).reshape(n_samples, -1), axis=0))
        if unique_samples < n_samples * 0.5:
            issues.append(f"🔴 Only {unique_samples}/{n_samples} unique samples (degenerate!)")
        
        # 诊断6：v 的熵
        # 低熵意味着高确定性
        mean_entropy = []
        for t in range(n_weeks):
            for s_idx in range(n_samples):
                v_t = v[s_idx, t, :]
                v_t = v_t[v_t > 1e-10]  # Remove zeros
                if len(v_t) > 0:
                    ent = -np.sum(v_t * np.log(v_t + 1e-10))
                    mean_entropy.append(ent)
        
        avg_entropy = np.mean(mean_entropy) if mean_entropy else 0
        max_possible_entropy = np.log(n_pairs)  # Uniform distribution
        entropy_ratio = avg_entropy / max_possible_entropy if max_possible_entropy > 0 else 0
        
        if entropy_ratio < 0.3:
            issues.append(f"⚠️ Low entropy ratio: {entropy_ratio:.3f} (very concentrated)")
        
        # 打印结果
        status = "✅" if not issues else "⚠️"
        print(f"Season {season:2d}: {status} samples={n_samples}, weeks={n_weeks}, pairs={n_pairs}")
        print(f"          v_var_mean={np.mean(v_var):.6f}, entropy_ratio={entropy_ratio:.3f}")
        
        if issues:
            for issue in issues:
                print(f"          {issue}")
            all_issues.extend([(season, i) for i in issues])
        
        print()
    
    # 总结
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if all_issues:
        print(f"\n🔴 Found {len(all_issues)} potential issues across all seasons")
        print("\nPossible causes:")
        print("  1. lambda_pl too high → model overfits to elimination events")
        print("  2. Circular validation → using training data for testing")
        print("  3. Prior too weak → posterior collapses to MLE")
        print("\nRecommendations:")
        print("  1. Reduce lambda_pl (try 1.0 or even 0.5)")
        print("  2. Implement leave-one-out cross-validation")
        print("  3. Check if v posterior has meaningful uncertainty")
    else:
        print("\n✅ No obvious issues detected")
    
    return all_issues


def check_validation_logic():
    """
    检查验证逻辑是否存在循环验证问题
    
    关键问题：
    - 模型训练时使用淘汰事件作为 likelihood
    - 验证时检查模型是否能"预测"这些淘汰事件
    - 这本质上是在问"模型能否复现它学到的东西"
    - 这不是真正的预测验证！
    """
    
    print("\n" + "=" * 70)
    print("VALIDATION LOGIC CHECK")
    print("=" * 70)
    
    print("""
⚠️ CRITICAL INSIGHT:

Your current validation is essentially asking:
  "Can the model reproduce the elimination events it was trained on?"
  
This is NOT a proper validation because:
  1. Elimination events are part of the LIKELIHOOD (training signal)
  2. The posterior v is CONSTRAINED to make these events likely
  3. Testing on the same events = circular validation

PROPER VALIDATION OPTIONS:

Option A: Leave-One-Out Cross-Validation (LOSO)
  - For each elimination event (s,t):
    1. Fit model WITHOUT this event's constraint
    2. Predict who would be eliminated
    3. Check if prediction matches reality
  - This tests TRUE predictive ability

Option B: Temporal Cross-Validation
  - Train on weeks 1..t-1
  - Predict elimination at week t
  - This tests forward prediction

Option C: Season-Level Cross-Validation
  - Train on 33 seasons
  - Test on held-out season
  - Requires assuming similar dynamics across seasons

WHAT YOUR 92.6% ACCURACY REALLY MEANS:
  - It shows the model CAN FIT the elimination constraints
  - It does NOT show the model can PREDICT eliminations
  - The "uncertainty" is likely underestimated
    """)


def suggest_fixes():
    """建议修复方案"""
    
    print("\n" + "=" * 70)
    print("RECOMMENDED FIXES")
    print("=" * 70)
    
    print("""
1. REDUCE LAMBDA_PL (Highest Priority)
   Current: lambda_pl = 2.0 (or higher)
   Try: lambda_pl = 0.5 or even 0.1
   
   Why: Lower lambda = softer likelihood = more uncertainty
   Effect: Coverage should drop from 100% to ~90%
           Accuracy should drop to more realistic 40-60%

2. IMPLEMENT PROPER CROSS-VALIDATION
   Instead of: P(elim | full posterior)
   Use: P(elim | posterior without this week's constraint)
   
   Code change needed in task1_metrics.py

3. REPORT POSTERIOR UNCERTAINTY HONESTLY
   Currently: Reporting point prediction accuracy
   Should: Report credible intervals and their interpretation
   
   Key message: "Given elimination outcomes, we infer vote shares
                with uncertainty quantified by posterior variance"

4. ADD SENSITIVITY ANALYSIS
   - How does accuracy change with lambda?
   - How does coverage change with prior variance?
   - This shows model is not overfit to specific hyperparameters

5. USE HELD-OUT VALIDATION FOR AT LEAST SOME SEASONS
   - Hold out seasons 28-34 (rank_save rule)
   - Train hyperparameters on seasons 1-27
   - Test on held-out seasons
    """)


if __name__ == "__main__":
    diagnose_samples(SAMPLES_DIR)
    check_validation_logic()
    suggest_fixes()
