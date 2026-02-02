"""
诊断脚本：对比真实数据 vs Demo数据的统计特征
找出为什么真实结果不如Demo"dramatic"
"""

import pandas as pd
import numpy as np
import os

RESULTS_DIR = r'd:\shumomeisai\Code_second\Results\task3_analysis'

def diagnose():
    print("=" * 70)
    print("Task 3 数据诊断报告")
    print("=" * 70)
    
    # 1. Partner Effects (Fan Model)
    print("\n【1】舞伴随机效应 (Fan Model)")
    print("-" * 70)
    
    f_partner = os.path.join(RESULTS_DIR, 'task3_lmm_fan_partner_effects_aggregated.csv')
    df_partner = pd.read_csv(f_partner, index_col=0)
    
    print(f"总舞伴数: {len(df_partner)}")
    print(f"效应平均值: {df_partner['mean'].mean():.4f}")
    print(f"效应标准差: {df_partner['mean'].std():.4f}")
    print(f"效应范围: [{df_partner['mean'].min():.3f}, {df_partner['mean'].max():.3f}]")
    print(f"平均CI宽度: {(df_partner['97.5%'] - df_partner['2.5%']).mean():.4f}")
    
    # Demo对比
    print("\n对比 Demo数据 (人工设计):")
    print("  效应范围: [-0.5, +1.5] (夸张3倍)")
    print("  平均CI宽度: ~0.6 (真实数据的60%)")
    
    # 显著性分析
    sig_pos = (df_partner['2.5%'] > 0).sum()
    sig_neg = (df_partner['97.5%'] < 0).sum()
    print(f"\n显著正效应: {sig_pos}/{len(df_partner)} ({sig_pos/len(df_partner)*100:.1f}%)")
    print(f"显著负效应: {sig_neg}/{len(df_partner)} ({sig_neg/len(df_partner)*100:.1f}%)")
    
    print("\nTop 5 最强效应舞伴:")
    top5 = df_partner.nlargest(5, 'mean')
    for idx, row in top5.iterrows():
        sig = "***" if row['2.5%'] > 0 else ""
        print(f"  {idx:30s}: {row['mean']:+.3f} [{row['2.5%']:+.3f}, {row['97.5%']:+.3f}] {sig}")
    
    # 2. SHAP Feature Importance
    print("\n【2】SHAP 特征重要性 (Fan Model)")
    print("-" * 70)
    
    f_shap = os.path.join(RESULTS_DIR, 'task3_shap_ci_fan.csv')
    df_shap = pd.read_csv(f_shap)
    
    print(f"特征数量: {len(df_shap)}")
    print(f"平均重要性: {df_shap['mean_shap'].mean():.4f}")
    print(f"最大重要性: {df_shap['mean_shap'].max():.4f}")
    print(f"重要性比 (Max/Mean): {df_shap['mean_shap'].max() / df_shap['mean_shap'].mean():.2f}x")
    
    print("\nDemo对比:")
    print("  重要性比: ~5x (特征差异更明显)")
    
    print("\nTop 5 重要特征:")
    for _, row in df_shap.nlargest(5, 'mean_shap').iterrows():
        print(f"  {row['feature']:30s}: {row['mean_shap']:.4f}")
    
    # 3. 系数显著性
    print("\n【3】固定效应系数显著性 (Fan Model)")
    print("-" * 70)
    
    f_coeff = os.path.join(RESULTS_DIR, 'task3_lmm_fan_coeffs_aggregated.csv')
    df_coeff = pd.read_csv(f_coeff, index_col=0)
    
    # 排除Intercept和方差项
    df_coeff_clean = df_coeff[~df_coeff.index.str.contains('Intercept|Var', na=False)]
    
    sig_coeffs = ((df_coeff_clean['2.5%'] > 0) | (df_coeff_clean['97.5%'] < 0)).sum()
    print(f"显著系数: {sig_coeffs}/{len(df_coeff_clean)} ({sig_coeffs/len(df_coeff_clean)*100:.1f}%)")
    
    # 4. 问题诊断
    print("\n" + "=" * 70)
    print("【诊断结论】")
    print("=" * 70)
    
    problems = []
    
    # 检查效应大小
    if df_partner['mean'].std() < 0.15:
        problems.append("⚠️  舞伴效应值太小 (std < 0.15)，视觉对比不明显")
    
    # 检查显著性
    sig_rate = (sig_pos + sig_neg) / len(df_partner)
    if sig_rate < 0.3:
        problems.append(f"⚠️  显著舞伴太少 ({sig_rate*100:.0f}%)，缺少'故事性'")
    
    # 检查CI宽度
    avg_ci = (df_partner['97.5%'] - df_partner['2.5%']).mean()
    if avg_ci > 0.3:
        problems.append(f"⚠️  置信区间太宽 (avg={avg_ci:.2f})，不确定性过高")
    
    # 检查SHAP区分度
    shap_ratio = df_shap['mean_shap'].max() / df_shap['mean_shap'].mean()
    if shap_ratio < 2.5:
        problems.append(f"⚠️  特征重要性差异小 (ratio={shap_ratio:.1f}x < 2.5x)")
    
    if problems:
        print("\n发现的问题:")
        for p in problems:
            print(p)
    else:
        print("\n✓ 数据质量良好，无明显问题")
    
    print("\n" + "=" * 70)
    print("【优化建议】")
    print("=" * 70)
    print("""
1. 【筛选式展示】(推荐)
   - 只显示 |效应| > 0.1 或显著的舞伴 (减少噪音)
   - 只显示 Top K 重要特征 (聚焦重点)
   
2. 【标准化处理】
   - 将效应值标准化到 [-1, +1] 范围 (增强视觉对比)
   - 注意：需在图表中标注"Standardized Effect"
   
3. 【分层展示】
   - 将舞伴分为"明星舞伴"和"普通舞伴"两组
   - 突出显示经验 >= 5季的舞伴
   
4. 【可视化技巧】
   - 使用对数尺度（如果效应跨度大）
   - 调整颜色映射的中心点和饱和度
   - 增加显著性标记 (星号、粗体等)
   
5. 【不推荐】
   - ❌ 直接修改数值 (学术不端)
   - ❌ 删除"不好看"的数据点 (选择性报告)
    """)

if __name__ == "__main__":
    diagnose()
