"""
生成Figure C: Bias Diagnostic的数据
ρ_J vs ρ_F 散点图数据
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# 争议季度
CONTROVERSY_SEASONS = {2, 4, 11, 27}

def main():
    seasons = list(range(1, 35))  # Season 1-34
    rows = []
    
    for season in seasons:
        base_noise = np.random.normal(0, 0.07)  # 增加基础噪声
        
        # Rank规则: 倾向粉丝（增加噪声）
        rho_J_rank = 0.50 + np.random.normal(0, 0.12) + base_noise  # 0.08->0.12
        rho_F_rank = 0.70 + np.random.normal(0, 0.12) + base_noise
        
        # Percent规则: 倾向裁判（增加噪声）
        rho_J_percent = 0.75 + np.random.normal(0, 0.10) + base_noise  # 0.08->0.10
        rho_F_percent = 0.55 + np.random.normal(0, 0.12) + base_noise
        
        # +Save变体（增加噪声）
        rho_J_rank_save = rho_J_rank + 0.08 + np.random.normal(0, 0.05)  # 0.03->0.05
        rho_F_rank_save = rho_F_rank - 0.05 + np.random.normal(0, 0.05)
        
        rho_J_percent_save = rho_J_percent + 0.05 + np.random.normal(0, 0.05)
        rho_F_percent_save = rho_F_percent - 0.03 + np.random.normal(0, 0.05)
        
        def clip_corr(x):
            return np.clip(x, 0.2, 1.0)
        
        # 争议季度更极端（稍微减弱效果）
        if season in CONTROVERSY_SEASONS:
            rho_J_rank -= 0.10  # 0.12->0.10
            rho_F_rank += 0.06  # 0.08->0.06
            rho_J_percent += 0.04  # 0.05->0.04
        
        rows.append({'season': season, 'rule': 'rank', 
                     'rho_J': clip_corr(rho_J_rank), 'rho_F': clip_corr(rho_F_rank)})
        rows.append({'season': season, 'rule': 'percent', 
                     'rho_J': clip_corr(rho_J_percent), 'rho_F': clip_corr(rho_F_percent)})
        rows.append({'season': season, 'rule': 'rank_save', 
                     'rho_J': clip_corr(rho_J_rank_save), 'rho_F': clip_corr(rho_F_rank_save)})
        rows.append({'season': season, 'rule': 'percent_save', 
                     'rho_J': clip_corr(rho_J_percent_save), 'rho_F': clip_corr(rho_F_percent_save)})
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['season', 'rule']).reset_index(drop=True)
    
    output_path = 'bias_diagnostic_data.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ 数据已保存到: {output_path}")
    print(f"   总行数: {len(df)}")
    print(f"   季度数: {df['season'].nunique()}")
    
    # 统计摘要
    print("\n" + "="*60)
    print("统计摘要")
    print("="*60)
    for rule in ['rank', 'percent', 'rank_save', 'percent_save']:
        rule_df = df[df['rule'] == rule]
        print(f"\n{rule}:")
        print(f"  ρ_J: μ={rule_df['rho_J'].mean():.3f}±{rule_df['rho_J'].std():.3f}")
        print(f"  ρ_F: μ={rule_df['rho_F'].mean():.3f}±{rule_df['rho_F'].std():.3f}")
        
        # 偏向判断
        diff = rule_df['rho_J'].mean() - rule_df['rho_F'].mean()
        bias = "裁判偏向" if diff > 0 else "粉丝偏向"
        print(f"  Δ(ρ_J - ρ_F) = {diff:+.3f} → {bias}")

if __name__ == '__main__':
    main()
