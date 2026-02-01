"""
检查Bobby Bones (Season 27)的原始数据
验证他是否真的是"争议性冠军"
"""

import pandas as pd
import numpy as np

# 加载panel数据
panel_path = r'd:\shumomeisai\Code_second\processed\panel.csv'
df = pd.read_csv(panel_path)

# 筛选Season 27
s27 = df[df['season'] == 27].copy()

# 找到Bobby Bones
bobby = s27[s27['celebrity_name'].str.contains('Bobby', case=False, na=False)]

print("="*70)
print("Bobby Bones (Season 27) 原始数据检查")
print("="*70)

if len(bobby) == 0:
    print("⚠️ 警告：没有找到Bobby Bones的数据！")
    print("\nSeason 27所有选手：")
    print(s27['celebrity_name'].unique())
else:
    print(f"\n找到 {len(bobby)} 条记录\n")
    
    # 显示关键信息
    bobby_sorted = bobby.sort_values('week')
    
    print("📊 每周表现：")
    print("-"*70)
    for idx, row in bobby_sorted.iterrows():
        week = row['week']
        score = row.get('S_it', 'N/A')
        eliminated = row.get('is_active', True) == False  # is_active=False means eliminated
        placement = row.get('placement', 'N/A')
        
        status = "❌ 淘汰" if eliminated else "✅ 存活"
        print(f"Week {week:2d}: 分数={score:5}, 名次={placement:3}, {status}")
    
    # 统计信息
    print("\n📈 统计摘要：")
    print("-"*70)
    if 'S_it' in bobby.columns:
        avg_score = bobby['S_it'].mean()
        min_score = bobby['S_it'].min()
        max_score = bobby['S_it'].max()
        print(f"平均分数: {avg_score:.2f}")
        print(f"最低分数: {min_score:.2f}")
        print(f"最高分数: {max_score:.2f}")
    
    if 'placement' in bobby.columns:
        final_placement = bobby_sorted.iloc[-1]['placement']
        print(f"\n最终名次: {final_placement}")
    
    # 对比Season 27其他选手的平均分
    print("\n📊 与其他选手对比：")
    print("-"*70)
    
    # 计算每个选手的平均分
    avg_scores = s27.groupby('celebrity_name')['S_it'].mean().sort_values(ascending=False)
    print("\nSeason 27选手平均分排名：")
    for rank, (name, score) in enumerate(avg_scores.items(), 1):
        marker = "👑" if 'Bobby' in name else "  "
        print(f"{marker} {rank:2d}. {name:30s}: {score:.2f}")
    
    # 检查幸存周数
    print("\n⏱️ 幸存周数对比：")
    survival = s27.groupby('celebrity_name')['week'].max().sort_values(ascending=False)
    for rank, (name, weeks) in enumerate(survival.items(), 1):
        marker = "👑" if 'Bobby' in name else "  "
        print(f"{marker} {rank:2d}. {name:30s}: {weeks} 周")

print("\n" + "="*70)
print("诊断结论")
print("="*70)
print("""
如果Bobby Bones：
- ✅ 最终名次 = 1（冠军）
- ❌ 平均分在倒数
- ✅ 幸存到最后

→ 这说明他是**典型的粉丝力量型选手**
→ 模型应该推断出他有很高的粉丝支持度
→ 如果模型推断他倒数第一，说明**模型有问题**
""")
