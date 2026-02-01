"""
直接使用controversy_portraits_data.csv绘制Figure B
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 配置
FOCAL_CONTESTANTS = {
    'Jerry Rice': {'season': 2, 'actual_placement': 2},
    'Billy Ray Cyrus': {'season': 4, 'actual_placement': 5},
    'Bristol Palin': {'season': 11, 'actual_placement': 3},
    'Bobby Bones': {'season': 27, 'actual_placement': 1},
}

RULES_CONFIG = {
    'rank': {'label': 'Rank', 'color': '#2166AC', 'hatch': None},
    'percent': {'label': 'Percent', 'color': '#B2182B', 'hatch': None},
    'rank_save': {'label': 'Rank + Save', 'color': '#4393C3', 'hatch': '//'},
    'percent_save': {'label': 'Percent + Save', 'color': '#D6604D', 'hatch': '//'},
}

RULE_ORDER = ['rank', 'percent', 'rank_save', 'percent_save']

# 加载数据
df = pd.read_csv('controversy_portraits_data.csv')

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

contestants = list(FOCAL_CONTESTANTS.keys())

for idx, name in enumerate(contestants):
    ax = axes[idx]
    contestant_data = df[df['contestant'] == name]
    
    # 准备数据
    plot_data = []
    positions = []
    colors = []
    
    for rule_idx, rule in enumerate(RULE_ORDER):
        rule_data = contestant_data[contestant_data['rule'] == rule]
        if len(rule_data) > 0:
            placements = rule_data['posterior_placement'].values
            plot_data.append(placements)
            positions.append(rule_idx)
            colors.append(RULES_CONFIG[rule]['color'])
    
    # 创建violin plot
    parts = ax.violinplot(plot_data, positions=positions, 
                         showmeans=True, showmedians=False, showextrema=False)
    
    # 自定义颜色
    for pc_idx, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[pc_idx])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        pc.set_linewidth(1.5)
        
        # 添加hatch
        rule = RULE_ORDER[positions[pc_idx]]
        if RULES_CONFIG[rule]['hatch']:
            pc.set_hatch(RULES_CONFIG[rule]['hatch'])
    
    # 风格化mean线
    parts['cmeans'].set_color('white')
    parts['cmeans'].set_linewidth(2)
    
    # 添加实际名次线
    actual = FOCAL_CONTESTANTS[name]['actual_placement']
    ax.axhline(y=actual, color='#2ca02c', linestyle='--', linewidth=2.5, 
              label=f'Actual Placement ({actual})')
    
    # 添加统计标注
    for rule_idx, rule in enumerate(RULE_ORDER):
        rule_data = contestant_data[contestant_data['rule'] == rule]
        if len(rule_data) > 0:
            placements = rule_data['posterior_placement'].values
            mean_p = np.mean(placements)
            p_win = np.mean(placements == 1)
            p_top3 = np.mean(placements <= 3)
            
            ann_text = f'μ={mean_p:.1f}\nP(Win)={p_win:.0%}\nP(Top3)={p_top3:.0%}'
            
            y_pos = max(placements) + 0.8
            ax.annotate(ann_text, xy=(rule_idx, y_pos), 
                       ha='center', va='bottom', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='gray', alpha=0.9))
    
    # 样式
    n_pairs = contestant_data['season'].iloc[0]  # 简化处理
    if name == 'Jerry Rice':
        n_pairs = 9
    elif name == 'Billy Ray Cyrus':
        n_pairs = 9
    elif name == 'Bristol Palin':
        n_pairs = 12
    else:  # Bobby Bones
        n_pairs = 11
    
    ax.set_ylim(0.5, n_pairs + 2)
    ax.set_xlim(-0.6, len(RULE_ORDER) - 0.4)
    ax.set_xticks(range(len(RULE_ORDER)))
    ax.set_xticklabels([RULES_CONFIG[r]['label'] for r in RULE_ORDER], fontsize=10)
    ax.set_ylabel('Final Placement (1 = Champion)', fontsize=11)
    ax.invert_yaxis()  # 1在顶部
    
    season = FOCAL_CONTESTANTS[name]['season']
    ax.set_title(f'{name} (Season {season})', fontsize=13, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=9)

# 全局图例
legend_elements = [
    mpatches.Patch(facecolor=RULES_CONFIG['rank']['color'], label='Rank', alpha=0.7, edgecolor='black'),
    mpatches.Patch(facecolor=RULES_CONFIG['percent']['color'], label='Percent', alpha=0.7, edgecolor='black'),
    mpatches.Patch(facecolor=RULES_CONFIG['rank_save']['color'], label='Rank + Save', 
                  alpha=0.7, edgecolor='black', hatch='//'),
    mpatches.Patch(facecolor=RULES_CONFIG['percent_save']['color'], label='Percent + Save', 
                  alpha=0.7, edgecolor='black', hatch='//'),
    Line2D([0], [0], color='#2ca02c', linestyle='--', linewidth=2.5, label='Actual Result'),
]

fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
          fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.98))

fig.suptitle('Controversy Portraits — Posterior Placement Distributions', 
            fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# 保存
output_path = 'fig_b_ideal.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ 图表已保存到: {output_path}")

plt.close()
