"""
使用CSV数据绘制Figure A热力图
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects

# 配置
CONTROVERSY_EVENTS = {
    (2, 5), (2, 6), (2, 7),
    (4, 3), (4, 4), (4, 5), (4, 6),
    (11, 3), (11, 5), (11, 7), (11, 9),
    (27, 4), (27, 6), (27, 8),
}

CONTROVERSY_SEASONS = {
    2: 'Jerry Rice (S2)',
    4: 'Billy Ray Cyrus (S4)',
    11: 'Bristol Palin (S11)',
    27: 'Bobby Bones (S27)',
}

RULE_SEGMENTS = {
    **{s: 'rank' for s in range(1, 3)},
    **{s: 'percent' for s in range(3, 28)},
    **{s: 'rank_save' for s in range(28, 35)},
}

RULE_COLORS = {
    'rank': '#4A90D9',
    'percent': '#F5A623',
    'rank_save': '#7B68EE',
}

def main():
    # 读取CSV
    df = pd.read_csv('reversal_heatmap_data.csv')
    print(f"✅ 加载数据: {len(df)} 行")
    
    # 构建矩阵
    seasons = sorted(df['season'].unique())
    max_week = int(df['week'].max())
    
    matrix = np.full((len(seasons), max_week), np.nan)
    s2r = {s: i for i, s in enumerate(seasons)}
    
    for _, row in df.iterrows():
        matrix[s2r[int(row['season'])], int(row['week']) - 1] = row['p_elim_diff']
    
    n_seasons = len(seasons)
    
    # 颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'reversal',
        ['#2166AC', '#67A9CF', '#D1E5F0', '#FDDBC7', '#EF8A62', '#B2182B'],
        N=256
    )
    cmap.set_bad(color='#F0F0F0')
    
    # 布局
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.6, max_week, 3.2, 0.8], wspace=0.05)
    
    ax_rule = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_anno = fig.add_subplot(gs[0, 2])
    ax_cb = fig.add_subplot(gs[0, 3])
    
    # 主热力图
    im = ax_main.imshow(matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                        interpolation='nearest', origin='upper')
    
    ax_main.set_xticks(range(max_week))
    ax_main.set_xticklabels(range(1, max_week + 1), fontsize=8)
    ax_main.set_xlabel('Week', fontsize=13, fontweight='bold', labelpad=8)
    
    ax_main.set_yticks(range(n_seasons))
    ax_main.set_yticklabels([f'S{s}' for s in seasons], fontsize=8)
    ax_main.set_ylabel('Season', fontsize=13, fontweight='bold', labelpad=8)
    
    # 争议事件星标
    for (s, w) in CONTROVERSY_EVENTS:
        if s in seasons:
            row = seasons.index(s)
            col = w - 1
            if 0 <= col < max_week:
                ax_main.text(col, row, '\u2605', ha='center', va='center',
                            fontsize=9, color='white', fontweight='bold',
                            path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')])
    
    # 网格
    ax_main.set_xticks(np.arange(-0.5, max_week, 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, n_seasons, 1), minor=True)
    ax_main.grid(which='minor', color='white', linewidth=0.3, alpha=0.5)
    ax_main.tick_params(which='minor', size=0)
    
    # 左侧规则时代条
    ax_rule.set_xlim(0, 1)
    ax_rule.set_ylim(-0.5, n_seasons - 0.5)
    ax_rule.invert_yaxis()
    
    for i, s in enumerate(seasons):
        rule = RULE_SEGMENTS.get(s, 'percent')
        ax_rule.barh(i, 1, height=1, color=RULE_COLORS.get(rule, '#CCC'),
                    edgecolor='white', linewidth=0.5)
    
    ax_rule.set_yticks([])
    ax_rule.set_xticks([])
    ax_rule.set_xlabel('Rule\nEra', fontsize=10, fontweight='bold', labelpad=6)
    
    # 右侧注释
    ax_anno.set_xlim(0, 1)
    ax_anno.set_ylim(-0.5, n_seasons - 0.5)
    ax_anno.invert_yaxis()
    ax_anno.axis('off')
    
    for s, name in CONTROVERSY_SEASONS.items():
        if s in seasons:
            row = seasons.index(s)
            ax_anno.annotate(f'  {name}', xy=(0.02, row), fontsize=9.5,
                           fontweight='bold', color='#B2182B', va='center')
    
    # 颜色条
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('P(Elimination Differs | Rank vs Percent)',
                fontsize=11, fontweight='bold', labelpad=12)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # 标题
    title = 'Rule Impact Heatmap\nRank vs. Percent Weekly Elimination Reversal Rate'
    sub = 'Redder = higher posterior probability of different elimination outcomes      \u2605 = known controversy event'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    ax_main.set_title(sub, fontsize=9.5, color='#555555', pad=14)
    
    # 图例
    patches = [
        mpatches.Patch(color=RULE_COLORS['rank'], label='Rank (S1-S2)'),
        mpatches.Patch(color=RULE_COLORS['percent'], label='Percent (S3-S27)'),
        mpatches.Patch(color=RULE_COLORS['rank_save'], label="Rank + Judges' Save (S28-S34)"),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
              frameon=True, edgecolor='#CCC', bbox_to_anchor=(0.45, 0.01))
    
    plt.subplots_adjust(top=0.89, bottom=0.09, left=0.06, right=0.95)
    
    # 保存
    fig.savefig('figa_from_csv.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ 已保存: figa_from_csv.png")
    plt.close()

if __name__ == '__main__':
    main()
