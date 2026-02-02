"""
Chart 1: The Verdict Matrix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
34 Seasons × 11 Weeks heatmap of every elimination prediction.
Green=Top-1 hit, Yellow=Top-2 hit, Red=Miss, Gray=No data.
Right margin: per-season accuracy bars.
Bottom margin: per-week accuracy bars.
Left color strip: voting regime indicator.

数据接口说明:
    将你的模型输出替换 ══ DATA INTERFACE ══ 区域中的 data 矩阵即可。
    data[s, w] 的编码:
        0 = 该赛季该周没有淘汰事件 (no data / season ended)
        1 = 模型 Top-1 预测命中 (最高淘汰概率的选手确实被淘汰)
        2 = 模型 Top-2 预测命中 (实际被淘汰者在前两高概率中)
        3 = 模型预测失误 (实际被淘汰者不在 Top-2 中)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

# ══════════════════════════════════════════════════════════════
# ══ DATA INTERFACE ── 替换此区域为你的真实模型输出 ══
# ══════════════════════════════════════════════════════════════

np.random.seed(42)

n_seasons = 34
max_weeks = 11

# 各赛季实际周数 (有淘汰事件的周数)
season_lengths = [6, 8, 9, 10, 10, 10, 10, 10, 10, 10,
                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                  11, 11, 11, 10, 10, 10, 10, 10, 10, 10,
                  10, 10, 10, 10]

# 投票制度: 'rank_early'=S1-2, 'percent'=S3-27, 'rank_save'=S28+
regime_map = {}
for s in range(n_seasons):
    if s <= 1:
        regime_map[s] = 'rank_early'
    elif s <= 26:
        regime_map[s] = 'percent'
    else:
        regime_map[s] = 'rank_save'

# ── 模拟数据 (替换为真实结果) ──
# 编码: 0=无数据, 1=Top1命中, 2=Top2命中, 3=Miss
data = np.zeros((n_seasons, max_weeks), dtype=int)

for s in range(n_seasons):
    for w in range(season_lengths[s]):
        r = np.random.random()
        if w >= season_lengths[s] - 1:
            # 决赛周: 稍难
            if r < 0.75: data[s, w] = 1
            elif r < 0.92: data[s, w] = 2
            else: data[s, w] = 3
        elif w <= 1:
            # 早期周: 稍难
            if r < 0.80: data[s, w] = 1
            elif r < 0.95: data[s, w] = 2
            else: data[s, w] = 3
        else:
            if r < 0.88: data[s, w] = 1
            elif r < 0.98: data[s, w] = 2
            else: data[s, w] = 3

# S26 特殊处理 (已知最难赛季)
for w in range(season_lengths[25]):
    r = np.random.random()
    if r < 0.45: data[25, w] = 1
    elif r < 0.72: data[25, w] = 2
    else: data[25, w] = 3

# ══════════════════════════════════════════════════════════════
# ══ END DATA INTERFACE ══
# ══════════════════════════════════════════════════════════════


# ── 计算边际准确率 ──
season_top1, season_top2 = [], []
for s in range(n_seasons):
    valid = data[s, data[s] > 0]
    if len(valid) > 0:
        season_top1.append(np.mean(valid == 1))
        season_top2.append(np.mean(valid <= 2))
    else:
        season_top1.append(0); season_top2.append(0)

week_top1, week_top2 = [], []
for w in range(max_weeks):
    valid = data[data[:, w] > 0, w]
    if len(valid) > 0:
        week_top1.append(np.mean(valid == 1))
        week_top2.append(np.mean(valid <= 2))
    else:
        week_top1.append(0); week_top2.append(0)

# ── 颜色方案 ──
cmap_colors = ['#F0F0F0', '#1B9E77', '#F4C430', '#E74C3C']
cmap = ListedColormap(cmap_colors)

regime_palette = {
    'rank_early': '#3498DB',
    'percent':    '#E67E22',
    'rank_save':  '#9B59B6',
}

# ── 构建图形 ──
fig = plt.figure(figsize=(16, 14), facecolor='#FAFAFA')
gs = fig.add_gridspec(2, 2, width_ratios=[11, 2.5], height_ratios=[34, 4],
                      hspace=0.08, wspace=0.08)

ax_main   = fig.add_subplot(gs[0, 0])
ax_right  = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, 0])
ax_corner = fig.add_subplot(gs[1, 1])

# ── 主热力图 ──
ax_main.imshow(data, cmap=cmap, vmin=0, vmax=3, aspect='auto', interpolation='nearest')

symbol_map = {1: ('✓', 'white'), 2: ('◐', '#333'), 3: ('✗', 'white'), 0: ('', '#CCC')}
for s in range(n_seasons):
    for w in range(max_weeks):
        sym, col = symbol_map[data[s, w]]
        if sym:
            ax_main.text(w, s, sym, ha='center', va='center',
                        fontsize=8, fontweight='bold', color=col)

ax_main.set_yticks(range(n_seasons))
ax_main.set_yticklabels([f'S{i+1}' for i in range(n_seasons)], fontsize=8, fontfamily='monospace')
ax_main.set_xticks(range(max_weeks))
ax_main.set_xticklabels([f'W{i+1}' for i in range(max_weeks)], fontsize=9, fontfamily='monospace')
ax_main.xaxis.set_ticks_position('top')
ax_main.xaxis.set_label_position('top')
ax_main.set_xlabel('Elimination Week', fontsize=11, fontweight='bold', labelpad=10)

# 左侧制度色条
for s in range(n_seasons):
    rc = regime_palette[regime_map[s]]
    ax_main.add_patch(plt.Rectangle((-0.5, s - 0.5), 0.15, 1,
                      facecolor=rc, edgecolor='none', clip_on=False, zorder=5))

# 网格线
for s in range(n_seasons + 1):
    ax_main.axhline(s - 0.5, color='white', linewidth=0.5)
for w in range(max_weeks + 1):
    ax_main.axvline(w - 0.5, color='white', linewidth=0.5)

# 高亮最难赛季 (S26)
hardest = 25
ax_main.add_patch(plt.Rectangle((-0.5, hardest - 0.5), max_weeks, 1,
                  facecolor='none', edgecolor='#E74C3C', linewidth=2.5, linestyle='--'))
ax_main.annotate('Hardest\nSeason', xy=(max_weeks - 0.5, hardest), xytext=(max_weeks + 0.3, hardest),
                fontsize=7, color='#E74C3C', fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2))

# ── 右侧面板: 各赛季准确率 ──
bar_y = np.arange(n_seasons)
ax_right.barh(bar_y, season_top2, height=0.8, color='#F4C430', alpha=0.5, label='Top-2')
ax_right.barh(bar_y, season_top1, height=0.8, color='#1B9E77', alpha=0.85, label='Top-1')
ax_right.set_xlim(0, 1.15)
ax_right.set_ylim(-0.5, n_seasons - 0.5)
ax_right.set_yticks([])
ax_right.set_xticks([0, 0.5, 1.0])
ax_right.set_xticklabels(['0%', '50%', '100%'], fontsize=8)
mean_t1 = np.mean(season_top1)
ax_right.axvline(mean_t1, color='#1B9E77', linestyle='--', linewidth=1.5, alpha=0.7)
ax_right.text(mean_t1 + 0.02, -2.2, f'μ={mean_t1:.1%}', fontsize=7, color='#1B9E77', fontweight='bold')
ax_right.set_xlabel('Season Accuracy', fontsize=9, fontweight='bold')
ax_right.invert_yaxis()
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)

# ── 底部面板: 各周准确率 ──
bar_x = np.arange(max_weeks)
ax_bottom.bar(bar_x, week_top2, width=0.8, color='#F4C430', alpha=0.5)
ax_bottom.bar(bar_x, week_top1, width=0.8, color='#1B9E77', alpha=0.85)
ax_bottom.set_ylim(0, 1.15)
ax_bottom.set_xlim(-0.5, max_weeks - 0.5)
ax_bottom.set_xticks(range(max_weeks))
ax_bottom.set_xticklabels([f'W{i+1}' for i in range(max_weeks)], fontsize=9, fontfamily='monospace')
ax_bottom.set_yticks([0, 0.5, 1.0])
ax_bottom.set_yticklabels(['0%', '50%', '100%'], fontsize=8)
ax_bottom.axhline(np.mean(week_top1), color='#1B9E77', linestyle='--', linewidth=1.5, alpha=0.7)
ax_bottom.set_ylabel('Week Accuracy', fontsize=9, fontweight='bold')
ax_bottom.spines['bottom'].set_visible(False)
ax_bottom.spines['right'].set_visible(False)

# ── 右下角: 图例 ──
ax_corner.axis('off')
legend_items = [
    ('✓', '#1B9E77', f'Top-1 Hit  ({mean_t1:.1%})'),
    ('◐', '#F4C430', f'Top-2 Hit  ({np.mean(season_top2):.1%})'),
    ('✗', '#E74C3C', f'Miss       ({1-np.mean(season_top2):.1%})'),
    ('',  '#F0F0F0', 'No Data'),
]
for i, (sym, col, label) in enumerate(legend_items):
    y = 0.85 - i * 0.22
    ax_corner.add_patch(plt.Rectangle((0.05, y - 0.08), 0.15, 0.16,
                       facecolor=col, edgecolor='#999', linewidth=0.5,
                       transform=ax_corner.transAxes))
    if sym:
        ax_corner.text(0.125, y, sym, ha='center', va='center', fontsize=10, fontweight='bold',
                      color='white' if col != '#F4C430' else '#333', transform=ax_corner.transAxes)
    ax_corner.text(0.28, y, label, ha='left', va='center', fontsize=8.5,
                  fontfamily='monospace', transform=ax_corner.transAxes)

regime_info = [('#3498DB', 'Rank (S1-2)'), ('#E67E22', 'Percent (S3-27)'), ('#9B59B6', 'Rank+Save (S28+)')]
for i, (col, label) in enumerate(regime_info):
    y = -0.05 - i * 0.2
    ax_corner.add_patch(plt.Rectangle((0.05, y - 0.06), 0.08, 0.12,
                       facecolor=col, edgecolor='none', transform=ax_corner.transAxes))
    ax_corner.text(0.18, y, label, ha='left', va='center', fontsize=7.5,
                  transform=ax_corner.transAxes, color='#555')

# ── 标题 ──
fig.suptitle('The Verdict Matrix: Every Elimination Prediction Across 34 Seasons',
            fontsize=16, fontweight='bold', y=0.97, color='#1a1a2e', fontfamily='serif')
fig.text(0.5, 0.945,
        'Each cell represents one elimination event · Model correctly identifies the eliminated contestant '
        f'{mean_t1:.1%} of the time (Top-1)',
        ha='center', fontsize=9.5, color='#666', style='italic')

plt.savefig('chart1_verdict_matrix.png', dpi=300, bbox_inches='tight',
           facecolor='#FAFAFA', edgecolor='none')
plt.close()
print("✅ Chart 1 saved: chart1_verdict_matrix.png")
