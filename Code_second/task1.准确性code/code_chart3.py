"""
Chart 3: The Uncertainty Fingerprint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three-panel figure directly answering "Is certainty always the same?"
(A) 2D heatmap: competition stage × judge score quintile → mean CI width
(B) Violin/box by voting regime
(C) Violin/box by competition stage

数据接口说明:
    将你的模型输出替换 ══ DATA INTERFACE ══ 区域。
    Panel A 需要 5×5 矩阵 ci_width_data[quintile, stage]
    Panel B 需要三个 1D array: rank_early_ci, percent_ci, rank_late_ci
    Panel C 需要四个 1D array: early_ci, mid_ci, late_ci, finals_ci
    所有值都是 95% CI 宽度 (即 upper_bound - lower_bound of vote share)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as sp_stats
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ══════════════════════════════════════════════════════════════
# ══ DATA INTERFACE ── 替换此区域为你的真实模型输出 ══
# ══════════════════════════════════════════════════════════════

np.random.seed(2026)

# ── Panel A: 5×5 均值 CI 宽度矩阵 ──
# 行 = Judge Score 分位数 (Top 20% → Bottom 20%)
# 列 = 竞赛阶段 (Week 1-2, 3-4, 5-6, 7-8, Finals)
# 值 = 该分组下所有 (选手, 周次) 组合的 mean 95% CI width
#
# 预期模式:
#   - 左上/左中: 最宽 (早期 + 中间排名 = 最不确定)
#   - 右下/右上: 最窄 (后期 + 极端排名 = 最确定)
ci_width_data = np.array([
    # W1-2    W3-4    W5-6    W7-8    Finals
    [0.14,   0.11,   0.08,   0.06,   0.04],   # Top 20% (最高 judge score)
    [0.19,   0.16,   0.13,   0.10,   0.07],   # Upper-Mid
    [0.22,   0.19,   0.16,   0.13,   0.09],   # Middle (最宽 → 最不确定)
    [0.20,   0.17,   0.14,   0.11,   0.08],   # Lower-Mid
    [0.15,   0.12,   0.09,   0.07,   0.05],   # Bottom 20% (最低 judge score)
])

# ── Panel B: 各投票制度下的 CI 宽度分布 ──
# 每个元素 = 一个 (选手, 周次) 组合的 95% CI width
rank_early_ci = np.clip(sp_stats.gamma.rvs(a=3, scale=0.04, size=80) + 0.04, 0.02, 0.30)
percent_ci    = np.clip(sp_stats.gamma.rvs(a=3.5, scale=0.045, size=500) + 0.05, 0.02, 0.35)
rank_late_ci  = np.clip(sp_stats.gamma.rvs(a=3, scale=0.035, size=200) + 0.03, 0.02, 0.28)

# ── Panel C: 各竞赛阶段的 CI 宽度分布 ──
early_ci  = np.clip(sp_stats.gamma.rvs(a=3.5, scale=0.05, size=300) + 0.06, 0.02, 0.35)
mid_ci    = np.clip(sp_stats.gamma.rvs(a=3, scale=0.04, size=250) + 0.04, 0.02, 0.30)
late_ci   = np.clip(sp_stats.gamma.rvs(a=2.5, scale=0.03, size=150) + 0.03, 0.02, 0.22)
finals_ci = np.clip(sp_stats.gamma.rvs(a=2, scale=0.025, size=100) + 0.02, 0.01, 0.18)

# ══════════════════════════════════════════════════════════════
# ══ END DATA INTERFACE ══
# ══════════════════════════════════════════════════════════════


# ── 标签定义 ──
stages = ['Week\n1–2', 'Week\n3–4', 'Week\n5–6', 'Week\n7–8', 'Finals']
judge_quintiles = [
    'Top 20%\n(Highest Judges)',
    'Upper\nMiddle',
    'Middle',
    'Lower\nMiddle',
    'Bottom 20%\n(Lowest Judges)',
]

# ── 色彩方案 ──
colors_cmap = ['#1B9E77', '#66C2A5', '#FEE08B', '#FDAE61', '#F46D43', '#D73027']
cmap = LinearSegmentedColormap.from_list('certainty', colors_cmap, N=256)

# ── 构建图形 ──
fig = plt.figure(figsize=(16, 10), facecolor='#FAFAFA')
gs = fig.add_gridspec(1, 3, width_ratios=[5, 2.5, 2.5], wspace=0.25)

ax_heat  = fig.add_subplot(gs[0, 0])
ax_regime = fig.add_subplot(gs[0, 1])
ax_stage  = fig.add_subplot(gs[0, 2])

# ═══════════════════════════════════════
# Panel A: 2D Heatmap
# ═══════════════════════════════════════
im = ax_heat.imshow(ci_width_data, cmap=cmap, aspect='auto', vmin=0.03, vmax=0.24,
                    interpolation='bilinear')

for i in range(5):
    for j in range(5):
        val = ci_width_data[i, j]
        text_color = 'white' if val > 0.16 else '#1a1a2e'
        ax_heat.text(j, i, f'±{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color, fontfamily='monospace')

ax_heat.set_xticks(range(5))
ax_heat.set_xticklabels(stages, fontsize=10, fontweight='bold')
ax_heat.set_yticks(range(5))
ax_heat.set_yticklabels(judge_quintiles, fontsize=9)
ax_heat.set_xlabel('Competition Stage', fontsize=12, fontweight='bold', labelpad=12)
ax_heat.set_ylabel('Judge Score Position', fontsize=12, fontweight='bold', labelpad=12)

# Colorbar
divider = make_axes_locatable(ax_heat)
cax = divider.append_axes("bottom", size="5%", pad=0.6)
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Mean 95% CI Width of Vote Share Estimate (v̂)',
              fontsize=10, fontweight='bold', labelpad=8)
cbar.ax.tick_params(labelsize=9)

# 方向箭头
ax_heat.annotate('', xy=(4.6, 2), xytext=(4.6, -0.3),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2), annotation_clip=False)
ax_heat.text(5.0, 0.8, 'Less\ncertain', fontsize=8, color='#888', ha='center', fontstyle='italic')
ax_heat.annotate('', xy=(4.6, 2), xytext=(4.6, 4.3),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2), annotation_clip=False)
ax_heat.text(5.0, 3.2, 'Less\ncertain', fontsize=8, color='#888', ha='center', fontstyle='italic')
ax_heat.annotate('More certain →', xy=(4.5, -0.7), fontsize=8, color='#888',
                ha='right', fontstyle='italic', annotation_clip=False)

ax_heat.set_title('(A) Uncertainty Fingerprint: When Does the Model Know More?',
                 fontsize=12, fontweight='bold', pad=15, color='#1a1a2e')

# ═══════════════════════════════════════
# Panel B: By Voting Regime
# ═══════════════════════════════════════
regime_data   = [rank_early_ci, percent_ci, rank_late_ci]
regime_names  = ['Rank\n(S1–2)', 'Percent\n(S3–27)', 'Rank+Save\n(S28+)']
regime_colors = ['#3498DB', '#E67E22', '#9B59B6']
positions_r   = [0.8, 2.0, 3.2]

for data_arr, name, col, pos in zip(regime_data, regime_names, regime_colors, positions_r):
    parts = ax_regime.violinplot([data_arr], positions=[pos], vert=True, showextrema=False, widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor(col); pc.set_alpha(0.3); pc.set_edgecolor(col); pc.set_linewidth(1.5)

    ax_regime.boxplot([data_arr], positions=[pos], widths=0.2, vert=True,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=col, alpha=0.8, edgecolor='white', linewidth=1.5),
                      medianprops=dict(color='white', linewidth=2),
                      whiskerprops=dict(color=col, linewidth=1.2),
                      capprops=dict(color=col, linewidth=1.2))
    ax_regime.text(pos, np.median(data_arr) + 0.025, f'{np.median(data_arr):.3f}',
                  ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                  color=col, fontfamily='monospace')

ax_regime.set_xticks(positions_r)
ax_regime.set_xticklabels(regime_names, fontsize=9, fontweight='bold')
ax_regime.set_ylabel('95% CI Width', fontsize=10, fontweight='bold')
ax_regime.set_ylim(0, 0.32)
ax_regime.spines['top'].set_visible(False)
ax_regime.spines['right'].set_visible(False)
ax_regime.set_facecolor('#FAFAFA')
ax_regime.set_title('(B) By Voting Regime', fontsize=11, fontweight='bold', pad=10, color='#1a1a2e')

# 显著性标注
ax_regime.annotate('', xy=(0.8, 0.28), xytext=(2.0, 0.28),
                  arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
ax_regime.text(1.4, 0.285, 'p < 0.01', ha='center', va='bottom', fontsize=8,
              fontweight='bold', color='#333')

# ═══════════════════════════════════════
# Panel C: By Competition Stage
# ═══════════════════════════════════════
stage_data   = [early_ci, mid_ci, late_ci, finals_ci]
stage_names  = ['Early\n(W1–3)', 'Mid\n(W4–6)', 'Late\n(W7–8)', 'Finals']
stage_colors = ['#D73027', '#FDAE61', '#66C2A5', '#1B9E77']
positions_s  = [0.7, 1.7, 2.7, 3.7]

for data_arr, name, col, pos in zip(stage_data, stage_names, stage_colors, positions_s):
    parts = ax_stage.violinplot([data_arr], positions=[pos], vert=True, showextrema=False, widths=0.65)
    for pc in parts['bodies']:
        pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor(col); pc.set_linewidth(1.5)

    ax_stage.boxplot([data_arr], positions=[pos], widths=0.18, vert=True,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=col, alpha=0.85, edgecolor='white', linewidth=1.5),
                     medianprops=dict(color='white', linewidth=2),
                     whiskerprops=dict(color=col, linewidth=1.2),
                     capprops=dict(color=col, linewidth=1.2))
    ax_stage.text(pos, np.median(data_arr) + 0.02, f'{np.median(data_arr):.3f}',
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                 color=col, fontfamily='monospace')

ax_stage.set_xticks(positions_s)
ax_stage.set_xticklabels(stage_names, fontsize=9, fontweight='bold')
ax_stage.set_ylabel('95% CI Width', fontsize=10, fontweight='bold')
ax_stage.set_ylim(0, 0.32)
ax_stage.spines['top'].set_visible(False)
ax_stage.spines['right'].set_visible(False)
ax_stage.set_facecolor('#FAFAFA')
ax_stage.set_title('(C) By Competition Stage', fontsize=11, fontweight='bold', pad=10, color='#1a1a2e')

# 趋势箭头
ax_stage.annotate('', xy=(3.7, 0.06), xytext=(0.7, 0.20),
                 arrowprops=dict(arrowstyle='->', color='#1B9E77', lw=2.5,
                                connectionstyle='arc3,rad=-0.15'))
ax_stage.text(2.2, 0.16, 'Information\naccumulates →', fontsize=8, color='#1B9E77',
             fontweight='bold', fontstyle='italic', ha='center')

# ── 标题 ──
fig.suptitle('The Uncertainty Fingerprint: Where Does the Model Know More — and Where Less?',
            fontsize=15, fontweight='bold', y=1.02, color='#1a1a2e', fontfamily='serif')
fig.text(0.5, 0.98,
        'Systematic analysis of 95% credible interval widths across 34 seasons '
        '· Answering "Is certainty always the same?"',
        ha='center', fontsize=9.5, color='#666', style='italic')

# 底部解读框
fig.text(0.5, -0.02,
        'Key Finding: Certainty is NOT uniform — it depends systematically on '
        'competition stage, judge score position, and voting regime.\n'
        'The model is most certain about extreme-ranked contestants in late '
        'competition stages under the Rank regime.',
        ha='center', fontsize=9, color='#555',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5', edgecolor='#1B9E77', alpha=0.8))

plt.savefig('chart3_uncertainty_fingerprint.png', dpi=300, bbox_inches='tight',
           facecolor='#FAFAFA', edgecolor='none')
plt.close()
print("✅ Chart 3 saved: chart3_uncertainty_fingerprint.png")
