"""
Chart 2: The Confidence Spectrum
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raincloud plot (half-violin + jitter strip + boxplot) showing the
distribution of model-predicted P(eliminate) for actually-eliminated
contestants, split by voting regime and pooled.

数据接口说明:
    将你的模型输出替换 ══ DATA INTERFACE ══ 区域。
    你需要提供三个 1D numpy array:
        rank_early_probs: S1-S2 中每次淘汰事件的模型预测概率
        percent_probs:    S3-S27 中每次淘汰事件的模型预测概率
        rank_late_probs:  S28-S34 中每次淘汰事件的模型预测概率
    每个元素 = 模型对实际被淘汰选手给出的 P(eliminate)
    例如: 某周有 10 人, 模型给被淘汰者的概率为 0.72, 则该事件的值为 0.72
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as sp_stats

# ══════════════════════════════════════════════════════════════
# ══ DATA INTERFACE ── 替换此区域为你的真实模型输出 ══
# ══════════════════════════════════════════════════════════════

np.random.seed(2026)

# S1-S2: Rank 制度 (约 14 次淘汰)
rank_early_probs = np.clip(
    np.concatenate([
        sp_stats.beta.rvs(5, 1.5, size=12),   # 大部分高置信命中
        sp_stats.beta.rvs(1.5, 3, size=2),     # 少量低置信
    ]), 0.01, 0.99)

# S3-S27: Percent 制度 (约 230 次淘汰)
percent_probs = np.clip(
    np.concatenate([
        sp_stats.beta.rvs(4, 1.8, size=198),   # ~86% 高置信
        sp_stats.beta.rvs(1.2, 2.5, size=32),  # ~14% 低置信/miss
    ]), 0.01, 0.99)

# S28-S34: Rank+JudgeSave 制度 (约 65 次淘汰)
rank_late_probs = np.clip(
    np.concatenate([
        sp_stats.beta.rvs(4.5, 1.6, size=57),  # ~87% 高置信
        sp_stats.beta.rvs(1.3, 2.8, size=8),   # ~13% 低置信
    ]), 0.01, 0.99)

# 各制度下随机猜测基线 (1/平均选手数)
random_baselines = {
    'rank_early': 1 / 10,    # S1-S2 平均 ~10 人
    'percent':    1 / 12,    # S3-S27 平均 ~12 人
    'rank_late':  1 / 11,    # S28-S34 平均 ~11 人
    'pooled':     1 / 11,    # 全部平均
}

# ══════════════════════════════════════════════════════════════
# ══ END DATA INTERFACE ══
# ══════════════════════════════════════════════════════════════


# ── 组织数据 ──
regimes = [
    'Rank\n(S1–S2)',
    'Percent\n(S3–S27)',
    'Rank + Judge Save\n(S28–S34)',
    'All Seasons\n(Pooled)',
]
all_data = [
    rank_early_probs,
    percent_probs,
    rank_late_probs,
    np.concatenate([rank_early_probs, percent_probs, rank_late_probs]),
]
baseline_keys = ['rank_early', 'percent', 'rank_late', 'pooled']

regime_colors       = ['#3498DB', '#E67E22', '#9B59B6', '#1B9E77']
regime_colors_light = ['#AED6F1', '#FAD7A0', '#D7BDE2', '#A3E4D7']

# ── 构建图形 ──
fig, axes = plt.subplots(4, 1, figsize=(14, 12), facecolor='#FAFAFA',
                          gridspec_kw={'hspace': 0.45})

for idx, (ax, arr, name, col, col_light) in enumerate(
    zip(axes, all_data, regimes, regime_colors, regime_colors_light)):

    n = len(arr)

    # ── 半小提琴 (KDE) ──
    kde_x = np.linspace(0, 1, 300)
    kde = sp_stats.gaussian_kde(arr, bw_method=0.12)
    kde_y = kde(kde_x)
    kde_y_norm = kde_y / kde_y.max() * 0.35

    y_base = 0.5
    ax.fill_between(kde_x, y_base, y_base + kde_y_norm,
                   color=col_light, alpha=0.7, edgecolor=col, linewidth=1.2)
    ax.plot(kde_x, y_base + kde_y_norm, color=col, linewidth=1.5)

    # ── 抖动散点 ──
    jitter = np.random.uniform(-0.15, 0.15, size=n)
    y_strip = 0.3 + jitter

    for x_val, y_val in zip(arr, y_strip):
        if x_val >= 0.5:
            dot_color, dot_alpha = col, 0.5
        else:
            dot_color, dot_alpha = '#E74C3C', 0.7
        ax.scatter(x_val, y_val, s=12, color=dot_color, alpha=dot_alpha,
                  edgecolors='none', zorder=3)

    # ── 箱线图 ──
    ax.boxplot([arr], positions=[0.5], widths=0.08, vert=False,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=col, alpha=0.8, edgecolor='white', linewidth=1.5),
               medianprops=dict(color='white', linewidth=2.5),
               whiskerprops=dict(color=col, linewidth=1.5),
               capprops=dict(color=col, linewidth=1.5))

    # ── 统计标注 ──
    median_val = np.median(arr)
    mean_val   = np.mean(arr)
    pct_above  = np.mean(arr > 0.5) * 100

    ax.axvline(median_val, color=col, linestyle='--', linewidth=1, alpha=0.5, ymin=0, ymax=0.95)

    # 随机基线
    rb = random_baselines[baseline_keys[idx]]
    ax.axvline(rb, color='#E74C3C', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(rb, 0.92, f'Random\n1/N≈{rb:.2f}', ha='center', va='top', fontsize=7,
           color='#E74C3C', fontweight='bold', transform=ax.get_xaxis_transform())

    # 统计框
    stats_text = f'n={n}  ·  median={median_val:.2f}  ·  mean={mean_val:.2f}  ·  P>0.5: {pct_above:.0f}%'
    ax.text(0.99, 0.95, stats_text, ha='right', va='top', fontsize=8.5,
           fontfamily='monospace', color='#555', transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#DDD', alpha=0.9))

    # 制度标签
    ax.text(-0.02, 0.5, name, ha='right', va='center', fontsize=11,
           fontweight='bold', color=col, transform=ax.transAxes)

    # ── 样式 ──
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.05, 0.92)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('#FAFAFA')

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    if idx == 3:
        ax.set_xlabel("Model's Predicted P(eliminate) for the Actually Eliminated Contestant",
                      fontsize=11, fontweight='bold', labelpad=8)
        ax.set_xticklabels(['0', '.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9', '1.0'],
                          fontsize=9)
    else:
        ax.set_xticklabels([])

    if idx < 3:
        ax.axhline(0.05, color='#DDD', linewidth=0.8)

# ── 标题 ──
total_n = sum(len(d) for d in all_data[:3])
fig.suptitle("The Confidence Spectrum: How Certain Is the Model When It's Right?",
            fontsize=16, fontweight='bold', y=0.98, color='#1a1a2e', fontfamily='serif')
fig.text(0.5, 0.945,
        f'Distribution of predicted elimination probabilities for actually-eliminated contestants '
        f'· {total_n} events across 34 seasons',
        ha='center', fontsize=9.5, color='#666', style='italic')

fig.text(0.5, 0.01,
        '● Colored dots: model assigned P > 0.5 (confident & correct)  ·  '
        '● Red dots: model assigned P < 0.5 (under-confident)  ·  '
        '⁞ Red dashed: random guess baseline',
        ha='center', fontsize=8.5, color='#777',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#DDD'))

plt.savefig('chart2_confidence_spectrum.png', dpi=300, bbox_inches='tight',
           facecolor='#FAFAFA', edgecolor='none')
plt.close()
print("✅ Chart 2 saved: chart2_confidence_spectrum.png")
