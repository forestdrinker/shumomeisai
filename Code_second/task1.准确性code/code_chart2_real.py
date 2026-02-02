"""
Chart 2: The Confidence Spectrum (Real Data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Raincloud plot (half-violin + jitter strip + boxplot) showing the
distribution of model-predicted P(eliminate) for CORRECTLY eliminated
contestants (Top-1 Hits), split by voting regime.

Data Source: Results/validation_results/detailed_predictions.csv
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import os

# ══════════════════════════════════════════════════════════════
# ══ DATA LOADING ══
# ══════════════════════════════════════════════════════════════

CSV_PATH = r'd:\shumomeisai\Code_second\Results\validation_results\detailed_predictions.csv'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\task1.准确性code'

# Load predictions
df = pd.read_csv(CSV_PATH)

# Filter for CORRECT predictions (Top-1 Hits)
# The chart title "How Certain Is the Model When It's Right?" implies we focus on Hits.
df_hits = df[df['accuracy'] == 1].copy()

# Extract Probabilities
# In detailed_predictions.csv, 'max_prob' is the probability of the predicted eliminated person.
# Since accuracy=1, predicted == actual, so 'max_prob' IS the probability of the eliminated person.
# Clip to (0.01, 0.99) to avoid numeric issues in KDE if any are exactly 0 or 1
df_hits['prob'] = np.clip(df_hits['max_prob'], 0.01, 0.99)

# Split by Regime
# Rank Early: S1-S2
# Percent: S3-S27
# Rank Late: S28-S34

rank_early_probs = df_hits[df_hits['season'] <= 2]['prob'].values
percent_probs    = df_hits[(df_hits['season'] >= 3) & (df_hits['season'] <= 27)]['prob'].values
rank_late_probs  = df_hits[df_hits['season'] >= 28]['prob'].values

print(f"Loaded Real Data:")
print(f"Rank Early (S1-S2): {len(rank_early_probs)} events")
print(f"Percent (S3-S27):   {len(percent_probs)} events")
print(f"Rank Late (S28-S34):{len(rank_late_probs)} events")

# Random Baselines (1 / Avg Contestants)
# Approx: S1-2 (~10), S3-27 (~12), S28-34 (~12)
random_baselines = {
    'rank_early': 1 / 10,
    'percent':    1 / 12,
    'rank_late':  1 / 12,
    'pooled':     1 / 12,
}

# ══════════════════════════════════════════════════════════════
# ══ PLOTTING LOGIC (Adapted from code_chart2.py) ══
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
    if n < 2: # Skip if no data
        continue

    # ── 半小提琴 (KDE) ──
    kde_x = np.linspace(0, 1, 300)
    try:
        kde = sp_stats.gaussian_kde(arr, bw_method=0.15) # Slightly wider BW for real data
        kde_y = kde(kde_x)
        kde_y_norm = kde_y / kde_y.max() * 0.35
    except:
        kde_y_norm = np.zeros_like(kde_x)

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
        ax.set_xlabel("Model's Predicted P(eliminate) for the Actually Eliminated Contestant (Correct Predictions Only)",
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
        f'Distribution of predicted elimination probabilities for correctly identified contestants '
        f'· {total_n} events across 34 seasons (Real Data)',
        ha='center', fontsize=9.5, color='#666', style='italic')

fig.text(0.5, 0.01,
        '● Colored dots: model assigned P > 0.5 (confident & correct)  ·  '
        '● Red dots: model assigned P < 0.5 (under-confident but correct)  ·  '
        '⁞ Red dashed: random guess baseline',
        ha='center', fontsize=8.5, color='#777',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#DDD'))

output_path = os.path.join(OUTPUT_DIR, 'chart2_confidence_spectrum_real.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight',
           facecolor='#FAFAFA', edgecolor='none')
plt.close()
print(f"✅ Chart 2 saved: {output_path}")

# Print Stats
print("\n" + "="*50)
print("CONFIDENCE STATS")
print("="*50)
print(f"Rank Early (S1-S2): Mean P={np.mean(rank_early_probs):.1%}")
print(f"Percent (S3-S27):   Mean P={np.mean(percent_probs):.1%}")
print(f"Rank Late (S28+):   Mean P={np.mean(rank_late_probs):.1%}")
print("="*50)
