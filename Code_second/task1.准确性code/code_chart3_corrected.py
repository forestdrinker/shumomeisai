"""
Chart 3: The Uncertainty Fingerprint (Corrected Data Source)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes:
1. Judge Score Direction handling:
   - S1-S2, S28+: v is Rank (Lower is Better).
   - S3-S27: v is Score (Higher is Better).
   This ensures Quintile 0 is always "Best Performance".
2. Metric: 95% CI Width of Softmax(u).

Output: chart3_uncertainty_fingerprint_real.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import glob
from scipy.special import softmax

# ══════════════════════════════════════════════════════════════
# ══ CONFIG ══
# ══════════════════════════════════════════════════════════════

NPZ_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\task1.准确性code'

# ══════════════════════════════════════════════════════════════
# ══ DATA EXTRACTION ══
# ══════════════════════════════════════════════════════════════

all_uncertainties = []

print("Scanning .npz files (Corrected Logic)...")
files = sorted(glob.glob(os.path.join(NPZ_DIR, 'season_*.npz')))

for f in files:
    try:
        base = os.path.basename(f)
        try:
            season_num = int(base.replace('season_', '').replace('.npz', ''))
        except:
            continue
            
        # Determine Regime & Score Direction
        # Rank Eras: v is Rank (Lower is Better) -> Ascending Sort
        # Percent Era: v is Score (Higher is Better) -> Descending Sort
        if season_num <= 2: 
            regime = 'Rank (S1-2)'
            ascending_rank = True 
        elif season_num >= 28:
            regime = 'Rank+Save (S28+)'
            ascending_rank = True
        else:
            regime = 'Percent (S3-27)'
            ascending_rank = False

        data = np.load(f)
        if 'u' not in data or 'v' not in data:
            continue
            
        u = data['u'] # (Samples, Weeks, People)
        v = data['v'] # (Samples, Weeks, People)
        
        # Take mean of v (Judge Score) over samples
        v_mean = np.mean(v, axis=0) # (Weeks, People)
        
        n_samples, n_weeks, n_people = u.shape
        week_values = data.get('week_values', np.arange(1, n_weeks+1))
        
        for t in range(n_weeks):
            week_num = week_values[t]
            
            # Stage
            if week_num <= 3: stage_cat = 'Early\n(W1–3)'; stage_idx = 0
            elif week_num <= 6: stage_cat = 'Mid\n(W4–6)'; stage_idx = 1
            elif week_num <= 8: stage_cat = 'Late\n(W7–8)'; stage_idx = 2
            else: stage_cat = 'Finals'; stage_idx = 3
            
            # Heat Map Col
            if week_num <= 2: heat_col = 0
            elif week_num <= 4: heat_col = 1
            elif week_num <= 6: heat_col = 2
            elif week_num <= 8: heat_col = 3
            else: heat_col = 4
            
            # Uncertainty (Softmax u) - Vote Share CI
            probs_t = softmax(u[:, t, :], axis=1) # (Samples, People)
            ci_lower = np.percentile(probs_t, 2.5, axis=0)
            ci_upper = np.percentile(probs_t, 97.5, axis=0)
            ci_widths = ci_upper - ci_lower
            
            # Quintile Grouping
            scores_t = v_mean[t, :] # (People,)
            
            # Determine Ranks (1=Best)
            if ascending_rank:
                # Lower score is Better (Rank 1 < Rank 10)
                ranks = sp_stats.rankdata(scores_t, method='min')
            else:
                # Higher score is Better (Score 30 > Score 20)
                ranks = sp_stats.rankdata(-scores_t, method='min')
                
            percentiles = ranks / len(ranks) # 1/N (Best) to 1.0 (Worst)
            
            for i in range(n_people):
                # Quintile 0: Top 20% Best Performers
                pct = percentiles[i]
                if pct <= 0.2: q_idx = 0
                elif pct <= 0.4: q_idx = 1
                elif pct <= 0.6: q_idx = 2
                elif pct <= 0.8: q_idx = 3
                else: q_idx = 4
                
                all_uncertainties.append({
                    'regime': regime,
                    'stage_cat': stage_cat,
                    'heat_col': heat_col,
                    'q_idx': q_idx,
                    'ci_width': ci_widths[i]
                })

    except Exception as e:
        print(f"Skipping {f}: {e}")

df = pd.DataFrame(all_uncertainties)

# ══════════════════════════════════════════════════════════════
# ══ PLOTTING ══
# ══════════════════════════════════════════════════════════════

# Heatmap Data
heatmap_matrix = np.zeros((5, 5))
for r in range(5):
    for c in range(5):
        subset = df[(df['q_idx'] == r) & (df['heat_col'] == c)]
        if len(subset) > 0:
            heatmap_matrix[r, c] = subset['ci_width'].mean()
        else:
            heatmap_matrix[r, c] = np.nan

# Plot setup matches previous
stages_heat = ['Week\n1–2', 'Week\n3–4', 'Week\n5–6', 'Week\n7–8', 'Finals\n(W9+)']
stages_violin = ['Early\n(W1–3)', 'Mid\n(W4–6)', 'Late\n(W7–8)', 'Finals']
judge_quintiles = ['Top 20%\n(Highest Judges)', 'Upper\nMiddle', 'Middle', 'Lower\nMiddle', 'Bottom 20%\n(Lowest Judges)']

colors_cmap = ['#1B9E77', '#66C2A5', '#FEE08B', '#FDAE61', '#F46D43', '#D73027']
# Reverse color map? Ideally Red=Uncertain(Wide), Green=Certain(Narrow).
# Check cmap list: #1B9E77 (Green) ... #D73027 (Red).
# So Low Value (0.03) -> Green. High Value (0.3) -> Red.
# This makes sense. Wide CI = Red = Uncertain.
cmap = LinearSegmentedColormap.from_list('certainty', colors_cmap, N=256)

fig = plt.figure(figsize=(16, 10), facecolor='#FAFAFA')
gs = fig.add_gridspec(1, 3, width_ratios=[5, 2.5, 2.5], wspace=0.25)

ax_heat  = fig.add_subplot(gs[0, 0])
ax_regime = fig.add_subplot(gs[0, 1])
ax_stage  = fig.add_subplot(gs[0, 2])

# Panel A
im = ax_heat.imshow(heatmap_matrix, cmap=cmap, aspect='auto', vmin=0.03, vmax=0.40) # Increased vmax to cover all data

for i in range(5):
    for j in range(5):
        val = heatmap_matrix[i, j]
        if np.isnan(val): continue
        text_color = 'white' if val > 0.25 else '#1a1a2e'
        # Fixed: Removed '±' to represent Full Metric Width (not margin of error)
        ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color, fontfamily='monospace')

ax_heat.set_xticks(range(5)); ax_heat.set_xticklabels(stages_heat, fontsize=10, fontweight='bold')
ax_heat.set_yticks(range(5)); ax_heat.set_yticklabels(judge_quintiles, fontsize=9)
ax_heat.set_xlabel('Competition Stage', fontsize=12, fontweight='bold', labelpad=12)
ax_heat.set_ylabel('Judge Score Position', fontsize=12, fontweight='bold', labelpad=12)
divider = make_axes_locatable(ax_heat)
cax = divider.append_axes("bottom", size="5%", pad=0.6)
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Mean 95% CI Width of Vote Share Estimate (Softmax(u))', fontsize=10, fontweight='bold')
ax_heat.set_title('(A) Uncertainty Fingerprint: Real Data Heatmap', fontsize=12, fontweight='bold', pad=15, color='#1a1a2e')

# Panel B (Regime)
reg_order = ['Rank (S1-2)', 'Percent (S3-27)', 'Rank+Save (S28+)']
reg_labels = ['Rank\n(S1–2)', 'Percent\n(S3–27)', 'Rank+Save\n(S28+)']
colors = ['#3498DB', '#E67E22', '#9B59B6']
pos = [1, 2, 3]

for i, r in enumerate(reg_order):
    d = df[df['regime'] == r]['ci_width'].values
    if len(d) == 0: continue
    parts = ax_regime.violinplot([d], positions=[pos[i]], showextrema=False, widths=0.7)
    for pc in parts['bodies']: pc.set_facecolor(colors[i]); pc.set_alpha(0.3)
    ax_regime.boxplot([d], positions=[pos[i]], widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.8), showfliers=False)

ax_regime.set_xticks(pos); ax_regime.set_xticklabels(reg_labels, fontweight='bold')
ax_regime.set_title('(B) By Voting Regime', fontsize=11, fontweight='bold')
ax_regime.spines['top'].set_visible(False); ax_regime.spines['right'].set_visible(False)

# Panel C (Stage)
stg_order = ['Early\n(W1–3)', 'Mid\n(W4–6)', 'Late\n(W7–8)', 'Finals']
stg_labels = ['Early', 'Mid', 'Late', 'Finals']
colors_s = ['#D73027', '#FDAE61', '#66C2A5', '#1B9E77']
pos_s = [1, 2, 3, 4]

for i, s in enumerate(stg_order):
    d = df[df['stage_cat'] == s]['ci_width'].values
    if len(d) == 0: continue
    parts = ax_stage.violinplot([d], positions=[pos_s[i]], showextrema=False, widths=0.7)
    for pc in parts['bodies']: pc.set_facecolor(colors_s[i]); pc.set_alpha(0.3)
    ax_stage.boxplot([d], positions=[pos_s[i]], widths=0.2, patch_artist=True, boxprops=dict(facecolor=colors_s[i], alpha=0.8), showfliers=False)

ax_stage.set_xticks(pos_s); ax_stage.set_xticklabels(stg_labels, fontweight='bold')
ax_stage.set_title('(C) By Competition Stage', fontsize=11, fontweight='bold')
ax_stage.spines['top'].set_visible(False); ax_stage.spines['right'].set_visible(False)

fig.suptitle('The Uncertainty Fingerprint: Where Does the Model Know More? (Corrected Source)', fontsize=15, fontweight='bold', y=1.02)

output_path = os.path.join(OUTPUT_DIR, 'chart3_uncertainty_fingerprint_real.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print(f"✅ Chart 3 saved: {output_path}")
