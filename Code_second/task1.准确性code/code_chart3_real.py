"""
Chart 3: The Uncertainty Fingerprint (Real Data)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Directly answering "Is certainty always the same?" using Real Posterior Samples.

Metric: 95% CI Width of "Vote Share" (Softmax(u)).
- u: Latent utility/logit vote share from posterior.
- v: Standardized Judge Score/Rank.

Panels:
(A) 2D Heatmap: Stage x Judge Quintile -> Mean CI Width
(B) Violin by Regime
(C) Violin by Stage
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

all_uncertainties = [] # List of (regime, stage, quintile, ci_width)

print("Scanning .npz files...")
files = sorted(glob.glob(os.path.join(NPZ_DIR, 'season_*.npz')))
# Filter logic inside loop to determine Regime

for f in files:
    try:
        # Extract season number
        base = os.path.basename(f)
        try:
            season_num = int(base.replace('season_', '').replace('.npz', ''))
        except:
            continue
            
        # Determine Regime
        if season_num <= 2: regime = 'Rank (S1-2)'
        elif season_num <= 27: regime = 'Percent (S3-27)'
        else: regime = 'Rank+Save (S28+)'
        
        data = np.load(f)
        if 'u' not in data or 'v' not in data:
            continue
            
        u = data['u'] # (Samples, Weeks, People) - check dims!
        v = data['v'] # (Samples, Weeks, People)
        
        # Verify shape
        # Based on inspection: (2000, 9, 11) -> (Samples, Time, People)
        # Note: Samples could be dimension 0
        n_samples, n_weeks, n_people = u.shape
        
        # Mean Judge Score (over samples)
        # v might vary by sample if imputed? But mean is safe.
        v_mean = np.mean(v, axis=0) # (Weeks, People)
        
        # Process Week by Week
        week_values = data.get('week_values', np.arange(1, n_weeks+1))
        
        for t in range(n_weeks):
            week_num = week_values[t]
            
            # Determine Stage
            # Early: W1-3, Mid: W4-6, Late: W7-8, Finals: W9+
            if week_num <= 3: stage_cat = 'Early\n(W1–3)'; stage_idx = 0
            elif week_num <= 6: stage_cat = 'Mid\n(W4–6)'; stage_idx = 1
            elif week_num <= 8: stage_cat = 'Late\n(W7–8)'; stage_idx = 2
            else: stage_cat = 'Finals'; stage_idx = 3 # Map W9+ to Finals
            
            # Mapping for Heatmap (5 columns: 1-2, 3-4, 5-6, 7-8, Finals)
            # 0: W1-2, 1: W3-4, 2: W5-6, 3: W7-8, 4: Finals
            if week_num <= 2: heat_col = 0
            elif week_num <= 4: heat_col = 1
            elif week_num <= 6: heat_col = 2
            elif week_num <= 8: heat_col = 3
            else: heat_col = 4
            
            # Get Posterior Probabilities via Softmax on People dimension (axis=2)
            # u[:, t, :] is (Samples, People)
            probs_t = softmax(u[:, t, :], axis=1) # (Samples, People)
            
            # Calculate 95% CI Width for each person
            # Percentiles along axis 0 (Samples)
            ci_lower = np.percentile(probs_t, 2.5, axis=0)
            ci_upper = np.percentile(probs_t, 97.5, axis=0)
            ci_widths = ci_upper - ci_lower
            
            # Get Scores and Ranks
            scores_t = v_mean[t, :]
            # Determine Quintile (Top 20% = Quintile 0)
            # Rank scores (descending)
            ranks = sp_stats.rankdata(-scores_t, method='min') # 1 = Highest Score
            percentiles = ranks / len(ranks) # 1/N to 1.0
            
            # Quintile 0: 0-0.2 (Top 20%)
            # Quintile 4: 0.8-1.0 (Bottom 20%)
            # Wait, percentiles via rankdata:
            # Score 100 (Best) -> Rank 1 -> 1/N (Small percentile)
            # Score 0 (Worst) -> Rank N -> 1.0 (Large percentile)
            # So Low Percentile = Top Score.
            
            for i in range(n_people):
                unc = ci_widths[i]
                pct = percentiles[i]
                
                # Assign Quintile (0=Top, 4=Bottom)
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
                    'ci_width': unc
                })
                
    except Exception as e:
        print(f"Skipping {f}: {e}")

print(f"Total Data Points: {len(all_uncertainties)}")
df = pd.DataFrame(all_uncertainties)

# ══════════════════════════════════════════════════════════════
# ══ DATA PREPARATION FOR PLOT ══
# ══════════════════════════════════════════════════════════════

# Panel A: 5x5 Matrix
# Rows: Quintile (0-4), Cols: Heat Col (0-4)
heatmap_matrix = np.zeros((5, 5))
for r in range(5):
    for c in range(5):
        subset = df[(df['q_idx'] == r) & (df['heat_col'] == c)]
        if len(subset) > 0:
            heatmap_matrix[r, c] = subset['ci_width'].mean()
        else:
            heatmap_matrix[r, c] = np.nan

# Panel B: Regime Data
rank_early_ci = df[df['regime'] == 'Rank (S1-2)']['ci_width'].values
percent_ci    = df[df['regime'] == 'Percent (S3-27)']['ci_width'].values
rank_late_ci  = df[df['regime'] == 'Rank+Save (S28+)']['ci_width'].values

# Panel C: Stage Data
early_ci  = df[df['stage_cat'] == 'Early\n(W1–3)']['ci_width'].values
mid_ci    = df[df['stage_cat'] == 'Mid\n(W4–6)']['ci_width'].values
late_ci   = df[df['stage_cat'] == 'Late\n(W7–8)']['ci_width'].values
finals_ci = df[df['stage_cat'] == 'Finals']['ci_width'].values

# ══════════════════════════════════════════════════════════════
# ══ PLOTTING ══
# ══════════════════════════════════════════════════════════════

# ── 标签定义 ──
stages_heat = ['Week\n1–2', 'Week\n3–4', 'Week\n5–6', 'Week\n7–8', 'Finals\n(W9+)']
stages_violin = ['Early\n(W1–3)', 'Mid\n(W4–6)', 'Late\n(W7–8)', 'Finals']

judge_quintiles = [
    'Top 20%\n(Highest Judges)',
    'Upper\nMiddle',
    'Middle',
    'Lower\nMiddle',
    'Bottom 20%\n(Lowest Judges)',
]

colors_cmap = ['#1B9E77', '#66C2A5', '#FEE08B', '#FDAE61', '#F46D43', '#D73027']
cmap = LinearSegmentedColormap.from_list('certainty', colors_cmap, N=256)

fig = plt.figure(figsize=(16, 10), facecolor='#FAFAFA')
gs = fig.add_gridspec(1, 3, width_ratios=[5, 2.5, 2.5], wspace=0.25)

ax_heat  = fig.add_subplot(gs[0, 0])
ax_regime = fig.add_subplot(gs[0, 1])
ax_stage  = fig.add_subplot(gs[0, 2])

# Panel A
im = ax_heat.imshow(heatmap_matrix, cmap=cmap, aspect='auto', vmin=0.03, vmax=0.30,
                    interpolation='bilinear')

for i in range(5):
    for j in range(5):
        val = heatmap_matrix[i, j]
        if np.isnan(val): continue
        text_color = 'white' if val > 0.20 else '#1a1a2e'
        ax_heat.text(j, i, f'±{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color, fontfamily='monospace')

ax_heat.set_xticks(range(5))
ax_heat.set_xticklabels(stages_heat, fontsize=10, fontweight='bold')
ax_heat.set_yticks(range(5))
ax_heat.set_yticklabels(judge_quintiles, fontsize=9)
ax_heat.set_xlabel('Competition Stage', fontsize=12, fontweight='bold', labelpad=12)
ax_heat.set_ylabel('Judge Score Position', fontsize=12, fontweight='bold', labelpad=12)

divider = make_axes_locatable(ax_heat)
cax = divider.append_axes("bottom", size="5%", pad=0.6)
cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label('Mean 95% CI Width of Vote Share Estimate (Softmax(u))',
              fontsize=10, fontweight='bold', labelpad=8)

ax_heat.set_title('(A) Uncertainty Fingerprint: Real Data Heatmap',
                 fontsize=12, fontweight='bold', pad=15, color='#1a1a2e')

# Panel B
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
ax_regime.set_title('(B) By Voting Regime', fontsize=11, fontweight='bold', pad=10, color='#1a1a2e')
ax_regime.set_ylim(0, 0.45) # Adjusted for real data range
ax_regime.spines['top'].set_visible(False); ax_regime.spines['right'].set_visible(False)
ax_regime.set_facecolor('#FAFAFA')

# Panel C
stage_data   = [early_ci, mid_ci, late_ci, finals_ci]
stage_names  = stages_violin
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
ax_stage.set_title('(C) By Competition Stage', fontsize=11, fontweight='bold', pad=10, color='#1a1a2e')
ax_stage.set_ylim(0, 0.45)
ax_stage.spines['top'].set_visible(False); ax_stage.spines['right'].set_visible(False)
ax_stage.set_facecolor('#FAFAFA')

# Final Touches
fig.suptitle('The Uncertainty Fingerprint: Where Does the Model Know More?',
            fontsize=15, fontweight='bold', y=1.02, color='#1a1a2e', fontfamily='serif')

output_path = os.path.join(OUTPUT_DIR, 'chart3_uncertainty_fingerprint_real.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight',
           facecolor='#FAFAFA', edgecolor='none')
plt.close()
print(f"✅ Chart 3 saved: {output_path}")
