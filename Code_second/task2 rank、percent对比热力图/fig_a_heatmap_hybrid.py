"""
Figure A: Rule Impact Heatmap (Hybrid Version)
==============================================
Combines:
1. Real Season Lengths (Structural Accuracy)
2. Synthetic "Ideal" Data Values (Visual Clarity/Paper Style)

This meets the user requirement: "Adjust season end times [to real], but keep data according to the reference image [idealized Demo data]."
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects
import argparse
import glob
import os

# ================================================================
# 1. Configuration
# ================================================================

# Real Season Lengths (extracted from replay_results)
REAL_SEASON_LENGTHS = {
    1: 5, 2: 7, 3: 9, 4: 9, 5: 9, 6: 9, 7: 9, 8: 10, 9: 9, 10: 9,
    11: 9, 12: 9, 13: 9, 14: 9, 15: 9, 16: 9, 17: 10, 18: 9, 19: 10,
    20: 9, 21: 10, 22: 9, 23: 10, 24: 9, 25: 9, 26: 3, 27: 8, 28: 10,
    29: 10, 30: 9, 31: 9, 32: 10, 33: 8, 34: 10
}

CONTROVERSY_EVENTS = {
    (2, 5): True, (2, 6): True, (2, 7): True,
    (4, 3): True, (4, 4): True, (4, 5): True, (4, 6): True,
    (11, 3): True, (11, 5): True, (11, 7): True, (11, 9): True,
    (27, 4): True, (27, 6): True, (27, 8): True,
}

CONTROVERSY_SEASONS = {
    2:  'Jerry Rice (S2)',
    4:  'Billy Ray Cyrus (S4)',
    11: 'Bristol Palin (S11)',
    27: 'Bobby Bones (S27)',
}

RULE_SEGMENTS = {
    **{s: 'rank' for s in range(1, 3)},
    **{s: 'percent' for s in range(3, 28)},
    **{s: 'rank_save' for s in range(28, 35)},
}

RULE_COLORS = {
    'rank':      '#4A90D9',
    'percent':   '#F5A623',
    'rank_save': '#7B68EE',
}

# ================================================================
# 2. Hybrid Data Generation
# ================================================================
def generate_hybrid_data():
    """Generate synthetic data constrained by REAL season lengths."""
    np.random.seed(42)
    rows = []

    for season in range(1, 35):
        n_weeks = REAL_SEASON_LENGTHS.get(season, 10) # Default to 10 if missing
        rule = RULE_SEGMENTS.get(season, 'percent')

        for week in range(1, n_weeks + 1):
            # --- Demo Logic from fig_a_heatmap.py ---
            if rule == 'rank':
                base_p = np.random.beta(2, 5) # Mean ~0.28
            elif rule == 'percent':
                base_p = np.random.beta(2.5, 4) # Mean ~0.38
            else:
                base_p = np.random.beta(3, 4) # Mean ~0.43

            # Amplify for controversy seasons
            if season in CONTROVERSY_SEASONS:
                late_ratio = week / n_weeks
                if late_ratio > 0.4:
                    base_p = min(1.0, base_p + np.random.beta(3, 2) * 0.4)

            if (season, week) in CONTROVERSY_EVENTS:
                base_p = min(1.0, base_p + 0.35) # Boost controversy spots

            rows.append({
                'season': season,
                'week': week,
                'p_elim_diff': np.clip(base_p, 0, 1)
            })

    return pd.DataFrame(rows)

# ================================================================
# 3. Matrix & Plot
# ================================================================
def build_heatmap_matrix(df):
    seasons = sorted(df['season'].unique())
    max_week = int(df['week'].max())

    matrix = np.full((len(seasons), max_week), np.nan)
    s2r = {s: i for i, s in enumerate(seasons)}

    for _, row in df.iterrows():
        matrix[s2r[int(row['season'])], int(row['week']) - 1] = row['p_elim_diff']

    return matrix, seasons, max_week

def plot_heatmap(matrix, seasons, max_week, output_path=None):
    n_seasons = len(seasons)
    
    # Custom diverging colormap
    # Cool (Low Reversal) -> Hot (High Reversal)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'reversal',
        ['#2166AC', '#67A9CF', '#D1E5F0', '#FDDBC7', '#EF8A62', '#B2182B'],
        N=256
    )
    cmap.set_bad(color='#F0F0F0') # Gray for NaN

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.6, max_week, 3.2, 0.8], wspace=0.05)

    ax_rule = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_anno = fig.add_subplot(gs[0, 2])
    ax_cb   = fig.add_subplot(gs[0, 3])

    # Main Heatmap
    im = ax_main.imshow(
        matrix, aspect='auto', cmap=cmap, vmin=0, vmax=1,
        interpolation='nearest', origin='upper'
    )

    ax_main.set_xticks(range(max_week))
    ax_main.set_xticklabels(range(1, max_week + 1), fontsize=9)
    ax_main.set_xlabel('Week', fontsize=13, fontweight='bold', labelpad=8)

    ax_main.set_yticks(range(n_seasons))
    ax_main.set_yticklabels([f'S{s}' for s in seasons], fontsize=9)
    ax_main.set_ylabel('Season', fontsize=13, fontweight='bold', labelpad=8)

    # Markers
    for (s, w) in CONTROVERSY_EVENTS:
        if s in seasons:
            row = seasons.index(s)
            col = w - 1
            if 0 <= col < max_week:
                # Ensure we only plot if data exists (not NaN)
                if not np.isnan(matrix[row, col]):
                    ax_main.text(
                        col, row, '\u2605', 
                        ha='center', va='center',
                        fontsize=10, color='white', fontweight='bold',
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground='black')]
                    )

    # Grid
    ax_main.set_xticks(np.arange(-0.5, max_week, 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, n_seasons, 1), minor=True)
    ax_main.grid(which='minor', color='white', linewidth=0.3, alpha=0.5)
    ax_main.tick_params(which='minor', size=0)

    # Rule Strip
    ax_rule.set_xlim(0, 1)
    ax_rule.set_ylim(-0.5, n_seasons - 0.5)
    ax_rule.invert_yaxis()
    for i, s in enumerate(seasons):
        rule = RULE_SEGMENTS.get(s, 'percent')
        ax_rule.barh(i, 1, height=1, color=RULE_COLORS.get(rule, '#CCC'),
                     edgecolor='white', linewidth=0.5)
    ax_rule.set_yticks([])
    ax_rule.set_xticks([])
    ax_rule.set_xlabel('Rule', fontsize=10, fontweight='bold')

    # Annotations
    ax_anno.set_xlim(0, 1)
    ax_anno.set_ylim(-0.5, n_seasons - 0.5)
    ax_anno.invert_yaxis()
    ax_anno.axis('off')
    for s, name in CONTROVERSY_SEASONS.items():
        if s in seasons:
            row = seasons.index(s)
            ax_anno.annotate(
                f'  {name}',
                xy=(0.02, row), fontsize=10, fontweight='bold', color='#B2182B', va='center'
            )

    # Colorbar
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('P(Elimination Differs | Rank vs Percent)', fontsize=11, fontweight='bold', labelpad=12)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

    # Titles
    fig.suptitle('Rule Impact Heatmap: Rank vs. Percent Weekly Elimination Reversal Rate', 
                 fontsize=15, fontweight='bold', y=0.98, color='#1a1a2e')
    ax_main.set_title('Redder = Higher Probability of Reversal  \u2605 = Controversy Event', 
                      fontsize=10, color='#555', pad=10)

    # Legend
    patches = [
        mpatches.Patch(color=RULE_COLORS['rank'],      label='Rank (S1-S2)'),
        mpatches.Patch(color=RULE_COLORS['percent'],    label='Percent (S3-S27)'),
        mpatches.Patch(color=RULE_COLORS['rank_save'],  label="Rank + Save (S28+)"),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10,
               frameon=True, edgecolor='#CCC', bbox_to_anchor=(0.5, 0.02))

    plt.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.95)

    base = str(output_path or 'fig_a_reversal_heatmap_hybrid').replace('.png', '')
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {base}.png")

# ================================================================
# 4. Main
# ================================================================
def main():
    print("Generating Hybrid Heatmap (Real Structure + Ideal Values)...")
    df = generate_hybrid_data()
    matrix, seasons, max_week = build_heatmap_matrix(df)
    plot_heatmap(matrix, seasons, max_week, output_path='task2 rank、percent对比热力图/fig_a_heatmap_hybrid')

if __name__ == '__main__':
    main()
