"""
Figure C: Bias Diagnostic
ρ_J vs ρ_F scatter plot showing Judge-bias vs Fan-bias for each (Season, Rule) pair.

Data Source: task2_metrics.csv → 'rho_F', 'rho_J' columns

Usage:
    python fig_c_bias_diagnostic.py --demo                    # Use synthetic demo data
    python fig_c_bias_diagnostic.py --metrics PATH            # Use real metrics CSV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import argparse
import os
from scipy import stats

# ============================================================================
# CONFIGURATION
# ============================================================================

# Rule display configuration with markers
RULES_CONFIG = {
    'rank': {
        'label': 'Rank', 
        'color': '#2166AC', 
        'marker': 'o', 
        'fillstyle': 'full',
        'size': 100
    },
    'percent': {
        'label': 'Percent', 
        'color': '#B2182B', 
        'marker': 's', 
        'fillstyle': 'full',
        'size': 100
    },
    'rank_save': {
        'label': 'Rank + Save', 
        'color': '#4393C3', 
        'marker': 'o', 
        'fillstyle': 'none',  # hollow
        'size': 120
    },
    'percent_save': {
        'label': 'Percent + Save', 
        'color': '#D6604D', 
        'marker': 's', 
        'fillstyle': 'none',  # hollow
        'size': 120
    },
}

RULE_ORDER = ['rank', 'percent', 'rank_save', 'percent_save']

# Controversy seasons to highlight
CONTROVERSY_SEASONS = {
    2: 'Jerry Rice',
    4: 'Billy Ray Cyrus', 
    11: 'Bristol Palin',
    27: 'Bobby Bones'
}

# ============================================================================
# DEMO DATA GENERATION
# ============================================================================

def generate_demo_metrics():
    """
    Generate synthetic metrics data demonstrating the expected pattern:
    - Rank rule: lower ρ_J, higher ρ_F (fan-biased, below diagonal)
    - Percent rule: higher ρ_J, lower ρ_F (judge-biased, above diagonal)
    - +Save variants: generally shift toward higher ρ_J
    """
    np.random.seed(42)
    
    seasons = list(range(1, 35))
    rows = []
    
    for season in seasons:
        # Base correlation values with some season variation
        base_noise = np.random.normal(0, 0.05)
        
        # Rank rule: tends to favor fans
        # Lower ρ_J (0.4-0.6), Higher ρ_F (0.6-0.8)
        rho_J_rank = 0.50 + np.random.normal(0, 0.08) + base_noise
        rho_F_rank = 0.70 + np.random.normal(0, 0.08) + base_noise
        
        # Percent rule: tends to favor judges
        # Higher ρ_J (0.65-0.85), Lower ρ_F (0.45-0.65)
        rho_J_percent = 0.75 + np.random.normal(0, 0.08) + base_noise
        rho_F_percent = 0.55 + np.random.normal(0, 0.08) + base_noise
        
        # +Save variants: shift ρ_J upward (more judge-aligned)
        rho_J_rank_save = rho_J_rank + 0.08 + np.random.normal(0, 0.03)
        rho_F_rank_save = rho_F_rank - 0.05 + np.random.normal(0, 0.03)
        
        rho_J_percent_save = rho_J_percent + 0.05 + np.random.normal(0, 0.03)
        rho_F_percent_save = rho_F_percent - 0.03 + np.random.normal(0, 0.03)
        
        # Clip to valid correlation range
        def clip_corr(x):
            return np.clip(x, -0.2, 1.0)
        
        # Make controversy seasons more extreme
        if season in CONTROVERSY_SEASONS:
            # More extreme fan bias for rank rule
            rho_J_rank -= 0.12
            rho_F_rank += 0.08
            # Less extreme for percent (closer to balanced)
            rho_J_percent += 0.05
        
        rows.append({
            'season': season, 'rule': 'rank',
            'rho_J': clip_corr(rho_J_rank), 'rho_F': clip_corr(rho_F_rank)
        })
        rows.append({
            'season': season, 'rule': 'percent',
            'rho_J': clip_corr(rho_J_percent), 'rho_F': clip_corr(rho_F_percent)
        })
        rows.append({
            'season': season, 'rule': 'rank_save',
            'rho_J': clip_corr(rho_J_rank_save), 'rho_F': clip_corr(rho_F_rank_save)
        })
        rows.append({
            'season': season, 'rule': 'percent_save',
            'rho_J': clip_corr(rho_J_percent_save), 'rho_F': clip_corr(rho_F_percent_save)
        })
    
    return pd.DataFrame(rows)


def load_real_metrics(metrics_path):
    """
    Load real metrics from task2_metrics.csv
    
    Expected columns: season, rule, rho_J, rho_F, ...
    """
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    df = pd.read_csv(metrics_path)
    
    # Validate required columns
    required_cols = ['season', 'rule', 'rho_J', 'rho_F']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_bias_diagnostic(df, output_path='fig_c_bias_diagnostic.png', is_demo=False):
    """
    Create a scatter plot with marginal distributions showing
    ρ_J vs ρ_F for each (Season, Rule) combination.
    """
    
    # Create figure with custom grid for marginals
    fig = plt.figure(figsize=(12, 10))
    
    # Grid spec: main plot + top marginal + right marginal
    gs = fig.add_gridspec(3, 3, width_ratios=[0.1, 4, 1], height_ratios=[1, 4, 0.1],
                         wspace=0.05, hspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 1])
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    ax_legend = fig.add_subplot(gs[0, 2])
    ax_legend.axis('off')
    
    # Determine axis limits
    all_rho = np.concatenate([df['rho_J'].values, df['rho_F'].values])
    lim_min = max(0, min(all_rho) - 0.1)
    lim_max = min(1, max(all_rho) + 0.1)
    
    # Plot each rule type
    for rule in RULE_ORDER:
        rule_data = df[df['rule'] == rule]
        if len(rule_data) == 0:
            continue
        
        config = RULES_CONFIG[rule]
        
        # Scatter plot
        if config['fillstyle'] == 'full':
            ax_main.scatter(
                rule_data['rho_F'], rule_data['rho_J'],
                c=config['color'], marker=config['marker'],
                s=config['size'], alpha=0.7, edgecolors='black',
                linewidths=1, label=config['label'], zorder=3
            )
        else:
            # Hollow markers
            ax_main.scatter(
                rule_data['rho_F'], rule_data['rho_J'],
                facecolors='none', edgecolors=config['color'],
                marker=config['marker'], s=config['size'],
                linewidths=2, alpha=0.8, label=config['label'], zorder=3
            )
        
        # Marginal distributions (KDE)
        if len(rule_data) > 2:
            # Top marginal (rho_F)
            kde_x = np.linspace(lim_min, lim_max, 100)
            try:
                kde_f = stats.gaussian_kde(rule_data['rho_F'])
                ax_top.fill_between(kde_x, kde_f(kde_x), alpha=0.3, color=config['color'])
                ax_top.plot(kde_x, kde_f(kde_x), color=config['color'], linewidth=1.5)
            except:
                pass
            
            # Right marginal (rho_J)
            try:
                kde_j = stats.gaussian_kde(rule_data['rho_J'])
                ax_right.fill_betweenx(kde_x, kde_j(kde_x), alpha=0.3, color=config['color'])
                ax_right.plot(kde_j(kde_x), kde_x, color=config['color'], linewidth=1.5)
            except:
                pass
    
    # Highlight controversy seasons with labels
    for season, name in CONTROVERSY_SEASONS.items():
        season_data = df[df['season'] == season]
        for _, row in season_data.iterrows():
            if row['rule'] == 'rank':  # Only label rank rule points
                ax_main.annotate(
                    f'S{season}',
                    xy=(row['rho_F'], row['rho_J']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=8, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                             alpha=0.7, edgecolor='none'),
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5)
                )
    
    # Reference line: y = x (Judge-bias vs Fan-bias equality)
    ax_main.plot([lim_min, lim_max], [lim_min, lim_max], 
                 'k--', linewidth=2, alpha=0.5, zorder=1, label='ρ_J = ρ_F')
    
    # Fill regions
    # Above diagonal: Judge-favoring (ρ_J > ρ_F)
    ax_main.fill_between([lim_min, lim_max], [lim_min, lim_max], [lim_max, lim_max],
                         color='#B2182B', alpha=0.05, zorder=0)
    # Below diagonal: Fan-favoring (ρ_F > ρ_J)
    ax_main.fill_between([lim_min, lim_max], [lim_min, lim_min], [lim_min, lim_max],
                         color='#2166AC', alpha=0.05, zorder=0)
    
    # Region labels
    ax_main.text(lim_max - 0.08, lim_min + 0.05, 'Fan-Favoring\n(ρ_F > ρ_J)', 
                fontsize=10, color='#2166AC', alpha=0.8, ha='right', va='bottom',
                style='italic')
    ax_main.text(lim_min + 0.05, lim_max - 0.05, 'Judge-Favoring\n(ρ_J > ρ_F)', 
                fontsize=10, color='#B2182B', alpha=0.8, ha='left', va='top',
                style='italic')
    
    # Styling main plot
    ax_main.set_xlabel('ρ_F (Correlation with Fan Ranking)', fontsize=12)
    ax_main.set_ylabel('ρ_J (Correlation with Judge Ranking)', fontsize=12)
    ax_main.set_xlim(lim_min, lim_max)
    ax_main.set_ylim(lim_min, lim_max)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3, linestyle=':')
    ax_main.set_axisbelow(True)
    
    # Style marginals
    ax_top.set_ylabel('Density', fontsize=9)
    ax_top.tick_params(labelbottom=False)
    ax_top.set_yticks([])
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    
    ax_right.set_xlabel('Density', fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xticks([])
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=RULES_CONFIG['rank']['color'],
               markersize=10, markeredgecolor='black', label='Rank'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=RULES_CONFIG['percent']['color'],
               markersize=10, markeredgecolor='black', label='Percent'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markersize=10, markeredgecolor=RULES_CONFIG['rank_save']['color'],
               markeredgewidth=2, label='Rank + Save'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
               markersize=10, markeredgecolor=RULES_CONFIG['percent_save']['color'],
               markeredgewidth=2, label='Percent + Save'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='ρ_J = ρ_F'),
    ]
    
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10, 
                    frameon=True, title='Rule Type', title_fontsize=11)
    
    # Title
    fig.suptitle('Bias Diagnostic — Judge vs Fan Alignment by Rule', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    return fig


def print_statistics(df):
    """Print summary statistics for the paper."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS FOR PAPER")
    print("="*70)
    
    print("\n--- Mean ρ by Rule (across all seasons) ---")
    summary = df.groupby('rule').agg({
        'rho_J': ['mean', 'std'],
        'rho_F': ['mean', 'std']
    }).round(3)
    print(summary)
    
    print("\n--- Systematic Bias Test ---")
    for rule in RULE_ORDER:
        rule_data = df[df['rule'] == rule]
        if len(rule_data) == 0:
            continue
        
        # Test: Is mean(ρ_J) significantly different from mean(ρ_F)?
        diff = rule_data['rho_J'].mean() - rule_data['rho_F'].mean()
        t_stat, p_val = stats.ttest_rel(rule_data['rho_J'], rule_data['rho_F'])
        
        bias_direction = "Judge-favoring" if diff > 0 else "Fan-favoring"
        print(f"  {RULES_CONFIG[rule]['label']:15s}: "
              f"Δ(ρ_J - ρ_F) = {diff:+.3f}, "
              f"t = {t_stat:.2f}, p = {p_val:.4f} → {bias_direction}")
    
    print("\n--- Rank vs Percent Comparison ---")
    rank_data = df[df['rule'] == 'rank']
    percent_data = df[df['rule'] == 'percent']
    
    if len(rank_data) > 0 and len(percent_data) > 0:
        # Paired comparison by season
        merged = rank_data.merge(percent_data, on='season', suffixes=('_rank', '_percent'))
        
        delta_rho_J = merged['rho_J_percent'].mean() - merged['rho_J_rank'].mean()
        delta_rho_F = merged['rho_F_rank'].mean() - merged['rho_F_percent'].mean()
        
        print(f"  Percent has higher ρ_J by: {delta_rho_J:+.3f} (more judge-aligned)")
        print(f"  Rank has higher ρ_F by: {delta_rho_F:+.3f} (more fan-aligned)")
        
        # Statistical test
        t_J, p_J = stats.ttest_rel(merged['rho_J_percent'], merged['rho_J_rank'])
        t_F, p_F = stats.ttest_rel(merged['rho_F_rank'], merged['rho_F_percent'])
        print(f"  t-test for ρ_J difference: t = {t_J:.2f}, p = {p_J:.4f}")
        print(f"  t-test for ρ_F difference: t = {t_F:.2f}, p = {p_F:.4f}")
    
    print("\n--- Controversy Seasons vs Others ---")
    controversy_mask = df['season'].isin(CONTROVERSY_SEASONS.keys())
    
    for rule in ['rank', 'percent']:
        rule_df = df[df['rule'] == rule]
        controv = rule_df[rule_df['season'].isin(CONTROVERSY_SEASONS.keys())]
        normal = rule_df[~rule_df['season'].isin(CONTROVERSY_SEASONS.keys())]
        
        if len(controv) > 0 and len(normal) > 0:
            diff_J = controv['rho_J'].mean() - normal['rho_J'].mean()
            diff_F = controv['rho_F'].mean() - normal['rho_F'].mean()
            print(f"  {RULES_CONFIG[rule]['label']:15s} - Controversy vs Normal:")
            print(f"    Δρ_J = {diff_J:+.3f}, Δρ_F = {diff_F:+.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Figure C: Bias Diagnostic')
    parser.add_argument('--demo', action='store_true', help='Use synthetic demo data')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Path to task2_metrics.csv')
    parser.add_argument('--output', type=str, default='fig_c_bias_diagnostic.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Determine data source
    if args.demo or args.metrics is None:
        print("="*60)
        print("DEMO MODE: Using synthetic data")
        print("="*60)
        df = generate_demo_metrics()
        is_demo = True
    else:
        print(f"Loading metrics from: {args.metrics}")
        df = load_real_metrics(args.metrics)
        is_demo = False
    
    # Print data info
    print(f"\nLoaded {len(df)} rows across {df['season'].nunique()} seasons")
    print(f"Rules present: {df['rule'].unique().tolist()}")
    
    # Print statistics
    print_statistics(df)
    
    # Create visualization
    fig = create_bias_diagnostic(df, args.output, is_demo)
    
    # plt.show()
    print(f"Finished generating {args.output}")


if __name__ == '__main__':
    main()