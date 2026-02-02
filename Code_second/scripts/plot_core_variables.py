"""
Core Variable Visualizations: u_init, u(t), v, b
Generates 4 comprehensive figures visualizing model uncertainty across all seasons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os
from glob import glob
from scipy.stats import spearmanr, binned_statistic

# Configuration
RESULTS_DIR = r'd:\shumomeisai\Code_second\Results'
POSTERIOR_DIR = os.path.join(RESULTS_DIR, 'posterior_samples')
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
FIG_DIR = os.path.join(RESULTS_DIR, 'figures', 'core_variables')
os.makedirs(FIG_DIR, exist_ok=True)

# Style
sns.set_context("talk", font_scale=1.0)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 200

# Rule eras
RULE_SEGMENTS = {
    **{s: 'Rank' for s in range(1, 3)},
    **{s: 'Percent' for s in range(3, 28)},
    **{s: 'Rank+Save' for s in range(28, 35)},
}

RULE_COLORS = {'Rank': '#4A90D9', 'Percent': '#F5A623', 'Rank+Save': '#7B68EE'}


def load_all_posteriors():
    """Load all posterior samples into a dictionary."""
    data = {}
    pattern = os.path.join(POSTERIOR_DIR, 'season_*.npz')
    files = sorted(glob(pattern), key=lambda x: int(x.split('season_')[1].split('.')[0]))
    
    for fpath in files:
        season = int(fpath.split('season_')[1].split('.')[0])
        npz = np.load(fpath, allow_pickle=True)
        data[season] = {
            'u_init': npz['u_init'],       # (R, N)
            'u': npz['u'],                 # (R, T, N)
            'v': npz['v'],                 # (R, T, N)
            'b': npz['b'],                 # (R, T, N)
            'pair_ids': npz['pair_ids'],   # (N,)
            'week_values': npz['week_values'],  # (T,)
        }
    return data


def load_panel():
    """Load panel.csv with placement and judge scores."""
    df = pd.read_csv(PANEL_PATH)
    return df


# =============================================================================
# Figure 1: Initial Popularity Heatmap
# =============================================================================
def plot_fig1_u_init_heatmap(posteriors, panel):
    """
    Heatmap of E[u_init] across all seasons and contestants.
    X-axis: Contestants sorted by final placement (1=Champion at left)
    Y-axis: Season
    """
    print("Generating Fig 1: u_init Heatmap...")
    
    # Build matrix
    seasons_list = sorted(posteriors.keys())
    max_contestants = max(len(d['pair_ids']) for d in posteriors.values())
    
    matrix = np.full((len(seasons_list), max_contestants), np.nan)
    std_matrix = np.full((len(seasons_list), max_contestants), np.nan)
    
    for i, season in enumerate(seasons_list):
        d = posteriors[season]
        u_init = d['u_init']  # (R, N)
        pair_ids = d['pair_ids']
        
        # Get placements for ordering
        season_panel = panel[panel['season'] == season].drop_duplicates(subset='pair_id')
        placement_map = season_panel.set_index('pair_id')['placement'].to_dict()
        
        # Order by placement
        ordered_indices = sorted(range(len(pair_ids)), key=lambda j: placement_map.get(pair_ids[j], 999))
        
        for rank, j in enumerate(ordered_indices):
            mean_u = np.mean(u_init[:, j])
            std_u = np.std(u_init[:, j])
            matrix[i, rank] = mean_u
            std_matrix[i, rank] = std_u
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'popularity', ['#2166AC', '#67A9CF', '#D1E5F0', '#FDDBC7', '#EF8A62', '#B2182B'], N=256
    )
    cmap.set_bad(color='#E0E0E0')
    
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Labels
    ax.set_yticks(range(len(seasons_list)))
    ax.set_yticklabels([f'S{s}' for s in seasons_list], fontsize=9)
    ax.set_ylabel('Season', fontsize=12, fontweight='bold')
    
    ax.set_xticks(range(max_contestants))
    ax.set_xticklabels([f'{i+1}' for i in range(max_contestants)], fontsize=8)
    ax.set_xlabel('Final Placement (1 = Champion)', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('E[u_init] (Posterior Mean Initial Popularity)', fontsize=11)
    
    # Baseline: Perfect correlation line (if u_init perfectly predicts placement, 
    # the matrix should be monotonically decreasing from left to right)
    # Add text annotation
    ax.text(0.02, 0.98, "Baseline: If u_init perfectly predicts rank,\ncolor should fade left→right (红→蓝)", 
            transform=ax.transAxes, fontsize=9, va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_title('Initial Popularity (u_init) Posterior Estimate by Season & Placement', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'Fig1_u_init_heatmap.png'), dpi=200, bbox_inches='tight')
    print("Saved Fig1_u_init_heatmap.png")
    plt.close()


# =============================================================================
# Figure 2: Popularity Trajectory Envelope
# =============================================================================
def plot_fig2_u_trajectory(posteriors, panel):
    """
    Spaghetti plot showing u(t) trajectories with uncertainty envelopes.
    Focus on champions and controversy contestants.
    """
    print("Generating Fig 2: u(t) Trajectory Envelope...")
    
    # Focus seasons with controversy
    focus_seasons = [2, 4, 11, 27]
    focus_names = {2: 'Jerry Rice', 4: 'Billy Ray Cyrus', 11: 'Bristol Palin', 27: 'Bobby Bones'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for ax_idx, season in enumerate(focus_seasons):
        ax = axes[ax_idx]
        d = posteriors[season]
        u = d['u']  # (R, T, N)
        pair_ids = d['pair_ids']
        weeks = d['week_values']
        
        R, T, N = u.shape
        
        # Get contestant info
        season_panel = panel[(panel['season'] == season)].drop_duplicates(subset='pair_id')
        placement_map = season_panel.set_index('pair_id')['placement'].to_dict()
        name_map = season_panel.set_index('pair_id')['celebrity_name'].to_dict()
        
        # Identify champion and focus contestant
        champion_pid = None
        focus_pid = None
        
        for pid in pair_ids:
            if placement_map.get(pid) == 1:
                champion_pid = pid
            name = name_map.get(pid, '')
            if focus_names[season].lower() in name.lower():
                focus_pid = pid
        
        # Background: all contestants (gray)
        for j in range(N):
            mean_u = np.mean(u[:, :, j], axis=0)
            ax.plot(weeks[:len(mean_u)], mean_u, color='gray', alpha=0.2, linewidth=1)
        
        # Champion (gold)
        if champion_pid is not None:
            j = list(pair_ids).index(champion_pid)
            mean_u = np.mean(u[:, :, j], axis=0)
            ci_lo = np.percentile(u[:, :, j], 5, axis=0)
            ci_hi = np.percentile(u[:, :, j], 95, axis=0)
            ax.fill_between(weeks[:len(mean_u)], ci_lo, ci_hi, color='gold', alpha=0.3)
            ax.plot(weeks[:len(mean_u)], mean_u, color='gold', linewidth=3, label=f'Champion: {name_map.get(champion_pid, "?")}')
        
        # Focus contestant (red)
        if focus_pid is not None and focus_pid != champion_pid:
            j = list(pair_ids).index(focus_pid)
            mean_u = np.mean(u[:, :, j], axis=0)
            ci_lo = np.percentile(u[:, :, j], 5, axis=0)
            ci_hi = np.percentile(u[:, :, j], 95, axis=0)
            ax.fill_between(weeks[:len(mean_u)], ci_lo, ci_hi, color='#D62728', alpha=0.3)
            ax.plot(weeks[:len(mean_u)], mean_u, color='#D62728', linewidth=3, label=f'Focus: {focus_names[season]}')
        
        # Baseline: Uniform u (dashed)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, label='Baseline: u=0 (Neutral)')
        
        ax.set_xlabel('Week', fontsize=11)
        ax.set_ylabel('u (Latent Popularity)', fontsize=11)
        ax.set_title(f'Season {season} ({RULE_SEGMENTS.get(season, "?")})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Popularity Dynamics u(t) for Controversy Seasons\n(90% CI Envelope)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'Fig2_u_trajectory.png'), dpi=200, bbox_inches='tight')
    print("Saved Fig2_u_trajectory.png")
    plt.close()


# =============================================================================
# Figure 3: Vote Share v vs Judge Rank rJ
# =============================================================================
def plot_fig3_v_vs_rJ(posteriors, panel):
    """
    Hexbin density plot of E[v] vs Judge Rank, colored by rule type.
    """
    print("Generating Fig 3: v vs rJ Scatter...")
    
    # Aggregate data
    rows = []
    for season, d in posteriors.items():
        v = d['v']  # (R, T, N)
        pair_ids = d['pair_ids']
        weeks = d['week_values']
        
        R, T, N = v.shape
        rule = RULE_SEGMENTS.get(season, 'Percent')
        
        season_panel = panel[panel['season'] == season]
        
        for t, week in enumerate(weeks):
            if t >= T:
                break
            week_panel = season_panel[season_panel['week'] == week]
            
            for j, pid in enumerate(pair_ids):
                row = week_panel[week_panel['pair_id'] == pid]
                if len(row) == 0:
                    continue
                
                s_it = row['S_it'].values[0]
                if pd.isna(s_it):
                    continue
                
                # Judge rank (lower = better)
                week_scores = week_panel[['pair_id', 'S_it']].dropna()
                week_scores = week_scores.sort_values('S_it', ascending=False)
                rJ = week_scores['pair_id'].tolist().index(pid) + 1 if pid in week_scores['pair_id'].values else np.nan
                
                if np.isnan(rJ):
                    continue
                
                mean_v = np.mean(v[:, t, j])
                
                rows.append({
                    'season': season,
                    'week': week,
                    'pair_id': pid,
                    'v': mean_v,
                    'rJ': rJ,
                    'rule': rule
                })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No data for Fig 3!")
        return
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    rules = ['Rank', 'Percent', 'Rank+Save']
    
    for ax, rule in zip(axes, rules):
        subset = df[df['rule'] == rule]
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        color = RULE_COLORS.get(rule, 'gray')
        
        ax.hexbin(subset['v'], subset['rJ'], gridsize=20, cmap='Blues', mincnt=1)
        
        # Baseline: Random (uniform v)
        n_contestants_avg = 10
        ax.axvline(1/n_contestants_avg, color='red', linestyle='--', label=f'Random v = 1/N ≈ {1/n_contestants_avg:.1%}')
        
        # Regression line
        if len(subset) > 10:
            z = np.polyfit(subset['v'], subset['rJ'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['v'].min(), subset['v'].max(), 50)
            ax.plot(x_line, p(x_line), color='red', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('E[v] (Posterior Mean Vote Share)', fontsize=11)
        ax.set_ylabel('Judge Rank (1 = Highest Score)', fontsize=11)
        ax.set_title(f'{rule} Rule Era', fontsize=12, fontweight='bold')
        ax.invert_yaxis()  # Higher rank (1) at top
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Vote Share (v) vs Judge Rank (rJ)\n(Lower rJ = Better Judges)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'Fig3_v_vs_rJ.png'), dpi=200, bbox_inches='tight')
    print("Saved Fig3_v_vs_rJ.png")
    plt.close()


# =============================================================================
# Figure 4: Badness Score Calibration
# =============================================================================
def plot_fig4_b_calibration(posteriors, panel):
    """
    Calibration curve: P(eliminated | b) vs observed elimination rate.
    """
    print("Generating Fig 4: b Calibration Curve...")
    
    # Aggregate elimination events
    rows = []
    
    for season, d in posteriors.items():
        b = d['b']  # (R, T, N)
        pair_ids = d['pair_ids']
        weeks = d['week_values']
        
        R, T, N = b.shape
        rule = RULE_SEGMENTS.get(season, 'Percent')
        
        season_panel = panel[panel['season'] == season]
        
        for t, week in enumerate(weeks):
            if t >= T:
                break
            
            # Get eliminated contestant this week
            eliminated_pids = []
            for pid in pair_ids:
                row = season_panel[(season_panel['pair_id'] == pid) & (season_panel['week'] == week)]
                if len(row) == 0:
                    continue
                elim_week = row['elim_week_by_score'].values[0]
                if not pd.isna(elim_week) and int(elim_week) == week:
                    eliminated_pids.append(pid)
            
            for j, pid in enumerate(pair_ids):
                # Skip if contestant already eliminated
                row = season_panel[(season_panel['pair_id'] == pid) & (season_panel['week'] == week)]
                if len(row) == 0 or row['is_active'].values[0] == False:
                    continue
                
                mean_b = np.mean(b[:, t, j])
                was_eliminated = 1 if pid in eliminated_pids else 0
                
                rows.append({
                    'season': season,
                    'week': week,
                    'pair_id': pid,
                    'b': mean_b,
                    'eliminated': was_eliminated,
                    'rule': rule
                })
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        print("No data for Fig 4!")
        return
    
    # Convert b to elimination probability via softmax: P = exp(b) / sum(exp(b)) per week
    # For simplicity, we'll bin by b values and check observed elimination rate
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    rules = ['Rank', 'Percent', 'Rank+Save']
    colors = [RULE_COLORS['Rank'], RULE_COLORS['Percent'], RULE_COLORS['Rank+Save']]
    
    for rule, color in zip(rules, colors):
        subset = df[df['rule'] == rule]
        if len(subset) < 20:
            continue
        
        # Bin by b values
        bins = np.percentile(subset['b'], np.linspace(0, 100, 11))
        bins = np.unique(bins)
        
        if len(bins) < 3:
            continue
        
        bin_means, bin_edges, _ = binned_statistic(subset['b'], subset['b'], statistic='mean', bins=bins)
        bin_elim_rate, _, _ = binned_statistic(subset['b'], subset['eliminated'], statistic='mean', bins=bins)
        bin_counts, _, _ = binned_statistic(subset['b'], subset['eliminated'], statistic='count', bins=bins)
        
        # Filter bins with enough samples
        mask = bin_counts >= 5
        bin_centers = (bin_edges[:-1] + bin_edges[1:])[mask] / 2
        bin_means = bin_means[mask]
        bin_elim_rate = bin_elim_rate[mask]
        
        # Normalize b to pseudo-probability (0-1 range for plotting)
        b_min, b_max = subset['b'].min(), subset['b'].max()
        b_norm = (bin_means - b_min) / (b_max - b_min + 1e-6)
        
        ax.plot(b_norm, bin_elim_rate, 'o-', color=color, linewidth=2, markersize=8, label=rule)
    
    # Baseline: Perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)
    
    # Random baseline: if b is uninformative, elimination rate = 1/N ≈ 10%
    ax.axhline(0.1, color='gray', linestyle=':', label='Random Baseline (~10%)', alpha=0.7)
    
    ax.set_xlabel('Normalized b (Badness Score, 0=Best, 1=Worst)', fontsize=12)
    ax.set_ylabel('Observed Elimination Rate', fontsize=12)
    ax.set_title('Badness Score (b) Calibration by Rule Era\n(Higher b → More Likely to be Eliminated)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'Fig4_b_calibration.png'), dpi=200, bbox_inches='tight')
    print("Saved Fig4_b_calibration.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("Loading data...")
    posteriors = load_all_posteriors()
    print(f"Loaded {len(posteriors)} seasons")
    
    panel = load_panel()
    print(f"Loaded panel with {len(panel)} rows")
    
    plot_fig1_u_init_heatmap(posteriors, panel)
    plot_fig2_u_trajectory(posteriors, panel)
    plot_fig3_v_vs_rJ(posteriors, panel)
    plot_fig4_b_calibration(posteriors, panel)
    
    print("\nAll figures saved to:", FIG_DIR)
