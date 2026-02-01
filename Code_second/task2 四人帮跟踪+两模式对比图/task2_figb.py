"""
Figure B: Controversy Portraits
Posterior placement distributions for 4 focal contestants across rule variants.

Data Source: season_{S}_{rule}.npz files → 'placements' array
             Filter to specific contestant's column index via 'pair_ids'
             
Usage:
    python fig_b_controversy_portraits.py --demo           # Use synthetic demo data
    python fig_b_controversy_portraits.py --data-dir PATH  # Use real replay results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import argparse
import os
import glob
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Focal contestants for controversy analysis
FOCAL_CONTESTANTS = {
    'Jerry Rice': {'season': 2, 'actual_placement': 2, 'pair_id': None},
    'Billy Ray Cyrus': {'season': 4, 'actual_placement': 5, 'pair_id': None},
    'Bristol Palin': {'season': 11, 'actual_placement': 3, 'pair_id': None},
    'Bobby Bones': {'season': 27, 'actual_placement': 1, 'pair_id': None},
}

# Rule display configuration
RULES_CONFIG = {
    'rank': {'label': 'Rank', 'color': '#2166AC', 'hatch': None},
    'percent': {'label': 'Percent', 'color': '#B2182B', 'hatch': None},
    'rank_save': {'label': 'Rank + Save', 'color': '#4393C3', 'hatch': '//'},
    'percent_save': {'label': 'Percent + Save', 'color': '#D6604D', 'hatch': '//'},
}

RULE_ORDER = ['rank', 'percent', 'rank_save', 'percent_save']

# ============================================================================
# DEMO DATA GENERATION
# ============================================================================

def generate_demo_data():
    """
    Generate synthetic placement distributions that demonstrate
    the expected patterns for controversial contestants.
    """
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = {}
    
    # Jerry Rice (S2): Fan favorite, judge skeptic
    # Under Rank: clusters near top (1-3)
    # Under Percent: shifts to middle (3-6)
    # Under +Save: shifts further (4-8)
    n_pairs_s2 = 12
    demo_data['Jerry Rice'] = {
        'rank': np.random.choice([1, 2, 2, 3, 3, 3, 4, 4, 5], n_samples, 
                                  p=[0.15, 0.25, 0.25, 0.15, 0.15, 0.15, 0.05, 0.03, 0.02]/np.sum([0.15, 0.25, 0.25, 0.15, 0.15, 0.15, 0.05, 0.03, 0.02])),
        'percent': np.random.choice([2, 3, 4, 4, 5, 5, 6, 6, 7, 8], n_samples,
                                     p=[0.05, 0.10, 0.15, 0.15, 0.20, 0.20, 0.08, 0.04, 0.02, 0.01]),
        'rank_save': np.random.choice([3, 4, 5, 5, 6, 6, 7, 7, 8, 9], n_samples,
                                       p=[0.05, 0.10, 0.15, 0.15, 0.20, 0.20, 0.08, 0.04, 0.02, 0.01]),
        'percent_save': np.random.choice([4, 5, 6, 6, 7, 7, 8, 8, 9, 10], n_samples,
                                          p=[0.05, 0.10, 0.15, 0.15, 0.20, 0.20, 0.08, 0.04, 0.02, 0.01]),
        'n_pairs': n_pairs_s2,
        'actual': 2
    }
    
    # Billy Ray Cyrus (S4): Similar pattern
    n_pairs_s4 = 11
    demo_data['Billy Ray Cyrus'] = {
        'rank': np.random.choice([3, 4, 5, 5, 6, 6, 7], n_samples,
                                  p=[0.08, 0.15, 0.25, 0.25, 0.15, 0.08, 0.04]),
        'percent': np.random.choice([5, 6, 7, 7, 8, 8, 9, 10], n_samples,
                                     p=[0.05, 0.12, 0.20, 0.20, 0.20, 0.13, 0.07, 0.03]),
        'rank_save': np.random.choice([6, 7, 8, 8, 9, 9, 10], n_samples,
                                       p=[0.08, 0.15, 0.25, 0.25, 0.15, 0.08, 0.04]),
        'percent_save': np.random.choice([7, 8, 9, 9, 10, 10, 11], n_samples,
                                          p=[0.08, 0.15, 0.25, 0.25, 0.15, 0.08, 0.04]),
        'n_pairs': n_pairs_s4,
        'actual': 5
    }
    
    # Bristol Palin (S11): Extreme fan favorite
    n_pairs_s11 = 12
    demo_data['Bristol Palin'] = {
        'rank': np.random.choice([1, 2, 3, 3, 4, 4, 5], n_samples,
                                  p=[0.12, 0.25, 0.30, 0.30, 0.15, 0.05, 0.03]/np.sum([0.12, 0.25, 0.30, 0.30, 0.15, 0.05, 0.03])),
        'percent': np.random.choice([3, 4, 5, 5, 6, 6, 7, 8], n_samples,
                                     p=[0.08, 0.15, 0.22, 0.22, 0.18, 0.08, 0.05, 0.02]),
        'rank_save': np.random.choice([4, 5, 6, 6, 7, 7, 8, 9], n_samples,
                                       p=[0.05, 0.12, 0.22, 0.22, 0.20, 0.10, 0.06, 0.03]),
        'percent_save': np.random.choice([5, 6, 7, 7, 8, 8, 9, 10], n_samples,
                                          p=[0.05, 0.12, 0.22, 0.22, 0.20, 0.10, 0.06, 0.03]),
        'n_pairs': n_pairs_s11,
        'actual': 3
    }
    
    # Bobby Bones (S27): THE controversial winner
    # Under Rank: High win probability (peak at 1)
    # Under Percent: Win probability collapses
    # Under +Save: Even more dramatic collapse
    n_pairs_s27 = 13
    demo_data['Bobby Bones'] = {
        'rank': np.random.choice([1, 1, 2, 2, 3, 3, 4, 5, 6], n_samples,
                                  p=[0.35, 0.35, 0.12, 0.12, 0.08, 0.08, 0.04, 0.02, 0.01]/np.sum([0.35, 0.35, 0.12, 0.12, 0.08, 0.08, 0.04, 0.02, 0.01])),
        'percent': np.random.choice([3, 4, 5, 5, 6, 6, 7, 8, 9], n_samples,
                                     p=[0.05, 0.10, 0.18, 0.18, 0.22, 0.12, 0.08, 0.05, 0.02]),
        'rank_save': np.random.choice([4, 5, 6, 6, 7, 7, 8, 9, 10], n_samples,
                                       p=[0.04, 0.08, 0.15, 0.15, 0.25, 0.15, 0.10, 0.05, 0.03]),
        'percent_save': np.random.choice([5, 6, 7, 8, 8, 9, 9, 10, 11], n_samples,
                                          p=[0.04, 0.08, 0.12, 0.18, 0.18, 0.20, 0.10, 0.06, 0.04]),
        'n_pairs': n_pairs_s27,
        'actual': 1
    }
    
    return demo_data


def load_real_data(data_dir, panel_path=None):
    """
    Load real placement data from .npz replay files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing season_{S}_{rule}.npz files
    panel_path : str, optional
        Path to panel.csv for pair_id -> name mapping
    
    Returns:
    --------
    dict : Placement distributions per contestant per rule
    """
    real_data = {}
    
    # Load panel for name mapping if available
    pair_name_map = {}
    if panel_path and os.path.exists(panel_path):
        df_panel = pd.read_csv(panel_path)
        pair_name_map = df_panel.set_index(['season', 'pair_id'])['celebrity_name'].to_dict()
    
    for name, info in FOCAL_CONTESTANTS.items():
        season = info['season']
        actual = info['actual_placement']
        real_data[name] = {'actual': actual}
        
        for rule in RULE_ORDER:
            fpath = os.path.join(data_dir, f"season_{season}_{rule}.npz")
            if not os.path.exists(fpath):
                print(f"  Warning: {fpath} not found")
                continue
            
            data = np.load(fpath, allow_pickle=True)
            placements = data['placements']  # (R, N)
            pair_ids = data['pair_ids']
            
            # Find contestant's column index
            # Try to match by name in pair_name_map
            target_col = None
            for col_idx, pid in enumerate(pair_ids):
                celeb_name = pair_name_map.get((season, pid), '')
                if name.lower() in celeb_name.lower() or celeb_name.lower() in name.lower():
                    target_col = col_idx
                    break
            
            # Fallback: use pair_id if provided
            if target_col is None and info['pair_id'] is not None:
                try:
                    target_col = list(pair_ids).index(info['pair_id'])
                except ValueError:
                    pass
            
            if target_col is not None:
                real_data[name][rule] = placements[:, target_col]
                real_data[name]['n_pairs'] = placements.shape[1]
            else:
                print(f"  Warning: Could not find {name} in season {season}")
    
    return real_data


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_controversy_portraits(data, output_path='fig_b_controversy_portraits.png', is_demo=False):
    """
    Create a 2x2 panel of violin plots showing posterior placement
    distributions for 4 controversial contestants.
    """
    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    contestants = list(FOCAL_CONTESTANTS.keys())
    
    for idx, name in enumerate(contestants):
        ax = axes[idx]
        contestant_data = data.get(name, {})
        
        if not contestant_data:
            ax.text(0.5, 0.5, f'No data for {name}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            continue
        
        # Prepare data for violin plot
        plot_data = []
        positions = []
        colors = []
        
        for rule_idx, rule in enumerate(RULE_ORDER):
            if rule in contestant_data:
                placements = contestant_data[rule]
                plot_data.append(placements)
                positions.append(rule_idx)
                colors.append(RULES_CONFIG[rule]['color'])
        
        if not plot_data:
            ax.text(0.5, 0.5, f'No placement data for {name}', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            continue
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=positions, 
                             showmeans=True, showmedians=False, showextrema=False)
        
        # Customize violin colors
        for pc_idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[pc_idx])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)
            pc.set_linewidth(1.5)
            
            # Add hatch for +Save variants
            rule = RULE_ORDER[positions[pc_idx]]
            if RULES_CONFIG[rule]['hatch']:
                pc.set_hatch(RULES_CONFIG[rule]['hatch'])
        
        # Style mean line
        parts['cmeans'].set_color('white')
        parts['cmeans'].set_linewidth(2)
        
        # Add actual placement line
        actual = contestant_data.get('actual')
        if actual:
            ax.axhline(y=actual, color='#2ca02c', linestyle='--', linewidth=2.5, 
                      label=f'Actual Placement ({actual})')
        
        # Add statistical annotations
        for rule_idx, rule in enumerate(RULE_ORDER):
            if rule in contestant_data:
                placements = contestant_data[rule]
                mean_p = np.mean(placements)
                p_win = np.mean(placements == 1)
                p_top3 = np.mean(placements <= 3)
                
                # Annotation text
                ann_text = f'μ={mean_p:.1f}\nP(Win)={p_win:.0%}\nP(Top3)={p_top3:.0%}'
                
                # Position annotation above violin
                y_pos = max(placements) + 0.8
                ax.annotate(ann_text, xy=(rule_idx, y_pos), 
                           ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                    edgecolor='gray', alpha=0.9))
        
        # Styling
        n_pairs = contestant_data.get('n_pairs', 13)
        ax.set_ylim(0.5, n_pairs + 2)
        ax.set_xlim(-0.6, len(RULE_ORDER) - 0.4)
        ax.set_xticks(range(len(RULE_ORDER)))
        ax.set_xticklabels([RULES_CONFIG[r]['label'] for r in RULE_ORDER], fontsize=10)
        ax.set_ylabel('Final Placement (1 = Champion)', fontsize=11)
        ax.invert_yaxis()  # 1 at top
        
        # Title with season info
        season = FOCAL_CONTESTANTS[name]['season']
        ax.set_title(f'{name} (Season {season})', fontsize=13, fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.set_axisbelow(True)
        
        # Add legend for actual placement line
        if actual:
            ax.legend(loc='lower right', fontsize=9)
    
    # Global legend for rule colors
    legend_elements = [
        mpatches.Patch(facecolor=RULES_CONFIG['rank']['color'], label='Rank', alpha=0.7, edgecolor='black'),
        mpatches.Patch(facecolor=RULES_CONFIG['percent']['color'], label='Percent', alpha=0.7, edgecolor='black'),
        mpatches.Patch(facecolor=RULES_CONFIG['rank_save']['color'], label='Rank + Save', 
                      alpha=0.7, edgecolor='black', hatch='//'),
        mpatches.Patch(facecolor=RULES_CONFIG['percent_save']['color'], label='Percent + Save', 
                      alpha=0.7, edgecolor='black', hatch='//'),
        Line2D([0], [0], color='#2ca02c', linestyle='--', linewidth=2.5, label='Actual Result'),
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.98))
    
    # Title
    demo_tag = ' [DEMO DATA]' if is_demo else ''
    fig.suptitle(f'Figure B: Controversy Portraits — Posterior Placement Distributions{demo_tag}', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    return fig


def print_statistics(data):
    """Print summary statistics for the paper."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS FOR PAPER")
    print("="*70)
    
    for name, contestant_data in data.items():
        season = FOCAL_CONTESTANTS[name]['season']
        actual = contestant_data.get('actual', '?')
        
        print(f"\n{name} (Season {season}, Actual: #{actual})")
        print("-" * 50)
        
        for rule in RULE_ORDER:
            if rule in contestant_data:
                placements = contestant_data[rule]
                
                mean_p = np.mean(placements)
                std_p = np.std(placements)
                p_win = np.mean(placements == 1)
                p_top3 = np.mean(placements <= 3)
                
                print(f"  {RULES_CONFIG[rule]['label']:15s}: "
                      f"E[Rank]={mean_p:.2f}±{std_p:.2f}, "
                      f"P(Win)={p_win:.1%}, P(Top3)={p_top3:.1%}")
        
        # Compute deltas if both rank and percent exist
        if 'rank' in contestant_data and 'percent' in contestant_data:
            delta_win = np.mean(contestant_data['rank'] == 1) - np.mean(contestant_data['percent'] == 1)
            delta_rank = np.mean(contestant_data['percent']) - np.mean(contestant_data['rank'])
            print(f"  → ΔP(Win) [Rank - Percent]: {delta_win:+.1%}")
            print(f"  → ΔE[Rank] [Percent - Rank]: {delta_rank:+.2f} (positive = worse under Percent)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Figure B: Controversy Portraits')
    parser.add_argument('--demo', action='store_true', help='Use synthetic demo data')
    parser.add_argument('--data-dir', type=str, default=None, 
                       help='Directory containing season_*_*.npz replay files')
    parser.add_argument('--panel', type=str, default=None,
                       help='Path to panel.csv for name mapping')
    parser.add_argument('--output', type=str, default='fig_b_controversy_portraits.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Determine data source
    if args.demo or args.data_dir is None:
        print("="*60)
        print("DEMO MODE: Using synthetic data")
        print("="*60)
        data = generate_demo_data()
        is_demo = True
    else:
        print(f"Loading real data from: {args.data_dir}")
        data = load_real_data(args.data_dir, args.panel)
        is_demo = False
    
    # Print statistics
    print_statistics(data)
    
    # Create visualization
    fig = create_controversy_portraits(data, args.output, is_demo)
    
    # plt.show()
    print(f"Finished generating {args.output}")

if __name__ == '__main__':
    main()






