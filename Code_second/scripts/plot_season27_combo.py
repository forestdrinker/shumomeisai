
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup Paths
BASE_DIR = r'd:\shumomeisai\Code_second'
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
FIG_DIR = os.path.join(RESULTS_DIR, 'final_figure_data')
os.makedirs(FIG_DIR, exist_ok=True)

# Aesthetic Config
def set_nature_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    # Aggressively Increased sizes (Huge)
    plt.rcParams['font.size'] = 18          # Was 14
    plt.rcParams['axes.labelsize'] = 22     # Was 16
    plt.rcParams['axes.titlesize'] = 26     # Was 20
    plt.rcParams['xtick.labelsize'] = 18    # Was 13
    plt.rcParams['ytick.labelsize'] = 18    # Was 13
    plt.rcParams['legend.fontsize'] = 18    # Was 13
    plt.rcParams['figure.titlesize'] = 32   # Was 24
    plt.rcParams['axes.linewidth'] = 1.5    # Thicker spines
    plt.rcParams['grid.linewidth'] = 0.8
    plt.rcParams['grid.alpha'] = 0.3
    pass

set_nature_style()


def plot_combo():
    print("Generating Season 27 Combo Figure (Huge Fonts)...")
    
    # Load Data
    full_path = os.path.join(BASE_DIR, 'season27_full_weekly_table.csv')
    bobby_path = os.path.join(BASE_DIR, 'season27_bobby_weekly.csv')
    
    if not os.path.exists(full_path) or not os.path.exists(bobby_path):
        print("CSV files not found.")
        return

    df_full = pd.read_csv(full_path).dropna(subset=['celebrity_name'])
    df_bobby = pd.read_csv(bobby_path).sort_values('week')

    # Setup 1x2 Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [1.5, 1]})
    
    # ==========================
    # LEFT: Full Season
    # ==========================
    unique_pairs = df_full[['pair_id', 'celebrity_name']].drop_duplicates().sort_values('pair_id')
    pair_map = unique_pairs.set_index('pair_id')['celebrity_name'].to_dict()
    pairs = unique_pairs['pair_id'].tolist()
    
    # Colors
    n_contestants = len(pairs)
    palette = sns.color_palette("husl", n_contestants)
    pair_color_map = {pid: palette[i] for i, pid in enumerate(pairs)}
    
    # Override Bobby Color
    BOBBY_COLOR = '#800080' # Purple
    
    final_x = df_full['week'].max()

    for pid in pairs:
        name = pair_map[pid]
        p_data = df_full[df_full['pair_id'] == pid].sort_values('week')
        if len(p_data) < 1: continue
        
        is_bobby = "Bobby" in name
        
        # Style
        color = BOBBY_COLOR if is_bobby else pair_color_map[pid]
        alpha_ribbon = 0.2 if is_bobby else 0.05
        alpha_line = 0.9 if is_bobby else 0.4
        lw = 5.0 if is_bobby else 2.0
        zorder = 10 if is_bobby else 2
        
        # Data
        x = p_data['week'].values
        y_fan = p_data['fan_rank_median'].values
        y_low = p_data['fan_rank_ci_low'].values.astype(float)
        y_high = p_data['fan_rank_ci_high'].values.astype(float)
        y_judge = p_data['judge_rank'].values
        
        # Plot
        ax1.fill_between(x, y_low, y_high, color=color, alpha=alpha_ribbon, ec='none')
        ax1.plot(x, y_fan, color=color, linewidth=lw, alpha=alpha_line, zorder=zorder)
        
        # Dots
        mask_j = ~np.isnan(y_judge)
        if np.any(mask_j):
            ax1.plot(x[mask_j], y_judge[mask_j], 'o', color=color, 
                    markeredgecolor='white', markeredgewidth=1.5, markersize=6 if not is_bobby else 10, 
                    alpha=0.8, linestyle=':', zorder=zorder)
                    
        # Inline Label (At the end of the line)
        if len(x) > 0:
            last_x = x[-1]
            last_y = y_fan[-1]
            
            # Label Props - Huge Sizes
            props = None
            fw = 'normal'
            fontsize = 16 # Was 12
            
            if is_bobby:
                props = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, linewidth=2.5)
                fw = 'bold'
                fontsize = 18 # Was 14
            
            # Place label slightly to the right of the last point
            ax1.text(last_x + 0.15, last_y, name, 
                    va='center', fontsize=fontsize, color=color, 
                    fontweight=fw, bbox=props, zorder=zorder+1)

    # Ax1 Formatting
    ax1.set_ylim(14, 0.5)
    # Reduced margin: max week + 1.5 is enough for most names (except maybe very long ones)
    ax1.set_xlim(df_full['week'].min()-0.2, final_x + 2.0) 
    ax1.set_xlabel('Week', weight='bold')
    ax1.set_ylabel('Rank (1 = Top)', weight='bold')
    ax1.set_title("The Hidden Current of Season 27", loc='left', pad=15)
    
    # Legend Ax1 (Moved Left as requested)
    # Using bbox_to_anchor to shift it left from the corner
    # (x, y) coordinates relative to axes. (0.8, 0.05) means 80% across, 5% up.
    from matplotlib.lines import Line2D
    custom_lines1 = [
        Line2D([0], [0], color='grey', lw=3, label='Fan Rank (Median)'),
        Line2D([0], [0], color='grey', marker='o', linestyle=':', markersize=8, label='Judge Rank')
    ]
    ax1.legend(handles=custom_lines1, loc='lower right', bbox_to_anchor=(0.85, 0.02), 
               frameon=True, fancybox=True, framealpha=0.9)

    # ==========================
    # RIGHT: Bobby Only
    # ==========================
    x_b = df_bobby['week'].values
    y_fan_b = df_bobby['fan_rank_median'].values
    y_low_b = df_bobby['fan_rank_ci_low'].values
    y_high_b = df_bobby['fan_rank_ci_high'].values
    y_judge_b = df_bobby['judge_rank'].values
    
    color_b = BOBBY_COLOR
    
    # Ribbon
    ax2.fill_between(x_b, y_low_b, y_high_b, color=color_b, alpha=0.25, ec='none', label='Fan Rank (95% CI)')
    # Line
    ax2.plot(x_b, y_fan_b, color=color_b, linewidth=5, alpha=1.0, label='Fan Rank (Median)')
    # Dots
    ax2.plot(x_b, y_judge_b, 'o', color='black', markerfacecolor='white', 
            markeredgewidth=3, markersize=12, label='Judge Rank', linestyle=':', linewidth=2.0)
            
    # Label - Increased Size
    ax2.text(x_b[-1]+0.2, y_fan_b[-1], "Bobby Bones", 
             va='center', fontsize=24, fontweight='bold', color=color_b,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color_b, linewidth=3))

    # Ax2 Formatting
    ax2.set_ylim(14, 0.5)
    ax2.set_xlim(x_b[0]-0.2, x_b[-1]+1.5)
    ax2.set_xlabel('Week', weight='bold')
    ax2.set_yticks([]) 
    ax2.set_title("Focus: The Rise of Bobby Bones", loc='left', pad=15)
    
    # Legend Ax2
    ax2.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
    
    # ==========================
    # Global Formatting
    # ==========================
    sns.despine(left=True, bottom=False, ax=ax1)
    sns.despine(left=True, bottom=False, ax=ax2)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, 'Figure_S27_Combo_Hero.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_combo()
