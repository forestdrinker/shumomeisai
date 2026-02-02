
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
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.linewidth'] = 1.0 
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.3
    pass

set_nature_style()

def plot_bobby_flow():
    print("Generating Figure 1 for Bobby Bones...")
    csv_path = os.path.join(BASE_DIR, 'season27_bobby_weekly.csv')
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Ensure sorted by week
    df = df.sort_values('week')
    
    name = "Bobby Bones"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Plot Ribbon (Fan Rank)
    x = df['week'].values
    y_fan = df['fan_rank_median'].values
    y_low = df['fan_rank_ci_low'].values
    y_high = df['fan_rank_ci_high'].values
    
    # Color: Gold/Orange for the "Unexpected Winner"
    color = '#E69F00' # Orange from Okabe-Ito palette or similar high contrast
    
    # Ribbon
    ax.fill_between(x, y_low, y_high, color=color, alpha=0.3, ec='none', label='Fan Rank (95% CI)')
    
    # Median Line
    ax.plot(x, y_fan, color=color, linewidth=4, alpha=0.9, label='Fan Rank (Median)')
    
    # 2. Plot Dots (Judge Rank)
    y_judge = df['judge_rank'].values
    ax.plot(x, y_judge, 'o', color='black', markerfacecolor='white', 
            markeredgewidth=2, markersize=8, label='Judge Rank', linestyle=':', linewidth=1)
            
    # Add annotations for gaps?
    # e.g. Week 4: Judge=7, Fan=1 -> Gap=6
    for i in range(len(x)):
        gap = y_judge[i] - y_fan[i]
        if gap > 3: # Significant gap
            # Draw vertical line?
            ax.vlines(x[i], y_fan[i], y_judge[i], color='red', alpha=0.3, linewidth=1, linestyle='--')
            # ax.text(x[i], (y_fan[i]+y_judge[i])/2, f"Δ{int(gap)}", color='red', fontsize=9, ha='right')

    # Formatting
    ax.set_ylim(14, 0.5) # Inverted
    ax.set_xticks(x)
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('Rank (1 = Top)', weight='bold')
    
    title = f"Figure 1 (Special): The Rise of {name}\nFan Support (Orange) vs Judge Scoring (Black Dots)"
    ax.set_title(title, loc='left', pad=15)
    
    ax.legend(loc='lower right', frameon=True)
    
    sns.despine(left=True, bottom=False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    out_path = os.path.join(FIG_DIR, 'Figure1_BobbyBones.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_bobby_flow()
