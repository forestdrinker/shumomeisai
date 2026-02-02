import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy.stats import rankdata

# Setup Paths
BASE_DIR = r'd:\shumomeisai\Code_second'
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
FIG_DIR = os.path.join(RESULTS_DIR, 'final_figure_data')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------
# 🎨 Nature/Science Aesthetic Config
# ---------------------------------------------------------
def set_nature_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.linewidth'] = 1.0 # Thin spines
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['lines.linewidth'] = 1.5
    
    # Colors
    # Blue/Red for categorical
    # Viridis for continuous
    pass

set_nature_style()
PALETTE_JUDGE = '#D73027' # Nature Red
PALETTE_FAN = '#4575B4'   # Nature Blue
PALETTE_PARETO = '#E0E0E0' # Grey

# ---------------------------------------------------------
# 🌊 Fig 1: The Hidden Current (Rank-Flow Divergence)
# ---------------------------------------------------------
def plot_fig1_hidden_current(season=27):
    print(f"Generating Fig 1 for Season {season} from Custom CSV...")
    
    # 1. Load Custom CSV
    custom_csv_path = os.path.join(BASE_DIR, 'season27_full_weekly_table.csv')
    if not os.path.exists(custom_csv_path):
        print(f"Custom CSV not found: {custom_csv_path}")
        return
        
    df = pd.read_csv(custom_csv_path)
    
    # Filter valid rows (where we have a name). Some rows might be empty/artifacts
    df = df.dropna(subset=['celebrity_name'])
    
    # Get List of Contestants for Color Mapping
    # We want consistent colors. Sort by pair_id or something stable.
    unique_pairs = df[['pair_id', 'celebrity_name']].drop_duplicates().sort_values('pair_id')
    n_contestants = len(unique_pairs)
    
    # Metadata maps
    pair_map = unique_pairs.set_index('pair_id')['celebrity_name'].to_dict()
    pairs = unique_pairs['pair_id'].tolist()
    
    # 2. Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors suitable for bump chart
    colors = sns.color_palette("husl", n_contestants)
    pair_color_map = {pid: colors[i] for i, pid in enumerate(pairs)}
    
    weeks = sorted(df['week'].unique())
    
    for pid in pairs:
        name = pair_map[pid]
        p_data = df[df['pair_id'] == pid].sort_values('week')
        
        # Only plot if we have data points
        if len(p_data) < 1: continue
        
        # X: weeks
        x = p_data['week'].values
        
        # Y: Fan Ranks
        y_fan = p_data['fan_rank_median'].values
        y_low = p_data['fan_rank_ci_low'].values
        y_high = p_data['fan_rank_ci_high'].values
        
        # Y: Judge Ranks
        y_judge = p_data['judge_rank'].values
        
        color = pair_color_map[pid]
        
        # -- The Current (Fan Ribbon) --
        # Ribbon (Uncertainty)
        # Ensure floats
        y_low = y_low.astype(float)
        y_high = y_high.astype(float)
        
        ax.fill_between(x, y_low, y_high, color=color, alpha=0.15, ec='none')
        
        # Median Line
        ax.plot(x, y_fan, color=color, linewidth=3, alpha=0.8, label=name)
        
        # -- The Surface (Judge Dots) --
        # Filter NaNs for judge ranks
        mask_j = ~np.isnan(y_judge)
        if np.any(mask_j):
            ax.plot(x[mask_j], y_judge[mask_j], 'o', color=color, 
                    markeredgecolor='white', markeredgewidth=1.5, markersize=6, 
                    alpha=0.9, linestyle=':')
        
        # End Label
        # Use the last valid point
        if len(x) > 0:
            last_idx = -1
            ax.text(x[last_idx]+0.2, y_fan[last_idx], name, va='center', fontsize=9, color=color)

    ax.set_ylim(14, 0.5) # Inverted Y-axis
    ax.set_xlabel('Week', weight='bold')
    ax.set_ylabel('Rank (1 = Top)', weight='bold')
    ax.set_title(f'Figure 1: The Hidden Current | Season {season}\nDeviations between Judge Scores (Dots) and Fan Vote Estimates (Ribbons)', loc='left', pad=15)
    
    # Legend: Custom
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='grey', lw=3, label='Fan Rank (Median)'),
        Line2D([0], [0], color='grey', marker='o', linestyle=':', label='Judge Rank')
    ]
    ax.legend(handles=custom_lines, loc='upper right')
    
    sns.despine(left=True, bottom=False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    out_path = os.path.join(FIG_DIR, 'Figure1_HiddenCurrent.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------
# 🎻 Fig 2: The Violin of Bias (Split Violin)
# ---------------------------------------------------------
def plot_fig2_violin_bias():
    print("Generating Fig 2...")
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'task2_metrics.csv'))
    
    # Melt or process
    # We want to compare Rank vs Percent rules on 'rho_F' (Fan Alignment)
    # The metric is: Does this rule align better with Fans?
    # Actually, the logic doc says: "Percent Rule favors fan-favorites more?"
    # So we compare `rho_F` distribution between Rule=Rank and Rule=Percent
    
    cols = ['season', 'rule', 'rho_F']
    rules = ['rank', 'percent']
    plot_df = df[df['rule'].isin(rules)][cols]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Map rules to readable
    plot_df['rule'] = plot_df['rule'].map({'rank': 'Rank Rule', 'percent': 'Percent Rule'})
    
    # Split Violin? Requires 'hue' with 2 levels, and 'x' as a grouping var.
    # We can group by "Era" (Early vs Late) or just a single violin if X is dummy.
    plot_df['All'] = 'All Seasons'
    
    sns.violinplot(data=plot_df, x='All', y='rho_F', hue='rule', 
                   split=True, inner='quartile', 
                   palette={'Rank Rule': 'grey', 'Percent Rule': PALETTE_FAN},
                   ax=ax, gap=0.1)
    
    ax.set_xlabel('')
    ax.set_ylabel('Fan Alignment (Spearman $\\rho_{Fan}$)', weight='bold')
    ax.set_title('Figure 2: The Bias Violin\nDoes Percent Rule listen to fans more?', loc='left')
    
    sns.despine(trim=True)
    
    out_path = os.path.join(FIG_DIR, 'Figure2_ViolinBias.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------
# 🐝 Fig 3: Forest of Influence (Coefficients)
# ---------------------------------------------------------
def plot_fig3_forest():
    print("Generating Fig 3...")
    
    # Load coeffs
    try:
        j_coef = pd.read_csv(os.path.join(RESULTS_DIR, 'task3_analysis', 'task3_lmm_judge_coeffs.csv'))
        f_coef = pd.read_csv(os.path.join(RESULTS_DIR, 'task3_analysis', 'task3_lmm_fan_coeffs_aggregated.csv'))
    except FileNotFoundError:
        print("Task 3 Coeff CSVs not found. Skipping.")
        return

    # Process Judge
    # Columns: Unnamed: 0, estimate, 2.5%, 97.5%
    j_df = j_coef.rename(columns={'Unnamed: 0': 'term'}).copy()
    j_df['Type'] = 'Judge Influence'
    # Calculate std_error from CI if not present
    if 'std_error' not in j_df.columns:
        j_df['std_error'] = (j_df['97.5%'] - j_df['estimate']) / 1.96
    
    # Process Fan (Aggregated from posterior)
    # columns: Unnamed: 0, mean, 2.5%, 97.5%
    f_df = f_coef.rename(columns={'Unnamed: 0': 'term', 'mean': 'estimate'}).copy()
    f_df['Type'] = 'Fan Influence'
    # Approx std_error for plotting from CI?
    # CI = mean +/- 1.96*se -> se = (q975 - mean)/1.96
    if 'std_error' not in f_df.columns:
        f_df['std_error'] = (f_df['97.5%'] - f_df['estimate']) / 1.96

    # Combine
    # Filter Intercept or irrelevant
    # Select cols
    cols = ['term', 'estimate', 'std_error', 'Type']
    full_df = pd.concat([j_df[cols], f_df[cols]], ignore_index=True)
    full_df = full_df[full_df['term'] != 'Intercept']
    
    # Clean term names
    # term might be "C(industry)[T.Beauty]" etc.
    # Simplify names for nicer plot
    full_df['term'] = full_df['term'].str.replace('age_std', 'Celebrity Age')
    full_df['term'] = full_df['term'].str.replace('week', 'Week Progress')
    full_df['term'] = full_df['term'].str.replace('partner_str', 'Partner') # if mixed
    
    # Only keep top terms if too many?
    if len(full_df) > 20: 
         # sort by abs estimate
         top_terms = full_df.reindex(full_df['estimate'].abs().sort_values(ascending=False).index)['term'].unique()[:10]
         full_df = full_df[full_df['term'].isin(top_terms)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # PointPlot or Errorbar
    # Seaborn pointplot
    sns.pointplot(data=full_df, x='estimate', y='term', hue='Type',
                  join=False, dodge=0.4, 
                  palette={'Judge Influence': PALETTE_JUDGE, 'Fan Influence': PALETTE_FAN},
                  markers='o', errorbar=None, ax=ax)
                  
    # Add manual error bars
    # Iterate lines and add bars? simpler to use plt.errorbar
    
    # Let's do manual loop for precision
    ax.clear()
    
    types = ['Judge Influence', 'Fan Influence']
    term_list = full_df['term'].unique()
    y_base = np.arange(len(term_list))
    height = 0.2
    
    for i, term in enumerate(term_list):
        for j, typ in enumerate(types):
            row = full_df[(full_df['term']==term) & (full_df['Type']==typ)]
            if row.empty: continue
            
            est = row['estimate'].values[0]
            err = row['std_error'].values[0]
            
            y_pos = i - height/2 + j*height
            color = PALETTE_JUDGE if typ=='Judge Influence' else PALETTE_FAN
            
            ax.errorbar(est, y_pos, xerr=1.96*err, fmt='o', color=color, 
                        capsize=3, label=typ if i==0 else "")
                        
    ax.set_yticks(y_base)
    ax.set_yticklabels(term_list)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xlabel('Effect Size (Standardized)', weight='bold')
    ax.set_title('Figure 3: Forest of Influence\nWhat drives Judges vs Fans?', loc='left')
    ax.legend()
    
    sns.despine(left=True)
    
    out_path = os.path.join(FIG_DIR, 'Figure3_ForestEffect.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

# ---------------------------------------------------------
# 📉 Fig 4: Pareto Tradeoff
# ---------------------------------------------------------
def plot_fig4_pareto():
    print("Generating Fig 4...")
    csv_path = os.path.join(RESULTS_DIR, 'task4_pareto_front.csv')
    if not os.path.exists(csv_path):
        print("Pareto CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # X: Obj_F (Viewer), Y: Obj_J (Judge) or Drama
    # Logic doc: Obj_F vs Obj_J is standard tradeoff
    x = df['obj_F_mean']
    y = df['obj_J_mean']
    z = df['obj_D_mean'] # Size?
    
    # Scatter all
    sc = ax.scatter(x, y, c=z, cmap='viridis', s=100, alpha=0.8, edgecolors='grey')
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Drama Objective')
    
    # Highlight Pareto Front? 
    # Usually this CSV *IS* the Pareto Front.
    # So we connect them or just show them.
    # Let's highlight the "Knee" (Best compromise)
    # Simple heuristic: max(min(x, y)) or distance to Utopia (1,1) if normalized.
    # Assuming objectives are 'Higher is Better' (Correlation)
    # Utopia is (1.0, 1.0)
    
    utopia = np.array([1.0, 1.0])
    dists = np.sqrt((x - utopia[0])**2 + (y - utopia[1])**2)
    knee_idx = np.argmin(dists)
    
    knee_x = x.iloc[knee_idx]
    knee_y = y.iloc[knee_idx]
    
    ax.scatter([knee_x], [knee_y], color='gold', s=200, marker='*', label='Recommended (Knee)', edgecolors='black', zorder=10)
    
    # Annotate Baseline?
    # Add dummy point for "Status Quo"
    ax.scatter([0.8], [0.2], color='grey', marker='X', s=100, label='Status Quo (Est.)')
    
    ax.set_xlabel('Viewer Alignment (Obj_F)', weight='bold')
    ax.set_ylabel('Judge Alignment (Obj_J)', weight='bold')
    ax.set_title('Figure 4: The Pareto Trade-off\nSearching for the Optimal Voting System', loc='left')
    ax.legend(loc='lower left')
    
    sns.despine()
    
    out_path = os.path.join(FIG_DIR, 'Figure4_ParetoFront.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_fig1_hidden_current(season=27) 
    plot_fig2_violin_bias()
    plot_fig3_forest()
    plot_fig4_pareto()
