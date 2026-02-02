
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Aspect Ratio and Style
sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['figure.dpi'] = 300

# Constants
SEASON = 27
RESULTS_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\figures\task1_internals'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'

def load_data(season):
    npz_path = os.path.join(RESULTS_DIR, f"season_{season}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Posterior samples not found at {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    samples = {k: data[k] for k in data.files}
    
    panel_df = pd.read_csv(PANEL_PATH)
    season_df = panel_df[panel_df['season'] == season].copy()
    
    pair_ids = samples['pair_ids']
    pid_map = {pid: i for i, pid in enumerate(pair_ids)}
    
    pair_names = {}
    for pid in pair_ids:
        name = season_df[season_df['pair_id'] == pid]['celebrity_name'].iloc[0]
        pair_names[pid_map[pid]] = name
        
    names_list = [pair_names[i] for i in range(len(pair_ids))]
    
    return samples, names_list, pair_ids, season_df

def plot_u_init(samples, names_list, output_dir):
    """
    Figure: u_init Posterior Distributions
    "The Starting Line" - Initial Popularity Advantage
    """
    print("Plotting u_init...")
    u_init_samples = samples['u'][:, 0, :] # (S, N) - u at t=0 is u_init
    # Note: Task 1 model: u = cumsum(u_innov_full). u[0] is u_init.
    
    # Sort by median
    medians = np.median(u_init_samples, axis=0)
    sort_idx = np.argsort(medians)
    
    sorted_samples = u_init_samples[:, sort_idx]
    sorted_names = [names_list[i] for i in sort_idx]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Boxplot
    # Prepare data for sns.boxplot
    # Melt? Or list of vectors
    data_list = [sorted_samples[:, i] for i in range(len(sorted_names))]
    
    ax.boxplot(data_list, vert=False, patch_artist=True, 
               boxprops=dict(facecolor="skyblue", color="black", alpha=0.7),
               medianprops=dict(color="black", linewidth=1.5),
               whiskerprops=dict(color="gray"),
               capprops=dict(color="gray"),
               showfliers=False)
    
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel(r"Initial Latent Popularity ($u_{init}$)")
    ax.set_title(f"The Starting Line: Initial Popularity Advantage (Season {SEASON})", fontsize=16, fontweight='bold')
    
    # Add zero line
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Neutral Popularity')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_u_init.png'))
    plt.close()

def plot_u_evolution(samples, names_list, output_dir):
    """
    Figure: u Trajectories (Raw Random Walk)
    "The Raw Drift" - Unconstrained Utility Evolution
    """
    print("Plotting u evolution...")
    u_samples = samples['u'] # (S, T, N)
    u_mean = np.mean(u_samples, axis=0)
    weeks = samples['week_values']
    
    # Identify top 5 movers (max - min range) or just top popular?
    # Let's highlight Top 3 and Bottom 3 final?
    final_u = u_mean[-1, :]
    sort_idx = np.argsort(final_u)
    top_indices = sort_idx[-3:]
    # bottom_indices = sort_idx[:3]
    highlight_indices = np.concatenate([top_indices]) # Just top 3 for clarity?
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i in range(len(names_list)):
        is_highlight = i in highlight_indices
        alpha = 1.0 if is_highlight else 0.15
        color = None if is_highlight else 'gray'
        lw = 3.0 if is_highlight else 1.0
        label = names_list[i] if is_highlight else None
        
        ax.plot(weeks, u_mean[:, i], color=color, alpha=alpha, linewidth=lw, label=label)
        
    ax.set_title(r"The Raw Drift: Evolution of Latent Utility ($u$)", fontsize=18, fontweight='bold')
    ax.set_ylabel(r"Latent Utility ($u$)", fontsize=14)
    ax.set_xlabel("Week", fontsize=14)
    ax.legend(title="Top Momentum", loc='upper left')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_u_evolution.png'))
    plt.close()

def plot_v_stacked(samples, names_list, output_dir):
    """
    Figure: v Vote Share Stacked Area
    "The Zero-Sum Game" - Vote Share Dynamics
    """
    print("Plotting v stacked...")
    v_samples = samples['v']
    v_mean = np.mean(v_samples, axis=0) # (T, N)
    weeks = samples['week_values']
    
    # We need to sort columns so the plot looks nice (smooth ribbons)
    # Sort by overall average share
    avg_share = np.mean(v_mean, axis=0)
    sort_idx = np.argsort(avg_share)[::-1] # Descending
    
    sorted_v = v_mean[:, sort_idx]
    sorted_names = [names_list[i] for i in sort_idx]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Palettes
    colors = sns.color_palette("tab20", len(names_list))
    
    ax.stackplot(weeks, sorted_v.T, labels=sorted_names, colors=colors, alpha=0.9)
    
    ax.set_title(r"The Zero-Sum Game: Vote Share Dynamics ($v$)", fontsize=18, fontweight='bold')
    ax.set_ylabel(r"Vote Share ($v$)", fontsize=14)
    ax.set_xlabel("Week", fontsize=14)
    ax.set_xlim(weeks[0], weeks[-1])
    ax.set_ylim(0, 1.0)
    
    # Legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_v_stacked.png'))
    plt.close()

def plot_b_risk(samples, names_list, pair_ids, season_df, output_dir):
    """
    Figure: b Badness/Risk Scores
    "The Signal" - Who is actually at risk?
    Compare b of Eliminated vs Safe
    """
    print("Plotting b risk...")
    b_samples = samples['b']
    b_mean = np.mean(b_samples, axis=0) # (T, N)
    weeks = samples['week_values']
    pid_map = {pid: i for i, pid in enumerate(pair_ids)}
    
    # Collect data points: (Week, b_val, status)
    data_points = []
    
    # Need elimination info
    # Reconstruct who was eliminated each week from season_df?
    # Actually we just want to know the status "Eliminated This Week" vs "Safe"
    
    # Only plot if active?
    # Hard to reconstruct active_mask perfectly without logic.
    # But b is usually 0 if inactive (per model logic `jnp.where(mask, r, 0.0)`).
    # So we can filter b != 0.
    
    for t_idx, w in enumerate(weeks):
        # Find who was eliminated this week
        # In panel_df, 'eliminated' column? No.
        # We can look at who disappears next week? Or use elim_events.json... too complex to load distinct.
        # Let's rely on b values. 
        # But we do want to distinctively mark the eliminated person.
        # Let's use logic: The person with MAX b is usually eliminated? Let's check.
        # Wait, I can just plot the trajectories.
        # Let's Plot High Risk Trajectories.
        
        pass

    # Simplified Approach: 
    # Just heatmap of b, but labeled "Elimination Risk Score (b)" -- Already done in fig3_risk_landscape.
    # What else? 
    # "Risk vs Judge Score" Scatter?
    # b = rJ + rF (roughly). 
    # Show contribution of Judge vs Fan to Badness? 
    # That's cool. Stacked Bar of Badness Components?
    # S = rJ. v = rF. 
    # Only for rank rule.
    # 27 is 'percent' rule? Let's check rule in runner output... "Rule=percent".
    # Percent rule: b = SoftRank(pJ + v). 
    # It's non-linear.
    
    # Let's stick to b evolution line chart for top 5 riskiest people.
    # Calculate avg b over season.
    avg_b = np.mean(b_mean, axis=0)
    riskiest_idx = np.argsort(avg_b)[::-1][:5] # Top 5 highest b
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i in range(len(names_list)):
        is_risky = i in riskiest_idx
        alpha = 1.0 if is_risky else 0.1
        color = None if is_risky else 'gray'
        lw = 2.5 if is_risky else 0.5
        label = names_list[i] if is_risky else None
        
        ax.plot(weeks, b_mean[:, i], color=color, alpha=alpha, linewidth=lw, label=label)
        
    ax.set_title(r"The Danger Zone: Elimination Risk Score ($b$) Evolution", fontsize=18, fontweight='bold')
    ax.set_ylabel(r"Risk Score ($b$)", fontsize=14)
    ax.set_xlabel("Week", fontsize=14)
    ax.legend(title="Highest Average Risk", loc='upper left')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_b_evolution.png'))
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Generating Internals for Season {SEASON}...")
    try:
        samples, names_list, pair_ids, season_df = load_data(SEASON)
        
        plot_u_init(samples, names_list, OUTPUT_DIR)
        plot_u_evolution(samples, names_list, OUTPUT_DIR)
        plot_v_stacked(samples, names_list, OUTPUT_DIR)
        plot_b_risk(samples, names_list, pair_ids, season_df, OUTPUT_DIR)
        
        print("\nInternal visualizations generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
