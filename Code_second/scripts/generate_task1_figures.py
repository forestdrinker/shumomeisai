
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set aesthetic style
sns.set_context("talk")
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Segoe UI'  # standard windows font
plt.rcParams['figure.dpi'] = 300

# Constants
SEASON = 27
RESULTS_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\figures\task1_advanced'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'

def logistic(x):
    return 1 / (1 + np.exp(-x))

def load_data(season):
    """Load posterior samples and panel data"""
    # 1. Load Posterior
    npz_path = os.path.join(RESULTS_DIR, f"season_{season}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Posterior samples not found at {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    samples = {k: data[k] for k in data.files}
    
    # 2. Load Panel
    panel_df = pd.read_csv(PANEL_PATH)
    season_df = panel_df[panel_df['season'] == season].copy()
    
    # Map pair_ids to names
    pair_ids = samples['pair_ids']
    pid_map = {pid: i for i, pid in enumerate(pair_ids)}
    
    # Create Name mapping (index -> name)
    # Use first occurrence of name for each pair
    pair_names = {}
    for pid in pair_ids:
        name = season_df[season_df['pair_id'] == pid]['celebrity_name'].iloc[0]
        pair_names[pid_map[pid]] = name
        
    names_list = [pair_names[i] for i in range(len(pair_ids))]
    
    return samples, season_df, names_list, pid_map

def plot_hidden_current(samples, names_list, output_path):
    """
    Figure 1: The Hidden Current (Latent Popularity v)
    Trajactories with 95% CI Ribbon
    """
    print("Generating Figure 1: The Hidden Current...")
    
    v_samples = samples['v'] # (S, T, N)
    weeks = samples['week_values']
    n_weeks = len(weeks)
    n_pairs = len(names_list)
    
    # Calculate stats
    v_mean = np.mean(v_samples, axis=0)
    v_low = np.percentile(v_samples, 2.5, axis=0)
    v_high = np.percentile(v_samples, 97.5, axis=0)
    
    # Determine top 5 by final week mean for coloring/highlighting
    final_means = v_mean[-1, :]
    top_indices = np.argsort(final_means)[-5:][::-1] # Top 5
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = sns.color_palette("husl", n_pairs)
    
    # Plot ribbons
    for i in range(n_pairs):
        is_top = i in top_indices
        alpha_ribbon = 0.2 if is_top else 0.05
        alpha_line = 1.0 if is_top else 0.3
        linewidth = 2.5 if is_top else 1.0
        label = names_list[i] if is_top else None
        
        # Color specific: if not top, make gray? 
        # Better: Color top 5 distinct, others gray.
        if is_top:
            color = colors[i]
            zorder = 10
        else:
            color = 'gray'
            zorder = 1
            
        ax.fill_between(weeks, v_low[:, i], v_high[:, i], color=color, alpha=alpha_ribbon, zorder=zorder)
        ax.plot(weeks, v_mean[:, i], color=color, label=label, linewidth=linewidth, zorder=zorder+1)
        
    ax.set_title(f"The Hidden Current: Latent Popularity Trajectories (Season {SEASON})", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Week", fontsize=14)
    ax.set_ylabel("Latent Vote Share (v)", fontsize=14)
    ax.legend(title="Top Contenders", loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    
    # Annotate "Uncertainty"
    # Find a wide ribbon and annotate
    
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig1_hidden_current.png'))
    print(f"Saved to {os.path.join(output_path, 'fig1_hidden_current.png')}")
    plt.close()

def plot_judge_vs_fan(samples, season_df, names_list, pid_map, output_path):
    """
    Figure 2: Judge vs Fan Dynamics (Quadrants)
    Average over season or per week trajectory?
    Let's do Season Average Position first for clarity, 
    maybe with arrows showing change from first half to second half.
    """
    print("Generating Figure 2: Judge vs Fan Dynamics...")
    
    v_samples = samples['v']
    v_mean_all = np.mean(v_samples, axis=0) # (T, N)
    
    # Compute Judge Ranks (from scores)
    weeks = samples['week_values']
    n_pairs = len(names_list)
    
    judge_ranks_avg = np.zeros(n_pairs)
    fan_ranks_avg = np.zeros(n_pairs)
    
    # Calculate average rank over weeks they were active
    rank_counts = np.zeros(n_pairs)
    
    for t, w in enumerate(weeks):
        # Fan Rank this week (based on v_mean)
        # Note: Rank 1 is best.
        # v is share, higher is better.
        # rankdata gives 1 for smallest. So we rank -v.
        from scipy.stats import rankdata
        f_ranks = rankdata(-v_mean_all[t, :])
        
        # Judge Rank
        # Get scores from season_df
        w_df = season_df[season_df['week'] == w]
        scores = np.zeros(n_pairs)
        mask = np.zeros(n_pairs, dtype=bool)
        
        for _, row in w_df.iterrows():
            if row['pair_id'] in pid_map:
                idx = pid_map[row['pair_id']]
                scores[idx] = row['S_it']
                mask[idx] = True
        
        # Rank scores (descending)
        # Only rank active
        active_scores = scores[mask]
        if len(active_scores) > 0:
            # Dense rank for judges usually? Or average? 
            # Let's use min rank (competetive)
            j_ranks_active = rankdata(-active_scores, method='min')
            
            # Map back
            curr_ptr = 0
            for i in range(n_pairs):
                if mask[i]:
                     judge_ranks_avg[i] += j_ranks_active[curr_ptr]
                     fan_ranks_avg[i] += f_ranks[i]
                     rank_counts[i] += 1
                     curr_ptr += 1
                     
    # Average
    mask_valid = rank_counts > 0
    judge_ranks_avg[mask_valid] /= rank_counts[mask_valid]
    fan_ranks_avg[mask_valid] /= rank_counts[mask_valid]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Quadrants
    mid = len(names_list) / 2
    ax.axhline(mid, color='black', alpha=0.3, linestyle='--')
    ax.axvline(mid, color='black', alpha=0.3, linestyle='--')
    
    # Quadrant Labels
    ax.text(1, 1, "Judge's Pet\n(High Judge, Low Fan)", ha='left', va='top', fontsize=12, color='gray')
    ax.text(len(names_list), 1, "Elite\n(High Both)", ha='right', va='top', fontsize=12, color='gray')
    ax.text(1, len(names_list), "At Risk\n(Low Both)", ha='left', va='bottom', fontsize=12, color='gray')
    ax.text(len(names_list), len(names_list), "People's Champion\n(Low Judge, High Fan)", ha='right', va='bottom', fontsize=12, color='gray')
    
    # Scatter
    # Note: Rank 1 is "High". So we invert axes or just know 1 is top/left.
    # Usually scatter plots: 0,0 is bottom left. 
    # But Ranks: 1 is top.
    # Let's Invert axes so Top-Right is Rank 1, 1 (Elite).
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    for i in range(n_pairs):
        if not mask_valid[i]: continue
        
        # Color by "Discrepancy" = Fan - Judge
        # If Fan(1) < Judge(10), Discrepancy = -9 (Good for fan)
        # Maybe color by category
        
        j = judge_ranks_avg[i]
        f = fan_ranks_avg[i]
        
        # Simple color
        ax.scatter(j, f, s=150, alpha=0.8, edgecolor='white', linewidth=1.5)
        
        # Label offset
        ax.text(j, f-0.2, names_list[i], ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("Judge Rank vs. Fan Rank (Season Average)", fontsize=16, fontweight='bold')
    ax.set_xlabel("Average Judge Rank (Lower is Better)", fontsize=12)
    ax.set_ylabel("Average Fan Rank (Lower is Better)", fontsize=12)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig2_judge_vs_fan.png'))
    print(f"Saved to {os.path.join(output_path, 'fig2_judge_vs_fan.png')}")
    plt.close()

def plot_risk_landscape(samples, names_list, output_path):
    """
    Figure 3: Risk Landscape (Badness Heatmap)
    Row: Contestant
    Col: Week
    Color: b (Badness)
    """
    print("Generating Figure 3: Risk Landscape...")
    
    b_samples = samples['b'] # (S, T, N)
    b_mean = np.mean(b_samples, axis=0) # (T, N)
    weeks = samples['week_values']
    
    # Prepare DataFrame for Heatmap
    # X: Week, Y: Name
    # We want matrix (N_names, N_weeks)
    heatmap_data = b_mean.T # (N, T)
    
    # Sort Names by Final Week Badness (or Final Rank if available)
    # Let's sort by Final Week Badness (Safe to Risky)
    # Lower Badness = Safer
    # The paper 'b' logic: High b -> High Elim Probability.
    
    final_b = heatmap_data[:, -1]
    # Filter out those already eliminated? They might have 0 or stationary b.
    # Let's sort by mean b? Or just use names_list order?
    # Sorting makes it look cleaner.
    sort_idx = np.argsort(final_b)
    
    heatmap_data_sorted = heatmap_data[sort_idx]
    names_sorted = [names_list[i] for i in sort_idx]
    
    # Create Mask for "Already Eliminated"
    # To do this correctly, we need to know when they were eliminated.
    # We can infer from samples['active_mask']? Not saved in npz directly, but part of loop.
    # Actually active_mask is likely constant across samples if not saved?
    # task1_runner doesn't save active_mask in npz.
    # But values of 'v' or 'b' might be conditioned.
    # In model, 'b' is calculated for all BUT masked?
    # task1_model: `b = jnp.stack(...) ... jnp.where(mask, r, 0.0)`
    # So if inactive, b=0? Or b is meaningless.
    # Typically badness is high for elim.
    # Let's check values.
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heatmap_data_sorted, cmap="RdYlBu_r", center=0, ax=ax, 
                xticklabels=weeks, yticklabels=names_sorted,
                cbar_kws={'label': 'Elimination Risk Score (b)'})
    
    ax.set_title("The Risk Landscape: Evolution of Elimination Probability", fontsize=16)
    ax.set_xlabel("Week")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'fig3_risk_landscape.png'))
    print(f"Saved to {os.path.join(output_path, 'fig3_risk_landscape.png')}")
    plt.close()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading data for Season {SEASON}...")
    try:
        samples, season_df, names_list, pid_map = load_data(SEASON)
        
        plot_hidden_current(samples, names_list, OUTPUT_DIR)
        plot_judge_vs_fan(samples, season_df, names_list, pid_map, OUTPUT_DIR)
        plot_risk_landscape(samples, names_list, OUTPUT_DIR)
        
        print("\nAll figures generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
