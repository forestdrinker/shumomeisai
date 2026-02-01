import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = r'd:\shumomeisai\Code_second\Results'
POSTERIOR_DIR = os.path.join(DATA_DIR, 'posterior_samples')
REPLAY_DIR = os.path.join(DATA_DIR, 'replay_results')
OUTPUT_DIR = os.path.join(DATA_DIR, 'audit_figures')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

SEASONS = list(range(1, 35))
RULE_A = "rank"
RULE_B = "percent"
Q_LIST = [1, 3, 5]

def load_npz(path):
    return np.load(path, allow_pickle=True)

def fan_topq_survival_mean(v_samples, elim_weeks, q, method="week1", weights=None):
    """
    v_samples: (S, T, N)
    elim_weeks: (S, N)
    return: scalar mean over universes
    """
    # Robustness: Handle size mismatch by slicing to min size
    n_v = v_samples.shape[0]
    n_e = elim_weeks.shape[0]
    n_min = min(n_v, n_e)
    
    if n_v != n_e:
        # print(f"  Note: Resizing samples {n_v} vs {n_e} -> {n_min}")
        v_samples = v_samples[:n_min]
        elim_weeks = elim_weeks[:n_min]
    
    S, T, N = v_samples.shape
    out = np.zeros(S)
    
    for s in range(S):
        v = v_samples[s]  # (T, N)
        if method == "week1":
            score = v[0]   # Week 1 votes
        elif method == "avg":
            # Simple avg if weights not provided
            score = np.mean(v, axis=0) 
        
        # Identify Top-q Index in this universe
        # Higher score = Better
        top_idx = np.argsort(-score)[:q]
        
        # Check survival of these candidates
        out[s] = elim_weeks[s, top_idx].mean()
        
    return out.mean()

def overall_survival_mean(elim_weeks):
    return elim_weeks.mean()

def main():
    print("Generating Fan Bias Delta Chart...")
    
    # Store delta values
    delta = {q: [] for q in Q_LIST}
    delta["avg"] = []
    valid_seasons = []

    for season in SEASONS:
        # Paths
        post_path = os.path.join(POSTERIOR_DIR, f"season_{season}.npz")
        path_A = os.path.join(REPLAY_DIR, f"season_{season}_{RULE_A}.npz")
        path_B = os.path.join(REPLAY_DIR, f"season_{season}_{RULE_B}.npz")
        
        # Check existence
        if not (os.path.exists(post_path) and os.path.exists(path_A) and os.path.exists(path_B)):
            # print(f"Skipping Season {season} (Missing files)")
            continue
            
        valid_seasons.append(season)
        
        # Load
        t1 = load_npz(post_path)
        v_samples = t1["v"] # v or v_samples
        
        A = load_npz(path_A)
        B = load_npz(path_B)
        
        elim_A = A["elim_weeks"]
        elim_B = B["elim_weeks"]
        
        # Top-q Calculation
        for q in Q_LIST:
            mA = fan_topq_survival_mean(v_samples, elim_A, q, method="week1")
            mB = fan_topq_survival_mean(v_samples, elim_B, q, method="week1")
            delta[q].append(mB - mA) # Positive = B (Percent) is better for fans
            
        # Avg Calculation
        delta["avg"].append(overall_survival_mean(elim_B) - overall_survival_mean(elim_A))
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Colors
    colors = {1: '#d62728', 3: '#ff7f0e', 5: '#2ca02c', 'avg': '#7f7f7f'}
    markers = {1: 'o', 3: '^', 5: 's', 'avg': 'x'}
    
    for q in Q_LIST:
        plt.plot(valid_seasons, delta[q], marker=markers[q], markersize=6, 
                 linewidth=1.5, label=f"Top-{q} (Fans Avg)", color=colors[q])
                 
    plt.plot(valid_seasons, delta["avg"], marker=markers['avg'], markersize=5, 
             linewidth=1, linestyle='--', label="Avg (All Contestants)", color=colors['avg'], alpha=0.7)
    
    plt.axhline(0, color='black', linewidth=1, linestyle='-')
    
    plt.title('Fan Bias Signal: Percent Rule vs Rank Rule\n(Δ Survival Weeks: Positive = Percent Rule Favors Fans)', fontsize=14)
    plt.xlabel('Season')
    plt.ylabel('Δ Survival Weeks (Percent - Rank)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = os.path.join(OUTPUT_DIR, "fan_bias_delta_topq_line.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_file}")

if __name__ == '__main__':
    main()
