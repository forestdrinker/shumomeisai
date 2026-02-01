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

def fan_topq_survival_mean(v_samples, elim_weeks, q, method="week1"):
    """
    v_samples: (S, T, N)
    elim_weeks: (S, N)
    """
    # Robustness: Handle size mismatch
    n_v = v_samples.shape[0]
    n_e = elim_weeks.shape[0]
    n_min = min(n_v, n_e)
    
    if n_v != n_e:
        v_samples = v_samples[:n_min]
        elim_weeks = elim_weeks[:n_min]
        
    S, T, N = v_samples.shape
    out = np.zeros(S)
    
    for s in range(S):
        # Determine Top-q indices based on Fan Votes (v)
        score = v_samples[s, 0, :]  # Week 1 votes as prosy for initial popularity
        top_idx = np.argsort(-score)[:q]
        
        # Calculate mean survival weeks for these Top-q candidates
        out[s] = elim_weeks[s, top_idx].mean()
        
    return out.mean() # Mean across parallel universes

def main():
    print("Generating Fan Bias Violin Plot...")
    
    delta = {q: [] for q in Q_LIST}
    
    for season in SEASONS:
        # Paths
        post_path = os.path.join(POSTERIOR_DIR, f"season_{season}.npz")
        path_A = os.path.join(REPLAY_DIR, f"season_{season}_{RULE_A}.npz")
        path_B = os.path.join(REPLAY_DIR, f"season_{season}_{RULE_B}.npz")
        
        if not (os.path.exists(post_path) and os.path.exists(path_A) and os.path.exists(path_B)):
            continue
            
        t1 = load_npz(post_path)
        v_samples = t1["v"]
        
        A = load_npz(path_A)
        B = load_npz(path_B)
        
        elim_A = A["elim_weeks"]
        elim_B = B["elim_weeks"]
        
        for q in Q_LIST:
            # Calculate mean survival for Top-q under Rule A (Rank) and Rule B (Percent)
            mA = fan_topq_survival_mean(v_samples, elim_A, q)
            mB = fan_topq_survival_mean(v_samples, elim_B, q)
            
            # Delta = Percent - Rank
            # Positive means Percent Rule keeps fan favorites longer
            delta[q].append(mB - mA)
            
    # Visualize
    data = [delta[1], delta[3], delta[5]]
    labels = ["Top-1", "Top-3", "Top-5"]
    positions = [1, 2, 3]
    
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    # Violin plot (Distribution shape)
    parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
        
    # Box plot overlay (Stats)
    ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
               boxprops=dict(facecolor='white', color='black'),
               medianprops=dict(color='black', linewidth=1.5),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'),
               flierprops=dict(marker='o', markerfacecolor='black', markersize=3))
               
    # Reference line y=0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Δ Survival Weeks (Percent - Rank)", fontsize=11)
    ax.set_title("Across-season Distribution of Fan Survival Advantage\n(Positive = Percent Rule is Friendlier to Fans)", fontsize=12)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "delta_survival_violin_box.png")
    plt.savefig(out_path, dpi=300)
    print(f"Chart saved to {out_path}")

if __name__ == '__main__':
    main()
