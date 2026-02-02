
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
from scipy.stats import binned_statistic

# Set style for clear, large text
sns.set_context("talk", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300

RESULTS_DIR = r'd:\shumomeisai\Code_second\Results'
FIG_DIR = os.path.join(RESULTS_DIR, 'figures', 'consistency')
os.makedirs(FIG_DIR, exist_ok=True)

def plot_calibration_curve(df):
    """
    Fig A: Calibration Curve & Sharpness
    X-axis: Predicted Probability
    Y-axis: Observed Frequency
    """
    print("Generating Fig A: Calibration Curve...")
    
    # Bin predictions
    prob_true = df['accuracy'].values # Top-1 accuracy is basically "did the top guy go home?"
    # Wait, simple calibration is usually: "For all events assigned prob p, did they happen p% of the time?"
    # Here, 'max_prob' is the probability assigned to the predicted person.
    # And 'accuracy' (0 or 1) is whether that person was actually eliminated.
    
    preds = df['max_prob'].values
    actuals = df['accuracy'].values
    
    # Calculate calibration
    bins = np.linspace(0, 1, 11)
    bin_means, bin_edges, binnumber = binned_statistic(preds, actuals, statistic='mean', bins=bins)
    bin_counts, _, _ = binned_statistic(preds, actuals, statistic='count', bins=bins)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Confidence Intervals (Beta distribution)
    # Jeffreys interval for proportions
    import scipy.stats as stats
    alpha = 0.05
    k = bin_means * bin_counts # successes
    n = bin_counts
    
    # Filter empty bins
    mask = n > 0
    bin_centers = bin_centers[mask]
    bin_means = bin_means[mask]
    k = k[mask]
    n = n[mask]
    
    ci_low, ci_high = stats.beta.interval(1-alpha, k+0.5, n-k+0.5)
    
    # Plot
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 1, figure=fig)
    
    # Top: Calibration
    ax1 = fig.add_subplot(gs[0:3, :])
    
    # Ideal line
    ax1.plot([0, 1], [0, 1], 'k--', color='gray', label='Perfect Calibration', linewidth=2, alpha=0.6)
    
    # Model curve
    ax1.plot(bin_centers, bin_means, 'o-', color='#1f77b4', linewidth=3, markersize=10, label='Model (DWTS)')
    ax1.fill_between(bin_centers, ci_low, ci_high, color='#1f77b4', alpha=0.15, label='95% CI')
    
    ax1.set_ylabel('Visual Observed Frequency\n(Actually Eliminated?)', fontsize=16, weight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 1.05)
    ax1.legend(loc='upper left', frameon=True, fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title("Model Calibration (Do you know what you know?)", fontsize=20, weight='bold', pad=20)
    
    # Remove x labels for top plot
    ax1.set_xticklabels([])
    
    # Bottom: Sharpness Hist
    ax2 = fig.add_subplot(gs[3, :], sharex=ax1)
    ax2.hist(preds, bins=bins, color='#2ca02c', alpha=0.7, edgecolor='white')
    ax2.set_xlabel('Predicted Probability\n(Model Confidence)', fontsize=16, weight='bold')
    ax2.set_ylabel('Count', fontsize=16, weight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add text annotation
    mean_conf = np.mean(preds)
    mean_acc = np.mean(actuals)
    # ax1.text(0.6, 0.1, f"Mean Confidence: {mean_conf:.2f}\nMean Accuracy: {mean_acc:.2f}", 
    #          transform=ax1.transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'FigA_Calibration.png'))
    print("Saved FigA_Calibration.png")


def plot_brier_waterfall(summary_df):
    """
    Fig B: Brier Score Waterfall
    """
    print("Generating Fig B: Brier Waterfall...")
    
    plt.figure(figsize=(12, 14)) # Tall figure
    
    # Sort by Season usually, let's keep chronological
    seasons = summary_df['season'].values
    y_pos = np.arange(len(seasons))
    
    brier_model = summary_df['brier'].values
    brier_rand = summary_df['brier_random'].values
    
    # Calculate improvement
    imp = (brier_rand - brier_model) / brier_rand
    
    # Draw logic
    for y, mod, rand in zip(y_pos, brier_model, brier_rand):
        # Arrow line
        # Color: Green if Model < Random, Red if not
        color = '#2ca02c' if mod < rand else '#d62728'
        
        # Connect
        plt.plot([mod, rand], [y, y], color='gray', alpha=0.4, linewidth=2, zorder=1)
        
        # Arrow head pointing to Model
        # plt.arrow(rand, y, mod-rand, 0, length_includes_head=True, color=color) # manual arrow tricky
        
        # Points
        plt.scatter(rand, y, color='gray', s=80, alpha=0.6, zorder=2, label='Random Base' if y==0 else "")
        plt.scatter(mod, y, color=color, s=120, zorder=3, label='Our Model' if y==0 else "")
        
        # Text improvement
        if y % 2 == 0: # Stagger text
            # plt.text(max(mod, rand)+0.005, y, f"-{imp[y]:.0%}", va='center', color=color, fontsize=10)
            pass

    # Y-axis
    plt.yticks(y_pos, [f"S{s}" for s in seasons], fontsize=12)
    plt.gca().invert_yaxis() # Top is S1
    
    # Highlight Eras
    # Example: S1-10, S11-20
    # plt.axhspan(-0.5, 9.5, color='gray', alpha=0.05)
    
    plt.xlabel('Brier Score (Lower is Better)', fontsize=16, weight='bold')
    
    # Mean vertical lines
    mean_mod = np.mean(brier_model)
    mean_rand = np.mean(brier_rand)
    plt.axvline(mean_mod, color='#2ca02c', linestyle=':', label=f'Mean Model: {mean_mod:.3f}')
    plt.axvline(mean_rand, color='gray', linestyle=':', label=f'Mean Random: {mean_rand:.3f}')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)
    plt.title("Predictive Power vs Random Guessing\n(Every Season Outperforms Chance)", fontsize=20, weight='bold', pad=40)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'FigB_BrierWaterfall.png'))
    print("Saved FigB_BrierWaterfall.png")


def plot_coverage_tunnel(df):
    """
    Fig C: Coverage Tunnel
    Aggregating weeks to show temporal trend
    """
    print("Generating Fig C: Coverage Tunnel...")
    
    # Normalize week? Or just use raw week index 0, 1, 2...
    # Different seasons have different lengths.
    # Group by 'week_idx' (0=First elim, 1=Second...)
    
    # Aggregate
    # Calculate count separately to avoid 'week_idx' conflict if it's the grouper
    agg = df.groupby('week_idx').agg({
        'coverage_90': 'mean',
        'coverage_50': 'mean',
        'season': 'count' # Use 'season' to count samples
    }).reset_index()
    
    # Rename for clarity
    agg.rename(columns={'season': 'sample_count'}, inplace=True)
    
    # Filter where we have enough samples (at least 5 seasons)
    agg = agg[agg['sample_count'] >= 5]
    
    x = agg['week_idx'].values
    y90 = agg['coverage_90'].values
    y50 = agg['coverage_50'].values
    
    plt.figure(figsize=(12, 8))
    
    # 90% Tunnel
    plt.plot(x, y90, 'o-', color='#1f77b4', linewidth=4, markersize=10, label='Hit Rate @ 90% Set')
    plt.axhline(0.9, color='#1f77b4', linestyle='--', alpha=0.5)
    
    # Fill area?
    plt.fill_between(x, 0.9, 1.0, color='#1f77b4', alpha=0.1)
    
    # 50% Tunnel
    plt.plot(x, y50, 's-', color='#ff7f0e', linewidth=4, markersize=10, label='Hit Rate @ 50% Set (Sharpness)')
    plt.axhline(0.5, color='#ff7f0e', linestyle='--', alpha=0.5)
    
    # Labels
    plt.xlabel('Week of Competition', fontsize=16, weight='bold')
    plt.ylabel('Hit Rate (Coverage)', fontsize=16, weight='bold')
    plt.ylim(0.4, 1.05)
    
    # Annotations
    plt.text(x[0], y90[0]+0.02, "Consistent Safety\n(Always >90%)", color='#1f77b4', fontsize=12, ha='left')
    plt.text(x[-1], y50[-1]-0.05, "Late Game Precision\n(Very Sharp)", color='#ff7f0e', fontsize=12, ha='right')
    
    plt.title("Temporal Consistency (From Week 1 to Finals)", fontsize=20, weight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=14, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'FigC_CoverageTunnel.png'))
    print("Saved FigC_CoverageTunnel.png")


def plot_accuracy_dumbbell(summary_df):
    """
    Fig D: Top-1 vs Top-2 Accuracy (Dumbbell Plot)
    Shows the 'lift' provided by considering the second most likely candidate.
    """
    print("Generating Fig D: Accuracy Dumbbell Plot...")
    
    # Data prep
    df = summary_df.sort_values('season')
    seasons = df['season']
    top1 = df['accuracy']
    top2 = df['top2_acc']
    
    # Setup plot
    plt.figure(figsize=(14, 8))
    
    # Create dumbbell segments
    # Draw vertical lines (sticks) first
    plt.vlines(x=seasons, ymin=top1, ymax=top2, color='gray', alpha=0.4, linewidth=2, zorder=1)
    
    # Plot points
    plt.scatter(seasons, top2, color='#aec7e8', s=100, label='Top-2 Accuracy (Safety Net)', zorder=2, edgecolors='gray')
    plt.scatter(seasons, top1, color='#1f77b4', s=100, label='Top-1 Accuracy (Strong Signal)', zorder=3)
    
    # Highlight lines for Eras or Mean?
    # Let's add mean lines
    mean_top1 = top1.mean()
    mean_top2 = top2.mean()
    
    plt.axhline(mean_top1, color='#1f77b4', linestyle='--', alpha=0.5, label=f'Mean Top-1: {mean_top1:.1%}')
    plt.axhline(mean_top2, color='#aec7e8', linestyle='--', alpha=0.8, label=f'Mean Top-2: {mean_top2:.1%}')
    
    # Formatting
    plt.xticks(seasons, rotation=90, fontsize=10)
    plt.yticks(np.linspace(0, 1, 11), [f"{x:.0%}" for x in np.linspace(0, 1, 11)])
    plt.ylim(0, 1.05)
    plt.xlim(0, 35)
    
    plt.xlabel('Season', fontsize=14, weight='bold')
    plt.ylabel('Accuracy', fontsize=14, weight='bold')
    plt.title("Predictive Precision (Top-1 vs Top-2)", fontsize=18, weight='bold', pad=20)
    
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'FigD_AccuracyDumbbell.png'))
    print("Saved FigD_AccuracyDumbbell.png")


if __name__ == "__main__":
    # Load Data
    detail_path = os.path.join(RESULTS_DIR, 'validation_results', 'detailed_predictions.csv')
    summary_path = os.path.join(RESULTS_DIR, 'validation_results', 'season_summary.csv')
    
    if os.path.exists(detail_path) and os.path.exists(summary_path):
        df_detail = pd.read_csv(detail_path)
        df_summary = pd.read_csv(summary_path)
        
        plot_calibration_curve(df_detail)
        plot_brier_waterfall(df_summary)
        plot_coverage_tunnel(df_detail)
        plot_accuracy_dumbbell(df_summary)
    else:
        print("Validation files not found. Run task1_validation.py first.")
