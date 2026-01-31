
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("poster", font_scale=0.8)
plt.rcParams['font.family'] = 'DejaVu Sans' # Safe font

RESULTS_DIR = r'd:\shumomeisai\Code_second\Results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def plot_task1():
    print("Plotting Task 1...")
    # Load Data
    val_path = os.path.join(RESULTS_DIR, 'validation_results', 'detailed_predictions.csv')
    if not os.path.exists(val_path):
        print("Task 1 data not found, skipping.")
        return
    
    df = pd.read_csv(val_path)
    
    # 1. Calibration Plot (Accuracy vs Confidence)
    # We use 'max_prob' (confidence of prediction) and 'accuracy' (correctness)
    plt.figure(figsize=(8, 6))
    
    # Create bins
    bins = np.linspace(0, 1, 11)
    df['conf_bin'] = pd.cut(df['max_prob'], bins=bins, labels=bins[1:])
    
    calibration = df.groupby('conf_bin', observed=False).agg({
        'max_prob': 'mean',
        'accuracy': 'mean',
        'season': 'count' # Support
    }).reset_index()
    
    sns.lineplot(data=calibration, x='max_prob', y='accuracy', marker='o', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.title('Task 1: Predictive Calibration\n(Accuracy vs Confidence)')
    plt.xlabel('Predicted Confidence (Max Prob)')
    plt.ylabel('Observed Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_calibration_curve.png'))
    plt.close()
    
    # Histogram of Probabilities
    plt.figure(figsize=(8, 6))
    sns.histplot(df['max_prob'], bins=20, kde=True)
    plt.title('Task 1: Prediction Confidence Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_confidence_hist.png'))
    plt.close()
    
    # 2. Accuracy over Weeks
    # We have 'week' and 'accuracy' columns directly
    acc_by_week = df.groupby('week')['accuracy'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=acc_by_week, x='week', y='accuracy', marker='o')
    plt.title('Task 1: Prediction Accuracy by Week')
    plt.ylabel('Top-1 Accuracy')
    plt.xlabel('Week of Season')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_accuracy_trend.png'))
    plt.close()

def plot_task2():
    print("Plotting Task 2...")
    metrics_path = os.path.join(RESULTS_DIR, 'task2_metrics.csv')
    if not os.path.exists(metrics_path):
        print("Task 2 metrics not found.")
        return
        
    df = pd.read_csv(metrics_path)
    
    # 1. Rule Comparison: Rho_F vs Rho_J
    agg = df.groupby('rule').agg({
        'rho_F': 'mean',
        'rho_J': 'mean',
        'upset_rate': 'mean'
    }).reset_index()
    
    melted = agg.melt(id_vars='rule', value_vars=['rho_F', 'rho_J'], var_name='Metric', value_name='Correlation')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x='rule', y='Correlation', hue='Metric', palette='viridis')
    plt.title('Task 2: Fairness Metrics by Rule\n(Judge vs Fan Alignment)')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task2_fairness_comparison.png'))
    plt.close()
    
    # 2. Upset Rate Comparison
    plt.figure(figsize=(8, 6))
    sns.barplot(data=agg, x='rule', y='upset_rate', palette='magma')
    plt.title('Task 2: Stability Analysis (Upset Rate)')
    plt.ylabel('Frequency of Upsets')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task2_upset_rate.png'))
    plt.close()
    
    # 3. Controversy Mockup (Reading Summary)
    summ_path = os.path.join(RESULTS_DIR, 'controversy_analysis_summary.csv')
    if os.path.exists(summ_path):
        summ = pd.read_csv(summ_path)
        # Parse 'rank_result' e.g. "2nd (p=0.46)" -> 2.0
        def parse_rank(s):
            try:
                if pd.isna(s): return np.nan
                return float(s.split('st')[0].split('nd')[0].split('rd')[0].split('th')[0])
            except:
                return np.nan
                
        plot_data = []
        for _, row in summ.iterrows():
            name = row['celebrity']
            # Plot Rank vs Percent (most contrast)
            r = parse_rank(row['rank_result'])
            p = parse_rank(row['percent_result'])
            if not np.isnan(r) and not np.isnan(p):
                plot_data.append({'Name': name, 'Rule': 'Rank', 'Rank': r})
                plot_data.append({'Name': name, 'Rule': 'Percent', 'Rank': p})
            
        if plot_data:
            p_df = pd.DataFrame(plot_data)
            plt.figure(figsize=(10, 6))
            sns.pointplot(data=p_df, x='Rule', y='Rank', hue='Name', markers='o')
            plt.gca().invert_yaxis()
            plt.title('Task 2: Controversy Slope Graph\nImpact of Rule (Rank vs Percent)')
            plt.ylabel('Expected Rank (Lower is Better)')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'task2_controversy_slope.png'))
            plt.close()

def plot_task3():
    print("Plotting Task 3...")
    # 1. LMM Coefficients
    lmm_path = os.path.join(RESULTS_DIR, 'task3_analysis', 'task3_lmm_fan_coeffs_aggregated.csv')
    if os.path.exists(lmm_path):
        df = pd.read_csv(lmm_path, index_col=0)
        # Filter for Industry terms
        ind_df = df[df.index.str.contains('industry')]
        # Clean index names
        ind_df = ind_df.copy()
        ind_df['Industry'] = ind_df.index.str.replace('C(industry)[T.', '', regex=False).str.replace(']', '', regex=False)
        ind_df = ind_df.sort_values('mean')
        
        plt.figure(figsize=(10, 8))
        # Errors
        yerr = [ind_df['mean'] - ind_df['2.5%'], ind_df['97.5%'] - ind_df['mean']]
        
        plt.errorbar(x=ind_df['mean'], y=ind_df['Industry'], xerr=yerr, fmt='o', color='b', capsize=5)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Task 3: Industry Fixed Effects on Fan Vote')
        plt.xlabel('Coefficient Value (95% CI)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'task3_lmm_industry_forest.png'))
        plt.close()

    # 2. Partner Effects
    pe_path = os.path.join(RESULTS_DIR, 'task3_analysis', 'task3_lmm_fan_partner_effects_aggregated.csv')
    if os.path.exists(pe_path):
        df = pd.read_csv(pe_path, index_col=0)
        df_sorted = df.sort_values('mean', ascending=False)
        top10 = df_sorted.head(10)
        bottom10 = df_sorted.tail(5) # Take bottom 5 to fit
        plot_df = pd.concat([top10, bottom10])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=plot_df['mean'], y=plot_df.index, palette='coolwarm')
        plt.axvline(x=0, color='k', linewidth=1)
        plt.title('Task 3: Top & Bottom Partner Effects (BLUPs)')
        plt.xlabel('Mean Effect on Vote Share')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'task3_partner_effects.png'))
        plt.close()

    # 3. SHAP
    shap_path = os.path.join(RESULTS_DIR, 'task3_analysis', 'task3_shap_ci_fan.csv')
    if os.path.exists(shap_path):
        df = pd.read_csv(shap_path)
        df = df.sort_values('mean_shap', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='mean_shap', y='feature', color='teal')
        plt.title('Task 3: Feature Importance (SHAP)')
        plt.xlabel('Mean |SHAP| Value')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'task3_shap_importance.png'))
        plt.close()

if __name__ == "__main__":
    plot_task1()
    plot_task2()
    plot_task3()
