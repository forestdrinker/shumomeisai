"""
Task 1 可视化脚本
生成论文所需的验证图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互模式
import os

# 配置
RESULTS_DIR = r'd:\shumomeisai\Code_second\validation_results'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\figures'


def plot_convergence_diagnostics(df_summary, output_dir):
    """
    图1: MCMC收敛诊断
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R-hat 分布
    ax1 = axes[0]
    seasons = df_summary['season']
    rhats = df_summary['max_rhat']
    
    colors = ['green' if r < 1.05 else ('orange' if r < 1.1 else 'red') for r in rhats]
    ax1.bar(seasons, rhats, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1.05, color='green', linestyle='--', label='Strict (1.05)')
    ax1.axhline(y=1.10, color='orange', linestyle='--', label='Acceptable (1.10)')
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Max R-hat', fontsize=12)
    ax1.set_title('(a) R-hat by Season', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0.95, max(1.2, rhats.max() * 1.1))
    
    # ESS 分布
    ax2 = axes[1]
    ess = df_summary['min_ess']
    
    colors = ['green' if e > 400 else ('orange' if e > 100 else 'red') for e in ess]
    ax2.bar(seasons, ess, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=400, color='green', linestyle='--', label='Ideal (400)')
    ax2.axhline(y=100, color='orange', linestyle='--', label='Minimum (100)')
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Min ESS', fontsize=12)
    ax2.set_title('(b) Effective Sample Size by Season', fontsize=14)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_convergence.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_convergence.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig_convergence.png/pdf")


def plot_prediction_performance(df_summary, output_dir):
    """
    图2: 预测性能
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    seasons = df_summary['season']
    
    # Accuracy
    ax1 = axes[0, 0]
    acc = df_summary['accuracy']
    ax1.bar(seasons, acc, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=acc.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {acc.mean():.3f}')
    ax1.axhline(y=0.1, color='gray', linestyle='--', label='Random (~10%)')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Top-1 Accuracy')
    ax1.set_title('(a) Elimination Prediction Accuracy')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Top-2 Accuracy
    ax2 = axes[0, 1]
    top2 = df_summary['top2_accuracy']
    ax2.bar(seasons, top2, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axhline(y=top2.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {top2.mean():.3f}')
    ax2.axhline(y=0.2, color='gray', linestyle='--', label='Random (~20%)')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Top-2 Accuracy')
    ax2.set_title('(b) Top-2 Elimination Accuracy')
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # Coverage
    ax3 = axes[1, 0]
    cov = df_summary['coverage_90']
    ax3.bar(seasons, cov, color='forestgreen', alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.90, color='red', linestyle='-', linewidth=2, label='Target: 0.90')
    ax3.axhline(y=cov.mean(), color='blue', linestyle='--', label=f'Mean: {cov.mean():.3f}')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('90% Coverage')
    ax3.set_title('(c) Credible Interval Coverage')
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # By Rule Type
    ax4 = axes[1, 1]
    rule_groups = df_summary.groupby('rule_segment').agg({
        'accuracy': 'mean',
        'top2_accuracy': 'mean',
        'coverage_90': 'mean'
    }).reset_index()
    
    x = np.arange(len(rule_groups))
    width = 0.25
    
    ax4.bar(x - width, rule_groups['accuracy'], width, label='Accuracy', color='steelblue')
    ax4.bar(x, rule_groups['top2_accuracy'], width, label='Top-2 Acc', color='darkorange')
    ax4.bar(x + width, rule_groups['coverage_90'], width, label='Coverage', color='forestgreen')
    
    ax4.set_xlabel('Rule Type')
    ax4.set_ylabel('Score')
    ax4.set_title('(d) Performance by Rule Type')
    ax4.set_xticks(x)
    ax4.set_xticklabels(rule_groups['rule_segment'])
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_prediction.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_prediction.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig_prediction.png/pdf")


def plot_calibration(df_preds, output_dir):
    """
    图3: Coverage 校准曲线
    """
    if df_preds.empty:
        print("  Skipped: No prediction data for calibration plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 计算不同覆盖率水平的实际覆盖
    # 这里简化处理，只展示 90% 的情况
    # 理想情况下需要重新计算不同阈值下的覆盖率
    
    observed_cov = df_preds['coverage_90'].mean()
    
    # 理想对角线
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # 实际点
    ax.scatter([0.9], [observed_cov], s=200, color='red', zorder=5, 
               label=f'90% CI: Observed={observed_cov:.3f}')
    
    ax.set_xlabel('Expected Coverage', fontsize=12)
    ax.set_ylabel('Observed Coverage', fontsize=12)
    ax.set_title('Posterior Calibration', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_calibration.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_calibration.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig_calibration.png/pdf")


def plot_uncertainty_by_week(df_preds, output_dir):
    """
    图4: 不确定性随周次变化
    """
    if df_preds.empty:
        print("  Skipped: No prediction data for uncertainty plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 按周聚合
    by_week = df_preds.groupby('week').agg({
        'entropy': ['mean', 'std'],
        'max_prob': ['mean', 'std'],
        'n_active': 'mean'
    }).reset_index()
    by_week.columns = ['week', 'entropy_mean', 'entropy_std', 'max_prob_mean', 'max_prob_std', 'n_active']
    
    # Entropy
    ax1 = axes[0]
    ax1.errorbar(by_week['week'], by_week['entropy_mean'], 
                 yerr=by_week['entropy_std'], fmt='o-', capsize=3, color='purple')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Prediction Entropy')
    ax1.set_title('(a) Uncertainty Over Weeks')
    ax1.grid(True, alpha=0.3)
    
    # Max Probability
    ax2 = axes[1]
    ax2.errorbar(by_week['week'], by_week['max_prob_mean'], 
                 yerr=by_week['max_prob_std'], fmt='s-', capsize=3, color='teal')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Max Predicted Probability')
    ax2.set_title('(b) Confidence Over Weeks')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig_uncertainty.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'fig_uncertainty.pdf'), bbox_inches='tight')
    plt.close()
    print("  Saved: fig_uncertainty.png/pdf")


def generate_latex_tables(df_summary, df_preds, output_dir):
    """
    生成 LaTeX 表格代码
    """
    lines = []
    
    # 表1: 收敛诊断
    lines.append("% Table: MCMC Convergence Summary")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{MCMC Convergence Diagnostics Summary}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Metric & All & Rank & Percent & Rank+Save \\\\")
    lines.append("\\hline")
    
    for rule in [None, 'rank', 'percent', 'rank_save']:
        if rule:
            subset = df_summary[df_summary['rule_segment'] == rule]
        else:
            subset = df_summary
        
        max_rhat = subset['max_rhat'].max()
        mean_rhat = subset['max_rhat'].mean()
        min_ess = subset['min_ess'].min()
        n_conv = (subset['max_rhat'] < 1.1).sum()
        n_total = len(subset)
        
        if rule is None:
            lines.append(f"Max R-hat & {max_rhat:.3f} & & & \\\\")
            lines.append(f"Mean R-hat & {mean_rhat:.3f} & & & \\\\")
            lines.append(f"Min ESS & {min_ess:.0f} & & & \\\\")
            lines.append(f"Converged (\\%) & {100*n_conv/n_total:.0f}\\% & & & \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    # 表2: 预测性能
    lines.append("% Table: Prediction Performance")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Elimination Prediction Performance}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append("Metric & Our Model & Random Baseline \\\\")
    lines.append("\\hline")
    
    mean_acc = df_summary['accuracy'].mean()
    mean_top2 = df_summary['top2_accuracy'].mean()
    mean_cov = df_summary['coverage_90'].mean()
    mean_brier = df_summary['mean_brier'].mean()
    
    lines.append(f"Top-1 Accuracy & {mean_acc:.1%} & $\\sim$10\\% \\\\")
    lines.append(f"Top-2 Accuracy & {mean_top2:.1%} & $\\sim$20\\% \\\\")
    lines.append(f"90\\% Coverage & {mean_cov:.1%} & -- \\\\")
    lines.append(f"Brier Score & {mean_brier:.3f} & -- \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # 保存
    with open(os.path.join(output_dir, 'latex_tables.tex'), 'w') as f:
        f.write('\n'.join(lines))
    
    print("  Saved: latex_tables.tex")


def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading validation results...")
    
    # 加载数据
    summary_path = os.path.join(RESULTS_DIR, 'season_summary.csv')
    preds_path = os.path.join(RESULTS_DIR, 'detailed_predictions.csv')
    
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found!")
        print("Please run task1_validation.py first.")
        return
    
    df_summary = pd.read_csv(summary_path)
    
    if os.path.exists(preds_path):
        df_preds = pd.read_csv(preds_path)
    else:
        df_preds = pd.DataFrame()
    
    print(f"Loaded {len(df_summary)} seasons, {len(df_preds)} prediction events")
    
    print("\nGenerating figures...")
    plot_convergence_diagnostics(df_summary, OUTPUT_DIR)
    plot_prediction_performance(df_summary, OUTPUT_DIR)
    plot_calibration(df_preds, OUTPUT_DIR)
    plot_uncertainty_by_week(df_preds, OUTPUT_DIR)
    
    print("\nGenerating LaTeX tables...")
    generate_latex_tables(df_summary, df_preds, OUTPUT_DIR)
    
    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
