"""
Task 3: 增强版可视化 (基于真实数据，优化视觉冲击力)
======================================================
优化策略:
1. 筛选式展示 - 只显示显著或大效应的舞伴/特征
2. 标准化处理 - 增强视觉对比度
3. 突出显著性 - 用视觉标记强调统计显著性
4. 优化配色 - 调整颜色映射范围
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import networkx as nx
import os
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = r'd:\shumomeisai\Code_second\Results\task3_analysis'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\task3\enhanced_output'

# 筛选阈值
PARTNER_EFFECT_THRESHOLD = 0.03  # 只显示 |效应| > 0.03 的舞伴
TOP_K_FEATURES = 10  # 只显示前10个重要特征
MIN_SEASONS = 3  # 网络图中只显示经验>=3季的舞伴

# Global Style
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'judge': '#2E86AB',
    'fan': '#E94F37',
    'positive': '#28A745',
    'negative': '#DC3545',
    'neutral': '#6C757D',
    'background': '#F8F9FA',
}

# =============================================================================
# DATA LOADING (复用之前的)
# =============================================================================

def load_real_data():
    """Load real analysis results from CSV files."""
    data = {}
    
    # Partner Effects
    f_j_partner = os.path.join(RESULTS_DIR, 'task3_lmm_judge_partner_effects.csv')
    if os.path.exists(f_j_partner):
        df_pj = pd.read_csv(f_j_partner)
        df_pj['ci_lower'] = df_pj['effect']
        df_pj['ci_upper'] = df_pj['effect']
        data['partner_judge'] = df_pj
    
    f_f_partner = os.path.join(RESULTS_DIR, 'task3_lmm_fan_partner_effects_aggregated.csv')
    if os.path.exists(f_f_partner):
        df_pf = pd.read_csv(f_f_partner, index_col=0).reset_index()
        df_pf.rename(columns={
            df_pf.columns[0]: 'partner',
            'mean': 'effect',
            '2.5%': 'ci_lower',
            '97.5%': 'ci_upper'
        }, inplace=True)
        data['partner_fan'] = df_pf
    
    # SHAP
    f_j_shap = os.path.join(RESULTS_DIR, 'task3_shap_ci_judge.csv')
    if os.path.exists(f_j_shap):
        data['shap_judge'] = pd.read_csv(f_j_shap)
        
    f_f_shap = os.path.join(RESULTS_DIR, 'task3_shap_ci_fan.csv')
    if os.path.exists(f_f_shap):
        data['shap_fan'] = pd.read_csv(f_f_shap)

    # Coefficients
    f_j_coeff = os.path.join(RESULTS_DIR, 'task3_lmm_judge_coeffs.csv')
    if os.path.exists(f_j_coeff):
        df_cj = pd.read_csv(f_j_coeff, index_col=0).reset_index()
        df_cj.rename(columns={'index': 'term'}, inplace=True)
        data['coeff_judge'] = df_cj
        
    f_f_coeff = os.path.join(RESULTS_DIR, 'task3_lmm_fan_coeffs_aggregated.csv')
    if os.path.exists(f_f_coeff):
        df_cf = pd.read_csv(f_f_coeff, index_col=0).reset_index()
        df_cf.rename(columns={'index': 'term'}, inplace=True)
        if 'mean' in df_cf.columns:
            df_cf.rename(columns={'mean': 'estimate'}, inplace=True)
        data['coeff_fan'] = df_cf

    # Panel Data
    if os.path.exists(PANEL_PATH):
        df_panel = pd.read_csv(PANEL_PATH)
        seasons_active = df_panel.groupby('ballroom_partner')['season'].nunique().to_dict()
        data['seasons_active'] = seasons_active
        
        if 'partner_fan' in data and not data['partner_fan'].empty:
            data['partners'] = data['partner_fan']['partner'].tolist()
    
    return data

# =============================================================================
# ENHANCED PLOTTING FUNCTIONS
# =============================================================================

def plot_caterpillar_enhanced(data, save_path):
    """增强版舞伴效应图 - 只显示大效应或显著的舞伴"""
    df_judge = data.get('partner_judge', pd.DataFrame()).copy()
    df_fan = data.get('partner_fan', pd.DataFrame()).copy()
    
    if df_judge.empty or df_fan.empty:
        print("Skipping Caterpillar: No data")
        return
    
    # 【优化1】筛选：只保留大效应或显著的舞伴
    df_fan['abs_effect'] = df_fan['effect'].abs()
    df_fan['is_significant'] = (df_fan['ci_lower'] > 0) | (df_fan['ci_upper'] < 0)
    
    df_fan_filtered = df_fan[
        (df_fan['abs_effect'] > PARTNER_EFFECT_THRESHOLD) | 
        df_fan['is_significant']
    ].copy()
    
    print(f"Partner Effects: 筛选后 {len(df_fan_filtered)}/{len(df_fan)} 舞伴")
    
    if len(df_fan_filtered) == 0:
        print("  Warning: No partners pass filter, lowering threshold...")
        df_fan_filtered = df_fan.nlargest(15, 'abs_effect')
    
    # 匹配Judge数据
    common_partners = df_fan_filtered['partner'].tolist()
    df_judge_filtered = df_judge[df_judge['partner'].isin(common_partners)]
    
    # 按Fan效应排序
    df_fan_filtered = df_fan_filtered.sort_values('effect', ascending=True).reset_index(drop=True)
    order = df_fan_filtered['partner'].tolist()
    df_judge_filtered = df_judge_filtered.set_index('partner').reindex(order).reset_index()
    
    # 【优化2】标准化效应值以增强视觉对比
    # 使用robust scaling (中位数和IQR)
    fan_median = df_fan_filtered['effect'].median()
    fan_iqr = df_fan_filtered['effect'].quantile(0.75) - df_fan_filtered['effect'].quantile(0.25)
    
    df_fan_filtered['effect_scaled'] = (df_fan_filtered['effect'] - fan_median) / (fan_iqr + 1e-6)
    df_fan_filtered['ci_lower_scaled'] = (df_fan_filtered['ci_lower'] - fan_median) / (fan_iqr + 1e-6)
    df_fan_filtered['ci_upper_scaled'] = (df_fan_filtered['ci_upper'] - fan_median) / (fan_iqr + 1e-6)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(order)*0.35)), sharey=True)
    fig.suptitle('Professional Partner Effects (Filtered & Enhanced)\n'
                 f'Showing {len(order)} partners with |Effect| > {PARTNER_EFFECT_THRESHOLD} or p<0.05',
                 fontsize=13, fontweight='bold')
    
    y_pos = np.arange(len(order))
    
    # Judge Model (原始值)
    ax = axes[0]
    for i, row in df_judge_filtered.iterrows():
        sig = (row['ci_lower'] > 0) or (row['ci_upper'] < 0)
        c = COLORS['positive'] if sig and row['effect'] > 0 else COLORS['negative'] if sig else COLORS['neutral']
        
        if row['ci_upper'] > row['ci_lower']:
            ax.hlines(y=i, xmin=row['ci_lower'], xmax=row['ci_upper'], color=c, lw=3 if sig else 2, alpha=0.8)
        ax.scatter(row['effect'], i, color=c, s=80 if sig else 50, zorder=5, edgecolor='white', linewidths=1.5)
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.4, lw=1)
    ax.set_title('Judge Score Model', fontweight='bold', color=COLORS['judge'], fontsize=12)
    ax.set_xlabel('Random Effect Estimate', fontsize=10)
    ax.grid(axis='x', alpha=0.3, ls=':')
    ax.set_facecolor(COLORS['background'])
    
    # Fan Model (标准化值)
    ax = axes[1]
    for i, row in df_fan_filtered.iterrows():
        sig = row['is_significant']
        c = COLORS['positive'] if sig and row['effect'] > 0 else COLORS['negative'] if sig else COLORS['neutral']
        
        ax.hlines(y=i, xmin=row['ci_lower_scaled'], xmax=row['ci_upper_scaled'], 
                 color=c, lw=3 if sig else 2, alpha=0.8)
        ax.scatter(row['effect_scaled'], i, color=c, s=80 if sig else 50, zorder=5, 
                  edgecolor='white', linewidths=1.5)
        
        # 标注显著性
        if sig:
            ax.text(row['ci_upper_scaled'] + 0.1, i, '*', fontsize=16, va='center', color=c, fontweight='bold')
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.4, lw=1)
    ax.set_title('Fan Vote Model (Standardized)', fontweight='bold', color=COLORS['fan'], fontsize=12)
    ax.set_xlabel('Standardized Effect (Robust Scaled)', fontsize=10)
    ax.grid(axis='x', alpha=0.3, ls=':')
    ax.set_facecolor(COLORS['background'])
    
    # Y-axis labels
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(order, fontsize=9)
    axes[0].set_ylabel('Professional Partner', fontsize=11)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['positive'], label='Significant Positive'),
        mpatches.Patch(color=COLORS['negative'], label='Significant Negative'),
        mpatches.Patch(color=COLORS['neutral'], label='Not Significant'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.03), fontsize=9, frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"✓ Enhanced Figure 1 saved: {save_path}")

def plot_shap_enhanced(data, save_path):
    """增强版SHAP图 - 只显示Top K特征"""
    shap_j = data.get('shap_judge', pd.DataFrame()).copy()
    shap_f = data.get('shap_fan', pd.DataFrame()).copy()
    
    if shap_j.empty or shap_f.empty:
        print("Skipping SHAP: No data")
        return
    
    # 【优化】只保留Top K特征
    top_features_j = set(shap_j.nlargest(TOP_K_FEATURES, 'mean_shap')['feature'])
    top_features_f = set(shap_f.nlargest(TOP_K_FEATURES, 'mean_shap')['feature'])
    top_features = sorted(top_features_j | top_features_f)
    
    shap_j_filtered = shap_j[shap_j['feature'].isin(top_features)]
    shap_f_filtered = shap_f[shap_f['feature'].isin(top_features)]
    
    print(f"SHAP Features: 筛选后 {len(top_features)} 个最重要特征")
    
    # 按平均重要性排序
    avg_importance = {}
    for feat in top_features:
        j_val = shap_j_filtered[shap_j_filtered['feature'] == feat]['mean_shap'].values
        f_val = shap_f_filtered[shap_f_filtered['feature'] == feat]['mean_shap'].values
        avg_importance[feat] = (j_val[0] if len(j_val) > 0 else 0) + (f_val[0] if len(f_val) > 0 else 0)
    
    order = sorted(top_features, key=lambda x: avg_importance[x])
    
    shap_j_sorted = shap_j_filtered.set_index('feature').reindex(order).reset_index()
    shap_f_sorted = shap_f_filtered.set_index('feature').reindex(order).reset_index()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, len(order)*0.5 + 2))
    
    y_pos = np.arange(len(order))
    h = 0.35
    
    bars_j = ax.barh(y_pos - h/2, shap_j_sorted['mean_shap'], h, 
                     label='Judge Model', color=COLORS['judge'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_f = ax.barh(y_pos + h/2, shap_f_sorted['mean_shap'], h, 
                     label='Fan Model', color=COLORS['fan'], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Error bars
    ax.errorbar(shap_j_sorted['mean_shap'], y_pos - h/2,
                xerr=[shap_j_sorted['mean_shap'] - shap_j_sorted['q2.5'], 
                      shap_j_sorted['q97.5'] - shap_j_sorted['mean_shap']],
                fmt='none', ecolor='black', capsize=2, alpha=0.6, lw=1)
    ax.errorbar(shap_f_sorted['mean_shap'], y_pos + h/2,
                xerr=[shap_f_sorted['mean_shap'] - shap_f_sorted['q2.5'], 
                      shap_f_sorted['q97.5'] - shap_f_sorted['mean_shap']],
                fmt='none', ecolor='black', capsize=2, alpha=0.6, lw=1)
    
    # 美化特征名
    feature_labels = {
        'rolling_avg_pJ': 'Rolling Avg Judge Score',
        'age': 'Celebrity Age',
        'partner_enc': 'Partner Encoding',
        'partner_embedding_1': 'Partner Embed Dim 1',
        'partner_embedding_2': 'Partner Embed Dim 2',
        'week_norm': 'Week Progress',
    }
    
    labels = [feature_labels.get(f, f) for f in order]
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=11)
    ax.set_title(f'Top {TOP_K_FEATURES} Feature Importance (SHAP)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=False)
    ax.grid(axis='x', alpha=0.3, ls=':')
    ax.set_facecolor(COLORS['background'])
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"✓ Enhanced Figure 2 saved: {save_path}")

def plot_heatmap_enhanced(data, save_path):
    """增强版系数热力图 - 优化配色和显著性标记"""
    cj = data.get('coeff_judge')
    cf = data.get('coeff_fan')
    
    if cj is None or cf is None or cj.empty or cf.empty:
        print("Skipping Heatmap: No data")
        return
    
    # 筛选行业系数 (排除Intercept和Var)
    terms = sorted([t for t in cj['term'] if t in cf['term'].values 
                   and 'Intercept' not in t and 'Var' not in t and 'C(industry)' in t])[:15]  # Top 15
    
    print(f"Coefficient Heatmap: {len(terms)} industry terms")
    
    if len(terms) == 0:
        print("  Warning: No industry terms found")
        return
    
    mat = np.zeros((len(terms), 2))
    sigs = np.zeros((len(terms), 2), dtype=bool)
    
    for i, t in enumerate(terms):
        subset_j = cj[cj['term'] == t]
        subset_f = cf[cf['term'] == t]
        if subset_j.empty or subset_f.empty:
            continue
            
        rj = subset_j.iloc[0]
        rf = subset_f.iloc[0]
        
        mat[i,0] = rj['estimate']
        sigs[i,0] = (rj['2.5%'] > 0) or (rj['97.5%'] < 0)
        
        mat[i,1] = rf['estimate']
        sigs[i,1] = (rf['2.5%'] > 0) or (rf['97.5%'] < 0)
    
    # 美化term名称
    term_labels = [t.replace('C(industry)[T.', '').replace(']', '') for t in terms]
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, len(terms)*0.45 + 2))
    
    # 【优化】使用Two-Slope Norm使0值居中
    vmax = np.max(np.abs(mat))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    im = ax.imshow(mat, cmap='RdBu_r', norm=norm, aspect='auto')
    
    # Text annotations
    for i in range(len(terms)):
        for j in range(2):
            val = mat[i,j]
            sig = sigs[i,j]
            txt = f"{val:.2f}"
            if sig:
                txt = f"**{txt}**"  # 模拟粗体
            color = 'white' if abs(val) > vmax*0.6 else 'black'
            ax.text(j, i, txt, ha='center', va='center', 
                   fontsize=9, color=color, fontweight='bold' if sig else 'normal')
    
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Judge Model', 'Fan Model'], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(term_labels, fontsize=10)
    
    ax.set_title('Industry Fixed Effects Comparison\\n(** = p<0.05)', 
                fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coefficient Estimate', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"✓ Enhanced Figure 3 saved: {save_path}")

def plot_network_enhanced(data, save_path):
    """增强版网络图 - 只显示经验丰富的舞伴"""
    if 'partner_fan' not in data or 'seasons_active' not in data:
        print("Skipping Network: Missing data")
        return
    
    df = data['partner_fan']
    effects = dict(zip(df['partner'], df['effect']))
    exp = data['seasons_active']
    
    # 【优化】只显示经验>=MIN_SEASONS的舞伴
    partners = [p for p in effects.keys() if p in exp and exp[p] >= MIN_SEASONS]
    
    print(f"Network: {len(partners)} partners with exp >= {MIN_SEASONS} seasons")
    
    if len(partners) == 0:
        print("  Warning: No experienced partners, showing all")
        partners = list(effects.keys())[:30]
    
    G = nx.Graph()
    for p in partners:
        G.add_node(p, size=exp.get(p,1)*60, color=effects.get(p,0))
    
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    
    fig, ax = plt.subplots(figsize=(14,12))
    
    sizes = [G.nodes[n]['size'] for n in G.nodes()]
    colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    # 【优化】使用标准化的颜色映射
    colors_scaled = [(c - np.mean(colors)) / (np.std(colors) + 1e-6) for c in colors]
    
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors_scaled, 
                          cmap='RdBu_r', vmin=-2, vmax=2, ax=ax, edgecolors='black', linewidths=1.5)
    
    # 标注Top效应舞伴
    top_partners = sorted(partners, key=lambda x: abs(effects[x]), reverse=True)[:12]
    labels = {n: n.split()[-1] for n in top_partners}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10, font_weight='bold')
    
    ax.axis('off')
    ax.set_title(f'Professional Partner Network (≥{MIN_SEASONS} Seasons Experience)\\n'
                f'Node Size = Experience, Color = Effect on Fan Votes (Standardized)',
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"✓ Enhanced Figure 4 saved: {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("Task 3 增强版可视化 (Filtered + Standardized)")
    print("="*70 + "\n")
    
    data = load_real_data()
    
    plot_caterpillar_enhanced(data, os.path.join(OUTPUT_DIR, 'fig1_partner_effects_enhanced.png'))
    plot_shap_enhanced(data, os.path.join(OUTPUT_DIR, 'fig2_shap_enhanced.png'))
    plot_heatmap_enhanced(data, os.path.join(OUTPUT_DIR, 'fig3_coeffs_enhanced.png'))
    plot_network_enhanced(data, os.path.join(OUTPUT_DIR, 'fig4_network_enhanced.png'))
    
    print("\n" + "="*70)
    print("完成！所有增强图表已保存至:")
    print(f"  {OUTPUT_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()
