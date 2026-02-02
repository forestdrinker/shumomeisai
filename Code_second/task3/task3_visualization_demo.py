"""
Task 3: Attribution Analysis - 五个高级可视化图表演示
======================================================
图表1: 舞伴效应对比图 (Caterpillar Plot with CI)
图表2: SHAP重要性对比条形图 (Dual Bar with CI)
图表3: 固定效应系数比较热力图 (Coefficient Heatmap)
图表4: 舞伴网络可视化 (Network Graph with Effect Size)
图表5: 不确定性传播瀑布图 (Uncertainty Waterfall)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和全局样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 定义配色方案
COLORS = {
    'judge': '#2E86AB',      # 蓝色 - 评委
    'fan': '#E94F37',        # 红橙色 - 粉丝
    'positive': '#28A745',   # 绿色 - 正效应
    'negative': '#DC3545',   # 红色 - 负效应
    'neutral': '#6C757D',    # 灰色 - 不显著
    'background': '#F8F9FA',
    'grid': '#DEE2E6'
}

# =============================================================================
# 模拟数据生成 (实际使用时替换为真实数据)
# =============================================================================

def generate_mock_data():
    """生成模拟数据用于演示"""
    np.random.seed(2026)
    
    # 舞伴名单 (真实DWTS舞伴)
    partners = [
        'Derek Hough', 'Mark Ballas', 'Maksim Chmerkovskiy', 'Val Chmerkovskiy',
        'Tony Dovolani', 'Cheryl Burke', 'Kym Johnson', 'Julianne Hough',
        'Witney Carson', 'Lindsay Arnold', 'Sharna Burgess', 'Peta Murgatroyd',
        'Emma Slater', 'Jenna Johnson', 'Alan Bersten', 'Gleb Savchenko',
        'Artem Chigvintsev', 'Sasha Farber', 'Brandon Armstrong', 'Daniella Karagach'
    ]
    
    # 1. 舞伴效应数据
    n_partners = len(partners)
    partner_effects_judge = {
        'partner': partners,
        'effect': np.random.randn(n_partners) * 0.8 + np.array([1.2, 0.9, 0.7, 0.8, 0.3, 0.6, 0.4, 1.0, 0.5, 0.3, 0.4, 0.5, 0.2, 0.3, 0.1, -0.1, 0.2, -0.2, 0.0, 0.6]),
        'ci_lower': None,
        'ci_upper': None
    }
    se = np.random.uniform(0.15, 0.4, n_partners)
    partner_effects_judge['ci_lower'] = partner_effects_judge['effect'] - 1.96 * se
    partner_effects_judge['ci_upper'] = partner_effects_judge['effect'] + 1.96 * se
    
    partner_effects_fan = {
        'partner': partners,
        'effect': np.random.randn(n_partners) * 0.6 + np.array([0.8, 1.1, 0.5, 1.0, 0.2, 0.7, 0.3, 0.9, 0.6, 0.4, 0.5, 0.6, 0.3, 0.4, 0.2, 0.1, 0.3, 0.0, 0.1, 0.5]),
        'ci_lower': None,
        'ci_upper': None
    }
    se_fan = np.random.uniform(0.2, 0.5, n_partners)
    partner_effects_fan['ci_lower'] = partner_effects_fan['effect'] - 1.96 * se_fan
    partner_effects_fan['ci_upper'] = partner_effects_fan['effect'] + 1.96 * se_fan
    
    df_partner_judge = pd.DataFrame(partner_effects_judge)
    df_partner_fan = pd.DataFrame(partner_effects_fan)
    
    # 2. SHAP重要性数据
    features = ['age', 'industry_Athlete', 'industry_Actor', 'industry_Singer', 
                'industry_TV_Personality', 'week_norm', 'rolling_avg_pJ', 'rolling_std_pJ',
                'partner_pagerank', 'partner_degree', 'partner_embedding_0', 'partner_embedding_1']
    
    shap_judge = pd.DataFrame({
        'feature': features,
        'mean_shap': [0.15, 0.12, 0.08, 0.06, 0.04, 0.18, 0.25, 0.10, 0.09, 0.07, 0.05, 0.03],
        'q2.5': [0.12, 0.08, 0.05, 0.03, 0.01, 0.14, 0.20, 0.06, 0.05, 0.04, 0.02, 0.01],
        'q97.5': [0.18, 0.16, 0.11, 0.09, 0.07, 0.22, 0.30, 0.14, 0.13, 0.10, 0.08, 0.05]
    })
    
    shap_fan = pd.DataFrame({
        'feature': features,
        'mean_shap': [0.08, 0.22, 0.15, 0.10, 0.12, 0.10, 0.12, 0.05, 0.18, 0.14, 0.08, 0.06],
        'q2.5': [0.05, 0.17, 0.11, 0.06, 0.08, 0.06, 0.08, 0.02, 0.13, 0.10, 0.04, 0.03],
        'q97.5': [0.11, 0.27, 0.19, 0.14, 0.16, 0.14, 0.16, 0.08, 0.23, 0.18, 0.12, 0.09]
    })
    
    # 3. 固定效应系数数据
    coeffs = ['Intercept', 'age_z', 'C(industry)[Athlete]', 'C(industry)[Actor]', 
              'C(industry)[Singer]', 'C(industry)[TV]', 'week_norm']
    
    coeff_judge = pd.DataFrame({
        'term': coeffs,
        'estimate': [0.50, -0.05, 0.03, 0.08, 0.05, -0.02, 0.15],
        '2.5%': [0.45, -0.10, -0.02, 0.03, 0.00, -0.07, 0.10],
        '97.5%': [0.55, 0.00, 0.08, 0.13, 0.10, 0.03, 0.20]
    })
    
    coeff_fan = pd.DataFrame({
        'term': coeffs,
        'estimate': [0.48, 0.02, 0.15, 0.12, 0.08, 0.10, 0.08],
        '2.5%': [0.42, -0.04, 0.08, 0.05, 0.02, 0.04, 0.02],
        '97.5%': [0.54, 0.08, 0.22, 0.19, 0.14, 0.16, 0.14]
    })
    
    # 4. 网络数据
    seasons_active = {p: np.random.randint(3, 15) for p in partners}
    
    # 5. 不确定性分解数据
    uncertainty_data = pd.DataFrame({
        'partner': partners[:10],
        'effect_mean': df_partner_fan['effect'].values[:10],
        'posterior_var': np.random.uniform(0.02, 0.08, 10),
        'model_var': np.random.uniform(0.01, 0.05, 10),
        'residual_var': np.random.uniform(0.005, 0.02, 10)
    })
    uncertainty_data['total_var'] = uncertainty_data['posterior_var'] + uncertainty_data['model_var'] + uncertainty_data['residual_var']
    
    return {
        'partner_judge': df_partner_judge,
        'partner_fan': df_partner_fan,
        'shap_judge': shap_judge,
        'shap_fan': shap_fan,
        'coeff_judge': coeff_judge,
        'coeff_fan': coeff_fan,
        'seasons_active': seasons_active,
        'uncertainty': uncertainty_data,
        'partners': partners
    }


# =============================================================================
# 图表1: 舞伴效应对比图 (Caterpillar Plot with CI)
# =============================================================================

def plot_caterpillar(data, save_path='figure1_caterpillar_plot.png'):
    """
    绘制舞伴随机效应的毛毛虫图(Forest Plot)
    展示评委模型vs粉丝模型的舞伴效应对比
    """
    df_judge = data['partner_judge'].copy()
    df_fan = data['partner_fan'].copy()
    
    # 按评委效应排序
    df_judge = df_judge.sort_values('effect', ascending=True).reset_index(drop=True)
    order = df_judge['partner'].tolist()
    df_fan = df_fan.set_index('partner').loc[order].reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    fig.suptitle('Figure 1: Professional Partner Effects on Performance\n(Caterpillar Plot with 95% Confidence Intervals)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    y_pos = np.arange(len(order))
    
    for ax, df, title, color in [(axes[0], df_judge, 'Judge Score Model', COLORS['judge']),
                                   (axes[1], df_fan, 'Fan Vote Model', COLORS['fan'])]:
        
        # 判断显著性
        significant_pos = (df['ci_lower'] > 0)
        significant_neg = (df['ci_upper'] < 0)
        not_significant = ~(significant_pos | significant_neg)
        
        # 绘制置信区间线
        for i, row in df.iterrows():
            if significant_pos.iloc[i]:
                c = COLORS['positive']
            elif significant_neg.iloc[i]:
                c = COLORS['negative']
            else:
                c = COLORS['neutral']
            
            ax.hlines(y=i, xmin=row['ci_lower'], xmax=row['ci_upper'], 
                     color=c, linewidth=2, alpha=0.7)
            ax.scatter(row['effect'], i, color=c, s=60, zorder=5, edgecolor='white', linewidth=1)
        
        # 零线
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # 设置
        ax.set_xlabel('Random Effect Estimate', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)
        ax.set_xlim(-1.5, 2.0)
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor(COLORS['background'])
    
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(order, fontsize=9)
    axes[0].set_ylabel('Professional Partner', fontsize=11)
    
    # 图例
    legend_elements = [
        mpatches.Patch(color=COLORS['positive'], label='Significant Positive (CI > 0)'),
        mpatches.Patch(color=COLORS['negative'], label='Significant Negative (CI < 0)'),
        mpatches.Patch(color=COLORS['neutral'], label='Not Significant')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 图表1已保存: {save_path}")
    return save_path


# =============================================================================
# 图表2: SHAP重要性对比条形图 (Dual Bar with CI)
# =============================================================================

def plot_shap_comparison(data, save_path='figure2_shap_comparison.png'):
    """
    绘制SHAP重要性对比图
    展示评委模型vs粉丝模型的特征重要性差异
    """
    shap_j = data['shap_judge'].copy()
    shap_f = data['shap_fan'].copy()
    
    # 按评委模型重要性排序
    shap_j = shap_j.sort_values('mean_shap', ascending=True).reset_index(drop=True)
    order = shap_j['feature'].tolist()
    shap_f = shap_f.set_index('feature').loc[order].reset_index()
    
    # 美化特征名
    feature_labels = {
        'age': 'Age',
        'industry_Athlete': 'Industry: Athlete',
        'industry_Actor': 'Industry: Actor',
        'industry_Singer': 'Industry: Singer',
        'industry_TV_Personality': 'Industry: TV Personality',
        'week_norm': 'Week Progress',
        'rolling_avg_pJ': 'Rolling Avg Score',
        'rolling_std_pJ': 'Rolling Std Score',
        'partner_pagerank': 'Partner PageRank',
        'partner_degree': 'Partner Network Degree',
        'partner_embedding_0': 'Partner Embedding Dim 1',
        'partner_embedding_1': 'Partner Embedding Dim 2'
    }
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(order))
    bar_height = 0.35
    
    # 绘制条形图
    bars_j = ax.barh(y_pos - bar_height/2, shap_j['mean_shap'], bar_height, 
                     label='Judge Model', color=COLORS['judge'], alpha=0.8)
    bars_f = ax.barh(y_pos + bar_height/2, shap_f['mean_shap'], bar_height, 
                     label='Fan Model', color=COLORS['fan'], alpha=0.8)
    
    # 绘制误差棒 (CI)
    ax.errorbar(shap_j['mean_shap'], y_pos - bar_height/2, 
                xerr=[shap_j['mean_shap'] - shap_j['q2.5'], shap_j['q97.5'] - shap_j['mean_shap']],
                fmt='none', color='black', capsize=3, capthick=1, alpha=0.6)
    ax.errorbar(shap_f['mean_shap'], y_pos + bar_height/2, 
                xerr=[shap_f['mean_shap'] - shap_f['q2.5'], shap_f['q97.5'] - shap_f['mean_shap']],
                fmt='none', color='black', capsize=3, capthick=1, alpha=0.6)
    
    # 标注差异显著的特征
    for i, feat in enumerate(order):
        j_val = shap_j.iloc[i]['mean_shap']
        f_val = shap_f.iloc[i]['mean_shap']
        diff = abs(f_val - j_val)
        if diff > 0.05:  # 差异阈值
            max_val = max(shap_j.iloc[i]['q97.5'], shap_f.iloc[i]['q97.5'])
            if f_val > j_val:
                ax.annotate('★', xy=(max_val + 0.01, i), fontsize=12, color=COLORS['fan'], va='center')
            else:
                ax.annotate('★', xy=(max_val + 0.01, i), fontsize=12, color=COLORS['judge'], va='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_labels.get(f, f) for f in order], fontsize=10)
    ax.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=11)
    ax.set_title('Figure 2: Feature Importance Comparison\n(SHAP Values with Bootstrap 95% CI)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor(COLORS['background'])
    ax.set_xlim(0, 0.35)
    
    # 添加注释
    ax.text(0.98, 0.02, '★ = Notable difference between models', 
            transform=ax.transAxes, fontsize=9, ha='right', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 图表2已保存: {save_path}")
    return save_path


# =============================================================================
# 图表3: 固定效应系数比较热力图 (Coefficient Heatmap)
# =============================================================================

def plot_coefficient_heatmap(data, save_path='figure3_coefficient_heatmap.png'):
    """
    绘制固定效应系数热力图
    展示评委模型vs粉丝模型的系数方向和显著性
    """
    coeff_j = data['coeff_judge'].copy()
    coeff_f = data['coeff_fan'].copy()
    
    # 合并数据
    terms = coeff_j['term'].tolist()[1:]  # 排除截距
    
    # 创建矩阵
    coeff_matrix = np.zeros((len(terms), 2))
    sig_matrix = np.zeros((len(terms), 2), dtype=bool)
    
    for i, term in enumerate(terms):
        # Judge
        row_j = coeff_j[coeff_j['term'] == term].iloc[0]
        coeff_matrix[i, 0] = row_j['estimate']
        sig_matrix[i, 0] = (row_j['2.5%'] > 0) or (row_j['97.5%'] < 0)
        
        # Fan
        row_f = coeff_f[coeff_f['term'] == term].iloc[0]
        coeff_matrix[i, 1] = row_f['estimate']
        sig_matrix[i, 1] = (row_f['2.5%'] > 0) or (row_f['97.5%'] < 0)
    
    # 美化标签
    term_labels = {
        'age_z': 'Age (Standardized)',
        'C(industry)[Athlete]': 'Industry: Athlete',
        'C(industry)[Actor]': 'Industry: Actor',
        'C(industry)[Singer]': 'Industry: Singer',
        'C(industry)[TV]': 'Industry: TV Personality',
        'week_norm': 'Week Progress'
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建发散色图
    cmap = LinearSegmentedColormap.from_list('diverging', 
                                              ['#2166AC', '#F7F7F7', '#B2182B'], N=256)
    
    # 绘制热力图
    vmax = max(abs(coeff_matrix.min()), abs(coeff_matrix.max()))
    im = ax.imshow(coeff_matrix, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    
    # 添加文本标注
    for i in range(len(terms)):
        for j in range(2):
            val = coeff_matrix[i, j]
            sig = sig_matrix[i, j]
            text = f'{val:.3f}'
            if sig:
                text += ' *'
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=11, 
                   fontweight='bold' if sig else 'normal', color=color)
    
    # 设置
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Judge Model', 'Fan Model'], fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels([term_labels.get(t, t) for t in terms], fontsize=11)
    
    ax.set_title('Figure 3: Fixed Effect Coefficients Comparison\n(* indicates 95% CI excludes zero)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Coefficient Estimate', fontsize=11)
    
    # 添加边框
    for i in range(len(terms) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    ax.axvline(0.5, color='white', linewidth=2)
    
    # 添加解释框
    interpretation = """
    Interpretation Guide:
    • Red = Positive effect (increases score/votes)
    • Blue = Negative effect (decreases score/votes)
    • * = Statistically significant (95% CI ≠ 0)
    • Compare columns to see Judge vs Fan differences
    """
    ax.text(1.35, 0.5, interpretation, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 图表3已保存: {save_path}")
    return save_path


# =============================================================================
# 图表4: 舞伴网络可视化 (Network Graph with Effect Size)
# =============================================================================

def plot_partner_network(data, save_path='figure4_partner_network.png'):
    """
    绘制舞伴网络图
    节点大小表示经验(出场季数)，颜色表示效应值
    """
    partners = data['partners']
    effects = dict(zip(data['partner_fan']['partner'], data['partner_fan']['effect']))
    seasons = data['seasons_active']
    
    # 创建网络
    G = nx.Graph()
    
    # 添加节点
    for p in partners:
        G.add_node(p, effect=effects.get(p, 0), seasons=seasons.get(p, 5))
    
    # 添加边 (模拟同季出现的舞伴)
    np.random.seed(42)
    for i, p1 in enumerate(partners):
        for j, p2 in enumerate(partners):
            if i < j and np.random.random() < 0.3:  # 30%概率连边
                weight = np.random.randint(1, 5)
                G.add_edge(p1, p2, weight=weight)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 节点属性
    node_sizes = [seasons.get(n, 5) * 80 for n in G.nodes()]
    node_colors = [effects.get(n, 0) for n in G.nodes()]
    
    # 边权重
    edge_weights = [G[u][v].get('weight', 1) * 0.5 for u, v in G.edges()]
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.3, edge_color='gray', ax=ax)
    
    # 绘制节点
    cmap = LinearSegmentedColormap.from_list('effect', 
                                              [COLORS['negative'], '#FFFFFF', COLORS['positive']], N=256)
    vmax = max(abs(min(node_colors)), abs(max(node_colors)))
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                    cmap=cmap, vmin=-vmax, vmax=vmax, 
                                    edgecolors='black', linewidths=1.5, ax=ax)
    
    # 标签 (只标注重要节点)
    important_nodes = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    labels = {n: n.split()[-1] for n, _ in important_nodes}  # 只显示姓氏
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax)
    
    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Partner Effect on Fan Votes', fontsize=11)
    
    # 图例 (节点大小)
    for size, label in [(400, '5 seasons'), (800, '10 seasons'), (1200, '15 seasons')]:
        ax.scatter([], [], s=size, c='gray', alpha=0.5, label=label, edgecolors='black')
    ax.legend(title='Experience', loc='upper left', fontsize=9, title_fontsize=10)
    
    ax.set_title('Figure 4: Professional Partner Network\n(Node size = Experience, Color = Effect on Fan Votes)', 
                 fontsize=13, fontweight='bold')
    ax.axis('off')
    
    # 添加统计摘要
    summary = f"""
    Network Statistics:
    • Nodes: {G.number_of_nodes()} partners
    • Edges: {G.number_of_edges()} co-appearances
    • Avg Degree: {np.mean([d for n, d in G.degree()]):.1f}
    • Top Effect: {max(effects.values()):.2f}
    • Bottom Effect: {min(effects.values()):.2f}
    """
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 图表4已保存: {save_path}")
    return save_path


# =============================================================================
# 图表5: 不确定性传播瀑布图 (Uncertainty Waterfall)
# =============================================================================

def plot_uncertainty_waterfall(data, save_path='figure5_uncertainty_waterfall.png'):
    """
    绘制不确定性分解瀑布图
    展示总不确定性的来源分解: Task1后验 + LMM模型 + 残差
    """
    df = data['uncertainty'].copy()
    df = df.sort_values('effect_mean', ascending=True).reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # ===== 左图: 堆叠条形图 (方差分解) =====
    ax1 = axes[0]
    
    partners = df['partner'].tolist()
    y_pos = np.arange(len(partners))
    
    # 标准化为比例
    df['prop_posterior'] = df['posterior_var'] / df['total_var']
    df['prop_model'] = df['model_var'] / df['total_var']
    df['prop_residual'] = df['residual_var'] / df['total_var']
    
    # 堆叠条形图
    bars1 = ax1.barh(y_pos, df['prop_posterior'], label='Task 1 Posterior Uncertainty', 
                     color='#E74C3C', alpha=0.8)
    bars2 = ax1.barh(y_pos, df['prop_model'], left=df['prop_posterior'], 
                     label='LMM Model Uncertainty', color='#3498DB', alpha=0.8)
    bars3 = ax1.barh(y_pos, df['prop_residual'], left=df['prop_posterior'] + df['prop_model'], 
                     label='Residual Uncertainty', color='#95A5A6', alpha=0.8)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(partners, fontsize=10)
    ax1.set_xlabel('Proportion of Total Variance', fontsize=11)
    ax1.set_title('Uncertainty Source Decomposition', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_facecolor(COLORS['background'])
    
    # ===== 右图: 效应估计 with 分层CI =====
    ax2 = axes[1]
    
    for i, row in df.iterrows():
        mean = row['effect_mean']
        total_se = np.sqrt(row['total_var'])
        posterior_se = np.sqrt(row['posterior_var'])
        model_se = np.sqrt(row['posterior_var'] + row['model_var'])
        
        # 绘制分层置信区间 (由窄到宽)
        # 最窄: 仅残差
        ax2.hlines(y=i, xmin=mean - 1.96*posterior_se, xmax=mean + 1.96*posterior_se,
                  color='#E74C3C', linewidth=6, alpha=0.8)
        # 中间: 残差 + 模型
        ax2.hlines(y=i, xmin=mean - 1.96*model_se, xmax=mean + 1.96*model_se,
                  color='#3498DB', linewidth=4, alpha=0.6)
        # 最宽: 全部
        ax2.hlines(y=i, xmin=mean - 1.96*total_se, xmax=mean + 1.96*total_se,
                  color='#95A5A6', linewidth=2, alpha=0.5)
        
        # 点估计
        ax2.scatter(mean, i, color='black', s=50, zorder=5)
    
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Partner Effect Estimate', fontsize=11)
    ax2.set_title('Layered Confidence Intervals', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_facecolor(COLORS['background'])
    
    # 图例
    legend_elements = [
        plt.Line2D([0], [0], color='#E74C3C', linewidth=6, label='Task 1 Posterior CI'),
        plt.Line2D([0], [0], color='#3498DB', linewidth=4, label='+ LMM Model CI'),
        plt.Line2D([0], [0], color='#95A5A6', linewidth=2, label='+ Residual (Total CI)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    fig.suptitle('Figure 5: Uncertainty Propagation from Task 1 to Task 3\n(Demonstrating End-to-End Certainty Quantification)', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    # 添加方法论说明
    method_note = """
    Methodology:
    The total uncertainty in partner effects combines:
    1. Task 1 posterior uncertainty (vote share inference)
    2. LMM random effect estimation uncertainty
    3. Residual model uncertainty
    
    This decomposition shows that Task 1 inference
    is the dominant source of uncertainty for most partners.
    """
    fig.text(0.5, -0.08, method_note, ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 图表5已保存: {save_path}")
    return save_path


# =============================================================================
# 主函数: 生成所有图表
# =============================================================================

def generate_all_figures(output_dir='d:\\shumomeisai\\Code_second\\task3\\demo_output'):
    """生成所有5个图表"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Task 3 Attribution Analysis - 高级可视化图表生成")
    print("=" * 60)
    
    # 生成模拟数据
    print("\n[1/6] 生成模拟数据...")
    data = generate_mock_data()
    print(f"  - 舞伴数量: {len(data['partners'])}")
    print(f"  - 特征数量: {len(data['shap_judge'])}")
    
    # 生成图表
    print("\n[2/6] 生成图表1: 舞伴效应对比图 (Caterpillar Plot)...")
    path1 = plot_caterpillar(data, f'{output_dir}/figure1_caterpillar_plot.png')
    
    print("\n[3/6] 生成图表2: SHAP重要性对比条形图...")
    path2 = plot_shap_comparison(data, f'{output_dir}/figure2_shap_comparison.png')
    
    print("\n[4/6] 生成图表3: 固定效应系数热力图...")
    path3 = plot_coefficient_heatmap(data, f'{output_dir}/figure3_coefficient_heatmap.png')
    
    print("\n[5/6] 生成图表4: 舞伴网络可视化...")
    path4 = plot_partner_network(data, f'{output_dir}/figure4_partner_network.png')
    
    print("\n[6/6] 生成图表5: 不确定性传播瀑布图...")
    path5 = plot_uncertainty_waterfall(data, f'{output_dir}/figure5_uncertainty_waterfall.png')
    
    print("\n" + "=" * 60)
    print("所有图表生成完成!")
    print("=" * 60)
    
    return [path1, path2, path3, path4, path5]


if __name__ == "__main__":
    paths = generate_all_figures()
    print("\n生成的文件列表:")
    for p in paths:
        print(f"  - {p}")
