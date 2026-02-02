"""
Task 3: Attribution Analysis - Visualization with REAL Data
======================================================
Visualizes the results from Task 3 (LMM and GBDT analysis).

Outputs:
1. Caterpillar Plot: Partner Random Effects (Judge vs Fan)
2. SHAP Comparison: Feature Importance (Judge vs Fan)
3. Coefficient Heatmap: Fixed Effects (Judge vs Fan)
4. Network Graph: Partner Experience & Fan Effect
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import os
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = r'd:\shumomeisai\Code_second\Results\task3_analysis'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\task3\real_output'

# Global Style
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'judge': '#2E86AB',      # Blue
    'fan': '#E94F37',        # Red/Orange
    'positive': '#28A745',   # Green
    'negative': '#DC3545',   # Red
    'neutral': '#6C757D',    # Grey
    'background': '#F8F9FA',
    'grid': '#DEE2E6'
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_real_data():
    """Load real analysis results from CSV files."""
    print("Loading real data...")
    
    data = {}
    
    # 1. Partner Effects
    # Judge
    f_j_partner = os.path.join(RESULTS_DIR, 'task3_lmm_judge_partner_effects.csv')
    if os.path.exists(f_j_partner):
        try:
            df_pj = pd.read_csv(f_j_partner)
            print(f"Loaded Judge Partner Effects: {len(df_pj)} rows")
            df_pj['ci_lower'] = df_pj['effect'] # No CI available
            df_pj['ci_upper'] = df_pj['effect']
            data['partner_judge'] = df_pj
        except Exception as e:
            print(f"Error loading {f_j_partner}: {e}")
            data['partner_judge'] = pd.DataFrame()
    else:
        print(f"Warning: Missing {f_j_partner}")
        data['partner_judge'] = pd.DataFrame()

    # Fan
    f_f_partner = os.path.join(RESULTS_DIR, 'task3_lmm_fan_partner_effects_aggregated.csv')
    if os.path.exists(f_f_partner):
        try:
            df_pf = pd.read_csv(f_f_partner, index_col=0) 
            df_pf = df_pf.reset_index().rename(columns={
                df_pf.index.name if df_pf.index.name else 'index': 'partner',
                'mean': 'effect', 
                '2.5%': 'ci_lower', 
                '97.5%': 'ci_upper'
            })
            print(f"Loaded Fan Partner Effects: {len(df_pf)} rows")
            data['partner_fan'] = df_pf
        except Exception as e:
            print(f"Error loading {f_f_partner}: {e}")
            data['partner_fan'] = pd.DataFrame()
    else:
        print(f"Warning: Missing {f_f_partner}")
        data['partner_fan'] = pd.DataFrame()

    # 2. SHAP
    f_j_shap = os.path.join(RESULTS_DIR, 'task3_shap_ci_judge.csv')
    if os.path.exists(f_j_shap):
        data['shap_judge'] = pd.read_csv(f_j_shap)
        print(f"Loaded Judge SHAP: {len(data['shap_judge'])} rows")
    else:
        data['shap_judge'] = pd.DataFrame()
        
    f_f_shap = os.path.join(RESULTS_DIR, 'task3_shap_ci_fan.csv')
    if os.path.exists(f_f_shap):
        data['shap_fan'] = pd.read_csv(f_f_shap)
        print(f"Loaded Fan SHAP: {len(data['shap_fan'])} rows")
    else:
        data['shap_fan'] = pd.DataFrame()

    # 3. Coefficients
    f_j_coeff = os.path.join(RESULTS_DIR, 'task3_lmm_judge_coeffs.csv')
    if os.path.exists(f_j_coeff):
        try:
            df_cj = pd.read_csv(f_j_coeff, index_col=0).reset_index()
            df_cj.rename(columns={'index': 'term'}, inplace=True)
            print(f"Loaded Judge Coeffs: {len(df_cj)} rows")
            print(f"Terms: {df_cj['term'].tolist()}")
            data['coeff_judge'] = df_cj
        except Exception as e:
            print(f"Error loading {f_j_coeff}: {e}")
            data['coeff_judge'] = pd.DataFrame()
    else:
        print(f"Warning: Missing {f_j_coeff}")
        data['coeff_judge'] = pd.DataFrame()
        
    f_f_coeff = os.path.join(RESULTS_DIR, 'task3_lmm_fan_coeffs_aggregated.csv')
    if os.path.exists(f_f_coeff):
        try:
            df_cf = pd.read_csv(f_f_coeff, index_col=0).reset_index()
            df_cf.rename(columns={'index': 'term'}, inplace=True)
            if 'mean' in df_cf.columns:
                df_cf.rename(columns={'mean': 'estimate'}, inplace=True)
            print(f"Loaded Fan Coeffs: {len(df_cf)} rows")
            print(f"Terms: {df_cf['term'].tolist()}")
            data['coeff_fan'] = df_cf
        except Exception as e:
             print(f"Error loading {f_f_coeff}: {e}")
             data['coeff_fan'] = pd.DataFrame()
    else:
        print(f"Warning: Missing {f_f_coeff}")
        data['coeff_fan'] = pd.DataFrame()

    # 4. Network / Panel Data
    if os.path.exists(PANEL_PATH):
        df_panel = pd.read_csv(PANEL_PATH)
        seasons_active = df_panel.groupby('ballroom_partner')['season'].nunique().to_dict()
        data['seasons_active'] = seasons_active
        print(f"Loaded Panel Data: {len(seasons_active)} partners")
        
        if 'partner_fan' in data and not data['partner_fan'].empty:
            data['partners'] = data['partner_fan']['partner'].tolist()
        else:
            data['partners'] = list(seasons_active.keys())
    
    return data

# =============================================================================
# PLOTTING FUNCTIONS 
# (Copied from demo, slightly adapted for robust key access)
# =============================================================================

def plot_caterpillar(data, save_path):
    df_judge = data.get('partner_judge').copy()
    df_fan = data.get('partner_fan').copy()
    
    if df_judge.empty or df_fan.empty:
        print("Skipping Caterpillar Plot: No data")
        return

    # Filter to intersection of partners
    partners_j = set(df_judge['partner'])
    partners_f = set(df_fan['partner'])
    common = list(partners_j & partners_f)
    
    df_judge = df_judge[df_judge['partner'].isin(common)]
    df_fan = df_fan[df_fan['partner'].isin(common)]
    
    # Sort by Judge effect
    df_judge = df_judge.sort_values('effect', ascending=True).reset_index(drop=True)
    order = df_judge['partner'].tolist()
    
    # Reorder fan to match
    df_fan = df_fan.set_index('partner').loc[order].reset_index()
    
    # Limit to top 30?
    if len(order) > 30:
        # Take top 15 and bottom 15? Or just top 30?
        # Let's take top 15 positive and bottom 15 negative from Judge effect
        top_15 = df_judge.iloc[-15:]
        bot_15 = df_judge.iloc[:15]
        subset = pd.concat([bot_15, top_15])
        
        # Re-sort subset
        subset = subset.sort_values('effect', ascending=True)
        order = subset['partner'].tolist()
        
        df_judge = subset
        df_fan = df_fan.set_index('partner').loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, len(order)*0.3)), sharey=True)
    
    y_pos = np.arange(len(order))
    
    for ax, df, title, color in [(axes[0], df_judge, 'Judge Model (Scores)', COLORS['judge']),
                                   (axes[1], df_fan, 'Fan Model (Votes)', COLORS['fan'])]:
        
        significant_pos = (df['ci_lower'] > 0)
        significant_neg = (df['ci_upper'] < 0)
        
        for i, row in df.iterrows():
            if significant_pos.loc[i]: c = COLORS['positive']
            elif significant_neg.loc[i]: c = COLORS['negative']
            else: c = COLORS['neutral']
            
            # Draw line only if width > 0
            if row['ci_upper'] > row['ci_lower']:
                ax.hlines(y=i, xmin=row['ci_lower'], xmax=row['ci_upper'], color=c, lw=2, alpha=0.7)
            
            ax.scatter(row['effect'], i, color=c, s=50, zorder=5)
            
        ax.axvline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_title(title, fontweight='bold', color=color)
        ax.grid(axis='x', alpha=0.3)
    
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(order, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 1 saved to {save_path}")

def plot_shap_comparison(data, save_path):
    shap_j = data.get('shap_judge', pd.DataFrame()).copy()
    shap_f = data.get('shap_fan', pd.DataFrame()).copy()
    
    if shap_j.empty or shap_f.empty: return

    # Sort by Judge importance
    shap_j = shap_j.sort_values('mean_shap', ascending=True)
    order = shap_j['feature'].tolist()
    shap_f = shap_f.set_index('feature').reindex(order).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 0.4 * len(order) + 2))
    y_pos = np.arange(len(order))
    h = 0.35
    
    ax.barh(y_pos - h/2, shap_j['mean_shap'], h, label='Judge', color=COLORS['judge'], alpha=0.8)
    ax.barh(y_pos + h/2, shap_f['mean_shap'], h, label='Fan', color=COLORS['fan'], alpha=0.8)
    
    # Error bars
    ax.errorbar(shap_j['mean_shap'], y_pos - h/2, 
                xerr=[shap_j['mean_shap'] - shap_j['q2.5'], shap_j['q97.5'] - shap_j['mean_shap']],
                fmt='none', ecolor='black', capsize=3)
    ax.errorbar(shap_f['mean_shap'], y_pos + h/2, 
                xerr=[shap_f['mean_shap'] - shap_f['q2.5'], shap_f['q97.5'] - shap_f['mean_shap']],
                fmt='none', ecolor='black', capsize=3)
                
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order)
    ax.set_xlabel('Mean |SHAP|')
    ax.legend()
    ax.set_title('Feature Importance (SHAP)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure 2 saved to {save_path}")

def plot_heatmap(data, save_path):
    cj = data.get('coeff_judge')
    cf = data.get('coeff_fan')
    
    if cj is None or cf is None: return

    # Intersect terms
    terms = sorted([t for t in cj['term'] if t in cf['term'].values and 'Intercept' not in t])
    
    print(f"Heatmap: Found {len(terms)} common terms")
    
    # Build array
    mat = np.zeros((len(terms), 2))
    sigs = np.zeros((len(terms), 2), dtype=bool)
    
    for i, t in enumerate(terms):
        # Debug trace
        # print(f"  Processing {i}: {t}")
        
        subset_j = cj[cj['term'] == t]
        if subset_j.empty:
            print(f"CRITICAL ERROR: Judge missing term '{t}'")
            continue
            
        subset_f = cf[cf['term'] == t]
        if subset_f.empty:
            print(f"CRITICAL ERROR: Fan missing term '{t}'")
            continue

        rj = subset_j.iloc[0]
        rf = subset_f.iloc[0]
        
        try:
            mat[i,0] = rj['estimate']
            sigs[i,0] = (rj['2.5%'] > 0) or (rj['97.5%'] < 0)
            
            mat[i,1] = rf['estimate']
            sigs[i,1] = (rf['2.5%'] > 0) or (rf['97.5%'] < 0)
        except KeyError as e:
            print(f"KeyError processing '{t}': {e}. Columns J: {rj.index}, F: {rf.index}")
            continue
        
    fig, ax = plt.subplots(figsize=(8, len(terms)*0.5 + 2))
    cmap = LinearSegmentedColormap.from_list('div', [COLORS['judge'], '#FFFFFF', COLORS['fan']], N=256) # Blue to Red ??? No, Blue to Red is Judge to Fan? 
    # Wait, coefficient sign: Red is positive (usually), Blue negative.
    # Color map should be Negative -> Positive.
    # Let's use Red=Negative, Green=Positive? Or Red/Blue
    cmap = LinearSegmentedColormap.from_list('div', ['#B2182B', '#F7F7F7', '#2166AC'], N=256) 
    # Actually, let's match the demo: Blue=Negative, Red=Positive?
    # Task 3 demo uses blue for Judge, Red for Fan.
    # But for heatmap values, we typically use RdBu where Red is Positive?
    # Let's use standard RdBu_r (Red=Positive, Blue=Negative)
    
    vmax = np.max(np.abs(mat))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    
    # Text
    for i in range(len(terms)):
        for j in range(2):
            val = mat[i,j]
            sig = sigs[i,j]
            txt = f"{val:.2f}" + ("*" if sig else "")
            ax.text(j, i, txt, ha='center', va='center', 
                    color='white' if abs(val)>vmax*0.5 else 'black')
            
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Judge', 'Fan'])
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms)
    
    plt.colorbar(im).set_label('Effect Size')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure 3 saved to {save_path}")

def plot_network(data, save_path):
    if 'partner_fan' not in data or 'seasons_active' not in data: return
    
    df = data['partner_fan']
    effects = dict(zip(df['partner'], df['effect']))
    exp = data['seasons_active']
    
    partners = list(effects.keys())
    # Subsample if too many
    if len(partners) > 50:
        partners = [p for p in partners if p in exp and exp[p] >= 3] # Filter by experience
    
    G = nx.Graph()
    for p in partners:
        G.add_node(p, size=exp.get(p,1)*50, color=effects.get(p,0))
        
    # Mock edges? Or assume fully disconnected unless we have data?
    # The demo mocked edges.
    # Real data: we don't have "co-occurrence" readily available here without panel processing.
    # Let's Skip edges or make fully connected components by era?
    # For now: Just nodes (scatter plot layout effectively)
    
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    fig, ax = plt.subplots(figsize=(12,12))
    
    sizes = [G.nodes[n]['size'] for n in G.nodes()]
    colors = [G.nodes[n]['color'] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors, cmap='RdBu_r', ax=ax, edgecolors='k')
    
    # Top 10 labels
    top_p = sorted(partners, key=lambda x: abs(effects[x]), reverse=True)[:15]
    labels = {n: n.split()[-1] for n in top_p}
    nx.draw_networkx_labels(G, pos, labels, ax=ax)
    
    ax.axis('off')
    plt.savefig(save_path)
    print(f"Figure 4 saved to {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    data = load_real_data()
    
    plot_caterpillar(data, os.path.join(OUTPUT_DIR, 'fig1_partner_effects.png'))
    plot_shap_comparison(data, os.path.join(OUTPUT_DIR, 'fig2_shap_importance.png'))
    plot_heatmap(data, os.path.join(OUTPUT_DIR, 'fig3_coeffs.png'))
    plot_network(data, os.path.join(OUTPUT_DIR, 'fig4_partner_network.png'))
    # Figure 5 skipped

if __name__ == '__main__':
    main()
