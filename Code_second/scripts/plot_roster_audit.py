import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Configuration
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
JSON_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\audit_figures'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    print("Loading datasets...")
    # A: Panel
    if os.path.exists(PANEL_PATH):
        df_panel = pd.read_csv(PANEL_PATH)
    else:
        # Fallback
        df_panel = pd.read_parquet(PANEL_PATH.replace('.csv', '.parquet'))
    
    # C: JSON
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    return df_panel, json_data

def analyze_seasons(df_panel, json_data):
    seasons = sorted(df_panel['season'].unique())
    
    coverage_stats = []
    membership_stats = []
    
    for s in seasons:
        # 1. Build Sets
        # Set A (Panel)
        df_s = df_panel[df_panel['season'] == s]
        set_A = set(df_s['pair_id'].unique())
        
        # Name Mapping (for JSON)
        # map name -> pair_id
        # Note: Need to handle duplicates/aliases if any. Assuming 1-to-1 for now.
        name_map = {}
        for _, row in df_s.iterrows():
            name_map[row['celebrity_name']] = row['pair_id']
            # Add simple processing? (lower, strip)
            
        # Set B (Model / Posterior)
        set_B = set()
        npz_path = os.path.join(POSTERIOR_DIR, f'season_{s}.npz')
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                set_B = set(data['pair_ids'])
            except:
                pass
                
        # Set C (JSON Elimination Logic)
        set_C_names = set()
        for k, v in json_data.items():
            # Key format: "1_1" or "1_10_final"
            parts = k.split('_')
            if int(parts[0]) == s:
                if 'active_names' in v: set_C_names.update(v['active_names'])
                if 'eliminated_names' in v: set_C_names.update(v['eliminated_names'])
                if 'finalists' in v: set_C_names.update(v['finalists'])
        
        set_C = set()
        for n in set_C_names:
            if n in name_map:
                set_C.add(name_map[n])
            # Unmapped names in JSON are technically in C but we can't link them to ID.
            # We will ignore them for ID-based logic but they represent a "Type 001" error (only in C).
            # But graph asks for roster sets. We stick to Pair IDs.
            
        # 2. Calculate Coverage Metrics
        # Denominator: |A| (Panel is Truth)
        len_A = len(set_A)
        if len_A == 0: len_A = 1 # Avoid div by zero
        
        overlap_AB = len(set_A & set_B)
        overlap_AC = len(set_A & set_C)
        overlap_ABC = len(set_A & set_B & set_C)
        
        coverage_stats.append({
            'Season': s,
            'B covers A': overlap_AB / len_A,
            'C covers A': overlap_AC / len_A,
            'ABC Overlap': overlap_ABC / len_A
        })
        
        # 3. Calculate Membership Distribution
        # Union of all known IDs
        all_ids = set_A | set_B | set_C
        
        counts = {
            '111': 0, '110': 0, '101': 0, '011': 0,
            '100': 0, '010': 0, '001': 0, '000': 0
        }
        
        for pid in all_ids:
            in_A = 1 if pid in set_A else 0
            in_B = 1 if pid in set_B else 0
            in_C = 1 if pid in set_C else 0
            
            key = f"{in_A}{in_B}{in_C}"
            counts[key] += 1
            
        # Append season info
        counts['Season'] = s
        membership_stats.append(counts)
        
    return pd.DataFrame(coverage_stats), pd.DataFrame(membership_stats)

def plot_heatmap(df_cov):
    print("Plotting Heatmap...")
    plt.figure(figsize=(8, 10))
    
    # Prepare data matrix
    # Rows: Season, Cols: Metric
    # Metrics: B covers A, C covers A, ABC Overlap
    data = df_cov.set_index('Season')[['B covers A', 'C covers A', 'ABC Overlap']]
    
    # Custom cmap: 0=Red/White, 1=Green
    # sns.heatmap default is good, but let's make 1.0 distinct
    sns.heatmap(data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1.0, 
                cbar_kws={'label': 'Coverage Ratio (Ratio to Panel Size)'})
    
    plt.title('Roster Coverage Consistency Heatmap\n(A=Panel, B=Model, C=Elim_JSON)', fontsize=14)
    plt.ylabel('Season')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Roster_Audit_Heatmap.png'), dpi=150)
    plt.close()

def plot_stacked_bar(df_mem):
    print("Plotting Stacked Bar...")
    
    # Categories to plot (exclude 000)
    categories = ['111', '110', '101', '011', '100', '010', '001']
    labels = {
        '111': '111: Perfect (A+B+C)',
        '110': '110: Missing in JSON (No Elim info)',
        '101': '101: Missing in Model (No Posterior)',
        '011': '011: Missing in Panel (Unknown Source)',
        '100': '100: Only in Panel (Orphan)',
        '010': '010: Only in Model (Ghost)',
        '001': '001: Only in JSON (Unmapped)'
    }
    
    # Colors
    # 111 -> Green
    # 110, 101, 011 -> Yellow/Orange
    # 100, 010, 001 -> Red
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd', '#d62728', '#8c564b', '#e377c2']
    
    df_plot = df_mem.set_index('Season')[categories]
    
    ax = df_plot.plot(kind='barh', stacked=True, figsize=(12, 10), color=colors)
    
    plt.title('Roster Membership Distribution by Season\n(Binary Key: A=Panel, B=Model, C=JSON)', fontsize=14)
    plt.xlabel('Count of Contestants')
    plt.ylabel('Season')
    plt.legend(labels=[labels[c] for c in categories], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Roster_Audit_StackedBar.png'), dpi=150)
    plt.close()

def main():
    df_panel, json_data = load_data()
    df_cov, df_mem = analyze_seasons(df_panel, json_data)
    
    print("Coverage Stats Head:")
    print(df_cov.head())
    
    print("Membership Stats Head:")
    print(df_mem.head())
    
    plot_heatmap(df_cov)
    plot_stacked_bar(df_mem)
    print(f"Figures saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
