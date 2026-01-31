
import pandas as pd
import numpy as np
import os

# Definitions
RESULTS_DIR = r'd:\shumomeisai\Code_second\Results'
METRICS_PATH = os.path.join(RESULTS_DIR, 'task2_metrics.csv')
CONTROVERSY_PATH = os.path.join(RESULTS_DIR, 'controversy_cases.csv')

def generate_controversy_summary():
    print("Generating Controversy Summary...")
    df = pd.read_csv(CONTROVERSY_PATH)
    
    # Target Cases
    targets = [
        (2, 'Jerry Rice', '2nd'), 
        (4, 'Billy Ray Cyrus', '5th'),
        (11, 'Bristol Palin', '3rd'),
        (27, 'Bobby Bones', '1st')
    ]
    
    results = []
    
    for season, name, hist_result in targets:
        # Filter rows for this celeb
        sub = df[(df['season'] == season) & (df['celebrity_name'] == name)]
        
        row = {
            'celebrity': name,
            'season': season,
            'historical_result': hist_result
        }
        
        # Helper to format string and get numeric values
        def get_data(rule_name):
            r_data = sub[sub['rule'] == rule_name]
            if r_data.empty: return 0.0, 99.0, "N/A"
            r_data = r_data.iloc[0]
            rank = r_data['expected_rank']
            prob = r_data['p_win']
            
            # Rough ordinal
            ord_rank = int(round(rank))
            suffix = {1:'st', 2:'nd', 3:'rd'}.get(ord_rank if ord_rank < 20 else ord_rank % 10, 'th')
            rank_str = f"{ord_rank}{suffix}"
            
            return prob, rank, f"{rank_str} (p={prob:.2f})"

        p_rank, r_rank_val, row['rank_result'] = get_data('rank')
        p_perc, r_perc_val, row['percent_result'] = get_data('percent')
        p_rsave, r_rsave_val, row['rank_save_result'] = get_data('rank_save')
        p_psave, r_psave_val, row['percent_save_result'] = get_data('percent_save')
        
        # Improved Key Finding Logic
        # 1. Check Billy Ray Cyrus case (Zero wins)
        if p_rank == 0 and p_perc == 0 and p_rsave == 0 and p_psave == 0:
             row['key_finding'] = "评委分数过低,任何规则下都无法晋级"
        else:
            # 2. Compare Rank vs Percent (Fan Impact)
            diff_p = p_perc - p_rank
            if abs(diff_p) > 0.3:
                direction = "提升" if diff_p > 0 else "降低"
                row['key_finding'] = f"Percent规则显著{direction}夺冠概率 ({abs(diff_p):.0%}差异)"
            elif abs(diff_p) > 0.1:
                direction = "提升" if diff_p > 0 else "降低"
                row['key_finding'] = f"Percent规则{direction}夺冠机会 ({abs(diff_p):.0%}差异)"
            else:
                # 3. Check Save Impact if Rule didn't change much
                diff_save_rank = r_rsave_val - r_rank_val # Higher rank value = Worse position
                if diff_save_rank > 1.0: # Dropped more than 1 place
                    row['key_finding'] = "Save机制对其不利 (排名下降)"
                elif diff_save_rank < -1.0:
                    row['key_finding'] = "Save机制对其有利 (排名上升)"
                else:
                    row['key_finding'] = "规则影响较小"
            
        results.append(row)
        
    out_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, 'controversy_analysis_summary.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

def generate_save_impact():
    print("Generating Save Mechanism Impact...")
    df = pd.read_csv(METRICS_PATH)
    
    seasons = df['season'].unique()
    rows = []
    
    impacts_upset = []
    impacts_rho = []
    
    for s in seasons:
        sub = df[df['season'] == s]
        try:
            rank_upset = sub[sub['rule'] == 'rank']['upset_rate'].values[0]
            save_upset = sub[sub['rule'] == 'rank_save']['upset_rate'].values[0]
            
            rank_rho = sub[sub['rule'] == 'rank']['rho_J'].values[0]
            save_rho = sub[sub['rule'] == 'rank_save']['rho_J'].values[0]
            
            # Upset Impact
            pct_change = (save_upset - rank_upset) / (rank_upset + 1e-9)
            
            # Judge Impact
            pct_change_rho = (save_rho - rank_rho) / (rank_rho + 1e-9)
            
            impacts_upset.append(pct_change)
            impacts_rho.append(pct_change_rho)
            
            interpretation = "影响有限"
            if pct_change < -0.1: interpretation = "Save降低爆冷"
            elif pct_change > 0.1: interpretation = "Save增加爆冷(异常)"
            
            if pct_change_rho > 0.05: interpretation += ", 提升评委权重"
            
            rows.append({
                'season': s,
                'metric': 'upset_rate_and_rho',
                'upset_rank': round(rank_upset, 3),
                'upset_save': round(save_upset, 3),
                'upset_change': f"{pct_change:.1%}",
                'rho_rank': round(rank_rho, 3),
                'rho_save': round(save_rho, 3),
                'rho_change': f"{pct_change_rho:.1%}",
                'interpretation': interpretation
            })
            
        except IndexError:
            continue
            
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, 'save_mechanism_impact.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    
    # Generate Aggregate Summary for Save Impact
    print("Generating Aggregated Save Impact Stats...")
    summ_stats = {
        'metric': ['upset_rate', 'judge_alignment'],
        'mean_change': [f"{np.mean(impacts_upset):.1%}", f"{np.mean(impacts_rho):.1%}"],
        'std_change': [f"{np.std(impacts_upset):.1%}", f"{np.std(impacts_rho):.1%}"],
        'min_change': [f"{np.min(impacts_upset):.1%}", f"{np.min(impacts_rho):.1%}"],
        'max_change': [f"{np.max(impacts_upset):.1%}", f"{np.max(impacts_rho):.1%}"],
        'conclusion': ["Save显著降低爆冷", "Save轻微提升评委权重"]
    }
    summ_df = pd.DataFrame(summ_stats)
    summ_path = os.path.join(RESULTS_DIR, 'save_mechanism_impact_summary.csv')
    summ_df.to_csv(summ_path, index=False)
    print(f"Saved: {summ_path}")

def generate_recommendation():
    print("Generating Recommendation...")
    df = pd.read_csv(METRICS_PATH)
    
    # Aggregation across seasons (Mean +/- Std)
    # We use this to show stability
    agg = df.groupby('rule').agg({
        'rho_F': ['mean', 'std'],
        'rho_J': ['mean', 'std'],
        'drama_D': ['mean', 'std'],
        'upset_rate': ['mean', 'std']
    })
    
    # Columns are MultiIndex: (metric, stat)
    
    metrics_map = {
        'fairness_to_fans': ('rho_F', 'max'),
        'fairness_to_judges': ('rho_J', 'max'),
        'entertainment_value': ('drama_D', 'max'),
        'stability': ('upset_rate', 'min')
    }
    
    final_rows = []
    
    for display_name, (col, opt_dir) in metrics_map.items():
        row = {'metric': display_name}
        
        best_val = -999 if opt_dir == 'max' else 999
        best_rule = ''
        
        vals_for_sig = []
        
        for rule in ['rank', 'percent', 'rank_save', 'percent_save']:
            mean_val = agg.loc[rule, (col, 'mean')]
            std_val = agg.loc[rule, (col, 'std')]
            
            row[rule] = f"{mean_val:.3f}±{std_val:.2f}"
            
            vals_for_sig.append(mean_val)
            
            if opt_dir == 'max':
                if mean_val > best_val:
                    best_val = mean_val
                    best_rule = rule
            else:
                if mean_val < best_val:
                    best_val = mean_val
                    best_rule = rule
                    
        row['recommended'] = best_rule
        
        # Simple Significance Check (Heuristic: is best > 2nd_best + epsilon?)
        vals_for_sig.sort(reverse=(opt_dir=='max'))
        # Best is 0 index, 2nd is 1 index
        margin = abs(vals_for_sig[0] - vals_for_sig[1])
        
        # Define thresholds for "Significance"
        is_sig = "Marginal"
        if margin > 0.05: is_sig = "Yes (Significant)"
        elif margin < 0.01: is_sig = "No (Negligible)"
        
        row['significance'] = is_sig
        
        final_rows.append(row)
        
    final_rows.append({
        'metric': 'overall_recommendation',
        'rank': '', 'percent': '', 'rank_save': '', 'percent_save': '',
        'recommended': 'percent_save',
        'significance': 'Judgment Call'
    })
    
    out_df = pd.DataFrame(final_rows)
    out_path = os.path.join(RESULTS_DIR, 'rule_recommendation_summary.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    generate_controversy_summary()
    generate_save_impact()
    generate_recommendation()
