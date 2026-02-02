
import pandas as pd
import numpy as np

csv_path = r'd:\shumomeisai\dancewith\2026_MCM_Problem_C_Data.csv'

def analyze_anomalies():
    df = pd.read_csv(csv_path)
    # Fix incomplete column matching if needed, though earlier I lowercased them.
    # Let's trust the columns are somewhat standard or reuse the cleaning logic.
    df.columns = [c.strip().lower() for c in df.columns]
    
    score_cols = [c for c in df.columns if 'score' in c]
    

    with open('anomalies_report.txt', 'w', encoding='utf-8') as f:
        f.write("--- 1. Score Range Analysis by Season ---\n")
        
        season_stats = []
        for season in sorted(df['season'].unique()):
            season_df = df[df['season'] == season]
            all_scores = season_df[score_cols].values.flatten()
            valid_scores = all_scores[~np.isnan(all_scores)]
            
            if len(valid_scores) > 0:
                season_stats.append({
                    'Season': season,
                    'Max': valid_scores.max(),
                    'Min': valid_scores.min(),
                    'Mean': round(valid_scores.mean(), 2)
                })
        
        stats_df = pd.DataFrame(season_stats)
        f.write(stats_df.to_string(index=False))
        f.write("\n\n--- 2. Missing Data Pattern Analysis ---\n")
        
        abnormal_missing_count = 0
        total_couples = len(df)
        
        for idx, row in df.iterrows():
            scores_by_week = {}
            for col in score_cols:
                if 'week' in col:
                    import re
                    week_match = re.search(r'week(\d+)', col)
                    if week_match:
                        w = int(week_match.group(1))
                        if w not in scores_by_week:
                            scores_by_week[w] = []
                        scores_by_week[w].append(row[col])
            
            sorted_weeks = sorted(scores_by_week.keys())
            week_presence = []
            for w in sorted_weeks:
                vals = scores_by_week[w]
                has_score = any(not pd.isna(v) for v in vals)
                week_presence.append(has_score)
                
            true_indices = [i for i, x in enumerate(week_presence) if x]
            if not true_indices:
                continue
                
            first_true = min(true_indices)
            last_true = max(true_indices)
            
            sub_sequence = week_presence[first_true : last_true+1]
            if False in sub_sequence:
                f.write(f"Gap found for S{row['season']} {row['celebrity_name']}: {week_presence}\n")
                abnormal_missing_count += 1
                
        f.write(f"\nTotal couples with 'Gap' in participation (Abnormal Missing): {abnormal_missing_count} / {total_couples}\n")

if __name__ == "__main__":
    analyze_anomalies()
