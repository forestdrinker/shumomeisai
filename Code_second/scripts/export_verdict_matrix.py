"""
Export Verdict Matrix State to CSV.
Saves the classification (Hit/Miss/Reason) for each Season-Week cell.
"""

import pandas as pd
import numpy as np
import os

PRED_PATH = r'd:\shumomeisai\Code_second\Results\validation_results\detailed_predictions.csv'
MISSING_PATH = r'd:\shumomeisai\Code_second\task1.准确性code\missing_data_report.csv'
OUTPUT_CSV = r'd:\shumomeisai\Code_second\task1.准确性code\verdict_matrix_state.csv'

def export_matrix():
    # Load data
    df_pred = pd.read_csv(PRED_PATH)
    missing_df = pd.read_csv(MISSING_PATH)
    
    n_seasons = 34
    max_weeks = 11
    
    # 1. Build Base Matrix from Predictions
    # 0=Empty, 1=Top1, 2=Top2, 3=Miss
    matrix = np.zeros((n_seasons, max_weeks), dtype=int)
    
    for _, row in df_pred.iterrows():
        s = int(row['season']) - 1
        w = int(row['week']) - 1
        
        if s >= n_seasons or w >= max_weeks:
            continue
            
        acc = row['accuracy']
        top2 = row['top2_acc']
        
        if acc == 1:
            matrix[s, w] = 1
        elif top2 == 1:
            matrix[s, w] = 2
        else:
            matrix[s, w] = 3
            
    # 2. Map Missing Reasons
    missing_map = {}
    for _, row in missing_df.iterrows():
        missing_map[(int(row['season']), int(row['week']))] = row['reason']
        
    # 3. Create Readable DataFrame
    output_rows = []
    
    for s in range(n_seasons):
        row_dict = {'Season': s + 1}
        for w in range(max_weeks):
            val = matrix[s, w]
            label = ""
            
            if val == 1:
                label = "Top-1 Hit"
            elif val == 2:
                label = "Top-2 Hit"
            elif val == 3:
                label = "Miss"
            else:
                # Check reason
                reason = missing_map.get((s+1, w+1), "Unknown")
                if reason == "Season Ended":
                    label = "Season Ended"
                elif reason == "Non-Elimination Week":
                    label = "Non-Elimination"
                elif reason == "Finale (Rank Only)":
                    label = "Finale"
                elif "Missing Prediction" in reason:
                    label = "Missing Data"
                else:
                    label = "No Data"
            
            row_dict[f'Week {w+1}'] = label
            
        output_rows.append(row_dict)
        
    # 4. Save
    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved Verdict Matrix state to: {OUTPUT_CSV}")
    
    # Print preview
    print(df_out.head().to_string())

if __name__ == '__main__':
    export_matrix()
