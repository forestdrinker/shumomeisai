"""
Analyze missing data reasons for Verdict Matrix.
Compares detailed_predictions.csv (Chart Source) with panel.csv (Ground Truth).
"""

import pandas as pd
import numpy as np

PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
PRED_PATH = r'd:\shumomeisai\Code_second\Results\validation_results\detailed_predictions.csv'

def analyze_missing():
    # Load data
    panel = pd.read_csv(PANEL_PATH)
    preds = pd.read_csv(PRED_PATH)
    
    # Create set of predicted (s, w)
    # Adjust prediction week to 1-based logic if needed
    # detailed_predictions.csv has 'week' column. Let's check S1.
    # S1 in preds: 2, 3, 5.
    # S1 actual elims: W2, W3, W4(Rachel), W5.
    pred_set = set(zip(preds['season'], preds['week']))
    
    # Analyze 34 Seasons x 11 Weeks
    reasons = []
    
    for s in range(1, 35):
        # Get season data from panel
        s_panel = panel[panel['season'] == s]
        if len(s_panel) == 0:
            print(f"Season {s}: No panel data")
            continue
            
        max_week_in_panel = s_panel['week'].max()
        
        # Get elimination info
        # elim_week_by_score contains the week a couple was eliminated
        elim_events = s_panel[['week', 'elim_week_by_score']].drop_duplicates()
        # Filter where elim_week is not NaN
        # Note: elim_week_by_score is valid for the row of the week they were eliminated?
        # Actually usually it's a property of the couple.
        # Let's verify: In S1, elim_week_by_score is 2 for Trista.
        # Is there a row where week=2 and elim_week_by_score=2?
        
        valid_elims = set()
        for idx, row in s_panel.iterrows():
            try:
                ew = float(row['elim_week_by_score'])
                if not pd.isna(ew):
                    valid_elims.add(int(ew))
            except:
                pass
        
        # Withdrawals? 
        # Look for 'withdrawal' in any column if exist, or infer from missing elim
        
        for w in range(1, 12):
            if (s, w) in pred_set:
                continue # Has data
            
            reason = "Unknown"
            
            if w > max_week_in_panel:
                reason = "Season Ended"
            else:
                # Week exists in panel
                if w in valid_elims:
                     # Elimination happened in reality, but missing in preds
                     reason = "Missing Prediction (Data Issue/Finale)"
                     # Check if it was finale
                     if w == max_week_in_panel:
                         reason = "Finale (Rank Only)"
                else:
                    # No elimination in reality
                    reason = "Non-Elimination Week"
            
            reasons.append({
                'season': s,
                'week': w,
                'reason': reason
            })
            
    # Output to CSV
    df = pd.DataFrame(reasons)
    df.to_csv(r'd:\shumomeisai\Code_second\task1.准确性code\missing_data_report.csv', index=False)
    
    # Print summary
    print("\nMissing Data Reasons Summary:")
    print(df.groupby('reason').size())
    
    # Print detailed report for first few seasons
    print("\nDetailed Report (Example S1-S5):")
    print(df[df['season'] <= 5].to_string())

if __name__ == '__main__':
    analyze_missing()
