import pandas as pd
import numpy as np
import json
import os

PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
SEASON = 11

def debug_loader():
    print(f"Loading Season {SEASON} from {PANEL_PATH}")
    df = pd.read_csv(PANEL_PATH)
    df_s = df[df['season'] == SEASON].copy()
    
    pair_ids = sorted(df_s['pair_id'].unique())
    print(f"Unique Pair IDs in DataFrame: {pair_ids}")
    
    for pid in pair_ids:
        name = df_s[df_s['pair_id'] == pid]['celebrity_name'].iloc[0]
        print(f"  PID {pid}: {name}")
        
    # Check logic from task1_runner.py
    weeks = sorted(df_s['week'].unique())
    print(f"Weeks: {weeks}")
    
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_data = json.load(f)
        
    print("\nChecking Elim Data:")
    for w in weeks:
        key = f"{SEASON}_{w}"
        if key in elim_data:
            print(f"  Week {w}: {elim_data[key].get('eliminated_names', [])}")
            
    key_final = f"{SEASON}_{max(weeks)}_final"
    if key_final in elim_data:
        print(f"  Finals: {elim_data[key_final].get('finalists', [])}")

if __name__ == "__main__":
    debug_loader()
