import pandas as pd
import numpy as np
import os
import glob

REPLAY_DIR = r'd:\shumomeisai\Code_second\Results\replay_results'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'

def check_weekly_diff():
    print("Checking weekly_diff.csv files...")
    for s in range(1, 35):
        fpath = os.path.join(REPLAY_DIR, f"season_{s}_weekly_diff.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            n_weeks = df['week'].nunique()
            if 'p_elim_diff' in df.columns:
                p_values = df['p_elim_diff'].unique()
                is_binary = all(v in [0.0, 1.0] for v in p_values)
                print(f"S{s}: {n_weeks} weeks, Binary={is_binary}, Values={sorted(p_values)[:5]}")
            else:
                 print(f"S{s}: Missing 'p_elim_diff' column")
        else:
            print(f"S{s}: MISSING weekly_diff.csv!")

def check_season_11_npz():
    print("\nChecking Season 11 NPZ for Bristol Palin...")
    fpath = os.path.join(REPLAY_DIR, "season_11_rank.npz")
    if os.path.exists(fpath):
        data = np.load(fpath, allow_pickle=True)
        pair_ids = data['pair_ids']
        print(f"Season 11 Pair IDs in NPZ: {pair_ids}")
        print(f"Pair IDs type: {type(pair_ids)}")
        if len(pair_ids) > 0:
            print(f"Element type: {type(pair_ids[0])}")
            
        # Check panel
        if os.path.exists(PANEL_PATH):
            df_panel = pd.read_csv(PANEL_PATH)
            s11_panel = df_panel[df_panel['season'] == 11]
            print("\nSeason 11 Panel Data (first 5):")
            print(s11_panel[['pair_id', 'celebrity_name']].head())
            
            # Check for Bristol Palin specifically
            bp = s11_panel[s11_panel['celebrity_name'].str.contains("Bristol", case=False)]
            print(f"\nBristol Palin in Panel:\n{bp[['pair_id', 'celebrity_name']]}")
            
            # Simulate matching logic
            pair_name_map = df_panel.set_index(['season', 'pair_id'])['celebrity_name'].to_dict()
            name = "Bristol Palin"
            season = 11
            
            print("\nMatching attempt:")
            for col_idx, pid in enumerate(pair_ids):
                # Ensure pid is same type as dict key (likely int)
                # If pair_ids are strings in NPZ but ints in key, that's the issue
                
                key = (season, pid)
                celeb_name = pair_name_map.get(key, 'NOT_FOUND')
                print(f"  Col {col_idx}, PID {pid} (type {type(pid)}) -> Key {key} -> Name: {celeb_name}")
                
    else:
        print("season_11_rank.npz not found")

if __name__ == "__main__":
    import sys
    with open('debug_log.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f
        check_weekly_diff()
        check_season_11_npz()
        sys.stdout = sys.__stdout__
    print("Log written to debug_log.txt")
