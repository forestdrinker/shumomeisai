import pandas as pd
import numpy as np
import json
import re
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

INPUT_FILE = r'd:\shumomeisai\Code_second\2026_MCM_Problem_C_Data.csv'
PANEL_OUTPUT = r'd:\shumomeisai\Code_second\panel.parquet'
ELIM_OUTPUT = r'd:\shumomeisai\Code_second\elim_events.json'

def load_and_preprocess():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Identify columns
    # We expect: season, celebrity_name, ballroom_partner (or similar)
    # And score columns: weekX_judgeY_score
    
    # Let's standardize column names for easier access if needed, but for now we look for the pattern
    # Pattern: week(\d+)_judge(\d+)_score
    
    score_cols = [c for c in df.columns if re.match(r'week\d+_judge\d+_score', c)]
    print(f"Found {len(score_cols)} score columns.")
    
    # Static columns (metadata)
    # We want to keep everything that is NOT a score column as ID/Metadata
    id_vars = [c for c in df.columns if c not in score_cols]
    
    print("Reshaping to long format...")
    # Melt
    df_long = df.melt(id_vars=id_vars, value_vars=score_cols, var_name='temp_idx', value_name='score')
    
    # Extract week and judge from column name
    # column format: weekX_judgeY_score
    # We extract X and Y
    # Regex extraction
    pat = r'week(\d+)_judge(\d+)_score'
    extracted = df_long['temp_idx'].str.extract(pat)
    df_long['week'] = extracted[0].astype(int)
    df_long['judge'] = extracted[1].astype(int)
    
    # Drop temp_idx
    df_long.drop(columns=['temp_idx'], inplace=True)
    
    # Rename basic columns to match spec if needed
    # Spec says: season, celebrity, partner
    # Data has: celebrity_name, ballroom_partner, season
    # We will standardize to: season, celebrity_name, ballroom_partner
    
    # 2. Pivot to get (season, week, celebrity, partner) -> [score_j1, score_j2, score_j3, score_j4]
    # Actually, the aggregated S_it calculation is easier if we have one row per (s, t, i) and columns or list for scores.
    # Group by (s, t, i)
    
    # Ensure scores are numeric
    df_long['score'] = pd.to_numeric(df_long['score'], errors='coerce')
    
    # Create the 'pair_id' or just use (celebrity, partner) as unique key
    # Let's create a unique ID for the pair within the season if not exists
    # Or just use the names.
    
    # Grouping keys
    group_keys = ['season', 'week', 'celebrity_name', 'ballroom_partner']
    # Include other metadata in group keys if we want to preserve them (like age, country)
    # Be careful not to explode rows if metadata is consistent. 
    # For now, let's just aggregate scores and join metadata later if needed, OR just include ALL id_vars in grouping.
    # However, 'placement' might be per season? Check if 'placement' is in id_vars.
    # Yes, placement should be in id_vars.
    
    print("Aggregating scores...")
    
    # Function to process the scores for a single (s,t,i)
    # We need:
    # 1. Check if all 4 are 0 or NaN -> Inactive
    # 2. Calculate Mean s_{i,t,j}
    
    # Pivot table: Index=(season, week, celebrity, partner, ...), Columns=judge, Values=score
    try:
        df_pivot = df_long.pivot_table(
            index=id_vars + ['week'], 
            columns='judge', 
            values='score', 
            aggfunc='first' # Should be unique per judge
        ).reset_index()
    except Exception as e:
        print(f"Pivot failed: {e}. duplicate entries?")
        # If duplicates, maybe aggregate?
        df_pivot = df_long.pivot_table(
             index=id_vars + ['week'], 
             columns='judge', 
             values='score', 
             aggfunc='mean'
        ).reset_index()

    # Determine "Active" Status (Hard Rule 0.2)
    # Scores are in columns 1, 2, 3, 4 (integers)
    judge_cols = [1, 2, 3, 4]
    available_judge_cols = [c for c in df_pivot.columns if isinstance(c, int)]
    print(f"Judges found: {available_judge_cols}")
    
    # Helper to check active
    def check_active(row):
        scores = [row[j] for j in available_judge_cols if not pd.isna(row[j])]
        if not scores: # All NaN
            return False
        if all(s == 0 for s in scores): # All 0
            return False
        # Special case: max=0 (same as all 0 if non-negative)
        if max(scores) == 0: 
            return False
        return True

    df_pivot['is_active'] = df_pivot.apply(check_active, axis=1)
    
    df_active = df_pivot[df_pivot['is_active']].copy()
    
    # Add pair_id (i in 1..n_s)
    # Sort by season then celebrity to be reproducible
    df_active.sort_values(by=['season', 'celebrity_name'], inplace=True)
    
    # Create dictionary mapping (season, celebrity) -> pair_id (1-based index)
    pair_map = {}
    for s in df_active['season'].unique():
        s_df = df_active[df_active['season'] == s]
        unique_celebs = sorted(s_df['celebrity_name'].unique())
        for idx, celeb in enumerate(unique_celebs):
            pair_map[(s, celeb)] = idx + 1
            
    df_active['pair_id'] = df_active.apply(lambda row: pair_map.get((row['season'], row['celebrity_name'])), axis=1)
    
    # Calculate S_it (Average of observed judges)
    df_active['S_it'] = df_active[available_judge_cols].mean(axis=1)
    
    # Count judges
    df_active['n_judges'] = df_active[available_judge_cols].notna().sum(axis=1)
    
    # Calculate Judge Percent (p^J_it) per (season, week)
    print("Calculating metrics...")
    
    season_week_sum = df_active.groupby(['season', 'week'])['S_it'].transform('sum')
    df_active['pJ_it'] = df_active['S_it'] / season_week_sum
    
    # Rule Segment Marking (0.3)
    def get_rule(s):
        if s <= 2: return 'rank'
        if 3 <= s <= 27: return 'percent'
        return 'rank_save'
        
    df_active['rule_segment'] = df_active['season'].apply(get_rule)
    
    # 0.4 Elimination Logic
    print("Extracting elimination events...")
    all_elim_events = {}
    
    seasons = df_active['season'].unique()
    
    for s in sorted(seasons):
        season_df = df_active[df_active['season'] == s]
        max_week = season_df['week'].max()
        
        # Determine finalists (A_s, Ts) - Ensure we look at the last active week survivors
        # Or specifically those active in max_week
        finalists_df = season_df[season_df['week'] == max_week]
        
        # Collect finalists with their placements if available
        # Check if 'placement' column exists safely
        if 'placement' in finalists_df.columns:
             # Sort by placement if possible (handle NaN or strings)
             # If placement is numeric, great. If string '1', '2', convert.
             # Note: 'placement' in csv is likely the FINAL placement for the season, replicated across all rows.
             # We should verify this assumption.
             pass
             
        finalists = finalists_df['celebrity_name'].tolist()
        
        # We also need the Final Ordering for Task 1 (pi_s)
        # Using 'placement' column.
        final_placement_map = {}
        if 'placement' in season_df.columns:
            # Get one value per celebrity (assuming it's constant for the pair)
            # We take the value from the last week row
             for celeb in finalists:
                 val = finalists_df[finalists_df['celebrity_name'] == celeb]['placement'].iloc[0]
                 final_placement_map[celeb] = val
        
        # Sort finalists by placement
        # Handle '1', '2', '8', '11' etc.
        try:
            finalists_sorted = sorted(finalists, key=lambda x: float(final_placement_map.get(x, 999)))
        except:
             # Fallback if placement format is weird
            finalists_sorted = finalists
            
        weeks = sorted(season_df['week'].unique())

        
        for i in range(len(weeks) - 1):
            t_curr = weeks[i]
            t_next = weeks[i+1]
            
            curr_celebs = set(season_df[season_df['week'] == t_curr]['celebrity_name'])
            next_celebs = set(season_df[season_df['week'] == t_next]['celebrity_name'])
            
            # Active set A_st
            active_list = list(curr_celebs)
            
            # Eliminated set E_st
            eliminated = list(curr_celebs - next_celebs)
            
            all_elim_events[f"{s}_{t_curr}"] = {
                "season": int(s),
                "week": int(t_curr),
                "active_count": len(active_list),
                "eliminated_count": len(eliminated),
                "active_names": active_list,
                "eliminated_names": eliminated
            }
        
        # Final week info
        all_elim_events[f"{s}_{max_week}_final"] = {
             "season": int(s),
             "week": int(max_week),
             "is_final": True,
             "finalists": finalists
        }

    # Save Elim Events
    with open(ELIM_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_elim_events, f, ensure_ascii=False, indent=2)
        
    print(f"Elimination events saved to {ELIM_OUTPUT}")
    
    # Save Panel
    # Rename columns to strictly match nice names if desired, but current are okay.
    # judge columns are integers (1,2,3,4). Let's rename them to judge_1, judge_2 etc for parquet compatibility (string cols preferred).
    
    df_active.columns = [str(c) if isinstance(c, int) else c for c in df_active.columns]
    
    # Ensure all object columns are strings
    for c in df_active.select_dtypes(include=['object']).columns:
        df_active[c] = df_active[c].astype(str)
        
    df_active.to_parquet(PANEL_OUTPUT, index=False)
    print(f"Panel saved to {PANEL_OUTPUT} with shape {df_active.shape}")

if __name__ == "__main__":
    load_and_preprocess()
