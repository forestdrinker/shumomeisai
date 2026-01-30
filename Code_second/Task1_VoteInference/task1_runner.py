
import pandas as pd
import numpy as np
import json
import argparse
import os
import joblib
from jax import random
import numpyro
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp

# Import model
from task1_model import probabilistic_model

# Enable 64-bit to avoid overflow in PL mechanism
numpyro.enable_x64()

PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'

def run_season(season, num_warmup=500, num_samples=1000, num_chains=1):
    print(f"--- Processing Season {season} ---")
    
    # Load Data
    df = pd.read_parquet(PANEL_PATH)
    df_s = df[df['season'] == season].copy()
    
    with open(ELIM_PATH, 'r', encoding='utf-8') as f:
        elim_data = json.load(f)
        
    # Dimensions
    # Weeks: 1..Ts
    weeks = sorted(df_s['week'].unique())
    n_weeks = len(weeks)
    week_map = {w: i for i, w in enumerate(weeks)} # week value -> 0-based index
    
    # Pairs: 1..N
    # We use pair_id from data_prep which is 1-based
    pair_ids = sorted(df_s['pair_id'].unique())
    n_pairs = len(pair_ids)
    pair_map = {p: i for i, p in enumerate(pair_ids)} # pair_id -> 0-based index
    
    # Create Name -> ID map for elim parsing
    # We need strictly (Celebrity Name) -> Index
    # Check uniqueness of name in season
    # Taking implicit mapping from first occurrence
    # Better: iterate df_s active rows
    name_to_id = {}
    for _, row in df_s.iterrows():
        p_idx = pair_map[row['pair_id']]
        name_to_id[row['celebrity_name']] = p_idx
        
    # Build Arrays
    # active_mask: (n_weeks, n_pairs)
    # observed_scores: (n_weeks, n_pairs)
    # judge_percents: (n_weeks, n_pairs)
    
    active_mask = np.zeros((n_weeks, n_pairs), dtype=bool)
    observed_scores = np.zeros((n_weeks, n_pairs))
    judge_percents = np.zeros((n_weeks, n_pairs))
    
    # Fill Data
    # Iterate rows
    # Note: data only contains ACTIVE rows.
    # We initialize with 0. Inactive will remain 0/False.
    
    for _, row in df_s.iterrows():
        t = week_map[row['week']]
        i = pair_map[row['pair_id']]
        
        active_mask[t, i] = True
        observed_scores[t, i] = row['S_it']
        judge_percents[t, i] = row['pJ_it']
        
    # Parse Elim Events
    # elim_data keys: "season_week" or "season_week_final"
    # Filter for this season
    
    elim_events_list = [] # List of (t, [indices])
    final_ranking = []
    
    # We rely on week_map. 
    # Final placement is special.
    
    max_week = max(weeks)
    
    # Regular weeks
    for w in weeks:
        # Check standard elim
        key = f"{season}_{w}"
        if key in elim_data:
            data = elim_data[key]
            # Eliminated names
            e_names = data.get('eliminated_names', [])
            e_indices = [name_to_id[n] for n in e_names if n in name_to_id]
            if e_indices:
                elim_events_list.append((week_map[w], e_indices))
    
    # Final
    key_final = f"{season}_{max_week}_final"
    if key_final in elim_data:
        # Finalists list
        # data['finalists'] is just list of names? 
        # But we need RANKING.
        # "Order them? ... final placement ranking observation".
        # data_prep.py logic for logic extraction was:
        # "assuming 'finalists' in json are just names"
        # AND "final placement" logic was added to data_prep later? 
        # Check data_prep.py:
        # It SAVED "finalists": finalists_list.
        # But did it save the ORDER? 
        # In the modified data_prep.py, I added code to sort finalists, but I didn't verify if it updated the JSON structure.
        # Let's check the json content logic in data_prep.py.
        # The JSON save line `all_elim_events[f"{s}_{max_week}_final"] = ... "finalists": finalists`
        # `finalists` variable came from `finalists_df['celebrity_name'].tolist()`.
        # AND I added sort logic `finalists_sorted`. 
        # Did I store `finalists_sorted` into the dict? 
        # Reviewing diff: 
        # I calculated `finalists_sorted` but the dict assignment used `finalists` (the original unsorted list from `season_df[...]['celeb_name']`). 
        # Mistake in data prep? 
        # Let's fix it here via pandas if possible.
        # I have df_s. I can look at 'placement' column for the finalists in the final week.
        
        final_week_df = df_s[df_s['week'] == max_week]
        # It should have 'placement' if it exists.
        if 'placement' in final_week_df.columns:
            # Sort by placement
            # Placement might be string '1', '2'. Convert to float.
            try:
                final_week_df['place_num'] = final_week_df['placement'].astype(float)
            except:
                # Handle 'Runner-up' etc if any? Assuming numeric/string numeric.
                 final_week_df['place_num'] = pd.to_numeric(final_week_df['placement'], errors='coerce').fillna(999)
            
            final_week_df = final_week_df.sort_values('place_num')
            final_ranking_names = final_week_df['celebrity_name'].tolist()
            final_ranking = [name_to_id[n] for n in final_ranking_names if n in name_to_id]
        else:
             print("Warning: No placement column found. Using arbitrary order for final.")
             # Fallback: Just use list from JSON (which might be arbitrary buffer order)
             final_names = elim_data[key_final].get('finalists', [])
             final_ranking = [name_to_id[n] for n in final_names if n in name_to_id]

    # Rule Segment
    # All rows in season should have same rule
    segment = df_s['rule_segment'].iloc[0]
    
    print(f"Data Prep Done. N={n_pairs}, T={n_weeks}, Rule={segment}")
    print(f"Elim Events: {len(elim_events_list)}")
    print(f"Finals: {len(final_ranking)}")
    
    # Run MCMC
    # Increase target_accept_prob to 0.95 to reduce divergences (User Feedback)
    kernel = NUTS(probabilistic_model, target_accept_prob=0.95)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, progress_bar=False)
    
    rng_key = random.PRNGKey(season)
    
    # JAX arrays
    active_mask_jax = jnp.array(active_mask)
    observed_scores_jax = jnp.array(observed_scores)
    judge_percents_jax = jnp.array(judge_percents)
    # elim_events needs to be passed carefully. 
    # Logic in model: `for t, elim_indices in elim_events_list:`
    # Python list is fine for `numpyro` model tracing (it unrolls loop).
    
    mcmc.run(rng_key, 
             n_pairs=n_pairs, 
             n_weeks=n_weeks, 
             active_mask=active_mask_jax,
             observed_scores=observed_scores_jax, 
             judge_percents=judge_percents_jax,
             elim_events=elim_events_list,
             final_ranking=final_ranking,
             rule_segment=segment
    )
    
    mcmc.print_summary()
    
    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    samples = mcmc.get_samples() 
    # Convert active_mask, etc to save too? 
    # Just save samples + pair_id map for interpretation
    
    save_path = os.path.join(OUTPUT_DIR, f"season_{season}.npz")
    # We want u, v. They are in samples.
    # Also save meta info like pair_ids to map back.
    
    # Convert samples to numpy
    samples_np = {k: np.array(v) for k, v in samples.items()}
    
    np.savez(save_path, 
             **samples_np, 
             pair_ids=pair_ids,
             week_values=weeks,
             season=season
    )
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', type=int, nargs='+', help='Seasons to run')
    parser.add_argument('--all', action='store_true', help='Run all seasons')
    parser.add_argument('--test-run', action='store_true', help='Short run for testing')
    
    args = parser.parse_args()
    
    if args.test_run:
        # Run s1 and s29
        # Warmup/Samples small
        seasons_to_run = [1, 29]
        # But wait, does data exist for s29? Data is s1..34?
        # Check data
        pass
        warmup = 50
        samples = 50
    else:
        seasons_to_run = args.seasons if args.seasons else []
        if args.all:
            # Read unique seasons from parquet
            df = pd.read_parquet(PANEL_PATH)
            seasons_to_run = sorted(df['season'].unique())
            
        warmup = 500
        samples = 1000
    
    for s in seasons_to_run:
        try:
             run_season(s, num_warmup=warmup, num_samples=samples)
        except Exception as e:
             print(f"Error running season {s}: {e}")
             import traceback
             traceback.print_exc()

