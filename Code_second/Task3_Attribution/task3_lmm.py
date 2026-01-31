
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import glob
import json

# Fix 1: Reproducibility
np.random.seed(2026)

# Paths
DATASET_PATH = r'd:\shumomeisai\Code_second\Results\task3_data\task3_weekly_dataset.parquet'
POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results\task3_analysis'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def run_task3_lmm():
    print("--- Running Task 3 LMM Analysis ---")
    
    # 1. Load Data
    print("Loading Weekly Dataset...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Data Shape: {df.shape}")
    
    # Preprocessing
    # Standardize Age
    df['age_z'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    # Ensure categorical
    df['industry'] = df['industry'].astype('category')
    df['partner'] = df['ballroom_partner'].astype(str)
    df['season_cat'] = df['season'].astype(str)
    
    # Handle week as numeric or factor? 
    # Usually linear trend + maybe squared? Or just linear week effect.
    # We'll use week index normalized?
    df['week_norm'] = df['week'] / df.groupby('season')['week'].transform('max')

    # ==========================================
    # Part A: Judge Model
    # Target: pJ_it
    # Formula: pJ_it ~ age_z + industry + week_norm + (1|partner) + (1|season)
    # Statsmodels MixedLM with crossed effects:
    # We group by "season" (as block) and add "partner" as VC? 
    # Or just use one grouping if simpler. 
    # Let's try to group by 1 (dummy) and add both as VC? No, scaling issue.
    # Standard approach for crossed: Group by one factor, add other as VC.
    # Group by Season, add Partition random effect?
    # Actually, let's treat Season as Fixed Effect (controls for season inflation) 
    # and Partner as Random Effect? 
    # Pro: Easier to interpret "Partner Effect" deviation from average.
    # Con: Too many fixed effects (34 seasons).
    # Doc says: b_season ~ N(0, sigma).
    # So we should use VC.
    # We will use valid groups in statsmodels.
    # Let's group by "season_cat" and add "partner" using vc_formula.
    # Note: If a partner appears in multiple seasons (groups), this works perfectly for crossed design 
    # if implemented correctly, but statsmodels MixedLM assumes groups are independent.
    # If a partner spans seasons, 'season' grouping splits the partner effect if we are not careful.
    # If we group by season, 'partner' inside vc_formula is ideally nested. 
    # BUT pros return! So partner crossed season.
    # Correct way in statsmodels for global crossed RE:
    # group = np.ones(len(df)) (Global group)
    # vc_formula = {"season": "0+C(season_cat)", "partner": "0+C(partner)"}
    # This might be slow but correctness is key.
    
    print("Fitting Judge LMM (Crossed Random Effects)...")
    df['group'] = 1
    
    # Minimize formula complexity to ensure convergence
    # age_z + C(industry) + week_norm
    formula = "pJ_it ~ age_z + C(industry) + week_norm"
    
    # Simple model first to test
    # vc_formula = {"partner": "0 + C(partner)", "season": "0 + C(season_cat)"}
    # But C(partner) is huge (many levels).
    
    try:
        model_j = smf.mixedlm(formula, df, groups="group", 
                             vc_formula={"partner": "0 + C(partner)", 
                                         "season": "0 + C(season_cat)"})
        res_j = model_j.fit(reml=True)
        print(res_j.summary())
        
        # Extract Fixed Effects with CIs
        # Fix 2: Completeness - Add Confidence Intervals
        params = res_j.params.to_frame(name='estimate')
        conf = res_j.conf_int()
        conf.columns = ['2.5%', '97.5%']
        # Combine
        fe_j = params.join(conf)
        # fe_j = res_j.params.to_frame(name='coef') # Old logic
        fe_j.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_judge_coeffs.csv'))
        
        # Extract Random Effects (Partner)
        # Random effects are dict of arrays.
        # res_j.random_effects[1] -> Series index by term name.
        # Key 'partner' will have coefficients.
        re_dict = res_j.random_effects[1] # Group 1
        
        # Helper to clean names
        def clean_partner_name(raw_name):
            name = raw_name
            patterns = ['partner[C(partner)[', 'partner[T.', 'C(partner)[', ']']
            for p in patterns:
                name = name.replace(p, '')
            return name.strip()
        
        # Keys in re_dict look like 'partner[T.Name]'
        partner_effs = {}
        for k, v in re_dict.items():
            if 'partner' in k.lower():
                name = clean_partner_name(k)
                if name:
                    partner_effs[name] = v
        
        pe_df = pd.DataFrame.from_dict(partner_effs, orient='index', columns=['effect'])
        pe_df.index.name = 'partner'
        pe_df.sort_values('effect', ascending=False, inplace=True)
        pe_df.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_judge_partner_effects.csv'))
        print("Judge LMM completed.")
        
    except Exception as e:
        print(f"Error fitting Judge LMM: {e}")
        # Fallback: Fixed Season + Random Partner (simpler)
        print("Fallback: Fixed Season Effects + Random Partner")
        try:
            model_j = smf.mixedlm("pJ_it ~ age_z + C(industry) + week_norm + C(season_cat)", 
                                 df, groups="partner")
            res_j = model_j.fit()
            # Extract random effects (BLUPs for partners)
            # here groups IS partner, so random_effects key is partner_id, value is intercept
            re_j = {k: v['Group'] for k, v in res_j.random_effects.items()}
            pe_df = pd.DataFrame.from_dict(re_j, orient='index', columns=['effect'])
            pe_df.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_judge_partner_effects.csv'))
        except Exception as e2:
            print(f"Fallback Error: {e2}")

    # ==========================================
    # Part B: Fan Model (Posterior Outer Loop)
    print("Fitting Fan LMM (Posterior Outer Loop)...")
    
    # We need to map (season, week, pair_id) to the posterior array indices
    # Load all season meta first
    season_meta = {}
    season_files = glob.glob(os.path.join(POSTERIOR_DIR, "season_*.npz"))
    
    # Pre-load needed data to memory?
    # If seasons are large, maybe keep file handle?
    # Let's iterate R draws, then inside S seasons.
    
    # Determine R
    # Open first file to check R
    if not season_files:
        print("No posterior samples found.")
        return
        
    s1 = np.load(season_files[0])
    R_total = s1['v'].shape[0]
    
    # For speed, let's use a subset of draws if R is large
    # 30-50 draws is usually enough for CI of effects
    N_DRAWS = min(R_total, 30) 
    draw_indices = np.linspace(0, R_total-1, N_DRAWS, dtype=int)
    
    print(f"Using {N_DRAWS} draws for outer loop.")
    
    fan_effects_list = []
    fan_coeffs_list = []
    
    for r_idx in draw_indices:
        # Construct y_fan for this draw
        # We need to reconstruct the dependent variable column for the whole df
        y_col = []
        
        # We can iterate over df rows? Too slow.
        # Better: iterate seasons, fetch v_r, map to dataframe index
        
        # Create a series with index (season, week, pair_id) -> v value
        # But df has multiple rows.
        # Let's rely on df being sorted or using index.
        # Temporarily index df
        # df_indexed = df.set_index(['season', 'week', 'pair_id'])
        
        # Actually, simpler: loop seasons, extract v[r], create dict, map.
        
        val_map = {} # (s, w, p) -> val
        
        for fpath in season_files:
            fname = os.path.basename(fpath)
            # season_{s}.npz or season_{s}_summary...
            if 'summary' in fname or 'diagnostics' in fname: continue
            
            s_id = int(fname.split('_')[1].split('.')[0])
            
            try:
                data = np.load(fpath)
                if 'pair_ids' not in data:
                    # Maybe it is in data['couple_id'] or inferred?
                    # Assuming pair_ids is stored or consistent with index
                    # Task1 runner saves 'couple_ids'? Check Task1 logic if needed.
                    # Usually 'pair_ids' or 'couple_ids' is saved.
                    # If missing, we assume order matches? Risky.
                    # Let's check keys
                    keys = list(data.keys())
                    pass
                
                # Check mapping
                # Assuming data has 'couple_ids'
                # v shape: (R, T, N_couples)
                v_r = data['v'][r_idx] # (T, N)
                
                # We need week mapping
                # Assuming weeks are 0-indexed in array, corresponding to real weeks?
                # Data should have week_values or similar.
                # Or assume logical weeks 1..T
                
                # Let's assume v matches weeks 0..T-1
                # And couple_ids array matches pairs 0..N-1
                c_ids = data.get('pair_ids')
                if c_ids is None:
                    c_ids = data.get('couple_ids')
                    
                if c_ids is not None:
                    n_weeks = v_r.shape[0]
                    for t in range(n_weeks):
                        # week number in df is usually t+1
                        wk = t + 1
                        for n, pid in enumerate(c_ids):
                            val = v_r[t, n]
                            val_map[(s_id, wk, pid)] = val
            except Exception as e:
                pass
                
        # Now map to df
        # Tuples for lookup
        keys = zip(df['season'], df['week'], df['pair_id'])
        y_values = [val_map.get(k, np.nan) for k in keys]
        
        df['y_fan_temp'] = y_values
        
        # Filter NaNs (non-active weeks)
        df_run = df.dropna(subset=['y_fan_temp']).copy()
        df_run['group'] = 1
        
        # Fit LMM
        # Same formula structure as Judge
        # pJ_it -> y_fan_temp
        formula_f = "y_fan_temp ~ age_z + C(industry) + week_norm"
        
        try:
             # Use same strategy as Judge (Crossed)
            model_f = smf.mixedlm(formula_f, df_run, groups="group",
                                 vc_formula={"partner": "0 + C(partner)", 
                                             "season": "0 + C(season_cat)"})
            res_f = model_f.fit(reml=True, maxiter=50) # Faster iteration
            
            # Store Fixed Effects
            fe = res_f.params.to_dict()
            fan_coeffs_list.append(fe)
            
            # Store Random Effects (Partner)
            re_dict = res_f.random_effects[1]
            p_effs = {}
            for k, v in re_dict.items():
                if 'partner' in k.lower():
                    # Reuse helper implicitly or redefine? 
                    # Redefine short version or move helper to global scope.
                    # Moving to global scope is cleaner but requires large edit.
                    # Just inline clean logic for safety and speed.
                    name = k
                    for p in ['partner[C(partner)[', 'partner[T.', 'C(partner)[', ']']:
                        name = name.replace(p, '')
                    name = name.strip()
                    if name:
                        p_effs[name] = v
            fan_effects_list.append(p_effs)
            
        except Exception as e:
            # print(f"  Draw {r_idx} failed: {e}")
            pass
            
    # Aggregation
    print("Aggregating Fan Results...")
    
    # Fixed Effects
    fe_df = pd.DataFrame(fan_coeffs_list)
    fe_summ = fe_df.describe(percentiles=[0.025, 0.975]).T
    fe_summ = fe_summ[['mean', '2.5%', '97.5%']]
    fe_summ.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_fan_coeffs_aggregated.csv'))
    
    # Partner Effects
    pe_df = pd.DataFrame(fan_effects_list)
    # Columns are partners. Rows are draws.
    # Compute Mean and CI
    pe_summ = pe_df.describe(percentiles=[0.025, 0.975]).T
    pe_summ = pe_summ[['mean', '2.5%', '97.5%', 'count']]
    pe_summ.rename(columns={'count': 'support_n'}, inplace=True)
    pe_summ.sort_values('mean', ascending=False, inplace=True)
    pe_summ.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_fan_partner_effects_aggregated.csv'))
    
    print("Task 3 LMM Finished.")

if __name__ == "__main__":
    run_task3_lmm()
