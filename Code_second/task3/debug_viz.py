import pandas as pd
import numpy as np
import os

RESULTS_DIR = r'd:\shumomeisai\Code_second\Results\task3_analysis'

def debug_coeffs():
    print("Debugging Coefficients Loading...")
    
    # Judge
    f_j = os.path.join(RESULTS_DIR, 'task3_lmm_judge_coeffs.csv')
    try:
        cj = pd.read_csv(f_j, index_col=0).reset_index()
        cj.rename(columns={'index': 'term'}, inplace=True)
        print(f"Judge Coeffs: {len(cj)} rows")
        print(cj.head())
    except Exception as e:
        print(f"Judge Load Error: {e}")
        cj = pd.DataFrame()

    # Fan
    f_f = os.path.join(RESULTS_DIR, 'task3_lmm_fan_coeffs_aggregated.csv')
    try:
        cf = pd.read_csv(f_f, index_col=0).reset_index()
        cf.rename(columns={'index': 'term'}, inplace=True)
        if 'mean' in cf.columns:
            cf.rename(columns={'mean': 'estimate'}, inplace=True)
        print(f"Fan Coeffs: {len(cf)} rows")
        print(cf.head())
    except Exception as e:
        print(f"Fan Load Error: {e}")
        cf = pd.DataFrame()
        
    # Intersect
    if 'term' in cj.columns and 'term' in cf.columns:
        terms = sorted([t for t in cj['term'] if t in cf['term'].values])
        print(f"Common terms ({len(terms)}): {terms}")
        
        for t in terms:
            print(f"Processing term: {t}")
            try:
                rj = cj[cj['term'] == t]
                if rj.empty:
                    print(f"  Judge missing {t}")
                else:
                    print(f"  Judge found: {rj.iloc[0]['estimate']}")
                    
                rf = cf[cf['term'] == t]
                if rf.empty:
                    print(f"  Fan missing {t}")
                else:
                    print(f"  Fan found: {rf.iloc[0]['estimate']}")
            except Exception as e:
                print(f"  Error accessing {t}: {e}")
    else:
        print("Missing 'term' column in one of the dataframes")

if __name__ == "__main__":
    debug_coeffs()
