
import pandas as pd
import numpy as np

def verify_outputs():
    print("--- Verifying Task 2 --")
    try:
        t2 = pd.read_csv(r'd:\shumomeisai\Code_second\Results\task2_metrics.csv')
        # Check Rounding (heuristic: check if string len of float is short, or just strict equality)
        # Check first 5 rows 'suspense_H'
        print("Suspense_H sample:", t2['suspense_H'].head().tolist())
        print("Drama_D sample:", t2['drama_D'].head().tolist())
    except Exception as e:
        print(f"Task 2 Error: {e}")

    print("\n--- Verifying Task 3 (Dataset) --")
    try:
        # Check parquet directly for cleaning
        t3 = pd.read_parquet(r'd:\shumomeisai\Code_second\Results\task3_data\task3_weekly_dataset.parquet')
        inds = t3['industry'].unique()
        print("Industries found:", sorted(map(str, inds)))
        
        if 'Beauty Pagent' in inds: print("FAIL: 'Beauty Pagent' still exists")
        else: print("PASS: 'Beauty Pagent' removed")
        
        if 'Con artist' in inds: print("FAIL: 'Con artist' still exists")
        else: print("PASS: 'Con artist' removed")
        
        if 'Personality' in inds: print("PASS: 'Personality' present")
    except Exception as e:
        print(f"Task 3 Error: {e}")

    print("\n--- Verifying Task 4 --")
    try:
        t4 = pd.read_csv(r'd:\shumomeisai\Code_second\Results\task4_pareto_front.csv')
        # Check Gamma NaNs
        if t4['gamma'].isnull().any():
            print("FAIL: NaNs found in gamma")
        else:
            print("PASS: No NaNs in gamma")
        
        print("Gamma sample:", t4['gamma'].head().tolist())
    except Exception as e:
        print(f"Task 4 Error: {e}")

if __name__ == "__main__":
    verify_outputs()
