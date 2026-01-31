
import os
import time
import subprocess
import sys

def run_script(script_name):
    print(f"[{time.strftime('%H:%M:%S')}] Starting {script_name}...")
    try:
        # Use simple subprocess call
        cmd = [sys.executable, script_name]
        subprocess.check_call(cmd)
        print(f"[{time.strftime('%H:%M:%S')}] Finished {script_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        sys.exit(1)

def main():
    print("=== Task 3: Attribution Analysis Runner ===")
    
    # 1. Data Prep
    run_script("task3_prep.py")
    
    # 2. LMM Analysis
    run_script("task3_lmm.py")
    
    # 3. GBDT Analysis
    run_script("task3_gbdt.py")
    
    print("=== Task 3 Completed Successfully ===")

if __name__ == "__main__":
    main()
