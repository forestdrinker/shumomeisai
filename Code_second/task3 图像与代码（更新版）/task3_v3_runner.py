"""
Task 3 Runner — V3 (Dual-Channel Attribution)
==============================================
Runs: prep → LMM → GBDT in sequence.
"""

import os, sys, time, subprocess

def run(script):
    print(f"\n{'─'*50}")
    print(f"[{time.strftime('%H:%M:%S')}] Starting {script}...")
    print(f"{'─'*50}")
    try:
        subprocess.check_call([sys.executable, script])
        print(f"[{time.strftime('%H:%M:%S')}] ✅ {script} finished.")
    except subprocess.CalledProcessError as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ {script} failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("  Task 3: Attribution Analysis (V3 — Dual Channel)")
    print("=" * 60)

    run("task3_v3_prep.py")
    run("task3_v3_lmm.py")
    run("task3_v3_gbdt.py")

    print(f"\n{'='*60}")
    print("  All Task 3 scripts completed successfully!")
    print(f"{'='*60}")
