
import sys
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

# Add parent directory to path to import Task4 modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from Task4_SystemDesign.task4_simulator import SeasonSimulatorFast
except ImportError:
    # Try alternate path if running from root
    sys.path.append(os.path.abspath('Task4_SystemDesign'))
    from task4_simulator import SeasonSimulatorFast

# Config
OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results'
PANEL_PATH = r'd:\shumomeisai\Code_second\Data\panel.parquet'
POSTERIORS_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
ELIM_PATH = r'd:\shumomeisai\Code_second\Data\elim_events.json'
RECOMMEND_PATH = r'd:\shumomeisai\Code_second\Results\task4_recommendations.json'

NOISE_LEVELS = np.linspace(0.0, 1.0, 11) # 0.0 to 1.0
N_SEASONS = 5 # Test on 5 seasons
SEASON_IDS = [1, 2, 3, 4, 5]
K_REPEATS = 10 # Repeats per noise level

def run_stress_test():
    print("--- Running Robustness Stress Test ---")
    
    # 1. Load Recommended Theta
    if not os.path.exists(RECOMMEND_PATH):
        print("Recommendations not found.")
        return
        
    with open(RECOMMEND_PATH, 'r') as f:
        recs = json.load(f)
    
    theta_knee = recs['knee_point']['theta']
    print(f"Knee Point Theta: {theta_knee}")
    
    # 2. Define Baseline Theta (e.g. Current System approximation)
    # Current system: Constant weights, Percent method (eta=1), No save
    theta_baseline = {
        'a': 0.1, # Flat curve
        'b': 0.0,
        'eta': 1.0, # Linear vote
        'gamma': 1.0, 
        'save_flag': 0 
    }
    print(f"Baseline Theta: {theta_baseline}")
    
    # 3. Initialize Simulator
    sim = SeasonSimulatorFast(PANEL_PATH, POSTERIORS_DIR, ELIM_PATH)
    for s in SEASON_IDS:
        sim.preload_season(s)
        
    results = []
    
    for kappa in NOISE_LEVELS:
        print(f"Testing Noise Level kappa={kappa:.2f}...")
        
        taus_knee = []
        taus_base = []
        
        for s in SEASON_IDS:
            # Ground Truth (No Noise)
            # Run simulation with 0 noise to get "Original" placement
            p0_knee, _ = sim.simulate(theta_knee, s, perturb_kappa=0.0)
            mp0_knee = np.mean(p0_knee, axis=0) # Mean placement
            
            p0_base, _ = sim.simulate(theta_baseline, s, perturb_kappa=0.0)
            mp0_base = np.mean(p0_base, axis=0)
            
            # Perturbed Runs
            for k in range(K_REPEATS):
                # Knee
                pp_knee, _ = sim.simulate(theta_knee, s, perturb_kappa=kappa)
                mpp_knee = np.mean(pp_knee, axis=0)
                tau_k, _ = kendalltau(mp0_knee, mpp_knee)
                taus_knee.append(tau_k)
                
                # Base
                pp_base, _ = sim.simulate(theta_baseline, s, perturb_kappa=kappa)
                mpp_base = np.mean(pp_base, axis=0)
                tau_b, _ = kendalltau(mp0_base, mpp_base)
                taus_base.append(tau_b)
        
        # Average across seasons and repeats
        res = {
            'noise_level': kappa,
            'tau_knee': np.mean(taus_knee) if taus_knee else 1.0,
            'tau_baseline': np.mean(taus_base) if taus_base else 1.0
        }
        results.append(res)
        
    # Save
    df = pd.DataFrame(results)
    path = os.path.join(OUTPUT_DIR, 'stress_test.csv')
    df.to_csv(path, index=False)
    print(f"Saved stress test results to {path}")
    print(df)

if __name__ == "__main__":
    run_stress_test()
