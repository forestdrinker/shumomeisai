
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = r'd:\shumomeisai\Code_second\Results'
PARETO_PATH = r'd:\shumomeisai\Code_second\Results\task4_pareto_front.csv'
RECOMMEND_PATH = r'd:\shumomeisai\Code_second\Results\task4_recommendations.json'

def analyze_task4():
    print("--- Analyzing Task 4 Results ---")
    
    if not os.path.exists(PARETO_PATH):
        print("Pareto front not found.")
        return
        
    df = pd.read_csv(PARETO_PATH)
    print(f"Loaded Pareto Front: {len(df)} points")
    
    # Identify Key Points
    # 1. Audience Favorite (Max Obj_F)
    idx_fan = df['obj_F_mean'].idxmax()
    fan_point = df.iloc[idx_fan].to_dict()
    
    # 2. Judge Favorite (Max Obj_J)
    idx_judge = df['obj_J_mean'].idxmax()
    judge_point = df.iloc[idx_judge].to_dict()
    
    # 3. Knee Point (Balanced)
    # Normalize objectives to [0, 1] relative to front
    cols = ['obj_F_mean', 'obj_J_mean', 'obj_D_mean', 'obj_R_mean']
    norm_df = df[cols].copy()
    
    min_vals = norm_df.min()
    max_vals = norm_df.max()
    
    # Avoid div by zero
    diffs = max_vals - min_vals
    diffs[diffs == 0] = 1.0
    
    norm_df = (norm_df - min_vals) / diffs
    
    # Ideal point is (1, 1, 1, 1) in normalized space (since we maximize all)
    ideal = np.ones(4)
    dists = norm_df.apply(lambda row: np.linalg.norm(row - ideal), axis=1)
    
    idx_knee = dists.idxmin()
    knee_point = df.iloc[idx_knee].to_dict()
    
    # Construct JSON
    def get_metrics_dict(row):
        cols = ['obj_F_mean', 'obj_J_mean', 'obj_D_mean', 'obj_R_mean',
                'obj_F_sd', 'obj_J_sd', 'obj_D_sd', 'obj_R_sd']
        d = {}
        for c in cols:
            if c in row:
                d[c] = row[c]
        return d
    
    def clean_theta(theta):
        # Replace NaN with None (null in JSON)
        cleaned = {}
        for k, v in theta.items():
            if k == 'gamma' and np.isnan(v):
                cleaned[k] = None
            else:
                cleaned[k] = v
        return cleaned

    recommendations = {
        "knee_point": {
            "theta": clean_theta({k: knee_point[k] for k in ['a','b','eta','gamma','save_flag']}),
            "metrics": get_metrics_dict(knee_point),
            "description": "Balanced trade-off closest to ideal point (1,1,1,1)."
        },
        "fan_favor_point": {
            "theta": clean_theta({k: fan_point[k] for k in ['a','b','eta','gamma','save_flag']}),
            "metrics": get_metrics_dict(fan_point),
            "description": "Maximizes alignment with audience choices."
        },
        "judge_favor_point": {
            "theta": clean_theta({k: judge_point[k] for k in ['a','b','eta','gamma','save_flag']}),
            "metrics": get_metrics_dict(judge_point),
            "description": "Maximizes alignment with technical judge scores."
        }
    }
    
    with open(RECOMMEND_PATH, 'w') as f:
        json.dump(recommendations, f, indent=2)
        
    print(f"Saved recommendations to {RECOMMEND_PATH}")
    
    # Optional: Plot Trade-off (F vs J)
    plt.figure(figsize=(8,6))
    plt.scatter(df['obj_F_mean'], df['obj_J_mean'], c='gray', alpha=0.5, label='Pareto Points')
    plt.scatter(knee_point['obj_F_mean'], knee_point['obj_J_mean'], c='red', marker='*', s=200, label='Knee')
    plt.scatter(fan_point['obj_F_mean'], fan_point['obj_J_mean'], c='blue', marker='s', s=100, label='Fan Fav')
    plt.scatter(judge_point['obj_F_mean'], judge_point['obj_J_mean'], c='green', marker='^', s=100, label='Judge Fav')
    plt.xlabel('Audience Alignment (Rho F)')
    plt.ylabel('Technical Alignment (Rho J)')
    plt.title('Task 4: Trade-off Analysis')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(OUTPUT_DIR, 'task4_tradeoff_plot.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    analyze_task4()
