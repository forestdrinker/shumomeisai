import pandas as pd
import numpy as np
import os
from scipy.stats import beta

DATA_DIR = r"d:\shumomeisai\Code_second\Results"
OUTPUT_DIR = r"d:\shumomeisai\Code_second\Results\final_figure_data"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_csv(df, name):
    if df is None or df.empty:
        print(f"Skipping {name}: Data empty")
        return
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"Saved {path}")

# Fig 1: Schematic - No data

# Fig 2: Validation
def export_fig2():
    print("Exporting Fig 2 data...")
    path = os.path.join(DATA_DIR, "task1_metrics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        save_csv(df, "MainFig2_Validation.csv")

# Fig 3: Posterior Shares
def export_fig3():
    print("Exporting Fig 3 data...")
    # Logic from plotting script: Season 27, top 6
    target_season = 27
    path = os.path.join(DATA_DIR, "posterior_samples", f"season_{target_season}_summary.csv")
    if not os.path.exists(path):
         target_season = 19
         path = os.path.join(DATA_DIR, "posterior_samples", f"season_{target_season}_summary.csv")
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        last_week = df['week'].max()
        finalists = df[df['week'] == last_week].sort_values('v_mean', ascending=False).head(6)
        top_ids = finalists['pair_id'].values
        # Filter for these pairs
        clean_df = df[df['pair_id'].isin(top_ids)].sort_values(['pair_id', 'week'])
        save_csv(clean_df, "MainFig3_Posterior_Shares.csv")

# Fig 4: Judge Weights (Simulated)
def export_fig4():
    print("Exporting Fig 4 data...")
    x = np.linspace(0, 1, 100)
    prior = beta.pdf(x, 2, 2)
    sim_posterior = beta.pdf(x, 15, 15)
    df = pd.DataFrame({
        'weight_w': x,
        'prior_density': prior,
        'posterior_density_sim': sim_posterior
    })
    save_csv(df, "MainFig4_Judge_Weights.csv")

# Fig 5: Rules
def export_fig5():
    print("Exporting Fig 5 data...")
    path = os.path.join(DATA_DIR, "task2_metrics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        save_csv(df, "MainFig5_Rule_Comparison.csv")

# Fig 6: Cases
def export_fig6():
    print("Exporting Fig 6 data...")
    path = os.path.join(DATA_DIR, "controversy_cases.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        targets = ['Jerry Rice', 'Bobby Bones', 'Bristol Palin', 'Billy Ray Cyrus']
        # Filter
        clean_df = df[df['celebrity_name'].isin(targets)]
        if clean_df.empty:
            clean_df = df # Fallback
        save_csv(clean_df, "MainFig6_Cases.csv")

# Fig 7: LMM
def export_fig7():
    print("Exporting Fig 7 data...")
    j_path = os.path.join(DATA_DIR, "task3_analysis", "task3_lmm_judge_coeffs.csv")
    f_path = os.path.join(DATA_DIR, "task3_analysis", "task3_lmm_fan_coeffs_aggregated.csv")
    
    if os.path.exists(j_path) and os.path.exists(f_path):
        jd = pd.read_csv(j_path, index_col=0)
        jd.index.name = 'term'
        jd = jd.reset_index().assign(Type="Judge")
        if 'coef' in jd.columns: jd = jd.rename(columns={'coef': 'estimate'})
        
        fd = pd.read_csv(f_path, index_col=0)
        fd.index.name = 'term'
        fd = fd.reset_index().assign(Type="Fan")
        if 'mean' in fd.columns: fd = fd.rename(columns={'mean': 'estimate'})
        
        combined = pd.concat([jd, fd], ignore_index=True)
        combined = combined[combined['term'] != 'Intercept']
        save_csv(combined, "MainFig7_LMM_Forest.csv")

# Fig 8: Task 4
def export_fig8():
    print("Exporting Fig 8 data...")
    # Panel A
    p_path = os.path.join(DATA_DIR, "task4_pareto_front.csv")
    if os.path.exists(p_path):
        df = pd.read_csv(p_path)
        save_csv(df, "MainFig8_Task4_Pareto.csv")
        
    # Panel B
    s_path = os.path.join(DATA_DIR, "stress_test.csv")
    if os.path.exists(s_path):
        df = pd.read_csv(s_path)
        save_csv(df, "MainFig8_Task4_Robustness.csv")

def main():
    export_fig2()
    export_fig3()
    export_fig4()
    export_fig5()
    export_fig6()
    export_fig7()
    export_fig8()

if __name__ == "__main__":
    main()
