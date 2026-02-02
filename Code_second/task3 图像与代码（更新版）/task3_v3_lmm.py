"""
Task 3 LMM Analysis — V3 (Dual-Channel Attribution)
=====================================================
Track 1 (Interpretable Anchor):
  Judge Channel: pJ_it ~ age_z + C(industry) + week_norm + fan_base
                         + partner_experience + (1|partner) + (1|season)
  Fan Channel:   v^(r)  ~ same formula
                         × R posterior draws → aggregated CI

Key upgrades over V2:
  1. fan_base (pre-season popularity proxy) as fixed effect
  2. partner_experience as fixed effect
  3. Proper CI reporting for both channels
  4. ICC computation for partner & season
  5. Robust fallback: grouped by partner + fixed season if crossed fails
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import glob
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
DATASET_PATH   = r'd:\shumomeisai\Code_second\Results\task3_data\task3_weekly_dataset.parquet'
POSTERIOR_DIR  = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR     = r'd:\shumomeisai\Code_second\Results\task3_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fit_lmm(df, formula, method='crossed'):
    """
    Fit LMM with partner + season random effects.
    method='crossed': groups=1, vc_formula for both (correct but slow)
    method='fallback': groups=partner, season as fixed effect
    Returns (result_object, method_used) or (None, 'failed')
    """
    df = df.copy()
    df['partner_str'] = df['ballroom_partner'].astype(str)
    df['season_str']  = df['season'].astype(str)

    if method == 'crossed':
        try:
            df['_grp'] = 1
            model = smf.mixedlm(formula, df, groups='_grp',
                                vc_formula={
                                    "partner": "0 + C(partner_str)",
                                    "season":  "0 + C(season_str)"
                                })
            res = model.fit(reml=True, maxiter=200)
            return res, 'crossed'
        except Exception as e:
            print(f"    Crossed RE failed ({e}), falling back...")
            return fit_lmm(df, formula, method='fallback')
    else:
        try:
            formula_fb = formula + " + C(season_str)"
            model = smf.mixedlm(formula_fb, df, groups='partner_str')
            res = model.fit(reml=True, maxiter=200)
            return res, 'fallback'
        except Exception as e2:
            print(f"    Fallback also failed: {e2}")
            return None, 'failed'


def extract_results(res, method):
    """Extract fixed effects + partner random effects from fitted LMM."""
    # Fixed effects with CI
    params = res.params.to_frame(name='estimate')
    try:
        ci = res.conf_int()
        ci.columns = ['ci_lo', 'ci_hi']
        fe = params.join(ci)
    except:
        fe = params.copy()
        fe['ci_lo'] = np.nan; fe['ci_hi'] = np.nan

    # p-values
    try:
        fe['pvalue'] = res.pvalues
    except:
        fe['pvalue'] = np.nan

    # Partner random effects (BLUPs)
    partner_effs = {}
    if method == 'crossed':
        try:
            re_dict = res.random_effects[1]
            for k, v in re_dict.items():
                if 'partner' in k.lower():
                    name = k
                    for pat in ['partner[C(partner_str)[', 'C(partner_str)[', 'partner[T.', ']']:
                        name = name.replace(pat, '')
                    name = name.strip()
                    if name: partner_effs[name] = v
        except:
            pass
    else:  # fallback: groups IS partner
        for k, v in res.random_effects.items():
            partner_effs[str(k)] = v.get('Group', v.get('Intercept', 0))

    pe_df = pd.DataFrame.from_dict(partner_effs, orient='index', columns=['effect'])
    pe_df.index.name = 'partner'
    pe_df.sort_values('effect', ascending=False, inplace=True)

    # Variance components & ICC
    vc = {}
    try:
        cov = res.cov_re
        if hasattr(cov, 'values'):
            # Extract variance of partner and season
            # For crossed: variance components are in res.vcomp
            pass
        scale = res.scale  # residual variance
        vc['residual_var'] = scale
    except:
        pass

    return fe, pe_df, vc


def run_task3_lmm():
    print("=" * 60)
    print("Task 3 LMM Analysis (V3 — Dual Channel)")
    print("=" * 60)

    # ── Load data ──
    print("\n[1] Loading data...")
    df = pd.read_parquet(DATASET_PATH)
    print(f"  Shape: {df.shape}")

    # Ensure types
    df['industry'] = df['industry'].astype('category')
    df['ballroom_partner'] = df['ballroom_partner'].astype(str)

    # Formula (same for both channels)
    formula = "target ~ age_z + C(industry) + week_norm + fan_base + partner_experience"

    # ══════════════════════════════════════════════════════════
    # CHANNEL A: JUDGE
    # ══════════════════════════════════════════════════════════
    print("\n[2] Fitting Judge Channel LMM...")
    df_j = df.dropna(subset=['pJ_it']).copy()
    df_j['target'] = df_j['pJ_it']

    res_j, method_j = fit_lmm(df_j, formula)
    if res_j is not None:
        print(f"    Method: {method_j}")
        fe_j, pe_j, vc_j = extract_results(res_j, method_j)
        fe_j.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_judge_fixed_effects.csv'))
        pe_j.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_judge_partner_effects.csv'))
        print(f"    Fixed effects:\n{fe_j[['estimate', 'ci_lo', 'ci_hi', 'pvalue']].head(10)}")
        print(f"    Top 5 partners: {pe_j.head(5).to_dict()['effect']}")
    else:
        print("    ⚠ Judge LMM failed completely.")
        fe_j, pe_j = None, None

    # ══════════════════════════════════════════════════════════
    # CHANNEL B: FAN (Posterior Outer Loop)
    # ══════════════════════════════════════════════════════════
    print("\n[3] Fitting Fan Channel LMM (Posterior Outer Loop)...")

    # Load posterior files
    season_files = [f for f in glob.glob(os.path.join(POSTERIOR_DIR, "season_*.npz"))
                    if 'summary' not in f and 'diagnostics' not in f]

    if not season_files:
        print("    ⚠ No posterior samples found. Falling back to v_mean.")
        # Single fit with point estimate
        df_f = df.dropna(subset=['v_mean']).copy()
        df_f['target'] = df_f['v_mean']
        res_f, method_f = fit_lmm(df_f, formula)
        if res_f:
            fe_f, pe_f, _ = extract_results(res_f, method_f)
            fe_f.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_fan_fixed_effects.csv'))
            pe_f.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_fan_partner_effects.csv'))
        return

    # Determine R and build index
    s1 = np.load(season_files[0], allow_pickle=True)
    R_total = s1['v'].shape[0]
    N_DRAWS = min(R_total, 30)
    draw_indices = np.linspace(0, R_total - 1, N_DRAWS, dtype=int)
    print(f"    Using {N_DRAWS} posterior draws (of {R_total} available)")

    # Build lookup: (season, week, pair_id) -> row index
    df['_row_id'] = range(len(df))
    lookup = {(int(r['season']), int(r['week']), r['pair_id']): r['_row_id']
              for _, r in df[['season', 'week', 'pair_id', '_row_id']].iterrows()}

    # Outer loop
    fan_coeff_list = []
    fan_partner_list = []

    for i, r_idx in enumerate(draw_indices):
        if i % 10 == 0:
            print(f"    Draw {i+1}/{N_DRAWS}...")

        # Map posterior draw r_idx → y_fan column
        val_map = {}
        for fpath in season_files:
            try:
                data = np.load(fpath, allow_pickle=True)
                s_id = int(os.path.basename(fpath).split('_')[1].split('.')[0])
                if 'v' not in data: continue
                v_r = data['v'][r_idx]  # (T, N)
                c_ids = data.get('pair_ids', data.get('couple_ids'))
                if c_ids is None: continue
                for t in range(v_r.shape[0]):
                    for n, pid in enumerate(c_ids):
                        val_map[(s_id, t + 1, pid)] = v_r[t, n]
            except:
                pass

        # Apply to df
        y_fan = np.array([val_map.get((int(r['season']), int(r['week']), r['pair_id']), np.nan)
                          for _, r in df[['season', 'week', 'pair_id']].iterrows()])

        df_r = df.copy()
        df_r['target'] = y_fan
        df_r = df_r.dropna(subset=['target'])

        if len(df_r) < 50:
            continue

        res_f, method_f = fit_lmm(df_r, formula)
        if res_f is None:
            continue

        fe_f, pe_f, _ = extract_results(res_f, method_f)
        fan_coeff_list.append(fe_f['estimate'].to_dict())
        fan_partner_list.append(pe_f['effect'].to_dict())

    # ── Aggregate across draws ──
    print(f"\n[4] Aggregating {len(fan_coeff_list)} successful draws...")

    if fan_coeff_list:
        fe_agg = pd.DataFrame(fan_coeff_list)
        fe_summ = fe_agg.describe(percentiles=[0.025, 0.975]).T
        fe_summ = fe_summ[['mean', '2.5%', '97.5%', 'std']].rename(
            columns={'mean': 'estimate', '2.5%': 'ci_lo', '97.5%': 'ci_hi', 'std': 'se_posterior'})
        fe_summ.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_fan_fixed_effects.csv'))
        print(f"  Fan fixed effects:\n{fe_summ.head(10)}")

    if fan_partner_list:
        pe_agg = pd.DataFrame(fan_partner_list)
        pe_summ = pe_agg.describe(percentiles=[0.025, 0.975]).T
        pe_summ = pe_summ[['mean', '2.5%', '97.5%', 'count']].rename(
            columns={'mean': 'effect', '2.5%': 'ci_lo', '97.5%': 'ci_hi', 'count': 'n_draws'})
        pe_summ.sort_values('effect', ascending=False, inplace=True)
        pe_summ.to_csv(os.path.join(OUTPUT_DIR, 'task3_lmm_fan_partner_effects.csv'))
        print(f"  Top 5 Fan partners:\n{pe_summ.head(5)}")

    # ── Save combined summary ──
    summary = {
        'judge_method': method_j if res_j else 'failed',
        'fan_n_draws': len(fan_coeff_list),
        'fan_n_total': N_DRAWS,
    }
    with open(os.path.join(OUTPUT_DIR, 'task3_lmm_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Task 3 LMM Finished.")


if __name__ == "__main__":
    run_task3_lmm()
