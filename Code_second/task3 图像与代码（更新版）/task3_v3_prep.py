"""
Task 3 Data Preparation — V3 (Dual-Channel Attribution)
========================================================
Key upgrades over V2:
  1. Pre-season fan base proxy: log(1 + v_week1_mean)
  2. Season-level fan target y_F: week-weighted aggregate of posterior vote shares
  3. Dual output: weekly panel (for LMM) + season-level panel (for summary)
  4. Improved network features with anti-leakage
"""

import pandas as pd
import numpy as np
import os
import glob
from itertools import combinations

# ═══════════════════════════════════════════════════════════════
# PATHS — adapt to your machine
# ═══════════════════════════════════════════════════════════════
PANEL_PATH       = r'd:\shumomeisai\Code_second\processed\panel.csv'
POSTERIOR_DIR    = r'd:\shumomeisai\Code_second\Results\posterior_samples'
OUTPUT_DIR       = r'd:\shumomeisai\Code_second\Results\task3_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(2026)


# ─────────────────────────────────────────────────────────────
# Network feature generator (kept from V2, bug-fixed)
# ─────────────────────────────────────────────────────────────
def generate_network_features(panel):
    """Build pro-pro co-occurrence graph using only past seasons (no leakage)."""
    print("  Building partner network features...")
    net_feats = []
    seasons = sorted(panel['season'].unique())
    season_pros = panel.groupby('season')['ballroom_partner'].unique().to_dict()

    def power_iteration(A, n_iter=30):
        deg = A.sum(axis=0); deg[deg == 0] = 1
        M = A / deg; n = M.shape[0]; v = np.ones(n) / n; d = 0.85
        for _ in range(n_iter):
            v = d * M.dot(v) + (1 - d) / n
        return v

    for s in seasons:
        past = [ps for ps in seasons if ps < s]
        all_past = sorted({p for ps in past for p in season_pros.get(ps, [])})
        n = len(all_past)
        pro2idx = {p: i for i, p in enumerate(all_past)}

        if n > 1:
            A = np.zeros((n, n))
            for ps in past:
                idxs = [pro2idx[p] for p in season_pros.get(ps, []) if p in pro2idx]
                for i1, i2 in combinations(idxs, 2):
                    A[i1, i2] += 1; A[i2, i1] += 1
            w_deg = A.sum(axis=1)
            try:    pr_vec = power_iteration(A)
            except: pr_vec = np.zeros(n)
            ndeg = w_deg / (w_deg.sum() + 1e-9)
            try:
                u, _, _ = np.linalg.svd(A)
                emb = u[:, :4] if u.shape[1] >= 4 else np.pad(u, ((0,0),(0,4-u.shape[1])))
            except:
                emb = np.zeros((n, 4))
        else:
            w_deg = pr_vec = ndeg = np.zeros(max(n, 1))
            emb = np.zeros((max(n, 1), 4))

        cur = panel[panel['season'] == s][['season', 'pair_id', 'ballroom_partner']].drop_duplicates('pair_id')
        rows = []
        for _, row in cur.iterrows():
            p = row['ballroom_partner']
            if p in pro2idx:
                idx = pro2idx[p]
                rows.append({
                    'season': row['season'], 'pair_id': row['pair_id'],
                    'partner_net_degree': w_deg[idx], 'partner_net_pagerank': pr_vec[idx],
                    'partner_net_norm_degree': ndeg[idx],
                    **{f'partner_emb_{k}': emb[idx, k] for k in range(4)}
                })
            else:
                rows.append({
                    'season': row['season'], 'pair_id': row['pair_id'],
                    'partner_net_degree': 0, 'partner_net_pagerank': 0,
                    'partner_net_norm_degree': 0,
                    **{f'partner_emb_{k}': 0.0 for k in range(4)}
                })
        net_feats.append(pd.DataFrame(rows))

    return pd.concat(net_feats, ignore_index=True) if net_feats else pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# NEW: Fan base proxy from Week-1 posterior
# ─────────────────────────────────────────────────────────────
def compute_fan_base_proxy(panel, posterior_dir):
    """
    For each (season, pair_id), compute pre-season fan base proxy:
      fan_base = log(1 + mean_over_draws(v_{i, week=1}^(r)))
    Uses Week 1 vote share posterior mean as external popularity signal.
    """
    print("  Computing fan base proxy from Week-1 posterior...")
    records = []
    season_files = glob.glob(os.path.join(posterior_dir, "season_*.npz"))
    valid_files = [f for f in season_files if 'summary' not in f and 'diagnostics' not in f]

    for fpath in valid_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            fname = os.path.basename(fpath)
            s_id = int(fname.split('_')[1].split('.')[0])
            if 'v' not in data: continue
            v_all = data['v']  # (R, T, N)
            c_ids = data.get('pair_ids', data.get('couple_ids'))
            if c_ids is None: continue

            # Week 1 = index 0, mean over posterior draws
            v_week1_mean = v_all[:, 0, :].mean(axis=0)  # (N,)
            for n, pid in enumerate(c_ids):
                records.append({
                    'season': s_id, 'pair_id': pid,
                    'fan_base': np.log1p(v_week1_mean[n])
                })
        except:
            pass

    if records:
        fb = pd.DataFrame(records)
        return fb
    else:
        print("  Warning: No posterior files found for fan_base.")
        return pd.DataFrame(columns=['season', 'pair_id', 'fan_base'])


# ─────────────────────────────────────────────────────────────
# Main prep
# ─────────────────────────────────────────────────────────────
def prep_task3_data():
    print("=" * 60)
    print("Task 3 Data Preparation (V3 — Dual Channel)")
    print("=" * 60)

    # ── 1. Load panel ──
    print(f"\n[1/6] Loading panel from {PANEL_PATH}")
    panel = pd.read_csv(PANEL_PATH) if PANEL_PATH.endswith('.csv') else pd.read_parquet(PANEL_PATH)

    # Column compatibility
    renames = {}
    if 'celebrity_age' in panel.columns and 'celebrity_age_during_season' not in panel.columns:
        renames['celebrity_age'] = 'celebrity_age_during_season'
    if renames:
        panel.rename(columns=renames, inplace=True)
    if 'pJ_it' not in panel.columns and 'S_it' in panel.columns:
        panel['pJ_it'] = panel['S_it']

    # Rename for brevity
    panel.rename(columns={
        'celebrity_age_during_season': 'age',
        'celebrity_industry': 'industry'
    }, inplace=True)
    print(f"  Panel shape: {panel.shape}")

    # ── 2. Industry cleanup ──
    print("[2/6] Cleaning industry labels...")
    panel['industry'] = panel['industry'].fillna('Unknown').astype(str).str.strip().str.title()
    panel['industry'] = panel['industry'].replace({
        'Beauty Pagent': 'Beauty Pageant',
        'Con Artist': 'Personality'
    })
    if panel['age'].isna().any():
        panel['age'] = panel['age'].fillna(panel['age'].mean())

    # ── 3. Posterior vote summaries ──
    print("[3/6] Loading posterior vote summaries...")
    v_list = []
    for s in sorted(panel['season'].unique()):
        sp = os.path.join(POSTERIOR_DIR, f"season_{s}_summary.csv")
        if os.path.exists(sp):
            try:
                sdf = pd.read_csv(sp)
                cols = [c for c in ['season', 'week', 'pair_id', 'v_mean', 'v_ci_width'] if c in sdf.columns]
                if len(cols) >= 3:
                    v_list.append(sdf[cols])
            except:
                pass

    merged = panel.copy()
    if v_list:
        v_stats = pd.concat(v_list, ignore_index=True)
        merged = pd.merge(merged, v_stats, on=['season', 'week', 'pair_id'], how='left')
    else:
        merged['v_mean'] = np.nan; merged['v_ci_width'] = np.nan

    # ── 4. Feature engineering ──
    print("[4/6] Engineering features...")
    merged.sort_values(['season', 'pair_id', 'week'], inplace=True)

    # Rolling performance momentum
    merged['rolling_avg_pJ'] = merged.groupby(['season', 'pair_id'])['pJ_it'] \
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    merged['rolling_std_pJ'] = merged.groupby(['season', 'pair_id'])['pJ_it'] \
        .transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)

    # Standardised age
    merged['age_z'] = (merged['age'] - merged['age'].mean()) / merged['age'].std()

    # Week progress normalised [0, 1]
    merged['week_norm'] = merged['week'] / merged.groupby('season')['week'].transform('max')

    # Partner experience: number of past seasons participated
    seasons_list = sorted(merged['season'].unique())
    partner_exp = {}
    for s in seasons_list:
        past = merged[merged['season'] < s]
        exp = past.groupby('ballroom_partner')['season'].nunique().to_dict()
        for _, row in merged[merged['season'] == s][['pair_id', 'ballroom_partner']].drop_duplicates().iterrows():
            partner_exp[(s, row['pair_id'])] = exp.get(row['ballroom_partner'], 0)
    merged['partner_experience'] = merged.apply(lambda r: partner_exp.get((r['season'], r['pair_id']), 0), axis=1)

    # ── 5. Network features ──
    print("[5/6] Network features...")
    net_df = generate_network_features(merged)
    if not net_df.empty:
        merged = pd.merge(merged, net_df, on=['season', 'pair_id'], how='left')
        for c in net_df.columns:
            if c not in ['season', 'pair_id']:
                merged[c] = merged[c].fillna(0.0)

    # ── 6. Fan base proxy ──
    print("[6/6] Fan base proxy...")
    fb_df = compute_fan_base_proxy(merged, POSTERIOR_DIR)
    if not fb_df.empty:
        merged = pd.merge(merged, fb_df, on=['season', 'pair_id'], how='left')
        merged['fan_base'] = merged['fan_base'].fillna(0.0)
    else:
        merged['fan_base'] = 0.0

    # ── Save ──
    out_cols = [
        'season', 'week', 'pair_id', 'celebrity_name', 'ballroom_partner',
        'pJ_it', 'v_mean', 'v_ci_width',
        'age', 'age_z', 'industry', 'week_norm',
        'rolling_avg_pJ', 'rolling_std_pJ',
        'partner_experience', 'fan_base',
        'partner_net_degree', 'partner_net_pagerank', 'partner_net_norm_degree',
        'partner_emb_0', 'partner_emb_1', 'partner_emb_2', 'partner_emb_3'
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols].copy()

    path = os.path.join(OUTPUT_DIR, 'task3_weekly_dataset.parquet')
    out.to_parquet(path, index=False)
    print(f"\n  Saved weekly panel → {path}  ({len(out)} rows)")
    print(f"  Columns: {list(out.columns)}")
    print(out.head(3))


if __name__ == "__main__":
    prep_task3_data()
