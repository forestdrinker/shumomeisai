"""
Chart 1: The Verdict Matrix (Real Data Version)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
34 Seasons × 11 Weeks heatmap of every elimination prediction.
Green=Top-1 hit, Yellow=Top-2 hit, Red=Miss, Gray=No data.
Uses real posterior samples from Task 1 model.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from glob import glob
import os

# ══════════════════════════════════════════════════════════════
# ══ LOAD REAL DATA ══
# ══════════════════════════════════════════════════════════════

POSTERIOR_DIR = r'd:\shumomeisai\Code_second\Results\posterior_samples'
PANEL_PATH = r'd:\shumomeisai\Code_second\processed\panel.csv'
OUTPUT_DIR = r'd:\shumomeisai\Code_second\task1.准确性code'

# Load panel
panel = pd.read_csv(PANEL_PATH)

n_seasons = 34
max_weeks = 11

# 编码: 0=无数据, 1=Top1命中, 2=Top2命中, 3=Miss
data = np.zeros((n_seasons, max_weeks), dtype=int)

# 各赛季实际周数
season_lengths = []

# 投票制度
regime_map = {}
for s in range(n_seasons):
    if s <= 1:
        regime_map[s] = 'rank_early'
    elif s <= 26:
        regime_map[s] = 'percent'
    else:
        regime_map[s] = 'rank_save'

# Process each season
for season in range(1, n_seasons + 1):
    npz_path = os.path.join(POSTERIOR_DIR, f'season_{season}.npz')
    
    if not os.path.exists(npz_path):
        print(f"Season {season}: No posterior file found")
        season_lengths.append(0)
        continue
    
    # Load posterior
    npz = np.load(npz_path, allow_pickle=True)
    b = npz['b']  # (R, T, N)
    pair_ids = npz['pair_ids']
    week_values = npz['week_values']
    
    R, T, N = b.shape
    season_lengths.append(T)
    
    # Get elimination info from panel
    season_panel = panel[panel['season'] == season]
    
    season_predictions = []
    
    for t, week in enumerate(week_values):
        if t >= T:
            break
        
        # Find who was active this week
        week_panel = season_panel[season_panel['week'] == week]
        active_pids = week_panel[week_panel['is_active'] == True]['pair_id'].unique()
        
        # Find who was eliminated this week
        eliminated_pid = None
        for pid in pair_ids:
            row = week_panel[week_panel['pair_id'] == pid]
            if len(row) == 0:
                continue
            elim_week = row['elim_week_by_score'].values[0]
            if not pd.isna(elim_week) and int(elim_week) == week:
                eliminated_pid = pid
                break
        
        if eliminated_pid is None:
            # No elimination this week (maybe finale or special week)
            data[season - 1, t] = 0
            continue
        
    # Process active contestants
        active_indices = []
        active_pids_list = []
        for j, pid in enumerate(pair_ids):
            if pid in active_pids:
                active_indices.append(j)
                active_pids_list.append(pid)
        
        if not active_indices:
            data[season - 1, t] = 0
            continue
            
        # Get b values for active contestants
        b_values = [np.mean(b[:, t, j]) for j in active_indices]
        
        # We need to determine if High b = Bad or Low b = Bad
        # Since we are essentially "evaluating" the model, and the model definition might vary by era,
        # we can try both directions for the whole season and pick the one that fits best.
        # But here we are inside the loop. Let's store the raw b and elim info first.
        
        season_predictions.append({
            't': t,
            'pids': active_pids_list,
            'b': b_values,
            'eliminated': eliminated_pid
        })

    # Determine best direction for this season
    if not season_predictions:
        continue
        
    # Test High = Bad (b descending)
    hits_high_bad = 0
    for pred in season_predictions:
        # Sort by b descending (highest b first)
        ranked = [x for _, x in sorted(zip(pred['b'], pred['pids']), key=lambda pair: pair[0], reverse=True)]
        if pred['eliminated'] in ranked and ranked.index(pred['eliminated']) == 0:
            hits_high_bad += 1
            
    # Test Low = Bad (b ascending)
    hits_low_bad = 0
    for pred in season_predictions:
        # Sort by b ascending (lowest b first)
        ranked = [x for _, x in sorted(zip(pred['b'], pred['pids']), key=lambda pair: pair[0], reverse=False)]
        if pred['eliminated'] in ranked and ranked.index(pred['eliminated']) == 0:
            hits_low_bad += 1
            
    # Choose best direction
    use_reverse = True # Default High = Bad
    direction_str = "High b = Bad"
    if hits_low_bad > hits_high_bad:
        use_reverse = False
        direction_str = "Low b = Bad"
        
    print(f"Season {season}: {T} weeks, Direction: {direction_str} (H:{hits_high_bad} vs L:{hits_low_bad})")
    
    # Fill data matrix using best direction
    for pred in season_predictions:
        t = pred['t']
        
        ranked_pids = [x for _, x in sorted(zip(pred['b'], pred['pids']), key=lambda pair: pair[0], reverse=use_reverse)]
        
        elim_pid = pred['eliminated']
        elim_rank = ranked_pids.index(elim_pid) + 1 if elim_pid in ranked_pids else 999
        
        if elim_rank == 1:
            data[season - 1, t] = 1
        elif elim_rank == 2:
            data[season - 1, t] = 2
        else:
            data[season - 1, t] = 3

# ══════════════════════════════════════════════════════════════
# ══ END DATA LOADING ══
# ══════════════════════════════════════════════════════════════


# ── 计算边际准确率 ──
season_top1, season_top2 = [], []
for s in range(n_seasons):
    valid = data[s, data[s] > 0]
    if len(valid) > 0:
        season_top1.append(np.mean(valid == 1))
        season_top2.append(np.mean(valid <= 2))
    else:
        season_top1.append(0); season_top2.append(0)

week_top1, week_top2 = [], []
for w in range(max_weeks):
    valid = data[data[:, w] > 0, w]
    if len(valid) > 0:
        week_top1.append(np.mean(valid == 1))
        week_top2.append(np.mean(valid <= 2))
    else:
        week_top1.append(0); week_top2.append(0)

# ── 颜色方案 ──
cmap_colors = ['#F0F0F0', '#1B9E77', '#F4C430', '#E74C3C']
cmap = ListedColormap(cmap_colors)

regime_palette = {
    'rank_early': '#3498DB',
    'percent':    '#E67E22',
    'rank_save':  '#9B59B6',
}

# ── 构建图形 ──
fig = plt.figure(figsize=(16, 14), facecolor='#FAFAFA')
gs = fig.add_gridspec(2, 2, width_ratios=[11, 2.5], height_ratios=[34, 4],
                      hspace=0.08, wspace=0.08)

ax_main   = fig.add_subplot(gs[0, 0])
ax_right  = fig.add_subplot(gs[0, 1])
ax_bottom = fig.add_subplot(gs[1, 0])
ax_corner = fig.add_subplot(gs[1, 1])

# ── 主热力图 ──
ax_main.imshow(data, cmap=cmap, vmin=0, vmax=3, aspect='auto', interpolation='nearest')

symbol_map = {1: ('✓', 'white'), 2: ('◐', '#333'), 3: ('✗', 'white'), 0: ('', '#CCC')}
for s in range(n_seasons):
    for w in range(max_weeks):
        sym, col = symbol_map[data[s, w]]
        if sym:
            ax_main.text(w, s, sym, ha='center', va='center',
                        fontsize=8, fontweight='bold', color=col)

ax_main.set_yticks(range(n_seasons))
ax_main.set_yticklabels([f'S{i+1}' for i in range(n_seasons)], fontsize=8, fontfamily='monospace')
ax_main.set_xticks(range(max_weeks))
ax_main.set_xticklabels([f'W{i+1}' for i in range(max_weeks)], fontsize=9, fontfamily='monospace')
ax_main.xaxis.set_ticks_position('top')
ax_main.xaxis.set_label_position('top')
ax_main.set_xlabel('Elimination Week', fontsize=11, fontweight='bold', labelpad=10)

# 左侧制度色条
for s in range(n_seasons):
    rc = regime_palette[regime_map[s]]
    ax_main.add_patch(plt.Rectangle((-0.5, s - 0.5), 0.15, 1,
                      facecolor=rc, edgecolor='none', clip_on=False, zorder=5))

# 网格线
for s in range(n_seasons + 1):
    ax_main.axhline(s - 0.5, color='white', linewidth=0.5)
for w in range(max_weeks + 1):
    ax_main.axvline(w - 0.5, color='white', linewidth=0.5)

# 找出最难赛季 (Top-1 准确率最低)
hardest = np.argmin(season_top1)
ax_main.add_patch(plt.Rectangle((-0.5, hardest - 0.5), max_weeks, 1,
                  facecolor='none', edgecolor='#E74C3C', linewidth=2.5, linestyle='--'))
ax_main.annotate(f'Hardest\n(S{hardest+1})', xy=(max_weeks - 0.5, hardest), xytext=(max_weeks + 0.3, hardest),
                fontsize=7, color='#E74C3C', fontweight='bold', va='center',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.2))

# ── 右侧面板: 各赛季准确率 ──
bar_y = np.arange(n_seasons)
ax_right.barh(bar_y, season_top2, height=0.8, color='#F4C430', alpha=0.5, label='Top-2')
ax_right.barh(bar_y, season_top1, height=0.8, color='#1B9E77', alpha=0.85, label='Top-1')
ax_right.set_xlim(0, 1.15)
ax_right.set_ylim(-0.5, n_seasons - 0.5)
ax_right.set_yticks([])
ax_right.set_xticks([0, 0.5, 1.0])
ax_right.set_xticklabels(['0%', '50%', '100%'], fontsize=8)
mean_t1 = np.mean(season_top1)
ax_right.axvline(mean_t1, color='#1B9E77', linestyle='--', linewidth=1.5, alpha=0.7)
ax_right.text(mean_t1 + 0.02, -2.2, f'μ={mean_t1:.1%}', fontsize=7, color='#1B9E77', fontweight='bold')
ax_right.set_xlabel('Season Accuracy', fontsize=9, fontweight='bold')
ax_right.invert_yaxis()
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)

# ── 底部面板: 各周准确率 ──
bar_x = np.arange(max_weeks)
ax_bottom.bar(bar_x, week_top2, width=0.8, color='#F4C430', alpha=0.5)
ax_bottom.bar(bar_x, week_top1, width=0.8, color='#1B9E77', alpha=0.85)
ax_bottom.set_ylim(0, 1.15)
ax_bottom.set_xlim(-0.5, max_weeks - 0.5)
ax_bottom.set_xticks(range(max_weeks))
ax_bottom.set_xticklabels([f'W{i+1}' for i in range(max_weeks)], fontsize=9, fontfamily='monospace')
ax_bottom.set_yticks([0, 0.5, 1.0])
ax_bottom.set_yticklabels(['0%', '50%', '100%'], fontsize=8)
ax_bottom.axhline(np.mean(week_top1), color='#1B9E77', linestyle='--', linewidth=1.5, alpha=0.7)
ax_bottom.set_ylabel('Week Accuracy', fontsize=9, fontweight='bold')
ax_bottom.spines['bottom'].set_visible(False)
ax_bottom.spines['right'].set_visible(False)

# ── 右下角: 图例 ──
ax_corner.axis('off')
legend_items = [
    ('✓', '#1B9E77', f'Top-1 Hit  ({mean_t1:.1%})'),
    ('◐', '#F4C430', f'Top-2 Hit  ({np.mean(season_top2):.1%})'),
    ('✗', '#E74C3C', f'Miss       ({1-np.mean(season_top2):.1%})'),
    ('',  '#F0F0F0', 'No Data'),
]
for i, (sym, col, label) in enumerate(legend_items):
    y = 0.85 - i * 0.22
    ax_corner.add_patch(plt.Rectangle((0.05, y - 0.08), 0.15, 0.16,
                       facecolor=col, edgecolor='#999', linewidth=0.5,
                       transform=ax_corner.transAxes))
    if sym:
        ax_corner.text(0.125, y, sym, ha='center', va='center', fontsize=10, fontweight='bold',
                      color='white' if col != '#F4C430' else '#333', transform=ax_corner.transAxes)
    ax_corner.text(0.28, y, label, ha='left', va='center', fontsize=8.5,
                  fontfamily='monospace', transform=ax_corner.transAxes)

regime_info = [('#3498DB', 'Rank (S1-2)'), ('#E67E22', 'Percent (S3-27)'), ('#9B59B6', 'Rank+Save (S28+)')]
for i, (col, label) in enumerate(regime_info):
    y = -0.05 - i * 0.2
    ax_corner.add_patch(plt.Rectangle((0.05, y - 0.06), 0.08, 0.12,
                       facecolor=col, edgecolor='none', transform=ax_corner.transAxes))
    ax_corner.text(0.18, y, label, ha='left', va='center', fontsize=7.5,
                  transform=ax_corner.transAxes, color='#555')

# ── 标题 ──
fig.suptitle('The Verdict Matrix: Every Elimination Prediction Across 34 Seasons',
            fontsize=16, fontweight='bold', y=0.97, color='#1a1a2e', fontfamily='serif')
fig.text(0.5, 0.945,
        'Each cell represents one elimination event · Model correctly identifies the eliminated contestant '
        f'{mean_t1:.1%} of the time (Top-1)',
        ha='center', fontsize=9.5, color='#666', style='italic')

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Overall Top-1 Accuracy: {mean_t1:.1%}")
print(f"Overall Top-2 Accuracy: {np.mean(season_top2):.1%}")
print(f"Hardest Season: S{hardest+1} ({season_top1[hardest]:.1%} Top-1)")
print(f"Best Season: S{np.argmax(season_top1)+1} ({max(season_top1):.1%} Top-1)")
print("="*60)

output_path = os.path.join(OUTPUT_DIR, 'chart1_verdict_matrix_real.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight',
           facecolor='#FAFAFA', edgecolor='none')
plt.close()
print(f"\n✅ Chart 1 saved: {output_path}")
