"""
Task 4 Figure 1: The Pareto Landscape
(a) Parallel Coordinates: 4-Objective Trade-off
(b) Fan vs Judge 2D Projection (bubble size=Drama, color=Robustness)

═══════ DATA INTERFACE ═══════
替换下方 pareto_* 和 key point 数据为真实 Optuna 输出:
  - pareto_F/J/D/R: Pareto 前沿上各解的四个目标值
  - dom_F/J/D/R: 被支配解的四个目标值
  - knee/fan_fav/judge_fav/baseline/current: 关键点的坐标
═══════════════════════════════
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

np.random.seed(2026)

# ══════════════════════════════════════════════════════
# DATA INTERFACE — 替换为 task4_pareto_front.csv 真实数据
# ══════════════════════════════════════════════════════

n_pareto = 45
n_dominated = 100

# Simulated Pareto front (F-J trade-off, D peaks mid-front, R peaks at knee)
t = np.linspace(0, 1, n_pareto)
pareto_F = 0.52 + 0.38 * t + np.random.normal(0, 0.015, n_pareto)
pareto_J = 0.88 - 0.38 * t + np.random.normal(0, 0.015, n_pareto)
pareto_D = 0.48 + 0.28 * np.sin(np.pi * t) + np.random.normal(0, 0.02, n_pareto)
pareto_R = 0.72 - 0.12 * np.abs(t - 0.45) + np.random.normal(0, 0.015, n_pareto)

# Dominated points
dom_F = np.random.uniform(0.42, 0.82, n_dominated)
dom_J = np.random.uniform(0.38, 0.78, n_dominated)
dom_D = np.random.uniform(0.32, 0.58, n_dominated)
dom_R = np.random.uniform(0.52, 0.72, n_dominated)

# Key points (from task4_recommendations.json)
knee      = {'F': 0.72, 'J': 0.71, 'D': 0.70, 'R': 0.73}
fan_fav   = {'F': 0.90, 'J': 0.51, 'D': 0.56, 'R': 0.66}
judge_fav = {'F': 0.53, 'J': 0.87, 'D': 0.53, 'R': 0.70}
baseline  = {'F': 0.64, 'J': 0.59, 'D': 0.44, 'R': 0.60}
current   = {'F': 0.67, 'J': 0.69, 'D': 0.51, 'R': 0.68}

# ══════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 8), facecolor='#FAFAF8')
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.14)

# ─── PALETTE ───
C_KNEE = '#D62728'
C_FAN  = '#1F77B4'
C_JDG  = '#2CA02C'
C_BASE = '#7F7F7F'
C_CUR  = '#9467BD'
C_DOM  = '#D5D5D5'

# ═══════════════════════════════════════════════════
# (a) PARALLEL COORDINATES
# ═══════════════════════════════════════════════════
ax = fig.add_subplot(gs[0, 0])

obj_names = ['Fan\nAlignment\n($\\rho_F$)', 'Judge\nAlignment\n($\\rho_J$)',
             'Drama\n($D$)', 'Robustness\n($R$)']
xs = np.arange(4)

# Dominated (faint)
for i in range(min(50, n_dominated)):
    ax.plot(xs, [dom_F[i], dom_J[i], dom_D[i], dom_R[i]],
            color=C_DOM, alpha=0.25, lw=0.6, zorder=1)

# Pareto front (gradient coloured by position on front)
from matplotlib.colors import LinearSegmentedColormap
cmap_pc = LinearSegmentedColormap.from_list('fj', [C_FAN, '#F0C040', C_JDG])
for i in range(n_pareto):
    c = cmap_pc(t[i])
    ax.plot(xs, [pareto_F[i], pareto_J[i], pareto_D[i], pareto_R[i]],
            color=c, alpha=0.45, lw=1.0, zorder=2)

# Key systems
key_pts = [
    (knee,      C_KNEE, '-',  3.5, '★ Knee (Recommended)'),
    (fan_fav,   C_FAN,  '--', 2.2, '■ Fan Favorite'),
    (judge_fav, C_JDG,  '--', 2.2, '▲ Judge Favorite'),
    (baseline,  C_BASE, ':',  2.0, '◆ 50/50 Baseline'),
    (current,   C_CUR,  '-.', 2.0, '● Current S28+ Rule'),
]
handles = []
for pt, col, ls, lw, label in key_pts:
    vals = [pt['F'], pt['J'], pt['D'], pt['R']]
    line, = ax.plot(xs, vals, color=col, ls=ls, lw=lw, zorder=10,
                    marker='o', ms=8, mfc=col, mec='white', mew=1.5, label=label)
    handles.append(line)
handles.append(plt.Line2D([0], [0], color=C_DOM, lw=1.2, label='  Dominated solutions'))

ax.set_xticks(xs)
ax.set_xticklabels(obj_names, fontsize=10, fontweight='bold')
ax.set_ylim(0.28, 0.98)
ax.set_ylabel('Objective Value  (higher = better)', fontsize=10.5, fontweight='bold')
ax.legend(handles=handles, loc='lower left', fontsize=7.8, ncol=2,
          framealpha=0.92, edgecolor='#CCC')
ax.grid(axis='y', alpha=0.25)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#FAFAF8')
ax.set_title('(a)  Parallel Coordinates: Four-Objective Trade-off',
             fontsize=11.5, fontweight='bold', pad=14, color='#1a1a2e')

# ═══════════════════════════════════════════════════
# (b) 2-D PROJECTION  F vs J,  size=D, color=R
# ═══════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])

# Dominated
ax2.scatter(dom_F[:60], dom_J[:60], s=16, c=C_DOM, alpha=0.30, edgecolors='none', zorder=1)

# Pareto front (bubble)
d_norm = (pareto_D - pareto_D.min()) / (pareto_D.max() - pareto_D.min() + 1e-9)
sizes = 40 + 280 * d_norm
sc = ax2.scatter(pareto_F, pareto_J, s=sizes, c=pareto_R,
                 cmap='RdYlGn', vmin=0.56, vmax=0.78,
                 alpha=0.72, edgecolors='white', lw=0.7, zorder=3)

# Pareto front curve
idx_sort = np.argsort(pareto_F)
ax2.plot(pareto_F[idx_sort], pareto_J[idx_sort], color='#555', lw=0.8, ls='--', alpha=0.35, zorder=2)

# Key points with annotations
annot_cfg = [
    (knee,      C_KNEE, '*',  320, 'Knee\n(Recommended)',   ( 0.02,  0.025)),
    (fan_fav,   C_FAN,  's',  150, 'Fan\nFavorite',         ( 0.015,-0.045)),
    (judge_fav, C_JDG,  '^',  150, 'Judge\nFavorite',       (-0.09,  0.020)),
    (baseline,  C_BASE, 'D',  120, '50/50\nBaseline',       (-0.10, -0.035)),
    (current,   C_CUR,  'o',  130, 'Current\n(S28+ Rule)',  (-0.11,  0.020)),
]
for pt, col, mk, ms, lab, (dx, dy) in annot_cfg:
    ax2.scatter(pt['F'], pt['J'], s=ms, c=col, marker=mk,
                edgecolors='white', lw=2, zorder=15)
    ax2.annotate(lab, xy=(pt['F'], pt['J']),
                 xytext=(pt['F']+dx, pt['J']+dy),
                 fontsize=7, fontweight='bold', color=col,
                 arrowprops=dict(arrowstyle='->', color=col, lw=1),
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=col, alpha=0.88))

# Regions
ax2.fill_between([0.83, 0.98], 0.35, 0.60, alpha=0.04, color=C_FAN)
ax2.text(0.89, 0.42, 'Fan\nZone', fontsize=9, color=C_FAN, alpha=0.45,
         ha='center', fontstyle='italic', fontweight='bold')
ax2.fill_between([0.40, 0.60], 0.80, 0.96, alpha=0.04, color=C_JDG)
ax2.text(0.49, 0.88, 'Judge\nZone', fontsize=9, color=C_JDG, alpha=0.45,
         ha='center', fontstyle='italic', fontweight='bold')

# Colorbar
cbar = plt.colorbar(sc, ax=ax2, shrink=0.55, pad=0.02, aspect=18)
cbar.set_label('Robustness ($R$)', fontsize=9, fontweight='bold')
cbar.ax.tick_params(labelsize=7.5)

# Size legend
for dv, dl in [(0.50, 'Low D'), (0.65, 'Med D'), (0.78, 'High D')]:
    sv = 40 + 280 * ((dv - pareto_D.min()) / (pareto_D.max() - pareto_D.min() + 1e-9))
    ax2.plot([], [], 'o', color='#999', ms=np.sqrt(max(sv, 1))/2.2, alpha=0.45, label=dl)
ax2.legend(loc='lower left', fontsize=7, title='Bubble = Drama', title_fontsize=7.5,
           framealpha=0.9, edgecolor='#CCC')

ax2.set_xlabel('Fan Alignment ($\\rho_F$)', fontsize=10.5, fontweight='bold')
ax2.set_ylabel('Judge Alignment ($\\rho_J$)', fontsize=10.5, fontweight='bold')
ax2.set_xlim(0.38, 0.97)
ax2.set_ylim(0.36, 0.95)
ax2.grid(alpha=0.18)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_facecolor('#FAFAF8')
ax2.set_title('(b)  Fan vs Judge Projection  (size = Drama, colour = Robustness)',
              fontsize=10.5, fontweight='bold', pad=14, color='#1a1a2e')

# ── Supertitle ──
fig.suptitle('The Pareto Landscape — Mapping the Design Space of Competition Rules',
             fontsize=15, fontweight='bold', y=1.01, color='#1a1a2e', fontfamily='serif')
fig.text(0.5, 0.975,
         '45 Pareto-optimal configurations across 4 objectives  ·  '
         'Knee point achieves the best balanced trade-off',
         ha='center', fontsize=9.2, color='#666', style='italic')

plt.savefig('/home/claude/fig1_pareto_landscape.png', dpi=250, bbox_inches='tight', facecolor='#FAFAF8')
plt.close()
print("✅ Figure 1 done")
