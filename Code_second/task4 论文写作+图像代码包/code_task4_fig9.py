"""
Task 4 Figure 2: The Recommended System's Blueprint
(a) Radar chart — 6-dimension comparison of 5 rule systems
(b) Dynamic weight curve w(t) — how judge/fan influence shifts over weeks
"""
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(17, 7.5), facecolor='#FAFAF8')
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.28)

# ─── PALETTE ───
C_KNEE = '#D62728'; C_FAN = '#1F77B4'; C_JDG = '#2CA02C'
C_BASE = '#7F7F7F'; C_CUR = '#9467BD'

# ═══════════════════════════════════════════════════
# (a) RADAR CHART  — 6 dimensions
# ═══════════════════════════════════════════════════
ax_r = fig.add_subplot(gs[0, 0], polar=True)

cats = ['Fan\nAlignment', 'Judge\nAlignment', 'Drama',
        'Robustness', 'Fairness', 'Simplicity']
N = len(cats)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

systems = {
    'Knee (Recommended)': ([0.72, 0.71, 0.70, 0.73, 0.76, 0.58], C_KNEE, '-',  3.2, 0.18),
    'Fan Favorite':       ([0.90, 0.51, 0.56, 0.66, 0.48, 0.68], C_FAN,  '--', 2.0, 0.06),
    'Judge Favorite':     ([0.53, 0.87, 0.53, 0.70, 0.63, 0.53], C_JDG,  '--', 2.0, 0.06),
    '50/50 Baseline':     ([0.64, 0.59, 0.44, 0.60, 0.54, 0.88], C_BASE, ':',  1.8, 0.04),
    'Current S28+':       ([0.67, 0.69, 0.51, 0.68, 0.70, 0.63], C_CUR,  '-.', 2.0, 0.08),
}

for name, (vals, col, ls, lw, fa) in systems.items():
    v = vals + vals[:1]
    ax_r.plot(angles, v, color=col, ls=ls, lw=lw, label=name, zorder=5)
    ax_r.fill(angles, v, color=col, alpha=fa)

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(cats, fontsize=9.5, fontweight='bold')
ax_r.set_ylim(0, 1.0)
ax_r.set_yticks([0.2, 0.4, 0.6, 0.8])
ax_r.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=7, color='#888')
ax_r.grid(color='#DDD', lw=0.5)
ax_r.spines['polar'].set_color('#DDD')
ax_r.set_facecolor('#FAFAF8')

ax_r.legend(loc='upper right', bbox_to_anchor=(1.38, 1.10), fontsize=8,
            framealpha=0.92, edgecolor='#CCC')
ax_r.set_title('(a)  Multi-Dimensional Comparison',
               fontsize=12, fontweight='bold', pad=25, color='#1a1a2e')

# ═══════════════════════════════════════════════════
# (b) WEIGHT CURVE  w(t)
# ═══════════════════════════════════════════════════
ax_w = fig.add_subplot(gs[0, 1])

def logistic(tt, a, b):
    return 1.0 / (1.0 + np.exp(-a * (tt - b)))

t_fine = np.linspace(0, 11, 300)

curves = [
    ('Knee  (a=0.8, b=4.2)',      0.8,  4.2, C_KNEE, '-',  3.2),
    ('Fan Fav  (a=0.3, b=8.0)',   0.3,  8.0, C_FAN,  '--', 2.0),
    ('Judge Fav  (a=1.8, b=1.0)', 1.8,  1.0, C_JDG,  '--', 2.0),
    ('50/50 Baseline',            0.0,  5.0, C_BASE, ':',  2.0),
]

for label, a, b, col, ls, lw in curves:
    w = np.ones_like(t_fine)*0.5 if a == 0 else logistic(t_fine, a, b)
    ax_w.plot(t_fine, w, color=col, ls=ls, lw=lw, label=label, zorder=5)

# Shaded zones
ax_w.fill_between(t_fine, 0.5, 1.0, alpha=0.04, color=C_JDG)
ax_w.fill_between(t_fine, 0.0, 0.5, alpha=0.04, color=C_FAN)
ax_w.text(0.4, 0.86, 'Judge-Weighted Zone', fontsize=10, color=C_JDG,
          alpha=0.45, fontstyle='italic', fontweight='bold')
ax_w.text(0.4, 0.14, 'Fan-Weighted Zone', fontsize=10, color=C_FAN,
          alpha=0.45, fontstyle='italic', fontweight='bold')

# Phase annotations for Knee
bbox_kw = dict(boxstyle='round,pad=0.3', alpha=0.90)
arr_kw  = dict(arrowstyle='->', lw=1.4)

ax_w.annotate('Phase 1 · Fan-Driven\n"Let the audience\npick early favorites"',
              xy=(1.5, logistic(1.5, 0.8, 4.2)),
              xytext=(0.2, 0.07), fontsize=7.8, color=C_KNEE, fontweight='bold',
              arrowprops={**arr_kw, 'color': C_KNEE},
              bbox={**bbox_kw, 'fc': '#FFF5F5', 'ec': C_KNEE})

ax_w.annotate('Phase 2 · Crossover\n"Stakes rise;\njudges gain voice"',
              xy=(4.2, 0.5),
              xytext=(2.8, 0.83), fontsize=7.8, color='#E67E22', fontweight='bold',
              arrowprops={**arr_kw, 'color': '#E67E22'},
              bbox={**bbox_kw, 'fc': '#FFF8EE', 'ec': '#E67E22'})

ax_w.annotate('Phase 3 · Merit-Driven\n"Best dancer wins\nthe trophy"',
              xy=(9, logistic(9, 0.8, 4.2)),
              xytext=(7.3, 0.96), fontsize=7.8, color=C_JDG, fontweight='bold',
              arrowprops={**arr_kw, 'color': C_JDG},
              bbox={**bbox_kw, 'fc': '#F0FFF4', 'ec': C_JDG})

# Midpoint
ax_w.axhline(0.5, color='#AAA', lw=0.7, ls=':', alpha=0.5)
ax_w.axvline(4.2, color=C_KNEE, lw=0.9, ls=':', alpha=0.3)
ax_w.scatter([4.2], [0.5], color=C_KNEE, s=70, zorder=10, edgecolors='white', lw=2)
ax_w.text(4.45, 0.44, '$b=4.2$\n(crossover)', fontsize=7.5, color=C_KNEE)

# Formula box
ax_w.text(8.5, 0.28,
          r'$w_t = \sigma\!\left(a\,(t - b)\right)$' + '\n'
          r'$C_{i,t} = w_t \cdot p_J + (1-w_t)\cdot \tilde{v}^{\,\eta}$',
          fontsize=9, fontfamily='serif', color='#333',
          bbox=dict(boxstyle='round,pad=0.4', fc='#FFFFF0', ec='#CCC', alpha=0.92))

ax_w.set_xlabel('Week of Competition', fontsize=10.5, fontweight='bold')
ax_w.set_ylabel('$w(t)$ = Judge Weight Proportion', fontsize=10.5, fontweight='bold')
ax_w.set_xlim(0, 11)
ax_w.set_ylim(-0.02, 1.02)
ax_w.set_xticks(range(1, 12))
ax_w.set_xticklabels([f'W{w}' for w in range(1, 12)], fontsize=8.5)
ax_w.grid(alpha=0.18)
ax_w.spines['top'].set_visible(False)
ax_w.spines['right'].set_visible(False)
ax_w.set_facecolor('#FAFAF8')
ax_w.legend(loc='center left', fontsize=7.8, framealpha=0.92, edgecolor='#CCC')
ax_w.set_title("(b)  Dynamic Weight Curve:  $w_t = \\sigma(a \\cdot (t - b))$",
               fontsize=11.5, fontweight='bold', pad=14, color='#1a1a2e')

# ── Supertitle ──
fig.suptitle("The Recommended System's Blueprint — What Makes It Work?",
             fontsize=15, fontweight='bold', y=1.01, color='#1a1a2e', fontfamily='serif')
fig.text(0.5, 0.975,
         'Early fan engagement  →  gradual judge authority  →  merit-based finals',
         ha='center', fontsize=9.5, color='#666', style='italic')

plt.savefig('/home/claude/fig2_system_blueprint.png', dpi=250, bbox_inches='tight', facecolor='#FAFAF8')
plt.close()
print("✅ Figure 2 done")
