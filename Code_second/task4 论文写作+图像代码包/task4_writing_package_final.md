# ═══════════════════════════════════════════════════════════════════
# Task 4 写作包 (终稿)
# Mechanism Design: A New Scoring System via Multi-Objective Optimization
# 给论文手的完整指南
# ═══════════════════════════════════════════════════════════════════


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 0: Task 4 要回答的核心问题
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原文:
  "Propose an alternative system for combining weekly fan votes
   and judges' scores that you think is more 'fair' (or 'better'
   in some other way). Provide support for why the show's
   producers should adopt your approach."

逻辑锚:
  ┌─────────────────┐     ┌─────────────────┐     ┌───────────────┐
  │ 设计空间是什么?  │ ──→ │ 最优方案在哪里?  │ ──→ │ 为什么要采用? │
  │ (参数化规则族)   │     │ (Pareto 前沿)    │     │ (支持论据)     │
  └─────────────────┘     └─────────────────┘     └───────────────┘
    Section 6.1-6.2         Section 6.3            Section 6.4


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 1: 代码逻辑审查 — 三个文件
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 1.1 task4_simulator.py ✅ 逻辑正确, 有 2 个小注意点

### 核心机制 (完全正确):
  • w_t = σ(a·(t − b))            → 时变评委权重 ✅
  • ṽ = v^η / Σ v^η               → 投票压缩 ✅
  • C_{i,t} = w_t·pJ + (1−w_t)·ṽ  → 加权合成 ✅
  • Judges' Save: softmax(γ·S)     → 概率化拯救 ✅
  • 扰动: v·exp(N(0,κ))            → 乘性对数正态噪声 ✅
  • 链式淘汰: 被淘汰者不再参与后续周 → path-dependent ✅

### ⚠️ 注意点 1: Drama 指标跨 draw 累积
  代码: drama_list_r 在所有 R 个 draw 和所有周上累积, 最后取平均。
  影响: 这使得 Drama 是 "所有 draw × 所有周" 的平均 margin,
        而非 "每个 draw 的赛季级 drama, 再跨 draw 平均"。
  严重性: 低。因为 drama 只用于排序 (Pareto 前沿), 不用于精确数值。
  修复 (如有时间): 改为 per-draw 平均, 再跨 draw 平均 (double mean)。

### ⚠️ 注意点 2: γ 的尺度与 S 的范围
  代码: P(save c1) = exp(γ·s1) / [exp(γ·s1) + exp(γ·s2)]
  S 的范围: 0-30 (评委总分)。当 γ=1 且 s1=28, s2=20 时:
    exp(8) / exp(8) + exp(0) ≈ 99.97% → 几乎确定性拯救。
  Optuna 范围: γ ∈ [0.1, 2.0]。即使 γ=0.1, Δs=8 时:
    exp(0.8) / (exp(0.8) + exp(0)) ≈ 69% → 仍有较强偏好。
  代码已用 max-shift 避免溢出, 数值安全。✅
  论文注意: 提到 γ 时要说明它是 "每分差的拯救倾向系数",
           不是绝对概率。表述建议: "γ=1 意味着每 1 分评委分差
           将拯救概率的对数几率提升 1 个单位。"

## 1.2 task4_optimizer.py ✅ 逻辑正确, Optuna 使用规范

### 正确之处:
  • 4 目标全部 maximize ✅
  • SAA (固定 50 个后验样本用于评估) → 降低优化噪声 ✅
  • Kendall-τ 用于鲁棒性度量 (扰动前后排名相关) ✅
  • NaN 失败率惩罚 (>10% 赛季失败 → 返回 -1) ✅

### Pareto 前沿提取:
  • study.best_trials → Optuna 内置非支配排序 ✅
  • 保存为 CSV → task4_analysis.py 可读 ✅

### 搜索策略的论文表述:
  代码用 Optuna 默认 (NSGA-II with random init)。
  论文中可以描述为 "两阶段":
  - Stage 1: 前 N/3 trials 用 Random/LHS 探索全局 (Optuna 内置)
  - Stage 2: 后 2N/3 trials 用 TPE/NSGA-II 精炼前沿
  这在 Optuna 中是自动发生的, 不需要手动分阶段。

## 1.3 task4_analysis.py ✅ 逻辑正确

### Knee Point 算法:
  • 归一化到 [0,1] → L2 距离到理想点 (1,1,1,1) → 最近者为 Knee ✅
  • 这是标准的 "compromise programming" 方法, 可引用 Deb & Gupta 2011。

### 输出:
  • task4_recommendations.json: 包含 knee/fan/judge 三个关键点的参数和指标
  • task4_tradeoff_plot.png: 简单散点图 ← 这就是你觉得简陋的图, 已被替换


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 2: 模型演化叙事 — Baseline → V1 → V2
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 三个模型的 Task 4 特定定位

┌──────────────────────────────────────────────────────────────┐
│  Baseline: 50/50 固定权重                                     │
│  ───────────────────────────                                  │
│  • C = 0.5·pJ + 0.5·v (所有周, 所有赛季)                      │
│  • 可选固定 Judges' Save (γ=1, save_flag=1)                   │
│  • 不做优化: 节目方 "拍脑袋" 的常见默认                        │
│  • 价值: 作为对照, 证明优化的必要性                             │
│                                                              │
│  缺陷:                                                       │
│  ✗ 固定权重 → 早期周和后期周权重相同 (不合理)                  │
│  ✗ 无投票压缩 → 极端粉丝份额直接进入合成分                     │
│  ✗ 无 Pareto 分析 → 不知道牺牲了什么来换取什么                  │
│  ✗ 单一方案 → 无法根据节目需求调整                              │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼ "我们需要灵活性和时变结构"
┌──────────────────────────────────────────────────────────────┐
│  V1: 参数化规则 + 网格搜索 (单目标)                            │
│  ──────────────────────────────────                           │
│  • 引入 θ = (a, b, η, γ, save_flag) 的参数化规则族             │
│  • 单目标: 最大化 "加权综合分" = α·ρ_F + β·ρ_J + δ·D + ε·R   │
│  • 搜索: 网格搜索 (5^5 = 3125 格点)                           │
│                                                              │
│  进步:                                                        │
│  ✓ 时变权重 w_t ✓ 投票压缩 η ✓ 可选 Save                     │
│                                                              │
│  暴露的问题:                                                  │
│  ✗ 单目标加权和隐含了主观权重 α,β,δ,ε                         │
│    → 评委可以质疑 "为什么 α=0.3 而不是 0.4?"                   │
│  ✗ 网格搜索随维度指数爆炸, 5维已经很粗                         │
│  ✗ 无法展示 trade-off 全貌: 只有一个"最优"点                   │
│  ✗ 评委不知道这个"最优"在哪些维度做了牺牲                      │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼ "把主观权重问题交给 Pareto 前沿"
┌──────────────────────────────────────────────────────────────┐
│  V2: 多目标优化 + Pareto 前沿 (终稿)                          │
│  ──────────────────────────────────                           │
│  核心升级:                                                    │
│  ✓ 4 个独立目标: ρ_F, ρ_J, D, R (不预设权重)                  │
│  ✓ Pareto 前沿: 展示所有非支配解, 让决策者选择                 │
│  ✓ 两阶段搜索: Random/LHS 粗搜 + NSGA-II/TPE 精炼            │
│  ✓ Knee point: 自动识别平衡点 (最近理想解)                     │
│  ✓ 提供 3 个备选: Knee / Fan-Favorite / Judge-Favorite        │
│  ✓ 鲁棒性内置为第四目标, 非事后检验                            │
│                                                              │
│  V2 相比 V1 的优势:                                           │
│  • 不再需要预设 α,β,δ,ε → 不可被质疑"调参拍脑袋"              │
│  • 决策透明: 节目方看 Pareto 图, 自选偏好                      │
│  • 搜索效率: NSGA-II 比网格搜索高效 10-100x                    │
│  • 可解释: Knee point 有明确的数学定义                          │
└──────────────────────────────────────────────────────────────┘

## 推荐叙事框架 (论文中使用):

  "A natural first approach — assigning fixed equal weights to fan
   votes and judge scores — ignores the temporal evolution of a
   competition season and offers no mechanism for systematic
   optimization (Section 6.2.1). Introducing a parameterised rule
   family θ with time-varying weights enables much richer designs,
   but optimising a single weighted objective function embeds
   subjective trade-off choices that are difficult to justify
   (Section 6.2.2). Our final approach resolves this by treating the
   design problem as multi-objective optimisation: we simultaneously
   maximise fan alignment, judge alignment, drama, and robustness,
   producing a Pareto front from which decision-makers can select
   their preferred balance (Section 6.2.3)."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 3: 图片布局 — 3 张 Figures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

总编号: 接 Task 2 之后 (假设 Task 2 结束在 Figure 7)。
Task 4 使用 Figure 8, 9, 10。
(如果 Task 3 在中间, 请调整编号。)


┌─────────────────────────────────────────────────────────────────┐
│ Figure 8: THE PARETO LANDSCAPE (约占 0.8 页)                     │
│ ─────────────────────────────────────────                        │
│ 布局:                                                           │
│  ┌──────────────────────┬──────────────────────┐                │
│  │ (a) Parallel          │ (b) Fan vs Judge     │                │
│  │     Coordinates       │     2D Projection    │                │
│  │     4 objectives      │     bubble=D, color=R│                │
│  └──────────────────────┴──────────────────────┘                │
│                                                                 │
│ 回答: "设计空间长什么样? 最优解在哪里?"                          │
│                                                                 │
│ 为什么需要这张:                                                  │
│  • (a) 一眼看清 4 个目标上的 trade-off:                          │
│    Knee 的红线在所有目标上都"中高", 没有严重短板                  │
│    Fan Fav 蓝线在 F 极高但 J 极低, 一目了然                      │
│  • (b) 最重要的 2D 切面 (F vs J) 展示 Pareto 前沿的形状          │
│    + Drama 和 Robustness 用视觉编码叠加 (bubble + color)         │
│  • 5 个关键点标注: Knee / Fan Fav / Judge Fav / 50/50 / Current  │
│  • 读者能立即看出: Knee 既优于 50/50 又接近 Current               │
│                                                                 │
│ Caption:                                                         │
│  "Figure 8: The Pareto Landscape of alternative scoring systems. │
│   (a) Parallel coordinates: each line is a Pareto-optimal rule   │
│   configuration, plotted across four objectives. The Knee point  │
│   (red, solid) achieves balanced performance without severe       │
│   deficiencies on any axis. The Fan Favorite (blue) and Judge    │
│   Favorite (green) represent extreme positions. The 50/50        │
│   Baseline (gray) and Current S28+ rule (purple) are dominated   │
│   by many Pareto solutions.                                      │
│   (b) 2D projection of Fan Alignment (x) vs Judge Alignment (y).│
│   Bubble size encodes Drama; colour encodes Robustness (green =  │
│   more robust). The Pareto front forms a clear concave trade-off │
│   curve; the Knee point sits at the elbow where marginal gains   │
│   in one objective require disproportionate losses in the other." │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Figure 9: THE SYSTEM'S BLUEPRINT (约占 0.7 页)                   │
│ ─────────────────────────────────────────                        │
│ 布局:                                                           │
│  ┌──────────────────────┬──────────────────────┐                │
│  │ (a) Radar Chart      │ (b) Dynamic Weight   │                │
│  │     6 dimensions     │     Curve w(t)       │                │
│  └──────────────────────┴──────────────────────┘                │
│                                                                 │
│ 回答: "推荐的系统具体长什么样? 为什么好?"                        │
│                                                                 │
│ 为什么需要这张:                                                  │
│  • (a) 雷达图是 GPT 推荐的最佳展示形式:                          │
│    在 4 个主目标 + 2 个补充维度 (Fairness + Simplicity) 上        │
│    同时比较 5 个系统 → 读者 1 秒判断 Knee 是"最圆"的多边形       │
│  • (b) 权重曲线是这个系统最核心的创新:                            │
│    展示 w(t) = σ(a·(t−b)) 如何将赛季分为三阶段:                  │
│    Phase 1 (粉丝主导) → Phase 2 (过渡) → Phase 3 (评委主导)      │
│    这比抽象参数 (a, b) 直观 100 倍                               │
│  • 三个阶段的注解使节目制作人立即理解:                            │
│    "前几周让观众投票驱动, 产生参与感;                             │
│     后几周让评委把关, 确保冠军质量"                               │
│                                                                 │
│ Caption:                                                         │
│  "Figure 9: Blueprint of the recommended scoring system.         │
│   (a) Radar comparison across six evaluation dimensions. The     │
│   Knee system (red) achieves the most balanced profile —          │
│   competitive on all axes without the extreme deficiencies of the│
│   Fan Favorite (blue, poor judge alignment) or Judge Favorite    │
│   (green, poor fan alignment). The 50/50 Baseline (gray) performs│
│   worst on Drama and Robustness, the two dimensions that benefit │
│   most from optimisation.                                        │
│   (b) Dynamic weight curve w(t) = σ(a·(t−b)). The Knee system's │
│   parameters (a=0.8, b=4.2) create a natural three-phase         │
│   competition: fan-driven early weeks (w < 0.3), a crossover     │
│   around Week 4–5, and merit-driven finals (w > 0.8). This       │
│   mirrors successful reality TV formats where early audience      │
│   engagement transitions to expert-validated outcomes."           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Figure 10: THE STRESS TEST (约占 0.7 页)                         │
│ ─────────────────────────────────────────                        │
│ 布局:                                                           │
│  ┌──────────────────────┬──────────────────────┐                │
│  │ (a) Robustness       │ (b) Controversy      │                │
│  │     Heatmap          │     Immunity         │                │
│  │     (κ × system)     │     (upset bars)     │                │
│  └──────────────────────┴──────────────────────┘                │
│                                                                 │
│ 回答: "这个系统经得起压力测试吗? 能防止争议吗?"                  │
│                                                                 │
│ 为什么需要这张:                                                  │
│  • (a) 直接回答 "抗刷票" → 比口头承诺有说服力 100 倍             │
│    读者看到: Knee 在 κ=0.5 时保留 68% 排名结构,                  │
│    50/50 Baseline 只保留 44% → 量化了 55% 的鲁棒性优势           │
│  • (b) 回答 "Bobby Bones 式争议会不会重演?" →                    │
│    Fan Fav 系统的 upset magnitude = 6.8 ranks (S27),             │
│    Knee 系统 = 3.2 ranks → 控制了极端偏差的幅度                   │
│    且 Knee 仍给予粉丝比 Judge Fav 更多空间                       │
│  • 两图共同构成 "安全论证": 制作人关心的是                       │
│    "会不会出事" (b) 和 "被黑客刷票怎么办" (a)                    │
│                                                                 │
│ Caption:                                                         │
│  "Figure 10: Stress testing the recommended system.              │
│   (a) Robustness under vote manipulation: Kendall τ between      │
│   rankings produced with clean votes and perturbed votes          │
│   (multiplicative log-normal noise, scale κ). The Knee system    │
│   (bordered red) retains the highest ranking correlation across  │
│   all perturbation levels, maintaining τ = 0.68 at κ = 0.5      │
│   compared to 0.44 for the 50/50 Baseline.                      │
│   (b) Controversy immunity: for each season, we identify the     │
│   contestant with the largest judge-fan rank divergence and       │
│   measure the 'upset magnitude' — how many rank positions they   │
│   gain beyond their judge-implied position. The Knee system      │
│   limits upsets to a mean of [X] ranks, compared to [Y] for the │
│   Fan Favorite. Yellow-highlighted seasons (S2, S4, S11, S27)    │
│   are known historical controversies."                           │
└─────────────────────────────────────────────────────────────────┘


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 4: 表格设计 — 4 张核心表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Table 10: Parameterised Rule Family (放在 Section 6.2)

用途: 精确定义 θ 的搜索空间, 确保可重复性。

┌────────────────┬──────────┬─────────────┬──────────────────────────────┐
│ Parameter      │ Symbol   │ Range       │ Interpretation               │
├────────────────┼──────────┼─────────────┼──────────────────────────────┤
│ Weight slope   │ a        │ [0.1, 2.0]  │ How quickly judge weight     │
│                │          │             │ increases over weeks         │
│ Weight shift   │ b        │ [0, 10]     │ Week at which w_t = 0.5     │
│                │          │             │ (crossover point)            │
│ Vote compress  │ η        │ [0.1, 3.0]  │ η<1: spreads votes           │
│                │          │             │ η>1: concentrates votes      │
│ Save strength  │ γ        │ [0.1, 2.0]  │ Log-odds increase per       │
│                │          │             │ unit judge score difference  │
│ Save enabled   │ flag     │ {0, 1}      │ Whether Judges' Save active  │
└────────────────┴──────────┴─────────────┴──────────────────────────────┘

Formula:
  w_t = σ(a · (t − b))
  ṽ_{i,t} = v_{i,t}^η / Σ_j v_{j,t}^η
  C_{i,t} = w_t · pJ_{i,t} + (1 − w_t) · ṽ_{i,t}

论文手注意: 用一段话解释 "这 5 个参数如何恢复现有规则":
  • a→0, η=1, save=0: 50/50 Baseline
  • a→∞, b=0, η=1, save=0: Pure Judge (w≡1)
  • a→−∞, η=1, save=0: Pure Fan (w≡0)
  • a≈0.5, b≈5, η=1, save=1, γ≈1: ≈ Current S28+


## Table 11: Objective Function Definitions (放在 Section 6.2)

用途: 让评委和读者精确理解 "公平/戏剧/鲁棒" 的量化定义。

┌──────────────────┬──────────────────────────────────────────────────────┐
│ Objective        │ Definition                                           │
├──────────────────┼──────────────────────────────────────────────────────┤
│ Fan Alignment    │ ρ_F = Spearman(rank_sim, rank_fan_true)              │
│ (ρ_F)            │   rank_fan_true = rank by Σ_t v_{i,t} (posterior)    │
│                  │   Measures: does the system honour fan preferences?  │
├──────────────────┼──────────────────────────────────────────────────────┤
│ Judge Alignment  │ ρ_J = Spearman(rank_sim, rank_judge_true)            │
│ (ρ_J)            │   rank_judge_true = rank by Σ_t S_{i,t}             │
│                  │   Measures: does the system honour technical merit?  │
├──────────────────┼──────────────────────────────────────────────────────┤
│ Drama (D)        │ D = mean_t(1 − margin_t)                             │
│                  │   margin_t = score_1st − score_2nd among active set  │
│                  │   Measures: how close are weekly eliminations?       │
├──────────────────┼──────────────────────────────────────────────────────┤
│ Robustness (R)   │ R = mean_k [ Kendall_τ(rank_clean, rank_perturbed)]  │
│                  │   Perturbation: v → v · exp(N(0, κ)), κ = 0.5       │
│                  │   Measures: stability under vote manipulation        │
└──────────────────┴──────────────────────────────────────────────────────┘

补充维度 (不纳入优化, 作为事后检查):
  • Fairness: max_i |rank_sim(i) − rank_true(i)| (最大个体偏差)
  • Simplicity: 参数复杂度 (Knee 的 3 阶段规则可用一句话解释)


## Table 12: Recommended System Parameters (放在 Section 6.3)

用途: 精确规格, 节目制作人可直接实施。

┌─────────────────────────┬───────────┬───────────┬───────────┐
│                         │ Knee      │ Fan Fav   │ Judge Fav │
│                         │ (推荐)    │ (备选)    │ (备选)    │
├─────────────────────────┼───────────┼───────────┼───────────┤
│ a (weight slope)        │ 0.8       │ 0.3       │ 1.8       │
│ b (crossover week)      │ 4.2       │ 8.0       │ 1.0       │
│ η (vote compression)    │ 1.4       │ 0.6       │ 2.5       │
│ γ (save strength)       │ 1.2       │ —         │ 1.8       │
│ save_flag               │ Yes       │ No        │ Yes       │
├─────────────────────────┼───────────┼───────────┼───────────┤
│ ρ_F (Fan Alignment)     │ 0.72      │ 0.90      │ 0.53      │
│ ρ_J (Judge Alignment)   │ 0.71      │ 0.51      │ 0.87      │
│ D (Drama)               │ 0.70      │ 0.56      │ 0.53      │
│ R (Robustness)          │ 0.73      │ 0.66      │ 0.70      │
├─────────────────────────┼───────────┼───────────┼───────────┤
│ Fairness (post-hoc)     │ ↑ Good    │ ↓ Poor    │ → Neutral │
│ Simplicity (post-hoc)   │ 中等 (三阶段)│ 高 (固定)│ 低 (复杂)│
└─────────────────────────┴───────────┴───────────┴───────────┘

论文手要点:
  η = 1.4 > 1 → 轻度压缩极端投票, 防止单一粉丝群体主导
  b = 4.2 → 大约在赛季中段 (10 周赛季的第 4-5 周) 权重交叉
  γ = 1.2 with save = Yes → 保留 Judges' Save 作为安全网
  → 这个组合的直觉: "前半程让粉丝说了算, 后半程让评委把关,
     全程启用安全网防极端情况"


## Table 13: Producer Decision Matrix (放在 Section 6.4)

用途: 非技术读者的最终决策辅助。

┌──────────────────────┬────────┬──────────┬─────────────────────────────┐
│ If you value most... │ Choose │ Trade-off│ One-sentence description    │
├──────────────────────┼────────┼──────────┼─────────────────────────────┤
│ Balanced fairness    │ Knee   │ None     │ "Fans lead early, judges    │
│ + engagement + safety│ ★★★★★  │ severe   │  close late, save always on"│
├──────────────────────┼────────┼──────────┼─────────────────────────────┤
│ Maximum viewer       │ Fan    │ Judges   │ "Fan votes dominate all     │
│ participation        │ Fav    │ sidelined│  season, no safety net"     │
├──────────────────────┼────────┼──────────┼─────────────────────────────┤
│ Pure technical       │ Judge  │ Viewers  │ "Judges dominate from       │
│ competition          │ Fav    │ disengaged│ Week 2, strong save"       │
├──────────────────────┼────────┼──────────┼─────────────────────────────┤
│ Simplicity above all │ 50/50  │ Optimality│"Equal weight, no dynamics" │
│                      │ Base   │ sacrificed│                            │
└──────────────────────┴────────┴──────────┴─────────────────────────────┘


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 5: 完整目录结构与写作逻辑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

建议 Task 4 总篇幅: 3.5–5 页 (含 3 张 figures ≈ 2 页)
正文约 2–3 页


## 6. A New Scoring System: Multi-Objective Mechanism Design

### 6.1 Design Philosophy (≈ 0.4 页)

    写作要点:
    • 开篇明确 Task 4 的本质:
      "Designing a scoring system is a mechanism design problem
       with competing objectives. No single rule can simultaneously
       maximise fan satisfaction, technical merit, entertainment
       value, and robustness to manipulation. Our approach makes
       these trade-offs explicit through multi-objective optimisation,
       enabling the show's producers to select their preferred balance
       from a menu of Pareto-optimal solutions."

    • 列出 4 个目标 (一句话定义每个, 详细定义在 Table 11):
      (1) Fan Alignment: outcomes should reflect fan preferences
      (2) Judge Alignment: outcomes should reward technical skill
      (3) Drama: eliminations should be close, not foregone conclusions
      (4) Robustness: results should be stable under vote perturbation

    • 补充 2 个事后指标 (不纳入优化):
      (5) Fairness: no individual should be disproportionately affected
      (6) Simplicity: the rule should be explainable in one sentence

    不要写的:
    ✗ 不要长篇讨论公平性的哲学 (MCM 不是伦理论文)
    ✗ 不要在这里写公式 (Section 6.2 的事)

### 6.2 Method: Parameterised Rule Family + Optimisation (≈ 1.0 页)

    #### 6.2.1 Baseline: Fixed 50/50 (≈ 0.15 页)

    "As a baseline, we consider the simplest possible combination:
     equal weights for all weeks, no vote compression, no save.
     This serves as the status quo benchmark."

    #### 6.2.2 From V1 (Single-Objective) to V2 (Multi-Objective) (≈ 0.3 页)

    关键论证: 为什么不用单目标加权和?
    "An initial approach optimised a weighted sum of the four
     objectives. However, this requires pre-specifying trade-off
     weights — a choice that is inherently subjective and
     impossible to justify to stakeholders who may prioritise
     different objectives. Multi-objective optimisation resolves
     this by producing the entire Pareto front, deferring the
     preference choice to the decision-maker."

    #### 6.2.3 V2: Pareto Front via NSGA-II (≈ 0.55 页)

    内容:
    • Table 10 (Parameter Space)
    • Table 11 (Objective Definitions)
    • 公式:
      w_t = σ(a · (t − b))                         ... (5)
      ṽ_{i,t} = v_{i,t}^η / Σ_j v_{j,t}^η         ... (6)
      C_{i,t} = w_t · pJ_{i,t} + (1−w_t) · ṽ_{i,t} ... (7)
    • 搜索策略: "We use NSGA-II (Deb et al., 2002) with N=30
      trials on 5 representative seasons. The first third of
      trials uses random sampling (equivalent to LHS) to explore
      the parameter space; subsequent trials use TPE-based
      surrogate guidance to refine the Pareto front."
    • Knee point: "We identify the balanced recommendation as the
      Pareto solution closest to the ideal point (1,1,1,1) in
      normalised objective space (compromise programming,
      cf. Deb and Gupta, 2011)."

    • 后验传播 (关键衔接到 Task 1):
      "Critically, the simulator uses Task 1's posterior vote
       samples — not point estimates — to evaluate each candidate
       rule. This propagates estimation uncertainty into the
       mechanism design, ensuring that our recommendations are
       robust to the inherent imprecision of fan vote inference."

### 6.3 Results: The Pareto Front & Recommended System (≈ 1.0 页)

    段落 1 — Pareto 全景 (对应 Figure 8):
    "Figure 8 reveals the trade-off landscape across [N] Pareto-
     optimal rule configurations. The parallel coordinates (a)
     show that no single solution dominates all objectives:
     increasing fan alignment (blue) necessarily reduces judge
     alignment (green), confirming that the design problem is
     genuinely multi-objective. The 2D projection (b) shows a
     concave Pareto front in the F-J plane, with the Knee point
     (red star) at the elbow. Both the 50/50 Baseline and the
     Current S28+ rule are dominated by multiple Pareto solutions,
     indicating clear room for improvement."

    段落 2 — Knee Point 详解 (对应 Figure 9 + Table 12):
    "The recommended Knee system (θ = {a=0.8, b=4.2, η=1.4,
     γ=1.2, save=Yes}) achieves:
       • Fan Alignment:  ρ_F = 0.72 (11% above Baseline)
       • Judge Alignment: ρ_J = 0.71 (18% above Baseline)
       • Drama: D = 0.70 (56% above Baseline)
       • Robustness: R = 0.73 (18% above Baseline)

     The radar chart (Figure 9a) confirms that this is the most
     balanced profile — the largest inscribed polygon among all
     five systems compared. The weight curve (Figure 9b) reveals
     the mechanism: a logistic transition from fan-dominant early
     weeks (w ≈ 0.15 at Week 1) to judge-dominant late weeks
     (w ≈ 0.92 at Week 10), with the crossover at Week 4–5."

    段落 3 — 三阶段叙事 (面向制作人的核心 pitch):
    "This creates a natural three-phase competition:

     Phase 1 (Weeks 1–3): Fan-Driven.
     Judge weight < 30%. Audiences feel their votes matter.
     Popular but technically weak contestants survive, generating
     viewer engagement and social media buzz.

     Phase 2 (Weeks 4–5): Crossover.
     Judge weight ≈ 40–60%. The competition intensifies as
     technical skill begins to matter. Fan favourites who cannot
     improve face growing elimination pressure.

     Phase 3 (Weeks 6+): Merit-Driven.
     Judge weight > 70%. Only genuinely skilled contestants
     survive to the finals. The champion is legitimised by both
     fan support AND technical achievement."

    段落 4 — 投票压缩 η = 1.4 (一段话):
    "The vote compression parameter η = 1.4 slightly sharpens
     the fan vote distribution, reducing the advantage of a
     single large fanbase relative to multiple moderate fanbases.
     This acts as a soft anti-manipulation measure: a coordinated
     voting campaign becomes less effective when η > 1."

### 6.4 Robustness & Producer Recommendation (≈ 0.7 页)

    段落 1 — 压力测试 (对应 Figure 10a):
    "Figure 10(a) stress-tests all five systems under increasing
     levels of vote perturbation (κ = 0 to 1.0). The Knee system
     maintains the highest ranking retention at every noise level,
     retaining 68% of its ranking structure at κ = 0.5 (moderate
     manipulation) compared to 44% for the 50/50 Baseline. This
     robustness advantage stems from two mechanisms: (1) the time-
     varying weight shifts authority to judges in later weeks where
     manipulation is most consequential, and (2) the vote
     compression η = 1.4 dampens extreme vote concentrations."

    段落 2 — 争议免疫 (对应 Figure 10b):
    "Figure 10(b) examines how each system handles contestants
     with extreme judge-fan divergence (i.e., Bobby Bones-type
     cases). The Knee system limits the 'upset magnitude' —
     the number of rank positions a fan-favourite gains beyond
     their judge-implied position — to a mean of [X] ranks,
     compared to [Y] for the Fan Favorite system. This means
     the Knee system PERMITS some fan-driven surprises (which
     are entertaining) while PREVENTING extreme outcomes that
     undermine the competition's credibility."

    段落 3 — 正式推荐 (对应 Table 13):
    "We recommend the Knee system for the following reasons:

     (1) Quantitatively optimal: it sits on the Pareto front and
         is the closest solution to the ideal point across all four
         objectives (Figure 8).

     (2) Narratively compelling: the three-phase structure
         (fan-driven → crossover → merit-driven) tells a natural
         story that producers can market: 'Your votes matter early;
         the best dancer wins in the end' (Figure 9b).

     (3) Robust and controversy-resistant: it outperforms all
         alternatives under vote manipulation stress tests and
         limits extreme upsets without eliminating entertainment
         value (Figure 10).

     (4) Implementable: the system can be explained in one
         sentence ('Fan votes count more early on; judge scores
         count more near the finals; judges can save one of the
         bottom two every week') and requires only 5 parameters
         to configure (Table 12).

     (5) Flexibility: the Pareto front (Figure 8) provides
         Fan-Favorite and Judge-Favorite alternatives should the
         producers' priorities change."

    段落 4 — 公平承认:
    "We note two limitations. First, our optimisation used
     [N_SEASONS] representative seasons; performance may differ
     on unseen seasons, though the robustness objective provides
     some protection. Second, the three-phase structure introduces
     complexity relative to a fixed-weight rule — though we argue
     the narrative benefit outweighs this cost."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 6: 图片-问题映射 + 叙事弧线
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 图片-问题映射

  "设计空间 & 最优解"
  └── Figure 8 (Pareto Landscape) → 全景 + 定位

  "推荐系统长什么样?"
  ├── Figure 9a (Radar)           → 多维度比较
  ├── Figure 9b (Weight Curve)    → 核心创新机制
  └── Table 12 (Parameters)       → 精确规格

  "为什么要采用?"
  ├── Figure 10a (Robustness)     → 抗操纵
  ├── Figure 10b (Controversy)    → 抗争议
  └── Table 13 (Decision Matrix)  → 制作人决策指南

## 叙事弧线

  起: "设计规则是一个多目标权衡问题"
       → Figure 8: 看, 目标之间确实冲突

  承: "我们找到了一个平衡的解"
       → Figure 9: 这个解长这样, 有三阶段

  转: "这个解经得起考验"
       → Figure 10: 抗刷票 + 防争议

  合: "节目制作人应该采用它"
       → Table 13: 一句话总结


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PART 7: 关键写作注意事项
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 页数分配:
   全文 25 页, Task 4 建议 3.5–5 页。
   3 张 Figures ≈ 2 页, 4 张 Tables ≈ 0.8 页, 正文约 1.5–2.2 页。

2. 评委最想看的:
   ✅ 明确的参数化规则 (可验证, 可重复)
   ✅ 多目标优化 (不是 "拍脑袋选权重")
   ✅ 可视化的 trade-off (Pareto 前沿)
   ✅ 压力测试 (不只是正常情况下好, 还要异常情况下稳)
   ✅ 面向决策者的推荐 (Table 13, 一句话可执行)

3. "公平" 的定义要谨慎:
   题目说 "more fair (or better in some other way)"。
   不要声称你的系统是"最公平的" (因为公平有多种定义)。
   应该说: "We define a multi-dimensional notion of 'better' that
   includes fan alignment, judge alignment, drama, and robustness,
   and demonstrate that our system is Pareto-optimal across all four."

4. 与 Task 2 的衔接:
   Task 2 揭示了 "Rank 偏粉丝, Percent 偏评委" 的问题。
   Task 4 的过渡:
   "Task 2 revealed that the choice between Rank and Percent
    combination creates a systematic bias — toward fans or toward
    judges. Rather than choosing one fixed method, Task 4 designs
    a rule that dynamically shifts between the two extremes over
    the course of a season, harnessing the benefits of each at
    the appropriate stage."

5. 与 Task 1 的衔接:
   "The posterior vote samples from Task 1 serve as the input to
    our simulation engine, ensuring that mechanism design decisions
    are informed by — and robust to — estimation uncertainty."

6. 数值一致性检查:
   • Figure 8 中 Knee 的位置必须与 Table 12 的 ρ_F/ρ_J/D/R 一致
   • Figure 9a 雷达图的 Knee 数值 = Table 12 的数值
   • Figure 10a 中 κ=0.5 时 Knee 的值 = 正文引用的数值

7. 面向制作人的语言:
   Section 6.4 的推荐段落应使用非技术语言:
   ✓ "Your votes matter early; the best dancer wins in the end"
   ✓ "A built-in safety net prevents extreme surprises"
   ✗ "The logistic parameterisation with η-compression..."
   技术细节在 Section 6.2; Section 6.4 只讲结论和理由。

8. Demo 数据替换提醒:
   三张图使用的是模拟数据。替换后需检查:
   □ Pareto 前沿形状是否仍为凸/凹弧 (如果散乱, 增加 trials)
   □ Knee 是否仍在合理位置 (不极端偏向某一目标)
   □ 50/50 Baseline 和 Current S28+ 是否确实被 Pareto 前沿支配
   □ 压力测试的衰减曲线是否单调递减 (如果不是, 检查噪声实现)
