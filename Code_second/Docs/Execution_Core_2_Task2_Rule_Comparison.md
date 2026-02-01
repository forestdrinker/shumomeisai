
Task 2｜规则反事实回放（Posterior-driven Monte Carlo Replay）

---
2.1 Task2 数据预处理（专项）
- 输入：
  1. 原始 panel（真实评委分 $S_{i,t}$、赛季结构 $T_s$、每周淘汰数 $m_{s,t}=|\mathcal E_{s,t}|$、决赛周 finalists/placements）
  2. Task1 输出的 posterior draws $\{v^{(r)}_{i,t}\}$（建议同时保留 posterior mean 便于做点估计对照）
- 反事实回放必须统一“节目结构”：
  - 每周淘汰数固定用真实 $m_{s,t}$（无淘汰/多淘汰周原样保留）
  - 决赛周以 finalists 的名次规则输出 placements（与 Task1 的 $\pi_s$ 口径一致）
- 反事实回放的 active 一致性处理（新增，必须）：
  - 回放过程中 active 集合 $\mathcal A^{(r)}_{s,t,R}$ 会随规则改变而改变；因此对每周使用的投票份额必须在当前 active 上重归一化：
$$
v^{(r)}_{i,t\mid A}=\frac{v^{(r)}_{i,t}}{\sum_{k\in A}v^{(r)}_{k,t}},\qquad i\in A
$$
  - Percent 合成时的评委百分比也必须在当前 active 上重算：
$$
p^{J}_{i,t\mid A}=\frac{S_{i,t}}{\sum_{k\in A}S_{k,t}},\qquad i\in A
$$
- 规则场景集合（建议明确写死，题面对应）：
$$
\mathcal R=\{\text{rank},\ \text{percent},\ \text{rank+save},\ \text{percent+save}\}
$$
（若你们最终只想报告 3 种，可在结果汇总时合并，但回放引擎保持 4 种最稳。）

---
2.2 Task2 核心公式（回放与指标）
(A) 回放算子（deterministic given $v$）
给定规则 $R\in\mathcal R$ 与某次后验抽样 $v^{(r)}$，定义：
$$
\mathrm{Replay}(s, R, v^{(r)})\to \text{placements }P^{(r)}_{s,R}\ \text{and eliminations }E^{(r)}_{s,t,R}
$$
- **实现约束（补充）：**回放时每周的合成/淘汰仅依赖当周评委分 $S_{i,t}$ 与当周 $v^{(r)}_{i,t\mid A}$，不引入“粉丝策略改变”。
(B) 偏向性：更偏观众还是更偏评委（保留，微调 active 口径）
设
$$
\bar v_i=\frac{1}{T_s}\sum_{t\le T_s} \mathbb E_r\big[v^{(r)}_{i,t}\big],\quad
\bar S_i=\frac{1}{T_s}\sum_{t\le T_s} \mathbb E_t\big[p^{J}_{i,t\mid \mathcal A_{s,t}}\big]
$$
（$\bar v$ 用后验均值即可；$\bar S$ 用真实 active 的评委百分比均值。）
Spearman（或 Kendall）相关：
$$
\rho_F(R)=\rho_S\big(\mathrm{rank}(\bar v),\ \mathrm{rank}(P_R)\big),\quad
\rho_J(R)=\rho_S\big(\mathrm{rank}(\bar S),\ \mathrm{rank}(P_R)\big)
$$
若 $\rho_F$ 更大 → 更“favor fans”；若 $\rho_J$ 更大 → 更“favor judges”。
注：$P_R$ 可取“预期名次”（对 draw 做平均）或“夺冠概率排序”。建议你们统一口径：用 $\mathbb E_r[P^{(r)}_{s,R}]$ 的名次或直接用 $P(\text{win})$ 排序。
(C) “冠军改变概率 / Top3 改变概率”（保留）
对规则 $R$ 与基准规则 $R_0$，用后验 MC：
$$
\mathbb P(\text{champion changes})\approx
\frac{1}{R}\sum_{r=1}^R \mathbf 1\big[P^{(r)}_{s,R}(1)\ne P^{(r)}_{s,R_0}(1)\big]
$$
Top3 类似。
(D) （新增）戏剧性/收视率视角指标（仅指标层，不改回放机制）
目的：量化“冲突与悬念”，支持第三问的娱乐性讨论。
(D1) 冲突度（Judge–Fan Disagreement）
对每周 $t$（在当前规则回放的 active 上）：
$$
D_{t}^{(r)}(R)=1-\rho_S\Big(\mathrm{rank}(S_{\cdot,t}),\ \mathrm{rank}(v^{(r)}_{\cdot,t\mid A})\Big)
$$
季级汇总（均值/后期均值）：
$$
\overline D^{(r)}(R)=\frac{1}{T_s-1}\sum_{t<T_s}D_t^{(r)}(R),\quad
\overline D_{\text{late}}^{(r)}(R)=\frac{1}{K}\sum_{t=T_s-K}^{T_s-1}D_t^{(r)}(R)
$$
(D2) 悬念感（Elimination Entropy，跨后验样本汇总算）
对每周淘汰者概率：
$$
\hat p_{s,t,R}(i)=\frac{1}{R}\sum_{r=1}^R \mathbf 1\big[E^{(r)}_{s,t,R}=i\big]
$$
熵：
$$
H_{s,t}(R)=-\sum_{i\in \mathcal A_{s,t}} \hat p_{s,t,R}(i)\log \hat p_{s,t,R}(i)
$$
并输出 $\overline H(R)$、$\overline H_{\text{late}}(R)$（同 $\overline D$ 的汇总方式）。
(D3) 爆冷率（Upset Rate，定义可执行）
一种简单可执行定义：淘汰者不在“评委排名最差的两名”中视作爆冷（或冠军评委均排名落在后25%视作爆冷夺冠）。
例如周级：
$$
\mathrm{Upset}_{t}^{(r)}(R)=\mathbf 1\big[E^{(r)}_{s,t,R}\notin \text{Bottom2ByJudge}(t)\big]
$$
季级爆冷率：
$$
\mathrm{UpsetRate}^{(r)}(R)=\frac{1}{T_s-1}\sum_{t<T_s}\mathrm{Upset}_{t}^{(r)}(R)
$$

---
2.3 Task2 伪代码（MC 回放）
Input: $panel_s$, 淘汰数序列 $\{m_{s,t}\}$, posterior draws $\{v^{(r)}\}$
Output: 差异热力图数据、冠军/Top3 改变概率、争议人物分布表、（新增）戏剧性指标分布
For each season $s$:
1. Precompute $S_{i,t}$ for all weeks; store baseline active sets $\mathcal A_{s,t}$
2. For each posterior draw $r = 1..R$:
For each rule scenario $R \in \{\text{rank, percent, rank+save, percent+save}\}$:
  - initialize active set $A \leftarrow \mathcal A_{s,1}$
  - initialize per-week logs: eliminated_list, conflict_list
  - For week $t = 1..T_s-1$:
a) restrict & renormalize votes on current active
- $v_A \leftarrow v^{(r)}_{\cdot,t}[A]$
- $v_A \leftarrow v_A / \sum v_A$
b) compute judge percent on current active (needed for percent mode)
- $p^J_A \leftarrow S_{\cdot,t}[A] / \sum_{i\in A} S_{i,t}$
c) compute combined under rule scenario R using current active A
- if R uses rank: build ranks from $S_A$ and $v_A$, combine
- if R uses percent: combine $p^J_A$ and $v_A$
d) (新增：记录当周冲突度，用于戏剧性指标)
- $D_t \leftarrow 1 - \mathrm{SpearmanCorr}(\mathrm{rank}(S_A),\mathrm{rank}(v_A))$
- append $D_t$ to conflict_list
e) eliminate $m_{s,t}$ couples
- if no “+save”:
eliminate lowest combined / highest badness (deterministic)
- else (“+save”):
first pick bottom-two by combined, then eliminate one by judge rule ($\gamma$ controls hard/soft)
f) update active set A (remove eliminated)
  - Final week:
rank remaining couples by combined → assign placements $P^{(r)}_{s,R}$
  - store replay artifacts:
    - eliminations $E^{(r)}_{s,t,R}$, placements $P^{(r)}_{s,R}$
    - conflict summaries: $\overline D^{(r)}(R)$, $\overline D_{\text{late}}^{(r)}(R)$
    - upset summaries: $\mathrm{UpsetRate}^{(r)}(R)$ (optional computed here or in aggregation)
3. Aggregate across draws (跨后验样本汇总，新增/扩展):
  - per-week elimination difference indicator (for heatmap)
  - champion/top3 change probability
  - $(\rho_F,\rho_J)$ distribution
  - (新增) Suspense entropy:
for each week t:
compute $\hat p_{s,t,R}(i)$ from $\{E^{(r)}_{s,t,R}\}$
compute $H_{s,t}(R)$, then $\overline H(R)$, $\overline H_{\text{late}}(R)$
  - (新增) drama metric summaries:
summarize distributions of $\overline D(R)$, $\overline D_{\text{late}}(R)$, $\overline H(R)$, $\overline H_{\text{late}}(R)$, UpsetRate(R)
4. For controversy names（Jerry Rice, Bobby Bones, …）extract their elimination-week / placement distribution across R
  - additionally report their drama context: average $D_t$ on weeks they were at risk (optional)
