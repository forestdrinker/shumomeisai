Task 4｜新赛制机制设计（Pareto + 多目标 BO）
4.1 Task4 数据预处理（专项）
- 输入：
  - panel（评委分）
  - Task1 posterior draws（票份额分布）
  - Task2 的 SeasonSimulator（回放引擎，必须稳定）
- 评价单位：对每个候选规则参数 (\theta)，在多季、多后验抽样上 Monte Carlo 估计 4 目标的均值±方差。

---
4.2 Task4 核心公式
(A) 参数化规则族（示例：(\theta=(a,b,\eta,\gamma,\text{save_flag}))）
- 票份额“压缩/放大”（公平/抗刷票常用）：
[
\tilde v_{i,t}(\eta)=\frac{v_{i,t}^{\eta}}{\sum_{k\in\mathcal A_{s,t}} v_{k,t}^{\eta}}
]
- 动态权重（logistic）：（周越后期越重视“技术/稳定性”或反之都可）
[
w_t=\sigma\big(a(t-b)\big),\quad a>0
]
- Percent 组合（可落地，易解释）：
[
C_{i,t}(\theta)=w_t,p^J_{i,t}+(1-w_t),\tilde v_{i,t}(\eta)
]
- Save：若 save_flag=1，则按 bottom-two + judges choice（(\gamma) 控强度）。
(B) 四目标（必须清晰可写）
采用最终修改意见中推荐的“可写版本”：
1. 观众对齐：
[
Obj_F(\theta)=\rho_S\big(\mathrm{rank}(\bar v),\ \mathrm{rank}(P_\theta)\big)
]
2. 技术对齐：
[
Obj_J(\theta)=\rho_S\big(\mathrm{rank}(\bar S),\ \mathrm{rank}(P_\theta)\big)
]
3. 戏剧性（margin 越小越刺激）：
[
Obj_D(\theta)=\frac{1}{T}\sum_t \left(1-\frac{margin_t}{\max margin}\right),
\quad margin_t=C_{1st,t}-C_{2nd,t}
]
4. 鲁棒性（抗操纵）：
令 (P_\theta^{pert}) 为对 votes 扰动后的名次，(d_K) 为 Kendall 距离（归一化）：
[
Obj_R(\theta)=1-\mathbb{E}\left[d_K(P_\theta, P_{\theta}^{pert})\right]
]
注意：这里的“扰动强度”建议单独记为 (\kappa^{pert})，避免与 soft-rank 的 (\kappa_J,\kappa_F) 混淆。

---
4.3 Task4 伪代码（两阶段搜索：LHS → MOBO）
Stage 0：准备
1. Freeze SeasonSimulator（来自 Task2）
2. Choose evaluation budget: N0 (粗搜点数), N1 (BO 迭代), R (posterior draws), K (vote perturbations)
Stage 1：粗搜铺形状（Latin Hypercube / Random）
1. Sample (\theta^{(1)},...,\theta^{(N0)}) via LHS
2. For each (\theta^{(n)}):
  - For each season s and posterior draw r: run Replay(s, θ, v^(r))
  - Estimate Obj_F, Obj_J, Obj_D
  - Robustness: for k=1..K perturb votes → Replay → compute Kendall distance → Obj_R
  - Store mean±sd of 4 objectives
3. Compute Pareto front from evaluated points
Stage 2：多目标 Bayesian Optimization 精炼（qEHVI / TPE）
1. Initialize MOBO with evaluated dataset
2. Repeat iter=1..N1:
  - propose next (\theta^*) maximizing acquisition (qEHVI/TPE) near Pareto
  - evaluate (\theta^*) using same MC protocol
  - update surrogate
3. Output:
  - Pareto front（用于主图）
  - Knee point（膝点推荐）+ 两个备选（偏观众/偏技术）

---