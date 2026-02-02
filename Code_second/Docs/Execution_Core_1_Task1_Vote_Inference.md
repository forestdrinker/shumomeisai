## Task 1｜票份额反演（NUTS MCMC 主线）执行核心

Baseline Model（对照基线）
每周独立的 Hard-constraint 可行域解：CSP/LP/最大熵单点票份额（无跨周平滑）
- 在每一周单独求一组 (v_{i,t})，只要求“淘汰机制与该周规则一致”（硬约束），不给时间连续性：
  - 对 S1–S2 与 S28+：采用 Rank 合成规则；
  - 对 S3–S27：采用 Percent 合成规则；
  - 对 S28+：采用“两阶段”硬约束版本（淘汰者必须位于 bottom2，且与评委倾向一致的情形优先）。
- 输出通常是单点解（或少量可行解采样），不强调后验校准，仅用于：
1325. 证明我们不是“凭空拟合”；
1326. 给软模型/贝叶斯模型提供 sanity check。
1327. 逻辑库
### 1.1 Task1 数据预处理（专项）
- **输入**：panel 中该季 $s$ 的所有 active 行 $(t,i)$、以及 $\mathcal E_{s,t}$（$t<T_s$）和决赛周 finalists 集合 $\mathcal F_s=\mathcal A_{s,T_s}$。
- **仅在 active 内建模**：每周 softmax 的归一化必须只对 $\mathcal A_{s,t}$ 做。
- **训练观测分两类**：
  1. 淘汰周观测：$t<T_s$ 且 $|\mathcal E_{s,t}|\ge 1$
  2. 决赛周名次观测：$t=T_s$ 时 finalists 的 placement 诱导的排序 $\pi_s$
该“淘汰事件 + 决赛排序”双观测结构，能在不引入“真实票数”的前提下增强可识别性，同时保持可实现性（不做 34 季巨模型）。

---

### 1.2 Task1 核心公式（LaTeX 最小骨架）

**(A) 人气随机游走 + 票份额映射**
$$
u_{i,1}\sim\mathcal N(0,\tau_u^2),\qquad
u_{i,t}=u_{i,t-1}+\epsilon_{i,t},\quad \epsilon_{i,t}\sim\mathcal N(0,\sigma_u^2)
$$
$$
v_{i,t}=\frac{\exp(u_{i,t})}{\sum_{k\in\mathcal A_{s,t}}\exp(u_{k,t})},\qquad i\in\mathcal A_{s,t}
$$

**(B) 可微 soft-rank（用于 rank/percent 规则的连续化）**
对同一周的某个“好分数” $x_i$（越大越好），定义：
$$
\widetilde{\mathrm{rank}}_{\kappa}(x_i)=1+\sum_{k\in\mathcal A_{s,t},k\ne i}\sigma\left(\frac{x_k-x_i}{\kappa}\right),\quad
\sigma(z)=\frac{1}{1+e^{-z}}
$$
- 直觉：若 $x_i$ 很大，则 $x_k-x_i$ 多为负，sigmoid 很小，rank 接近 1（最好）。
- 你们的 $\kappa$ 指导（避免拍脑袋）：
  $$
  \kappa \approx \frac{c}{\Delta},\quad \Delta=\mathrm{Std}\big(S_{i,t}; i\in\mathcal A_{s,t}\big),\ c\in[2,4]
  $$
  并用 CV-Brier 微调。

**(C) 规则映射 → “坏度”$b_{i,t}$（越大越危险）**
Rank 合并（赛季 1–2 & 28+）：
$$
\tilde r^J_{i,t}=\widetilde{\mathrm{rank}}_{\kappa_J}(S_{i,t}),\quad
\tilde r^F_{i,t}=\widetilde{\mathrm{rank}}_{\kappa_F}(v_{i,t})
$$
$$
b_{i,t}=\tilde r^J_{i,t}+\tilde r^F_{i,t}
$$

Percent 合并（赛季 3–27）：
$$
C_{i,t}=p^J_{i,t}+v_{i,t}\quad (\text{只差常数权重，排序不变})
$$
$$
b_{i,t}=\widetilde{\mathrm{rank}}_{\kappa_C}(C_{i,t})
$$

**(D) 淘汰似然（支持多淘汰）：Plackett–Luce 选“最差”**
令当周淘汰数 $m_{s,t}=|\mathcal E_{s,t}|$。将淘汰集合按任意顺序写成 $(e_{t,1},...,e_{t,m})$，定义：
$$
$$
\mathbb P_{\text{main}}(\mathbf e_t\mid \mathbf b_t)=
\prod_{\ell=1}^{m_{s,t}}
\frac{\exp(\lambda b_{e_{t,\ell},t})}
{\sum_{i\in \mathcal A_{s,t}\setminus\{e_{t,1},...,e_{t,\ell-1}\}}\exp(\lambda b_{i,t})}
$$
**Mixture Model (爆冷机制)**：引入 $\rho$ (e.g., 0.05) 代表随机淘汰概率：
$$
\mathbb P(\mathbf e_t\mid \mathbf b_t) = (1-\rho)\mathbb P_{\text{main}}(\mathbf e_t\mid \mathbf b_t) + \rho \cdot \text{Unif}(\mathbf e_t \mid \mathcal A_{s,t})
$$
- $\lambda$ 越大越接近硬淘汰，越小越“软”。$\rho$ 负责兜底 Coverage。
- $\lambda$ 越大越接近硬淘汰（winner-take-all），越小越“软”。

**(E) Judges’ Save（两段机制的概率版，避免硬阈值）**
对 save 周（默认 $s\ge 28$）：先用上式选 bottom-two（最差二人），再由评委在二人中决策。
用同一个“选最差”的 PL 思路表示 bottom-two 的有序抽样：
$$
\mathbb P(z_1=i,z_2=j\mid \mathbf b_t)=
\frac{\exp(\lambda b_{i,t})}{\sum_k\exp(\lambda b_{k,t})}\cdot
\frac{\exp(\lambda b_{j,t})}{\sum_{k\ne i}\exp(\lambda b_{k,t})}
$$
评委偏好强度 $\gamma$（越大越偏向“淘汰评委更不认可者”）：
$$
\mathbb P(e=i\mid \{i,j\})=
\frac{\exp(\gamma \tilde r^J_{i,t})}{\exp(\gamma \tilde r^J_{i,t})+\exp(\gamma \tilde r^J_{j,t})}
$$
边缘化 bottom-two 顺序（实现时用 log-sum-exp）：
$$
$$
\mathbb P(e=i)=\sum_{j\ne i}\Big[\mathbb P(z_1=i,z_2=j)\mathbb P(e=i\mid\{i,j\})
+\mathbb P(z_1=j,z_2=i)\mathbb P(e=i\mid\{i,j\})\Big]
$$
同样应用 Mixture Model：
$$
\mathbb P_{\text{final}}(e=i) = (1-\rho)\mathbb P_{\text{save}}(e=i) + \rho \frac{1}{|\mathcal A_{s,t}|}
$$

**(F) 决赛周名次（final placement）观测：Plackett–Luce 排序似然**
对 finalists 集合 $\mathcal F_s$ 的观测名次（placement）给出排序 $\pi_s$（1st,2nd,...），用“好度”$a_{i,T_s}=-b_{i,T_s}$：
$$
\mathbb P(\pi_s\mid \mathbf a_{T_s})=
\prod_{k=1}^{|\mathcal F_s|}
\frac{\exp(\lambda_{\text{fin}} a_{\pi_k,T_s})}
{\sum_{j\in \mathcal F_s\setminus\{\pi_1,...,\pi_{k-1}\}}\exp(\lambda_{\text{fin}} a_{j,T_s})}
$$

---

### 1.3 Task1 诊断/验证所需指标（必须落盘）

**(A) Posterior Predictive Coverage（淘汰事件的覆盖率）**
对每个淘汰周 $(s,t)$，由后验采样估计：
$$
\hat p_{s,t}(i)=\mathbb P(e_{s,t}=i\mid \text{data})
$$
令 $\mathcal C_{90}(s,t)$ 为累计概率达到 0.9 的最小集合，则：
$$
\mathrm{Coverage}_{90}=\frac{1}{N}\sum_{(s,t)}\mathbf 1\big[\text{true eliminated}\in\mathcal C_{90}(s,t)\big]
$$
（这是你们最终修改意见里要求的“可执行 coverage 定义”）。

**(B) Consistency 指标（建议同时给 4 个）**
- **Acc**：$\mathbf 1[\arg\max_i \hat p_{s,t}(i)=e^{\text{true}}_{s,t}]$
- **Top-2**：真实淘汰者是否在预测概率最高的 2 人内
- **Brier**：
  $$
  \mathrm{Brier}=\frac{1}{N}\sum_{(s,t)}\sum_{i\in\mathcal A_{s,t}}(\hat p_{s,t}(i)-\mathbf 1[i=e^{\text{true}}_{s,t}])^2
  $$
- **平均 CI 宽度（certainty）**：对 $v_{i,t}$ 取 $q_{0.95}-q_{0.05}$

**(C) 可识别性 PCR（Posterior Contraction Ratio）**
对关键量（例如某季 vote share 的均值或某个 finalist 的 $v_{i,t}$）：
$$
\mathrm{PCR}:=\frac{\mathrm{Var}(\text{prior})}{\mathrm{Var}(\text{posterior})}
$$
PCR 接近 1 → 数据对该量信息弱；PCR 大 → 数据约束强。

---

### 1.4 Task1 伪代码（NUTS + 自动诊断 + 回退 + 校准套件）
该结构与“最终修改意见”一致：分赛季并行、R-hat/ESS 自动化、VI 仅做 warm start/回退、LOSO+Temporal+轻量SBC+Coverage。

**Algorithm 1：Per-Season NUTS Inference（主线）**
```text
Input: season s 的 panel_s, {E_st | t < Ts}, finalists 排序 pi_s, 超参数/先验设定
Output: posterior samples {u_it, v_it}, summary, diagnostics

1. Build active sets
For each week (t): compute A_st, compute S_it, pJ_it for i in A_st

2. Define likelihood
For each (t < Ts):
  - if (|E_st| == 0): skip (no info)
  - else if save enabled: use Judges’ Save marginal likelihood
  - else: use PL elimination likelihood with badness b_it determined by rule segment
142: For final week: add PL ranking likelihood on pi_s
143: 
144: ** Patch Update (Mixture) **:
145:   - Calculate p_main (PL or Save)
146:   - Calculate p_unif = 1 / |Active|
147:   - Likelihood = logsumexp( log(1-rho)+log(p_main), log(rho)+log(p_unif) )

3. Run NUTS (NumPyro/PyMC)
  - warmup W, samples R, chains C

4. Diagnostics (must log & plot)
  - compute R-hat, ESS for key scalars + random walk states
  - if any (R-hat > 1.01) or ESS too low:
    - Plan A: increase warmup / samples
    - Plan B: reparameterize random walk (non-centered)
    - Plan C: run full-rank VI warm start -> rerun NUTS

5. Posterior summary
  - store mean/sd/q05/q95 for v_it and u_it
  - store per-week uncertainty width

6. Posterior predictive
  - estimate p_st(i) for each 淘汰周
  - compute Acc/Top2/Brier/Coverage_90 + PCR

7. Return all artifacts
```

**Algorithm 2：Validation Suite（LOSO + Temporal + Light SBC）**
```text
1. LOSO
For each hold-out season (s0):
  - use remaining seasons to tune global hyperparams (e.g., c in kappa 公式, lambda prior scale) via CV-Brier
  - run Algorithm 1 on s0
  - record OOS Acc/Top2/Brier/Coverage/PCR

2. Temporal
  - tune hyperparams on S1–S27
  - evaluate on S28–S34（跨机制变化）

3. Light SBC（每段抽若干季）
For each regime segment in {Rank, Percent, Save}:
  - repeat R times: sample from prior -> simulate eliminations/placements -> run inference -> rank histogram

4. 输出 Validation Dashboard 所需数据（图/表在 Step3 绑定）
```
