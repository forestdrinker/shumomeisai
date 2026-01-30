Task 3｜归因（LMM 锚定 + GBDT bootstrap SHAP CI）

---
3.1 Task3 数据预处理（专项）
- 构造“周级样本”（推荐）：每行 = active 的 ((s,t,i))
- 两个目标（与逻辑库一致）：
  - 评委目标：(;y^J_{i,t}=p^J_{i,t};)（周内归一后更可比）
  - 观众目标（修改：引入 Task1 后验不确定性外循环，不再只用单点均值）：
    - 基础量仍来自 Task1 的 vote share 后验样本 ({v^{(r)}{i,t}}{r=1}^R)
    - 在周级层面定义：
[
y^{F,(r)}{i,t}=v^{(r)}{i,t}\quad(\text{或回放口径一致时在 active 上重归一化})
]
    - 并保留点估计用于对照：
[
y^F_{i,t}=\mathbb E[v_{i,t}\mid\text{posterior}]
]
- 特征 (X_{i,t})（最低配必须有）：
  - celebrity_age_during_season（标准化）
  - industry/state/country one-hot（或 target encoding）
  - week index、remaining contestants count
  - 表现衍生：rolling mean/slope/volatility（对 (p^J) 与 (v) 都可做）
  - partner id（用于 LMM random effect；GBDT 中可用 embedding）
  - **（可选但推荐，支持 certainty-aware）：**加入 Task1 的投票不确定性特征，例如 (\mathrm{CIwidth}(v_{i,t})) 或 (\mathrm{Var}(v_{i,t}))，用于解释“粉丝支持的可信度差异”（不改变主标签外循环方案，仅作为补充特征）
- 网络特征（CS 加分项）：明星–舞伴二部图 (G) 的中心性/Node2Vec embedding（给 GBDT & 可视化）。
- 数据切分：Group CV by season（防跨季泄漏）。

---
3.2 Task3 核心公式
(A) LMM（主结论口径：舞伴效应带 CI）
对 (k\in{J,F})：
- 评委目标（不变）：
[
y^{(J)}{i,t}=X{i,t}\beta^{(J)} + b^{(J)}{\text{partner}(i)} + b^{(J)}{\text{season}(s)} + \varepsilon^{(J)}_{i,t}
]
- 观众目标（修改：对每个 posterior draw 做外循环拟合）：
对每个 (r=1,\dots,R)：
[
y^{(F,(r))}{i,t}=X{i,t}\beta^{(F,(r))} + b^{(F,(r))}{\text{partner}(i)} + b^{(F,(r))}{\text{season}(s)} + \varepsilon^{(F,(r))}_{i,t}
]
随机效应与噪声（结构不变）：
[
b^{(k)}{\text{partner}}\sim\mathcal N(0,\sigma^2{\text{partner},k}),\quad
b^{(k)}{\text{season}}\sim\mathcal N(0,\sigma^2{\text{season},k}),\quad
\varepsilon^{(k)}_{i,t}\sim\mathcal N(0,\sigma^2_k)
]
- 输出（修改：对 fan 侧系数/舞伴效应做跨 (r) 汇总得到最终 CI）：
  - 对评委侧：直接输出 (\hat b^{(J)}_{\text{partner}}) 的排名 + CI（Top10/Bottom10 + (\Delta)）。
  - 对粉丝侧：对每个 partner 的 (\hat b^{(F,(r))}{\text{partner}}) 做跨 (r) 汇总：
[
\text{CI}{95}\big(b^{(F)}{\text{partner}}\big)=\big[q{0.025}({\hat b^{(F,(r))}}),\ q_{0.975}({\hat b^{(F,(r))}})\big]
]
并输出“粉丝侧带飞/拖累”的 Top10/Bottom10（连同不确定性）。
解释口径：这样得到的 fan-side 置信区间同时包含（i）LMM估计误差与（ii）Task1 票份额后验不确定性带来的标签不确定性。
(B) GBDT + bootstrap SHAP CI（非线性洞见）
- 模型：XGBoost/LightGBM 分别拟合 (y^J) 与 (y^F)
- **评委侧（不变）：**直接用 (y^J) 做 Group-CV + bootstrap SHAP CI。
- 粉丝侧（修改：posterior propagation 的两种实现，默认用省算力版）：
  - **默认（省算力、易落地）：**训练/调参使用点估计 (y^F=\mathbb E[v])，但在 bootstrap 中额外对 posterior draws 做抽样，把票不确定性注入 SHAP CI：
    - 每次 bootstrap (b)：先抽一批 season 形成 block bootstrap 集，再随机抽一个 draw (r_b)（或抽若干 draws 取平均）生成标签 (y^{F,(r_b)})
    - 训练 (\to) 计算 SHAP importance (I^{(b)}_f)
    - 最终输出：(\mathrm{mean}(I_f)) 与 ([q_{0.025},q_{0.975}])
  - **严格（算力更高，可选）：**对多个 (r) 重复训练/评估并汇总 SHAP（双层汇总：bootstrap × posterior）。
（这是最终修改意见的硬要求：SHAP 必须配 CI。）

---
3.3 Task3 伪代码
1. Build weekly dataset
  - (D_J={(X_{i,t},y^J_{i,t},season,partner)})
  - (D_F^{(r)}={(X_{i,t},y^{F,(r)}_{i,t},season,partner)},\ r=1..R)
  - Also keep point estimate (D_F={(X_{i,t},y^F_{i,t},season,partner)}) for baseline/ablation
2. Fit LMM for Judge (once):
  - fixed effects: (X)
  - random intercepts: (1|partner) + (1|season)
  - extract partner effects + CI + support_n
3. Fit LMM for Fan with posterior propagation (外循环，新增):
For r = 1..R:
  - fit LMM on (D_F^{(r)}) with same structure
  - store (\hat\beta^{(F,(r))}), (\hat b^{(F,(r))}{partner}), (\hat b^{(F,(r))}{season})
  - aggregate across r:
    - for each coefficient/effect: compute mean + q05/q95 (or q025/q975)
    - output partner effect ranking with CI (fan-side)
4. Fit GBDT twice with GroupKFold(season):
  - Judge: train/eval on (D_J); record CV metrics (RMSE/MAE)
  - Fan: train/eval on (D_F) (point estimate) as main model; record CV metrics
  - (optional) ablation: no-network vs network embeddings
5. Bootstrap SHAP CI (block by season):
For b = 1..B:
  - sample seasons with replacement → build bootstrap set
  - train Judge-GBDT on bootstrap set; compute SHAP importance vector (I^{J,(b)})
  - Fan-GBDT:
    - draw r_b ~ Uniform({1..R}) (新增：posterior draw sampling)
    - build labels (y^{F,(r_b)}) on the bootstrap set
    - train Fan-GBDT; compute SHAP importance vector (I^{F,(b)})
6. Output:
  - LMM effect tables:
    - Judge: partner effects + CI
    - Fan: partner effects aggregated across r (posterior-propagated CI) + support_n
    - fixed-effect coefficients for Judge vs Fan (direction & magnitude comparison)
  - GBDT performance + ablation matrix（无网络/有网络）
  - SHAP diverging bar + CI 的作图数据（Judge and Fan）

---