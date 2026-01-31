下面给你一套**“8 张主图（每张可含 2–4 个子图）”的最终配置**：它能把题目 5 个任务全部覆盖、形成闭环证据链，同时在**准确性（无泄漏/可复现/有不确定性）**与**O 奖级审美（克制、统一、信息密度高但不乱）**上都站得住。

我会用“**图题 → 回答题目哪一问 → 画什么 → 关键标注 → 评委会看的硬指标**”来描述。所有设计都严格对齐题目要求：估计粉丝票、验证一致性与不确定性、比较 rank vs percent、争议案例、职业舞者/明星特征影响、提出更公平/更精彩的新系统。 

---

# 先给一条 O 奖级通用审美规范（你们 8 张图统一套用）

* **统一视觉语法**：同一含义同一视觉编码（例如：粉丝=实线，评委=虚线；基线=灰；推荐方案=加粗/星标）。
* **每张图只讲一个结论**（最多两个），图注第一句就是 take-away。
* **所有比较都带不确定性**：后验区间 / bootstrap CI / 跨季误差条，至少一种。
* **避免“看起来很强但其实泄漏”**：尤其是验证图、SHAP/特征重要性图，必须明确“只用历史信息/前向预测”。（你们代码里已经把“循环验证”列为严重风险并给出修正方案） 
* **版式建议**：A4 双栏论文常用：单栏图宽≈8–9cm、双栏图宽≈17–18cm；所有图导出矢量 PDF + 300dpi PNG 双份。

---

# 8 张“最完美主图”清单（我建议你们就按这个做最终版）

## Fig.1 模型与推断流程图（Graphical Abstract / Pipeline）

**回答**：你们怎么从“评委分 + 淘汰结果”推断出“粉丝票份额”，以及规则如何进入模型。对应 Task 1 的方法说明（评委最先看的可信度入口）。 

**画什么（建议 1 张双栏图，3 块区域）**

1. Data → Panel（S_it、pJ_it、active mask、淘汰事件）
2. Latent popularity random walk：(u_{t}\rightarrow v_{t}=\text{softmax}(u_t))
3. Likelihood（淘汰机制）+ 输出（v 后验、置信区间、w_judge、反事实回放）

**关键标注**

* “Fan votes are inferred (latent)”
* “Temporal predictive validation (t-1 → t)”（点名你们采用前向验证，避免自证） 
* 如果你们纳入权重估计：标出 (w_{\text{judge}}, w_{\text{fan}})（见你们带 Beta 先验的权重模型） 

**评委硬指标**

* 一眼看懂：输入是什么、推断什么、验证怎么做、后续比较怎么做。

---

## Fig.2 “真正预测”的验证与校准（Temporal Predictive Validation + Calibration）

**回答**：Task 1 两个核心问：

* 你估的粉丝票是否“能产生与实际淘汰一致的结果”？
* 不确定性有多大、是否随周/人变化？ 

**必须用“前向预测验证”，不要再用样本内自证**：你们已经在脚本里写清楚“原验证是循环验证”，修正为：用 (u_{t-1}) 后验 + 随机游走预测 (v_t)，再预测 t 周淘汰。 

**画什么（推荐 2×2 子图，双栏）**

* (a) Top-1 / Top-2 淘汰命中率（按赛季或按规则段聚合）+ Random baseline 虚线
* (b) 90% 可信集覆盖率（目标=0.90 的水平线）
* (c) Brier Score / Brier Skill Score（相对随机基线）
* (d) Reliability diagram（把预测淘汰概率分箱 → 实际淘汰频率），展示“校准而非只看准确率”

**关键标注**

* 明确写：Validation = Temporal Predictive (t−1 → t) 
* 图注写清：可信集定义、分箱方式、样本量（季×周）

**评委硬指标**

* 覆盖率接近标称值（0.90附近）比“100%完美”更可信。
* 校准曲线是否接近对角线。

---

## Fig.3 粉丝票后验与不确定性“长相”（Posterior Fan Vote Shares + Uncertainty Heterogeneity）

**回答**：Task 1 “certainty 是否对所有人/周相同？”——必须用一张图让评委直观看到“不确定性是异质的”。 

**画什么（建议：单季示例 + 结构化可读）**

* 选一个代表季（建议 Season 27 因题面点名争议、叙事强） 
* 画 **Top-K 选手（比如 K=6）** 的 (v_{it}) 后验均值轨迹 + 95% credible band
* 叠加：每周淘汰事件竖线（或被淘汰者标记“×”）
* 右侧（或下方）加一个小子图：该季“平均 CI 宽度/熵”随周变化（越到后期通常更确定）

**关键标注**

* 阴影=95% credible interval（写清分位数口径）
* 被淘汰后 (v=0) 的处理方式写在图注

**评委硬指标**

* 曲线是否遵守 simplex（份额非负、同周总和=1）且在淘汰节点上逻辑一致（你们诊断脚本强调这一点） 

---

## Fig.4 评委 vs 粉丝“实际影响力占比”后验（Posterior of (w_{\text{judge}}) Across Rule Segments）

**回答**：Task 2 的“哪种方法更偏向粉丝/评委？”——最有力的图不是“换规则冠军变没变”，而是**直接估出权重**。题目关心“谁更主导结果”。 

你们权重模型已经写了：(w_{\text{judge}}\sim \text{Beta}(2,2))，并输出 (w_{\text{judge}}, w_{\text{fan}})。 

**画什么（推荐：点区间 + 分组密度）**

* 主图：每个赛季一个点（后验均值）+ 95%CI，y 轴 (w_{\text{judge}}\in[0,1])，加 y=0.5 中线
* 分组副图：按 rule_segment（rank / percent / rank_save）画 ridge/density（或 box/violin），比较整体偏移

**关键标注**

* 在 x 轴背景做分段色块：S1–2(rank)、S3–27(percent)、S28+（合理假设为 rank+save），与题面叙述一致 
* 图注写：这是“从淘汰数据反推的有效权重”，不是节目官方权重

**评委硬指标**

* “偏向粉丝/评委”的结论必须来自可量化统计量（后验区间、显著偏离 0.5）。

---

## Fig.5 规则对比总览（Rank vs Percent vs Save variants）：稳定性 × 偏向性 × 节目效果

**回答**：Task 2 “对每一季同时应用两种方法比较结果，是否一方更偏向粉丝？” 

你们 replay + analysis 已经把核心指标做出来了：

* 结果变化概率：p_champion_change、p_top3_change
* 偏向性：(\rho_F, \rho_J)（模拟名次 vs 粉丝偏好/评委偏好相关）
* 戏剧性/悬念：drama_D、suspense_H、upset_rate 等 

**画什么（推荐：3 行小提琴/箱线 + 误差点，双栏）**

* Row 1（Outcome Stability）：p_champion_change、p_top3_change（相对 baseline=rank）
* Row 2（Bias）：(\rho_F) 与 (\rho_J) 的分布（每季一个点，规则汇总成分布）
* Row 3（Show Quality）：drama_D、suspense_H、upset_rate 的分布

**关键标注（非常重要，决定“准确性”）**

* 图注写清：这些指标来自对后验样本的回放模拟（replay），并说明“每规则每季的模拟次数”。你们 replay 逻辑确实支持抽样/子集模拟。 
* “upset_rate”的单位要写清（是率不是概率）

**评委硬指标**

* 同一指标同一尺度，色条/坐标统一。
* 对比必须是“同口径”，尤其是 baseline 的定义要写清（你们 analysis 里 baseline=rank 处理了 change=0）。 

---

## Fig.6 争议案例四宫格（Jerry Rice / Billy Ray Cyrus / Bristol Palin / Bobby Bones）

**回答**：Task 2 指名要你们分析争议明星：不同方法是否导致相同结果？引入“评委救人(bottom2 save)”会怎样？ 

**画什么（建议：4 行 × 2 列，小倍数图，非常 O 奖叙事）**
每一行一个案例（S2 Jerry Rice；S4 Billy Ray Cyrus；S11 Bristol Palin；S27 Bobby Bones）：

* 左列：时间轴（周）上 **Judge share pJ_it** vs **Inferred fan share v_it**（两条线 + CI 带），并用标记指出“judge bottom 2 但未淘汰”的周
* 右列：该案例在不同规则下的 **p_win / p_top3 / 期望名次** 对比条形图（带 CI），并用一条细线标出“真实名次/真实淘汰周”

你们的 per-celebrity 指标在 task2_analysis 里已经按规则输出了（p_win、p_top3、exp_rank 等）。 

**关键标注**

* 图注写：这些是“反事实回放结果”，不是节目真实公布的粉丝票（题面也说粉丝票未知） 
* 把“save”机制单独作为一种规则列（rank_save/percent_save），你们 replay 支持 save 规则并可设 gamma。 

**评委硬指标**

* 争议案例必须做到“证据链完整”：**分歧（pJ vs v）→ 规则机制差异 → 结果差异（概率/名次）**。

---

## Fig.7 Task 3：职业舞者与明星特征影响（Judge vs Fan 的“同/不同”）

**回答**：Task 3 “pro dancer 与明星特征（年龄/行业等）影响多大？对评委分与粉丝票是否一样？” 

这里我建议你们把“SHAP 蝴蝶图”从主图退下来，主图用更能回答题意、也更不容易被质疑的 **混合效应模型（LMM）森林图**。你们 LMM 脚本已经按 judge 与 fan 两套模型在做（含交叉随机效应思路）。 

**画什么（推荐：左右并排两列森林图）**

* 左：Judge model 固定效应（age_z、industry、week_norm）+ 关键 pro dancer 随机效应（Top 10/Bottom 10）
* 右：Fan model 同样结构（从 v 后验抽样/点估计得到 fan proxy），同样展示
* 最下面加一条：Pro dancer effect（fan） vs Pro dancer effect（judge）的散点图 + 相关系数（展示“是否一致”）

**关键标注**

* 明确哪些是 fixed effects、哪些是 random effects（评委非常在意解释口径）
* 行业类别太多时：只展示“相对基准行业”的 Top 6 绝对效应，剩余放附录

**评委硬指标**

* 能直接回答“同不同”：如果某些 pro dancer 在 fan 强但 judge 弱（或反之），这图会非常加分。

---

## Fig.8 Task 4：新规则推荐（Pareto 前沿 + 推荐点 + 鲁棒性曲线）

**回答**：Task 4 “提出更公平/更精彩的新系统并给出支持”。 

你们 task4_optimizer 里是四目标优化（对齐粉丝、对齐评委、戏剧性、鲁棒性），并且给出了 Pareto 前沿与 knee 点选择逻辑。  
模拟器也明确用了固定 seed 的抽样池（SAA 思路）来稳定优化评估，这一点在答辩时很好讲。 

**画什么（推荐 2×2 子图，双栏，作为压轴）**

* (a) Pareto scatter：x=(\rho_F)，y=(\rho_J)，点大小=Drama D，点颜色=Robustness R
  标出 3 个点：Knee（推荐）、Fan-favor、Judge-favor（你们 analysis 就是这么选的） 
* (b) Robustness curve：扰动强度 κ（投票噪声）→ Kendall tau（名次一致性），画 baseline vs 推荐方案（带区间带）
* (c) 推荐规则的“周权重曲线” (w_t=\sigma(a(t-b)))（让方案可执行、可解释）
* (d) 一张小表/小卡片（在图内角落）：推荐参数 (\theta={a,b,\eta,\text{save_flag},\gamma}) + 四目标均值±SD（来自你们优化日志）

**关键标注**

* 鲁棒性必须写清：是“同一规则在噪声扰动下的稳定性”，还是“相对现行系统的相似度”；建议你们采用前者（噪声=0 时 tau≈1）来避免概念混淆。你们优化里用 Kendall tau 做扰动前后名次一致性，本身就支持这种画法。 

**评委硬指标**

* 这是最容易拿高分的一张：**把“公平/偏向/精彩/稳定”四者的权衡可视化**，并且给出明确可执行参数。

---

# 你们现有 8 张图如何“升级成这一套”（快速对照）

如果按你们目前 MainFig1–8 来映射：

* **保留并升级**：MainFig2（→Fig3）、MainFig3（→Fig8(a) 的一部分）、MainFig5（→Fig5）、MainFig7（→Fig6）
* **建议主图替换/重做**：

  * MainFig1 → 必须换成 Fig.2（前向预测验证+校准），因为你们自己也写明“循环验证风险与修正方案”。 
  * MainFig4 → 建议换成 Fig.7（LMM 森林图），更贴 Task3，也更不容易被质疑。 
  * MainFig6 → 建议让位给 Fig.4（w_judge 后验）或并入 Fig.5（规则总览）
  * MainFig8 → 并入 Fig.8（鲁棒性曲线作为 (b) 子图），但口径要按上面修正

---
