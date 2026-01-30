
下面我按你给的旧版公式/伪代码结构，给出“能让工程同学直接改代码”的修改意见：哪些地方替换/新增、新增哪些变量、伪代码第几步插入什么分支。不需要改的我就不动。

---
1.1 数据预处理（专项）——建议改动点
✅ 保留不变
- active 集合 (\mathcal A_{s,t}) 只在 active 内 softmax 归一化
- “淘汰事件 + 决赛排序”双观测结构保留
🔧 新增两个“制度标记”字段（让后面规则层可控）
你现在预处理输入里只有 (\mathcal E_{s,t})、(\mathcal F_s)。建议再加：
1. 合成规则模式
[
\text{mode}_s\in{\text{Rank},\text{Percent}}
]
默认：S1–S2 Rank；S3–S27 Percent；S28+ Rank（并在验证里做敏感性）。
2. 是否启用 Judges’ Save 的周标记（按周，不按季“一刀切”）
[
\text{save}_{s,t}\in{0,1}
]
原因：现实里 S28+ 也不是每一周都必然 save（有些周规则不同、双淘汰等），你用按季默认容易误伤。
代码实现最简单：预处理阶段构造一个 save_weeks 列表/字典；没有信息就按默认规则启用，但留口子让你做敏感性。

---
1.2 核心公式（LaTeX 最小骨架）——逐段修改建议
(A) 人气随机游走 + 票份额映射
✅ 不改。

---
(B) soft-rank 连续化
✅ 不改定义。
🔧 只建议：把 (\kappa) 的“每周自适应”写清楚（你已有 (\Delta=\mathrm{Std}(S)) 的想法很好）。如果你代码里目前是全季常数 (\kappa)，建议改成：
- kappa_week = c / Std(score_week)（score_week 对 rank 模式用 (S)，对 percent 模式用 (C)）

---
(C) 规则映射 → “坏度” (b_{i,t})
✅ 你现在 Rank/Percent 的 (b_{i,t}) 定义本身是对的。
🔧 建议唯一改动：把“模式”显式参数化，避免 if 写死季号。
建议写法（逻辑不变，但对代码更友好）
- if mode_s == "rank": 用你 Rank 合并那套
- else: 用 Percent 合并那套

---
(D) 淘汰似然（PL 选“最差”）
✅ 保留，但我建议加一个“制作人喜欢冲突/爆冷”的混合项（这是你想加的“现实随机性”，而且代码很好加）。
🔧 新增参数：爆冷/干预率 (\rho\in[0,1])
直觉：大多数时候按规则走（PL），少数时候随机性更大（节目效果/突发因素）。
把原来的
[
\mathbb P(\mathbf e_t\mid \mathbf b_t)=\text{PL}(\mathbf e_t;\mathbf b_t,\lambda)
]
改成（替换一行即可）：
[
\mathbb P(\mathbf e_t\mid \mathbf b_t)=
(1-\rho)\cdot \text{PL}(\mathbf e_t;\mathbf b_t,\lambda)
+\rho\cdot \text{Unif}(\mathbf e_t\mid \mathcal A_{s,t})
]
其中 (\text{Unif}(\mathbf e_t\mid \mathcal A_{s,t})) 是“从 active 里等概率抽出 (m_{s,t}) 个淘汰者”的概率（实现时用组合数/或按顺序抽样的等价形式即可）。
这一步就是把“制作人喜欢冲突/爆冷”的考量落成一个参数 (\rho)：
(\rho) 越大，越允许“规则解释不通但节目发生了”的周存在，而不会强行把票份额推到极端。

---
(E) Judges’ Save（两段机制）——两处小修 + 一处关键增强
✅ 你现有结构非常接近方案A
- bottom-two 用 PL 抽样 ✅
- 评委在二人中投票 ✅
- 边缘化顺序 ✅
🔧 1) 修正一个容易写错的指数项（强烈建议你检查代码）
你写的是：
[
\mathbb P(e=i\mid {i,j})=
\frac{\exp(\gamma,\tilde r^J_{i,t})}{\exp(\gamma,\tilde r^J_{i,t})+\exp(\gamma,\tilde r^J_{j,t})}
]
这里指数写法不标准，工程里容易误实现。建议改成清晰版本（推荐二选一）：
版本1（用评委“坏度”做logit）：
[
\mathbb P(e=i\mid{i,j})
=\frac{\exp(\gamma\cdot \tilde r^J_{i,t})}{\exp(\gamma\cdot \tilde r^J_{i,t})+\exp(\gamma\cdot \tilde r^J_{j,t})}
]
解释：评委越觉得他差（rank 越大），越可能淘汰他。
版本2（用评委原始分差做logit，更数值稳定）：
[
\mathbb P(e=i\mid{i,j})
=\sigma!\left(\gamma\cdot (S_{j,t}-S_{i,t})\right)
]
解释：如果 (S_i) 更低，则 (S_j-S_i>0)，淘汰 (i) 概率更大。
如果你们担心评委分尺度跨周变化大，用版本2更稳。

---
🔧 2) 把“制作人冲突/爆冷”同样加入 Save 周（跟(D)一致）
现在 Save 周里你的淘汰概率是“bottom2 + judge choice”的严格机制。建议加同样的混合项（非常好加）：
把你最后的
[
\mathbb P(e=i)=\text{SaveMarginal}(i;\mathbf b_t,\lambda,\gamma)
]
改成：
[
\mathbb P(e=i)=
(1-\rho)\cdot \text{SaveMarginal}(i;\mathbf b_t,\lambda,\gamma)
+\rho\cdot \text{Unif}(i\mid \mathcal A_{s,t})
]
这样 Save 周也允许少量“不可解释周”，避免把 (v_{i,t}) 拉爆。

---
🔧 3) （可选但推荐）让 (\rho) 或 (\lambda) 随“冲突程度”变化
如果你想更贴“制作人更爱在冲突大时搞事”，可以不引入新参数，只做一个确定性调度：
定义当周冲突度：
[
d_t=\mathrm{Corr}\big(\tilde r^J_{\cdot,t},\tilde r^F_{\cdot,t}\big)
]
然后设
- 冲突越大（相关越低），(\lambda_t) 越小（淘汰更软）或 (\rho_t) 越大（爆冷更多）
工程上最省事：先不做这步，只用全局 (\rho)（足够写进论文解释）。

---
(F) 决赛周名次观测
✅ 不改。
如果你已经在代码里实现了 (\lambda_{\text{fin}})，很好。

---
1.3 诊断/验证指标——只加不改（对你的现有体系很友好）
你的 Coverage/Acc/Top2/Brier/CI宽度/PCR 都很好。基于新增的 (\rho) 和 Save 机制，我建议只补两类“必须落盘”的指标：
🔧 新增 (D) 爆冷参数的后验与校验
1. (\rho) 的后验均值/CI（告诉读者这节目“有多不按套路”）
2. 按周 posterior predictive 的 log score / NLL（你已有 (\hat p_{s,t}(i))，直接能算）
🔧 新增 (E) Save 周专项一致性
只针对 (\text{save}_{s,t}=1) 的周，报告：
- Save 周的 Coverage(_{90})、Top2、Brier（单独一组）
- （可选）真实淘汰者落在 bottom2 的后验概率均值（模型自洽检查）

---
1.4 伪代码——“最少改动”的可执行 patch
下面按你 Algorithm 1 的步骤给插入式修改（基本就是加两个 if 和一个 mixture 计算）。

---
Algorithm 1：Per-Season NUTS Inference（主线）——修改版要点
Step 1：Build active sets
✅ 保留
🔧 加两行（从预处理拿制度）：
- mode_s = regime_mode[s]
- save_flag[t] = save_weeks.get((s,t), default_save_rule(s,t))

---
Step 2：Define likelihood（这里是核心改动点）
你现在是：
- save enabled → Judges’ Save marginal likelihood
- else → PL elimination likelihood
🔧 改成下面三层结构（代码非常直观）：
(2.1) 先算当周 badness (b_{i,t})
- b_t = badness_rank(...) if mode_s == rank
- b_t = badness_percent(...) if mode_s == percent
(2.2) 根据是否 save 走不同“主机制”
- if save_flag[t]==1：p_main = SaveMarginal(i | b_t, λ, γ)（你已有）
- else：p_main = PL_elim(i | b_t, λ)（你已有）
(2.3) 加入爆冷混合（制作人冲突）
- p = (1-ρ)*p_main + ρ*p_uniform
- 用 p 作为该周淘汰观测的 likelihood
这一步只需要你在代码里多写一个 rho，和一个 uniform 概率项。

---
Step 4：Diagnostics
✅ 保留
🔧 新增一个“参数级别诊断清单”
- 把 (\rho,\lambda,\gamma) 也放进 (\widehat R)、ESS key scalars 列表（避免这些关键参数没混好）

---
Step 6：Posterior predictive
✅ 保留
🔧 Save 周预测概率 (\hat p_{s,t}(i)) 要用你新的混合 (p)（否则指标不一致）

---
Algorithm 2：Validation Suite——只需加一段“制度敏感性”
你已经有 Temporal（S1–S27 tune，S28–S34 eval），很好。
🔧 建议再加一个小循环（非常省事但论文很加分）：
- 对 switch_season ∈ {28,29}：
  - 设定 mode_s 在 switch 前后不同
  - 重跑关键季（只跑少数季也行）
  - 报告主结论/coverage 是否稳健

---
最后：你改代码时的“最小变更清单”
如果只允许做最少改动，我建议按优先级：
1. 加 per-week save_flag（避免 S28+ 全周硬套）
2. 在淘汰似然里加混合爆冷 (\rho)（D 和 E 都加）
3. 把评委二选一概率公式改成可实现的 logit（避免指数写错）
4. Validation 加制度敏感性（switch=28/29）（少跑几季也行）

---
如果你把你现有代码里 **likelihood 那段函数/类（尤其是 SaveMarginal 部分）**贴两屏出来，我可以按你当前变量名、函数签名，给你一份“逐行替换”的 patch 版本（保持你工程结构不变）。