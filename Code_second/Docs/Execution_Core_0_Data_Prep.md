# Step 2｜V2.0 New Master Library（Execution Core：清洗规则 + 核心公式 + 伪代码）

下面给 coder/论文手一套可直接落地实现的“执行核心库”。我先给全局数据预处理规范（所有 Task 共用），再分别给 Task1–4 的：
1. Data Preprocessing（硬规则）
2. Core LaTeX（主模型最小公式骨架）
3. Pseudo-code（不含 Python 的实现流程）

依据：你给的“最终修改意见（V2裁决：分赛季 NUTS + LOSO/Temporal/Coverage/SBC、PCR、κ指导、Task4四目标公式等）”与题面规则描述。

---

## 1. 全局数据预处理（Shared Data Engineering）

### 0.1 输入与索引统一
- **原始 CSV**：一行 = `(season, celebrity, partner)` 组合；评分字段为 `weekX_judgeY_score`，X=1..11，Y=1..4；评分可能为小数（多舞平均/bonus 平均摊）。
- **统一索引**（用你给的字典口径）：
  - 赛季：$s=1,\dots,34$
  - 周：$t=1,\dots,T_s$（每季周数不同）
  - 组合（明星–舞伴）：$i\in\{1,\dots,n_s\}$
  - 单评委分：$s_{i,t,j}$（可缺失）
  - 评委汇总分：$S_{i,t}$
  - 当周仍在赛集合：$\mathcal A_{s,t}$
  - 当周淘汰集合：$\mathcal E_{s,t}$
  - 潜在人气：$u_{i,t}$，票份额：$v_{i,t}$

### 0.2 缺失与“0 分”截断：必须硬处理
题面明确两类“非信息”必须区分：
- **N/A（NaN）**：
  1. 第 4 评委不存在；
  2. 该季根本没有那一周（例如 season 1 只有 6 周，week7–11 为 N/A）。
- **0 分**：表示该选手已淘汰，后续周的 0 分是“右截断/不再参赛”，不是“真实得分=0”。必须 mask 掉。

**硬规则**：
- 对每个 $(s,i,t)$，若该周四位评委分数全为 0（或 max=0），则判定 **不在赛**：$i\notin\mathcal A_{s,t}$。
- 若该周评委分全为 NaN（该季无此周），也判定不在赛。
- 仅对 $\mathcal A_{s,t}$ 内的数据计算任何统计量（percent、rank、loss、特征工程）。

### 0.3 面板化（宽表 → 长表）
构造统一长表 panel（最关键，所有 Task 共用）：
每行 = $(s,t,i)$ 且 $i\in\mathcal A_{s,t}$，字段建议最少包含：
- `season`, `week`, `pair_id`, `celebrity_name`, `ballroom_partner`, `placement`
- `judge_scores`（4 维含 NaN）、`n_judges`
- **评委聚合分**（建议用均值，避免评委人数变化影响尺度）：
  $$
  S_{i,t}=\frac{1}{|\mathcal J_{i,t}|}\sum_{j\in \mathcal J_{i,t}} s_{i,t,j},\quad \mathcal J_{i,t}=\{j:\ s_{i,t,j}\ \text{observed}\}
  $$
- **评委百分比**（只在当周 active 内归一）：
  $$
  p^J_{i,t}=\frac{S_{i,t}}{\sum_{k\in\mathcal A_{s,t}} S_{k,t}}
  $$
- **规则段标记**（用于 Task1/2/验证）：
  - $m_s=\text{rank}$ 若 $s\in\{1,2\}$；
  - $m_s=\text{percent}$ 若 $s\in\{3,\dots,27\}$；
  - $m_s=\text{rank}$ 且 $\text{save}_s=1$ 若 $s\ge 28$（题面说精确切换季未知，但“合理假设是 28”）。

### 0.4 淘汰集合 $\mathcal E_{s,t}$ 的抽取（用于 Task1 观测）
令每季长度 $T_s$ 为该季存在数据的最大周。然后对 $t<T_s$：
$$
\mathcal E_{s,t}=\mathcal A_{s,t}\setminus \mathcal A_{s,t+1}
$$
- 允许：$|\mathcal E_{s,t}|=0$（无淘汰周）、$|\mathcal E_{s,t}|>1$（多淘汰周）。
- 决赛周 $t=T_s$：不定义淘汰，改用 final placement 排名观测（见 Task1 公式）。
