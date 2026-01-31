# 📊 数据源深度对比报告：官方数据 vs 队友处理版 (Data Comparison Report)

**日期:** 2026-01-31  
**版本:** 2.0 (Deep Dive)  
**对象:** `Data/2026_MCM_Problem_C_Data.csv` (Official) vs `processed/panel.csv` (Teammate)

---

## 1. 结构与维度对比 (Structure & Dimensions)

最本质的区别在于数据组织形式：官方数据是**宽表 (Wide)**，队友处理版是**长表/面板 (Long/Panel)**。

| 维度 (Dimension)  | 官方原始数据 (Official Data)                     | 队友处理版 (Teammate Processed)            | 我们的处理必要性 (Why?)                                      |
| :---------------- | :----------------------------------------------- | :----------------------------------------- | :----------------------------------------------------------- |
| **行定义 (Row)**  | **每对选手一行** (One Row per Couple)            | **每人每周一行** (One Row per Couple-Week) | 时间序列模型 (Task 1/2/4) 必须基于“周”粒度进行状态转移分析。 |
| **列数量 (Cols)** | **51+ 列** (week1_judge1... week10_judge4)       | **23 列** (season, week, score, status...) | 原始宽表难以直接进行统计回归；长表更适合 Pandas/SQL 处理。   |
| **分数存储**      | 展开为数十列 (`week1_judge1`, `week2_judge1`...) | 收敛为标准列 (`judge_score_1`... `S_it`)   | 极大地简化了代码逻辑，支持任意长度的赛季。                   |

---

## 2. 关键内容清洗与衍生 (Cleaning & Derivation)

为了将“人类可读”的官方表格转换为“机器可算”的面板数据，我们执行了以下核心逻辑：

### 2.1 状态掩码生成 (Status Masking - Critical!)
*   **官方源**: 仅有一列文本 `results` (例如 "Eliminated Week 4", "1st Place", "Withdrew").
*   **队友版**: 解析该文本，生成了以下关键布尔列：
    *   `is_active`: 标记该周选手是否仍在比赛中 (True/False)。这是所有后续 Brier Score 计算的分母基础。
    *   `elim_week_by_score`: 提取出的具体被淘汰周数 (例如 `4`)。
    *   `placement`: 最终排名的数值化 (1, 2, 3...)。
*   **价值**: 没有这个解析，模型就不知道谁在第几周“活着”，无法计算任何生存概率。

### 2.2 评委分数标准化 (Score Normalization)
*   **官方源**: 包含 `N/A` 文本，且某些周可能是 3 个评委，某些是 4 个。
*   **处理逻辑**:
    1.  将 `N/A` 转换为 `NaN` 或 `0` (视缺赛情况而定)。
    2.  **S_it (总分)**: 队友版计算了当周总分 (或加权均分)，统一了 3/4 评委的差异。
    3.  我们在此基础上进一步计算了 `pJ_it` (Share)，解决了不同满分制的可比性问题。

### 2.3 文本与分类清洗 (Text Hygiene)
*   **Industry**: 官方数据包含 `Beauty Pagent` (拼写错误) 和不一致的大小写。
    *   **处理**: 全局统一为 Title Case，修正拼写 (`Pageant`)，归并同类项 (`Con artist` -> `Personality`)。
*   **Homestate/Country**: 官方数据存在缺失 (`,,England`) 或格式不一。
    *   **处理**: 进行了基础的填充和对齐，便于 Task 3 的 LMM 归因分析。

---

## 3. 最终数据清单 (Final Data Dictionary)

我们用于生成最终图表的数据集 (`processed/panel.csv` + 我们的补丁) 包含以下黄金字段：

| 字段名                      | 来源     | 描述                                        |
| :-------------------------- | :------- | :------------------------------------------ |
| `season`, `week`, `pair_id` | **衍生** | 主键索引 (Primary Key)。                    |
| `S_it`                      | **计算** | 选手当周评委总分 (部分归一化)。             |
| `is_active`                 | **解析** | 核心掩码，用于过滤非活跃样本。              |
| `industry`                  | **清洗** | 修正后的行业分类，用于固定效应模型。        |
| `pJ_it`                     | **补全** | (代码运行时补全) 评委打分份额，核心特征列。 |

---

## 4. 结论 (Conclusion)

官方数据提供了原始的**“比赛记录”** (Record)，而队友处理版（加上我们的补丁）将其转化为了**“分析就绪”** (Analytics-Ready) 的数据集。

这个转换过程（Pivoting + Parsing + Cleaning）是本次建模工作成功的基石，保证了模型不会被 `N/A` 或错误的文本格式报错打断。
