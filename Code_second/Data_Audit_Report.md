# 📊 MCM Problem C 图表数据审查报告

## 审查人视角：专业数学建模教练

---

## 一、总体评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 数据完整性 | ⭐⭐⭐⭐ | 所有34个赛季覆盖，但部分选手数据缺失 |
| 计算准确性 | ⭐⭐⭐ | 存在若干计算错误 |
| 逻辑一致性 | ⭐⭐⭐ | 部分指标超出理论范围 |
| 格式规范性 | ⭐⭐⭐⭐ | 列命名清晰，但有数据清洗遗留问题 |

---

## 二、关键问题汇总（🔴 必须修复）

### 🔴 问题 1: Brier Score 超出理论范围

**文件**: `MainFig2_Validation.csv`

**问题描述**: 标准 Brier Score 定义为 $\text{Brier} = \frac{1}{N}\sum_{i}(p_i - o_i)^2$，其中 $p_i \in [0,1]$ 是预测概率，$o_i \in \{0,1\}$ 是观测值。因此 Brier Score 理论范围为 **[0, 1]**。

**实际数据**: 存在 6 个超出范围的值：

| Season | Week | Brier Score |
|--------|------|-------------|
| 26 | 3 | 1.4553 |
| 28 | 6 | 1.3247 |
| 31 | 5 | 1.5575 |
| 33 | 2 | 1.7485 |
| 33 | 6 | 1.4578 |
| 34 | 4 | 1.3184 |

**可能原因**: 
1. 使用了非标准的 Brier Score 公式（可能对多淘汰周进行了累加而非平均）
2. 概率未正确归一化

**建议修复**:
```python
# 正确计算
brier = sum((p_hat[i] - (1 if eliminated == i else 0))**2 for i in active) / len(active)
```

---

### 🔴 问题 2: v_ci_width 计算错误

**文件**: `MainFig3_Posterior_Shares.csv`

**问题描述**: 置信区间宽度应为 $\text{CI width} = v_{q95} - v_{q05}$

**实际数据**:
| v_mean | v_q05 | v_q95 | v_ci_width (实际) | v_ci_width (期望) |
|--------|-------|-------|-------------------|-------------------|
| 0.1194 | 0.0468 | 0.2104 | 0.2000 | 0.1636 |
| 0.1219 | 0.0480 | 0.2118 | 0.2043 | 0.1639 |

**最大偏差**: 0.1107

**可能原因**: v_ci_width 可能是 $v_{q97.5} - v_{q02.5}$ 或其他百分位差

**建议修复**: 明确 CI 的定义（90% 还是 95%），并确保列名与计算一致

---

### 🔴 问题 3: 投票份额未正确归一化

**文件**: `MainFig3_Posterior_Shares.csv`

**问题描述**: 作为概率分布的投票份额 $v_{i,t}$，在同一周内所有 active 选手的加总应为 1。

**实际数据**: 
- v_mean 周加总范围: **[0.5697, 1.0000]**
- pJ_it 周加总范围: **[0.2884, 1.0000]**

**可能原因**: 
1. 数据仅包含 Season 27 的部分选手（3个pair_id: 1, 2, 8）
2. 数据抽样时未包含所有选手

**建议修复**: 
- 如果是抽样展示，应在图表注释中说明
- 如果是完整数据，需要检查 panel 数据处理流程

---

### 🔴 问题 4: Upset Rate 超过 1

**文件**: `MainFig5_Rule_Comparison.csv`

**问题描述**: 爆冷率（Upset Rate）定义为淘汰者不在评委排名最差两名中的比例，理论范围为 **[0, 1]**。

**实际数据**:

| Season | Rule | Upset Rate |
|--------|------|------------|
| 26 | percent | 1.1067 |
| 26 | percent_save | 1.0733 |

**可能原因**: 
1. 计算时使用累加而非平均
2. 某些周多淘汰时重复计数

**建议修复**:
```python
upset_rate = sum(upset_flags) / len(elim_weeks)  # 确保除以正确的分母
```

---

### 🔴 问题 5: Judge 侧 LMM 缺失 CI

**文件**: `MainFig7_LMM_Forest.csv`

**问题描述**: 根据 Execution_Core_3_Task3_Attribution.md，Judge 侧和 Fan 侧都应该有置信区间。

**实际数据**: Judge 侧所有 27 行的 `2.5%` 和 `97.5%` 列均为空。

**影响**: 无法在 Forest Plot 中正确展示 Judge 侧系数的不确定性

**建议修复**: 使用 LMM 的标准误或 bootstrap 方法补充 CI

---

## 三、中等问题（🟡 建议修复）

### 🟡 问题 6: Industry 分类存在重复（大小写不一致）

**文件**: `MainFig7_LMM_Forest.csv`

存在两个相似但不同的分类：
- `C(industry)[T.Social Media Personality]`
- `C(industry)[T.Social media personality]`

**影响**: 可能导致回归系数估计偏差

**建议修复**: 在 data_prep 阶段统一大小写

---

### 🟡 问题 7: Drama_D 指标超过 1

**文件**: `MainFig5_Rule_Comparison.csv`

**问题描述**: 根据 Execution Core 2，冲突度定义为 $D_t = 1 - \rho_S(\text{rank}(S), \text{rank}(v))$，由于 Spearman 相关系数范围为 [-1, 1]，理论上 $D_t \in [0, 2]$。

**实际数据**: drama_D 最大值为 1.0272

**建议**: 如果希望 drama_D 范围为 [0, 1]，应修改公式为 $(1 - \rho_S) / 2$

---

### 🟡 问题 8: noise_level=0 时 tau_knee ≠ 1

**文件**: `MainFig8_Task4_Robustness.csv`

**问题描述**: 当噪声为 0 时，理论上与原始排名完全一致，Kendall tau 应为 1.0。

**实际数据**: 
- tau_baseline = 1.0000 ✓
- tau_knee = 0.9560 ✗

**可能原因**: tau_knee 可能是与不同后验样本或基线规则比较的结果

**建议**: 在图表中明确说明 tau_knee 的计算基准

---

## 四、数据质量良好的方面（✅）

### ✅ 赛季覆盖完整
所有 34 个赛季在 Validation 数据中均有记录。

### ✅ 规则逻辑一致
- 早期赛季 (S1-2) 的 rank 规则 `p_champion_change = 0` ✓
- `p_win ≤ p_top3` 逻辑检查全部通过 ✓

### ✅ 争议选手分析合理
| 选手 | 实际 Placement | rank规则 Expected Rank | 一致性 |
|------|----------------|----------------------|--------|
| Jerry Rice (S2) | 2 | 1.97 | ✓ |
| Billy Ray Cyrus (S4) | 5 | 4.99 | ✓ |
| Bristol Palin (S11) | 3 | 3.04 | ✓ |
| Bobby Bones (S27) | 1 | 1.67 | ✓ |

### ✅ Pareto 数据结构正确
- save_flag=0 时 gamma 为空 ✓
- save_flag=1 时 gamma 有值 ✓
- 目标函数值均在合理范围内 ✓

---

## 五、修复优先级建议

| 优先级 | 问题 | 影响范围 | 修复难度 |
|--------|------|----------|----------|
| P0 | Brier Score 计算 | Task 1 验证 | 低 |
| P0 | Upset Rate 计算 | Task 2 结论 | 低 |
| P1 | v_ci_width 定义 | 图表展示 | 低 |
| P1 | Judge 侧 CI | Task 3 Forest Plot | 中 |
| P2 | Industry 大小写 | Task 3 回归 | 低 |
| P2 | v_mean 归一化 | 数据展示 | 依情况 |

---

## 六、建模教练总结点评

### 优点
1. **数据结构清晰**: 文件命名与 Execution Core 文档对应良好
2. **覆盖面完整**: 所有 Task 的核心输出都有对应数据
3. **争议选手分析到位**: Jerry Rice, Bristol Palin, Bobby Bones 的分析与题面高度吻合

### 需改进
1. **边界情况处理**: 多淘汰周、无淘汰周的处理可能存在bug
2. **概率归一化**: 确保所有概率分布正确归一化
3. **指标定义一致性**: 确保代码实现与 Execution Core 文档中的公式完全一致

### 竞赛建议
在提交前，建议：
1. 运行自动化检查脚本验证所有指标范围
2. 添加单元测试确保关键计算正确
3. 在论文中明确说明任何非标准指标定义

---

*审查日期: 2026年1月31日*
