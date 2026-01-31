# 项目产出客观汇总报告

本文件对 Task 1 至 Task 4 的输出文件、计算指标及实现特征进行客观汇总。

## Task 1: Vote Inference (投票推断)

### 输出文件
- **路径**: `Results/task1_metrics.csv`
- **路径**: `Results/validation_results/` (目录)

### 计算指标 (metrics.csv)
该文件包含以下字段：
- **标识符**: `season`, `week`
- **状态量**: `n_active` (当周活跃选手), `n_elim` (当周淘汰数)
- **推断性能指标**:
  - `coverage_90`: 90%置信区间覆盖率 (0 或 1)
  - `accuracy`: 准确率
  - `top2_acc`: 前两名准确率
  - `brier`: Brier得分
  - `post_var_v_mean`: 后验均值方差
  - `avg_ci_width`: 平均置信区间宽度

### 实现特征
- 实现了针对 DWTS 投票数据的概率模型推断。
- 包含验证逻辑，计算了上述准确性和不确定性指标。

---

## Task 2: Rule Comparison (规则比较)

### 输出文件
- **路径**: `Results/task2_metrics.csv`
- **路径**: `Results/replay_results/` (目录，包含各赛季回放NPZ文件)
- **路径**: `Results/controversy_cases.csv` (已验证存在)

### 计算指标
**1. 规则评估指标 (`task2_metrics.csv`)**:
该文件包含以下指标：
- **概率指标**: `p_champion_change`, `p_top3_change`
- **相关性指标**: `rho_F` (与观众票数相关性), `rho_J` (与评委分数相关性)
- **赛制动态指标**: `drama_D` (戏剧性/冲突度), `upset_rate` (爆冷率), `suspense_H` (悬念熵)

**2. 争议人物指标 (`controversy_cases.csv`)**:
该文件包含分赛季、规则、选手的详细统计：
- **标识符**: `season`, `rule`, `pair_id`, `celebrity_name`
- **预测指标**: `p_win` (夺冠概率), `p_top3` (前三概率), `expected_rank` (期望排名), `expected_survival_weeks` (期望存活周数)

### 实现特征
- **回放场景**: 包含 `rank`, `percent`, `rank_save`, `percent_save` 四种场景。
- **样本覆盖**: 覆盖 S1-S34 赛季。
- **数据源**: 使用 `posterior_samples` 和 `elim_events.json`。
- **逻辑细节**:
  - Active集合重归一化: 当和为0时采用均匀分布。
  - Rank计算: 使用硬排名 (hard rank) 叠加。
  - Judges' Save: 实现了确定性逻辑 (淘汰评委分较低者)，未应用概率参数 $\gamma$。
  - 爆冷机制: 未实现混合模型参数 $\rho$。
  - 相关性计算: 使用 Kendall $\tau$ 相关系数。

---

## Task 3: Attribution Analysis (归因分析)

### 输出文件
- **数据集**: `Results/task3_data/task3_weekly_dataset.parquet`
- **LMM 分析结果**:
  - `Results/task3_analysis/task3_lmm_judge_coeffs.csv`
  - `Results/task3_analysis/task3_lmm_judge_partner_effects.csv`
  - `Results/task3_analysis/task3_lmm_fan_coeffs_aggregated.csv`
  - `Results/task3_analysis/task3_lmm_fan_partner_effects_aggregated.csv`
- **SHAP 分析结果**:
  - `Results/task3_analysis/task3_shap_ci_fan.csv`
  - `Results/task3_analysis/task3_shap_ci_judge.csv`
  - `Results/task3_analysis/task3_gbdt_cv_metrics.json`

### 内容特征
- **数据处理**: 包含周级样本构造、评委分数归一化、年龄标准化及滚动特征 (`rolling_avg_pJ` 等)。
- **LMM (线性混合模型)**:
  - 包含交叉随机效应 (Partner + Season)。
  - 粉丝侧模型使用了 30 个后验样本 (Posterior Draws) 的外循环计算。
  - CI (置信区间) 包含了跨样本的分布信息 (2.5% / 97.5% 分位数)。
  - 舞伴名称已进行格式清洗。
- **GBDT + SHAP**:
  - 使用 GroupKFold (按赛季分组) 进行交叉验证。
  - SHAP 值计算采用了 Bootstrap 重采样，且包含后验不确定性。
- **缺失项**: 未包含网络特征 (Network Features)。

---

## Task 4: System Design (系统设计/优化)

### 输出文件
- **评估数据**: `Results/task4_theta_evaluations.parquet` (包含均值和标准差列)
- **Pareto 前沿**: `Results/task4_pareto_front.csv`
- **推荐结果**: `Results/task4_recommendations.json`
- **可视化**: `Results/task4_tradeoff_plot.png`

### 优化目标
定义了四个优化目标：
1. **Obj_F**: 观众一致性 (Fan Alignment)
2. **Obj_J**: 评委一致性 (Judge Alignment)
3. **Obj_D**: 戏剧性 (Drama) - 计算前两名分差的补数
4. **Obj_R**: 鲁棒性 (Robustness) - 在主要偏差下的稳定性

### 实现特征
- **模拟器**: `SeasonSimulatorFast` 类实现了全季快速回放。
- **优化算法**: 采用 MOBO (多目标贝叶斯优化)，使用 Optuna 框架。
- **搜索策略**: 两阶段搜索 (Stage 1 LHS/Random + Stage 2 TPE/NSGAII)。
- **鲁棒性**: 包含了投票扰动测试 ($K=5, \kappa=0.5$)。
- **推荐方案**: 输出了以下三类参数配置：
  - **Balanced (Knee Point)**: 均衡方案，开启 Save 机制。
  - **Fan Favorite**: 侧重观众偏好，后期介入，强力压缩差异。
  - **Judge Favorite**: 侧重评委偏好，早期介入，放大差异，关闭 Save 机制 (参数显示为 `null`)。
