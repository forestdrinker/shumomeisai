# Task 3 验证清单：归因与舞伴效应分析

**主题**: 使用线性混合模型 (LMM) 和 SHAP 值量化分析名人特征、职业及舞伴对观众投票的影响。

**目标**: 提供统计模型输出的原始数据，供评审员评估各变量的统计显著性及效应大小。

## 1. 线性混合模型 (LMM) 结果
**目标变量**: 观众投票份额 ($v_{i,t}$)
**方法**: 包含赛季 (Season) 和舞伴 (Partner) 交叉随机效应的 LMM 模型。

### A. 固定效应 (人口统计学与行业)
**文件**: `Results/task3_analysis/task3_lmm_fan_coeffs_aggregated.csv` (预览)

下表展示了各特征对投票份额的效应估计。
*   **Intercept**: 基准截距。
*   **C(industry)**: 相对于参考类别 (Activist/Actor) 的行业系数。
*   **week_norm**: 选手留存周数（时间趋势）的系数。

```csv
,mean,2.5%,97.5%
Intercept,0.059,0.043,0.075
C(industry)[T.Astronaut],0.064,-0.023,0.153
C(industry)[T.Athlete],0.018,-0.004,0.056
C(industry)[T.Musician],0.073,-0.122,0.296
C(industry)[T.Social media personality],0.198,-0.047,0.442
age_z,-0.019,-0.031,-0.010
week_norm,0.168,0.159,0.176
partner Var,0.379,0.173,0.789
season Var,0.108,0.043,0.207
```

### B. 随机效应 (舞伴效应)
**文件**: `Results/task3_analysis/task3_lmm_fan_partner_effects_aggregated.csv` (预览)

下表列出了控制名人自身特征后，专业舞伴的随机效应估计值 (BLUPs)。
*   **mean**: 对投票份额的平均边际影响估计。
*   **support_n**: 后验样本支撑量。

```csv
,mean,2.5%,97.5%,support_n
Jonathan Roberts,0.240,0.119,0.466,30.0
Daniella Karagach,0.104,0.002,0.181,30.0
Emma Slater/Kaitlyn Bristowe (week 9),0.083,0.007,0.255,30.0
Derek Hough,0.082,0.025,0.135,30.0
Jenna Johnson,0.061,0.005,0.128,30.0
Alan Bersten,0.042,-0.035,0.123,30.0
Kym Johnson,0.040,-0.004,0.093,30.0
...
Nick Kosovich,-0.062,-0.088,-0.029,30.0
Louis van Amstel,-0.064,-0.095,-0.035,30.0
```

## 2. SHAP 可解释性分析
**文件**: `Results/task3_analysis/task3_shap_ci_fan.csv` (预览)

GBDT 模型预测投票份额时的特征重要性（绝对值均值）。

```csv
feature,mean_shap,q2.5,q97.5
rolling_avg_pJ,0.042,0.034,0.053
age,0.021,0.009,0.037
partner_enc,0.018,0.009,0.031
industry_enc,0.008,0.002,0.015
rolling_std_pJ,0.006,0.001,0.011
week,0.001,0.000,0.003
```

## 3. 数学评估检查项

供数学模型评审员使用的客观检查点：

1.  **方差分量 (Variance Components)**:
    *   记录 `partner Var` (舞伴方差) 与 `season Var` (赛季方差) 的数值。
    *   计算二者的比率。

2.  **统计显著性 (Statistical Significance)**:
    *   检查 `age_z` (年龄标准化值) 的 95% 置信区间 (`2.5%` 至 `97.5%`) 是否包含 0。
    *   检查 `C(industry)[T.Social media personality]` 的 95% 置信区间是否包含 0。

3.  **效应值分布 (Effect Size Distribution)**:
    *   记录随机效应均值 (`mean`) 最高的 3 位舞伴姓名。
    *   记录随机效应均值最低的 3 位舞伴姓名。

4.  **模型一致性 (Model Consistency)**:
    *   对比 LMM 中 `week` 相关系数的符号与 SHAP 中 `week` 的重要性。
    *   对比 LMM 中 `age` 相关系数的符号与 SHAP 中 `age` 的重要性。
