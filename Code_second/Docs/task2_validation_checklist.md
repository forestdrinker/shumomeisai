# Task 2 Validation Checklist: Rule Comparison & Fairness Analysis

**Topic**: Comparative analysis of elimination rules ("Rank" vs "Percent" vs "Save" variations) to evaluate their impact on fairness, consistency, and entertainment value ("Drama").

**Objective**: Provide a mathematical reviewer with the raw simulation data to assess whether alternate rules would have produced "fairer" or "more predictable" outcomes compared to history.

## 1. Core Evaluation Metrics
**File**: `Results/task2_metrics.csv` (Preview Below)

This tables aggregates simulation results ($N=100$ runs per season/rule) to compare rule performance.

*   **rho_F / rho_J**: Spearman correlation with final Vote Share ($v$) and Judge Scores ($S$). Higher = More consistent with raw popularity/talent.
*   **upset_rate**: Probability that a contestant with better "combined score" than the survivor is eliminated.
*   **drama_D**: "Drama" metric (1 - Margin of Safety). Higher = Closer calls/More suspense.
*   **p_champion_change**: Probability that the historical winner *loses* under this rule.

```csv
season,rule,p_champion_change,p_top3_change,rho_F,rho_J,drama_D,drama_D_late,upset_rate,suspense_H,suspense_H_late
1,rank,0.0,0.0,0.8369999999999997,0.585,0.8908518999257499,0.2833333333333333,0.286,0.9145269492532891,0.9145269492532891
1,rank_save,0.27,0.73,0.6379999999999999,0.7749999999999999,0.8324938335252551,0.3566666666666667,0.068,0.8079558763817053,0.8079558763817053
1,percent,0.15,0.62,0.7779999999999998,0.49000000000000005,0.8526625759888868,0.25166666666666665,0.33399999999999996,0.9392017001033266,0.9392017001033266
1,percent_save,0.15,0.61,0.8589999999999999,0.7109999999999999,0.9000594921263111,0.3,0.24,0.8285924826628159,0.8285924826628159
2,rank,0.0,0.0,0.8618333333333333,0.7696666666666667,0.7794626984876729,0.8551294866407178,0.13142857142857142,0.9667422771342175,0.8013491407770278
2,rank_save,0.12,0.69,0.8333333333333331,0.8756666666666668,0.7921275190029096,0.809662053365946,0.038571428571428576,0.9424403143559497,0.8291678064108478
2,percent,0.46,0.39,0.9043333333333333,0.8118333333333334,0.7913389047393086,0.8336589688327298,0.13857142857142857,1.129181680308502,1.277930271517699
2,percent_save,0.45,0.23,0.9008333333333333,0.866,0.8438745618122183,0.9309922369016698,0.06142857142857143,0.9571709828144424,0.9606052101235217
3,rank,0.0,0.0,0.9096969696969698,0.919151515151515,0.5826096320642065,0.5185923600231369,0.1611111111111111,1.0221277434898803,0.9554107846256917
3,rank_save,0.29,0.36,0.8848484848484844,0.9167272727272727,0.5384872726819577,0.4698752524757789,0.05444444444444444,0.6575370231949352,0.6406602554020395
3,percent,0.26,0.66,0.957090909090909,0.9202424242424242,0.6771078668691349,0.7069445742851994,0.2911111111111111,1.0996059850819901,1.162737337564246
3,percent_save,0.15,0.71,0.9425454545454545,0.9367272727272726,0.6667279948253672,0.6974114257450198,0.20444444444444446,0.8561549840217725,1.1196095283141456
4,rank,0.0,0.0,0.905,0.8723333333333332,0.44760355607021224,0.385763224809629,0.060000000000000005,0.9126164064751959,1.0695322206991618
4,rank_save,0.31,0.16,0.9153333333333334,0.9143333333333333,0.45109675091618856,0.40912336700708873,0.003333333333333333,0.49900366844226046,0.7319561269923854
4,percent,0.27,0.37,0.9408333333333334,0.9025000000000002,0.42314641115613505,0.37733095920765,0.1611111111111111,1.10618683078483,1.1453804703598773
4,percent_save,0.25,0.18,0.936,0.9206666666666666,0.46119471109885396,0.3669795950231869,0.07333333333333333,0.8015012191654752,0.7271960717209969
```

## 2. Injustice Analysis (Counterfactuals)
**File**: `Results/controversy_cases.csv` (Preview Below)

This file tracks specific contestants in specific seasons to see how their fate changes under different rules.

*   **p_win**: Probability of winning the season under the given rule.
*   **expected_rank**: Average rank achieved in simulations (Lower is better).
*   **expected_survival_weeks**: How long they survive on average.

```csv
season,rule,pair_id,celebrity_name,p_win,p_top3,expected_rank,expected_survival_weeks
1,percent,1,Evander Holyfield,0.0,0.73,3.32,4.41
1,percent_save,1,Evander Holyfield,0.0,0.44,3.68,3.76
1,rank,1,Evander Holyfield,0.0,0.67,3.4,4.27
1,rank_save,1,Evander Holyfield,0.0,0.25,4.37,2.88
1,percent,2,Joey McIntyre,0.0,0.6,3.04,4.56
1,percent_save,2,Joey McIntyre,0.0,0.61,3.36,4.25
1,rank,2,Joey McIntyre,0.0,0.55,3.37,4.18
1,rank_save,2,Joey McIntyre,0.0,0.92,2.93,4.99
1,percent,3,John O'Hurley,0.07,0.67,2.74,4.86
1,percent_save,3,John O'Hurley,0.09,1.0,1.91,6.0
1,rank,3,John O'Hurley,0.08,0.81,2.3,5.43
1,rank_save,3,John O'Hurley,0.19,1.0,1.81,6.0
1,percent,4,Kelly Monaco,0.93,1.0,1.08,5.99
1,percent_save,4,Kelly Monaco,0.91,0.95,1.24,5.8
1,rank,4,Kelly Monaco,0.92,0.97,1.14,5.91
1,rank_save,4,Kelly Monaco,0.81,0.83,1.63,5.39
1,percent,5,Trista Sutter,0.0,0.0,4.82,2.18
1,percent_save,5,Trista Sutter,0.0,0.0,4.81,2.19
1,rank,5,Trista Sutter,0.0,0.0,4.79,2.21
```

## 3. Mathematical Evaluation Questions

For the reviewing agent/judge:

1.  **Metric Stability**: Do the metrics (rho_F, rho_J) show consistent trends across seasons, or is there high variance?
2.  **Trade-off Verification**: Does the "Save" mechanism (`rank_save`) consistently lower the `upset_rate` compared to the standard `rank` rule?
3.  **Impact Quantification**: Is the `p_champion_change` significant enough (>0.0) to justify the argument that "Rules Matter"?
4.  **Pareto Efficiency**: Can we identify a rule that maximizes `rho_J` (Judge agreement) without sacrificing `rho_F` (Fan agreement)?
