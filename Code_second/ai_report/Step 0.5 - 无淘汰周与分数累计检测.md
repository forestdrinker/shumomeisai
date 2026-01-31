无淘汰周与分数累计检测报告
1. 检测到的无淘汰周 (No Elimination Weeks)
通过对比相邻周活跃选手人数，我发现了多个赛季存在无淘汰周，常见模式如下：

Week 1: 绝大多数赛季 (Season 16-34 等) 第一周通常不淘汰人。
Mid-Season: 部分赛季在 Week 3, 5, 6 等存在非淘汰周 (可能是迪士尼之夜等特别主题)。
Season 32: 甚至在 Week 10 (半决赛前夕) 也没有淘汰。
2. 分数累计检测 (Cumulative Scores)
我检查了无淘汰周的下一周 (Week t+1) 的平均分是否显著高于本周 (Week t)，以判断数据集中分数是否已经包含了上一周的累计分。

结果:

绝大多数情况: Ratio (Week t+1 / Week t) 在 1.0 - 1.2 之间。这意味着分数是单周独立的（例如满分30/40分），并没有将前一周的分数直接加总到这一周的数据列中。
唯一异常: Season 34 Week 2 (Ratio 1.70, 19.86 vs 11.71)。这可能反映了分制的改变（如从双评委变四评委？）或者单纯是第一周分特别低。
3. 结论与建议
数据性质: dataset 中的 weekX_judgeY_score 是当周独立分数，不是累计分。
Task 1 建模: 在处理 Task 1 (票份额反演) 时，对于无淘汰周，我们需要手动实现**“分数携带” (Score Carryover)** 逻辑。即：
Effective_Score_Week_2 = Score_Week_1 + Score_Week_2 而不是指望数据里已经加好了。

下一步: 所有预处理逻辑验证完毕。我将开始编写核心的预处理脚本 src/preprocessing/clean_data.py 和 src/preprocessing/build_bundle.py，将原始 CSV 转换为干净的 Parquet 面板数据。是否开始？

