数据探索报告 (Data Exploration Report)
我已成功加载并分析了 data/raw/2026_MCM_Problem_C_Data.csv 数据集。以下是详细的探索结果：

1. 数据基本信息
行数: 421 (选手参赛记录)
列数: 53
主要字段:
celebrity_name, ballroom_partner: 选手与舞伴信息
season: 赛季 (1-34)
results, placement: 最终排位与结果
week1_judge1_score ... week11_judge4_score: 详细的每周裁判评分 (宽表格式)
2. 赛季统计摘要
各赛季的规模（选手）、持续时长（周数）及决赛人数如下：

Season	Contestants	Weeks	Finalists
1	6	6	2
15	13	10	3
26	10	4	3 (Athletes Special)
27	13	9	4
32	14	11	5
(注：Season 26 为全明星/运动员特别赛季，赛程较短)			
3. results 字段分析
唯一值列表:

'1st Place', '2nd Place', '3rd Place', '4th Place', '5th Place'
'Eliminated Week 1' ... 'Eliminated Week 11'
'Withdrew'
4. 异常值与数据质量检查
退赛关键词: 最初检查 "withdraw/quit/injury" 未匹配到行，但人工检查发现 Withdrew 是明确的类别之一。需在后续处理中将 Withdrew 视为特殊的淘汰事件。
排名矛盾:
Placement=1 且 Result="Eliminated" 的情况：0 例 (数据一致性良好)。
5. 随机抽样详情 (Seasons 1, 15, 27)
Season 1: Kelly Monaco
Result: 1st Place (冠军)
Placement: 1
Scores:
Week 1: [5.0, 4.0, 4.0]
... (分数逐周上升)
Week 6: [9.0, 10.0, 10.0]
Season 15: Drew Lachey (全明星赛季)
Result: Eliminated Week 3
Placement: 11
Scores:
Week 1-3: 分数在 7.0 - 8.0 之间
Week 4+: 无分数 (0.0 或 NaN)
Season 27: Bobby Bones
Result: 1st Place
Placement: 1
Scores:
虽然早期分数波动 (Week 4 有 6.0 分)，但后期稳定在 8.0-9.0 分区间。
接下来的计划: 下一步我将开始 Task 1: Preprocessing，主要工作包括：

将宽表 (Wide Format) 转换为长表 (Long Format) 以便模型处理。
清洗 Withdrew 状态，将其与正常淘汰区分。
处理缺失的裁判分数（如部分赛季只有3位裁判）。