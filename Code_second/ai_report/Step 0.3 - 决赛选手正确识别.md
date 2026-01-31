决赛选手识别验证报告
1. 验证结果概述
我对比了两种识别 Finalists 的方法：

基于分数 (Score): 识别在最后一轮（Max Week）仍有分数的选手。
基于排名 (Placement): 选取与“方法1”相同数量的 Top K 排名选手。
结论: 所有赛季 (Season 1 - 34) 的两种方法结果完全一致 (Consistent)。这意味着没有出现“有名次但最后一周没分”或“最后一周有分但名次极其靠后”的异常情况。

2. 详细发现：Results 字段的"矛盾"
虽然方法一致，但通过您的第三点要求，我发现了一些“语义上的矛盾”。部分选手虽然参加了最后一周的比赛 (是事实上的 Finalist)，但 results 字段标记为 Eliminated（通常是决赛当晚被淘汰的第4名）。

Results 标记为 "Eliminated" 但归类为 Finalist 的选手列表 (Placement=4):

Season	Star	Placement	Results	解释
16	Alexandra Raisman	4	Eliminated Week 10	决赛夜淘汰
17	Bill Engvall	4	Eliminated Week 11	决赛夜淘汰
18	James Maslow	4	Eliminated Week 10	决赛夜淘汰
19	Bethany Mota	4	Eliminated Week 11	决赛夜淘汰
21	Carlos PenaVega	4	Eliminated Week 11	决赛夜淘汰
23	Jana Kramer	4	Eliminated Week 11	决赛夜淘汰
25	Drew Scott	4	Eliminated Week 10	决赛夜淘汰
3. 处理建议
在后续建模中，建议采纳您的核心原则：不依赖 results 字段的文本描述。

对于 identify_finalists 函数，我们将使用 Method 2 (Last Week Score) 作为主要逻辑，因为它能准确捕获所有“打满全场”的选手。
上述 7 位选手将被视为 Finalists 参与 Task 1 的全部周次反演，这对模型训练是有益的（提供了完整的时间序列）。
下一步: 我将把这个逻辑固化到 src/preprocessing/build_bundle.py 中。是否继续？

