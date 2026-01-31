桥接函数 (Bridge) 已实现
它是连接各个 Task 的安全阀，确保数据流转正确。 已在 src/utils/bridge.py 中实现了以下函数，并通过了单元测试 (tests/test_bridge.py)：

renorm_votes(v_vec, A):
确保 Task1 反演出的全量投票份额，在 Task2 回放不同 Active 集合时能正确归一化。
包含非负检查和总和为1的强断言。
renorm_judge(S_vec, A):
类似地处理评委分数，防止 NaN 或 0分导致计算错误。
combine_score(rule, S_sub, ...):
统一了两种赛制 (Rank/Percent) 的合成分计算接口。
封装了核心的 soft_rank 调用，保证评估标准一致。
测试结果:

所有的随机性测试和一致性测试均通过。
异常输入（NaN, 负值）能正确抛出错误。
下一步: 现在核心数学工具 (core_functions) 和数据桥接工具 (bridge) 都已就绪。 我们可以正式编写 Observed Elimination Model —— 即 Task 1 的核心概率模型类。 是否继续实现 VoteShareModel 类？