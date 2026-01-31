核心数学函数已实现
已在 src/utils/core_functions.py 中实现了核心算法，并通过了单元测试 (tests/test_core_functions.py)。

实现的函数:

soft_rank(x, kappa): 可微软排名，支持反向传播梯度。
compute_kappa(scores): 基于分数方差自适应计算温度参数。
compute_badness(mode, S, v, params): 计算淘汰危险度指数，支持 Rank 和 Percent 两种赛制模式。
验证结果:

所有的单元测试（Rank计算准确性、Kappa自适应性、Badness单调性）均已通过。
下一步: 可以开始实现 VoteShareModel 类 (src/task1_inference/model.py)，将这些核心函数组合进完整的概率模型中。 是否继续？