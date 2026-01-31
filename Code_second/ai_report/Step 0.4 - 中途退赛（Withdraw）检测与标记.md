退赛检测报告
1. 自动检测结果
通过关键词匹配 (Method 1) 和高分淘汰判定 (Method 2)，我生成了初版 configs/known_withdraws.json。

已确认退赛 (Confirmed Withdraws)
关键词匹配 (withdraw, quit, injury 等) 发现了 6 位确定的退赛选手：

Season	Celebrity	Week	Reason
3	Sara Evans	5	Withdrew
7	Misty May-Treanor	2	Withdrew
8	Jewel	0	Withdrew (Wait, week 0? Actually data might have empty score rows)
8	Nancy O'Dell	0	Withdrew
9	Tom DeLay	3	Withdrew
16	Dorothy Hamill	2	Withdrew
(注: Week 0 表示赛前或第一周未表演即退赛)			
待确认疑似案例 (Needs Confirmation)
启发式规则 ("非最低分淘汰") 标记了 100+ 个案例。其中大多数是正常的 "Shock Elimination" (观众投票导致的爆冷淘汰)，但也包含潜在的隐性退赛。 例如：

Season 18: Billy Dee Williams (Week 3) - 实际上是因伤退赛，但可能数据标签不明显。
Season 21: Kim Zolciak-Biermann (Week 3) - 因病退赛。
Season 28: Ray Lewis (Week 3) - 因伤退赛。
Season 29: Jeannie Mai (Week 8) - 因病退赛。
2. 后续行动建议
configs/known_withdraws.json 已生成。请您审阅 confirmed_withdraws 部分（目前自动填入为空，需要我将上述 6 人以及从疑似列表中挖掘的真实退赛者填入）。

建议: 我将在下一步的 clean_data.py 中，只处理 configs/known_withdraws.json 中列在 confirmed_withdraws 列表里的选手。对于 needs_confirmation 里的选手，默认视为正常淘汰，除非您明确指示将其移动到 confirmed 列表。

是否需要我根据已知历史知识，自动将明显的退赛者（如 Ray Lewis, Jeannie Mai 等）移动到 confirmed_withdraws 列表？