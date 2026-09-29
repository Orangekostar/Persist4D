# 原提示词逐项验收（完成）

依据：`docs/0929/ReScene_Code_First_Audit/ReScene_Code_First_Audit.md`。
主代理已逐项核对原文、最终 diff、真实产物和验证输出；未将局部测试通过等同于全任务完成。

| 要求 | 当前证据 | 状态 |
|---|---|---|
| 实际仓库/官方 evaluator 部分 void 反例 | BASELINE_PROBE.json；原始探针在修改前执行 | 已核实 |
| 按 role/H 量化真实缺口，不能把全部 unknown 当受影响 | RELABEL_AUDIT.json、ORIGINAL_Q_SCORE_CROSSTAB.json | 已核实 |
| 检查原锁定 Q 头是否给缺口候选升分 | ORIGINAL_Q_SCORE_CROSSTAB.json，包含检查点 SHA256 | 已核实，不等同 AP 归因 |
| 分离几何质量、阈值资格、未知/歧义和全阈值不计分 | short_module_data.py；相关回归覆盖 0、0.6、1，出现/消失及分数/顺序不变性 | 已核实 |
| Q1/Q2 并集标量政策、Q3 矩阵掩码 | IMPLEMENTATION_PLAN.md、short_module_training.py、掩码梯度测试 | 已核实 |
| 修复 Q 时冻结 M；先量化 M 连带影响 | 481 份缓存逐项比较，原 M 分配差异为零；TRAIN 扩展分配变化 62 项 | 已核实 |
| 拆分预测与标签身份、直接依赖恢复检查、防止 publish-only 误恢复 | relabel 无 R1 前向；新标签 namespace；当前 checkout 恢复反例测试；最终 diff | 已核实 |
| 评分缓存覆盖 apply_module、head、materializer、官方指标完整源码/版本、spec、输入/权重 | short_module_eval_identity.py、short_module_identity.py、输入内容校验及失效测试；36 份结果当前身份一致 | 已核实 |
| 原父模型、输入、初始化、抽样、1500 步重训 Q；同输入前后对照 | Q_TRAINING_STATUS.json、Q_MATCHED_COMPARISON.json；实际抽样文件和初始参数身份相同；检查点 SHA 全部通过 | 已核实 |
| 原锁定头无训练增强核验，同扫描/顺序/标签、独立 B0 | NOAUG_EXPORT.json、NOAUG_INPUT_PARITY.json、NOAUG_FIXED_HEADS.json；94 输入、14 份完整结果；六头 SHA 与原清单相同 | 已核实 |
| M 受影响时仅做必要 M0/M1，共享分配和原配方 | M_CONTROL_STATUS.json、M_TRAINING_STATUS.json、M_MATCHED_COMPARISON.json 四组完整对照；指标状态复算通过 | 已核实 |
| 不改分数替换、±2 残差、全 slot 分母；不重提旧正则 bug；不扩成 Q+M/网格 | 最终 diff：头部代码未修改、geometry_loss 未修改；固定实验范围 | 已核实 |
| 证据有界结论、成本、原报告/锁点/tag 保留 | CONCLUSION.md、REPORT.md、VERIFIED_RESULTS.json；72 指标状态复算、68 测试通过；旧 tag 仍为 4bf00902 | 已核实 |

已知非本次引入的测试问题：TEST_NOTES.md 记录了数据软链接导致的既有路径序列化
测试失败，且已在原基线 worktree 复现。本次没有修复或隐藏该无关行为。
