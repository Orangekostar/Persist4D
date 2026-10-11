# 代码与实测证据的对应

原始模型、数据、损失、优化器及训练调度数值源保持四条轨迹的锁定身份；
本轮新增的是固定检查点评估、部署资产导出、机制/记忆测量和审计入口。
原数值身份由`protocol/BUDGET_LOCK.json`各任务及五个部署权重回执中的`training_identity.code`绑定。
源Git提交提供完整上下文，不把初始预检报告中的零训练状态当作当前结果。

| 新入口（仓库scripts目录） | 固定输入与作用 | 可验收证据 |
|---|---|---|
| native_long_postselection.py | 读取已结束的四臂CAL/SEL；按原规则分析，不重新训练/选点 | ANALYSIS、CAL_ALL、SEL_LOCKED、PAIRED_DELTAS；230状态SHA |
| native_long_confirmation.py / confirmation_worker.py / confirmation_inference.py | 锁定四个检查点、三个正式总体及输入顺序；原始官方预测路径 | FORMAL_LOCK/POPULATION、3560输入、FORMAL_AUDIT |
| native_long_deployment_validation.py / deployment_worker.py | 完整任务资产、固定CAL资源测量、A/B→A新进程重复 | 12项运行队列、PROFILE系列、RUNTIME_AUDIT |
| native_long_memory_bridge.py / memory_bridge_worker.py / memory_bridge_preflight.py | 同权重完整前缀、局部强关联和同候选持久状态；GT仅评价对齐 | 645前缀CPU证明、2×129序列计划、共享候选SHA与身份事件 |
| native_long_mechanism.py / mechanism_worker.py | 30个CAL嵌套输入×5行＋2梯度探针；固定权重与原目标 | MECHANISM_AUDIT、75官方状态、30开关配对、72采样调用 |
| native_long_mechanism_replay.py | 仅对原记录21/25，两模式各两次新进程重复 | 8项固定衍生计划、0.3GPUh上限、原152项不覆盖 |
| native_long_confirmation_audit.py / bridge_audit.py | 官方充分状态独立复算；桥接按保存身份事件复算计数和场景配对区间 | 正式596状态/32总体；桥接184状态/40总体均通过 |
| native_long_mechanism_audit.py / mechanism_replay_audit.py | 私有预测/目标重算官方状态与候选诊断；严格检查配对假设 | 28严格对齐＋2混杂，12重复源/30两两比较全部保留 |
| native_long_delivery.py | E2负结果选点导出；五份模型重定位编码器/评价taxonomy并严格加载 | PORTABLE_ASSET_AUDIT、LFS_MODEL_DOWNLOAD_RECEIPT |
| native_long_evidence_package.py | 所有审计完成后仅归档官方充分统计，并核验每成员与整个ZIP | 1093状态已归档、成员/整体SHA核验；排除逐点张量/数据/权重 |

反馈off干预仍计算并记录原反馈，但返回未更新特征；记录的computed反馈范数不是off实际施加的范数。
上游输入/真值不一致或捕获缺失仍令审计失败；真实上游数值差异保留为RUNTIME_CONFOUNDED，
不通过容差、删行或重复中挑选匹配对来变成因果PASS。梯度范数汇总仅允许JSON排序导致的浮点求和舍入差，
这与逐张量SHA一致性是不同检查。
`alignment.computed_feedback_exact`指完整已捕获记录一致（输入SHA、范数、形状计数等）；
未保存完整反馈增量张量，不能把该字段解释为增量张量逐元素相等。
`attribution_eligible`表示通过这些已测上游条件，不是对所有后续算子数值重复性的保证。

桥接把公共真值恢复限制在评价适配层。tracker.step仅收到预测特征、类别后验、置信度与预测mask；
GT不进入候选选择或关联。两个局部系统共享同一官方候选侧记录和平均置信度发布器。
完整前缀的AP来自联合预测，因此与局部方案还包含上下文和发布方式差异。

原始测量源一旦由计划绑定不在运行中改动；错误尝试另行保留并计费，不能覆盖成成功记录。
所有新增GPU队列复用原worker.lock、charged_process、节点配额及1800 GPUh上限。
Luna只运行只读探针，主代理负责诊断、实现、解释、最终核验和集成。

`SELECTED_CHECKPOINT_CURRICULUM.json`核实E3/2970仍为T2课程阶段。
`NUMERICAL_PRECISION_SCOPE.json`核实Lightning 32-true与原Concerto内部FP16 FlashAttention并存；
二者均为保留的实现事实，没有通过改变训练行为生成更有利的新结果。
