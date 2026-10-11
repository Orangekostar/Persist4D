# 主代理逐项验收

对照原`native_long_retrain_v1/EXECUTION_INSTRUCTION.md`第0–15节及后续已授权集群预算/论文证据工作。
审查日期2026-10-11。测量和资产整理已完成；实际Git发布身份由交接文件及仓库外最终回执核验。科学限制不因交付完成而解除。

| 项目 | 实际状态 | 证据与限制 |
|---|---|---|
| 目标与四臂 | SEED45_TRAINING_COMPLETE | E0/E1/E2/E3均29,700更新；无组合模型。E2短时退化、E3低于最佳E1均保留。 |
| 源版本/隔离 | PASS_BINDING | 原数值训练源与四条锁定身份一致；新工作位于独立worktree/new branch；原用户稿未改。 |
| 数据总体 | PASS_POPULATION | 361 TRAIN、8 CAL、8 SEL，真实不同扫描及原全局身份；正式场景与本轮TRAIN无重叠，历史验证重叠明确。 |
| 初始化/目标 | PASS_MEASURED | 新共同任务初始化、编码器仅预训练加载；冻结参数但保留原训练模式；raw_sum与12aux真实核验。 |
| 原生数值 | FRESH_PROCESS_NONREPEATABILITY_OBSERVED | 先前固定单输入顺序重复一致，扩大面板后2/30对照混杂；8项重复证实同协议张量变化，两个输入AP/REC相同。无容差PASS。 |
| 物理批量 | PASS_MEASURED_FALLBACK | 完整大T5每卡batch2 OOM，统一每卡batch1、两卡、累积16；有效批量32，未删大样本或改体素/query。 |
| 阶段采样S | MEASURED_WITH_NEGATIVE_RESULT | 实际训练cap触发且E2有分层调用；CAL72调用均触发，均匀/分层均无阶段遗漏；SEL T1−5.4287pp。 |
| 特征反馈F | CONNECTED_NOT_UNIVERSALLY_BENEFICIAL | 已训练T2/T5梯度非零、原解码器梯度有限、无参数更新；全部30对照保留，28严格对齐、2混杂；同一步/最佳基线同时列出。 |
| 训练/恢复 | COMPLETE_WITH_RUNTIME_LIMITATION | 四条轨迹终点/optimizer draw身份已核验；保留全部日志版本和失败成本。重复尝试的F范数有差异，不能声称完整训练bitwise恢复。 |
| CAL/SEL | COMPLETE_LOCKED | 40 CAL、6 SEL、230官方状态；CAL锁定后SEL，E3按既定门槛进入正式确认；未用诊断重选检查点。 |
| 选点课程阶段 | VERIFIED_LIMITATION | E3/2970与E1/2970尚处3RScan仅T2阶段；虽完整轨迹到29,700，选中点的长输入指标不能声称来自T3–T5监督。 |
| 正式确认 | COMPLETE_AUDITED | 四点×890=3560输入、596状态、32总体重算；全部分母/顺序/权重/官方指标源绑定一致。 |
| 独立种子 | NOT_REPLICATED_BUDGET | 完整E1/E3 seed46预测1034.5019 GPUh超过余额；未启动缩短版冒充复验。 |
| 成本 | VERIFIED_FINAL_COST | 总上限1800、预留8；训练历史基数1707.342516592207＋唯一新增事件＋活跃占用；失败也计费。最终1713.8830547291614 GPUh，余86.11694527083863，live=0，失败包含在内。 |
| 嵌套长时诊断 | COMPLETE_WITH_SCOPE_LIMIT | 30条模型/序列轨迹、370对象记录；absent含标签过滤，raw-query变化不等于持久ID错误；仅TRAIN/CAL解释性材料。 |
| 部署profile | COMPLETE_MEASURED | 同A40固定CAL输入，20组汇总、80原始观测含20次预热、4次进程setup；时间秒/显存字节，IO等单列。 |
| 已训练记忆桥接 | COMPLETE_AUDITED | E0/5940与E1/23760各129序列、1032前向；184状态/40指标复算。645前缀公共真值对齐PASS，125窗口标签差异均保留。 |
| 测试/代码审核 | PASS_FOCUSED | 本轮12文件41测试通过、31新Python文件Ruff通过；不以测试数量推断精度。原数值源保持绑定。 |
| 调度/监控 | COMPLETE_NO_ACTIVE | 原训练控制器与全部推理队列正常结束；Luna第332轮确认全部COMPLETE、ACTIVE清空后停止轮询。worker锁、账本和原配额均保留。 |
| 模型资产 | REMOTE_LFS_BYTES_VERIFIED | 五份完整任务/必要buffer资产，精确重载＋独立空缓存远端下载535,331,523字节；编码器另行获取。 |
| 指标/轻量结果 | COMPLETE_VERIFIED_PACKAGE | 1093官方状态、444测量回执、109训练日志文件；全部成员/整体SHA回读通过，所有正负行保留。 |
| Git发布 | SEE_HANDOFF_AND_EXTERNAL_RECEIPT | 本页随实验A提交；交接记录A，B包含最终清单；实际push/new tag与remote SHA见仓库外最终回执。旧tag不改写。 |
| 文稿/结论 | NOT_READY_FOR_CVPR | 已生成独立中文实验整合稿；缺等参数控制、独立种子、更多外部场景、长历史/饱和和完整投稿包；仅保留有实际证据的主张。 |

已测结果不确认新的官方最佳性能，也不证明普遍部署可重复性。固定方案的失败/不足作为结果交付；
无法在现有预算内做完整独立重训时明确保留NOT_REPLICATED_BUDGET，不能用工程检查替代科学复验。
