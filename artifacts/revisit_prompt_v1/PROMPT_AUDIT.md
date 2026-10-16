# MPR-V1 提示词执行审计

主代理已完成原指令 §19 的十二项内容审核。代码、锁定实验、统计、部署前向和本地交付核对完成，研究结论为 `SCREEN_NO_GO`。A/B 发布状态保持 `PENDING_EXTERNAL_VERIFICATION`；实际远端 Git/LFS 字节、Release 能力及最终状态只由运行根 publication/PUBLICATION_RECEIPT.json 记录，本页不预填发布成功。

原文为 [EXECUTION_INSTRUCTION.md](EXECUTION_INSTRUCTION.md)，SHA256 `08cf2d2c83f1bd93d3120623f3ea9884fddd0f1fb13af843677b92160cc0070e`；原 ZIP 五项 checksum 见 [INPUT_PACKAGE_VERIFICATION.json](INPUT_PACKAGE_VERIFICATION.json)。[PROMPT_REVIEW_SOURCE.md](PROMPT_REVIEW_SOURCE.md) 的静态审阅不等于实际训练或收益验证。BASE 为 `6b1b779172afdeb51e2719a737cd05e7eeeca982`；本轮分支 `research/persist4d-mpr-v1-formal-20261014`，tag `persist4d-mpr-v1-formal-20261014-v1`。旧预算阻塞轮及其已发布引用保持原身份。

[FORMAL_DELIVERY_REVIEW_PROGRESS.md](FORMAL_DELIVERY_REVIEW_PROGRESS.md) 和早期审查 JSON 保留其日期、当时 pending 状态及证据范围，不追溯改写历史。以下审核结合 [PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 的实际源码审阅与最终执行证明，不把任何旧快照扩大为最终验收。

## 1. 首个掩码前是否真正使用历史检索

结论：实现及有限内容干预有实际证据，收益机制未成立。models/rescene.py 在 initialize_queries 后、首次 mask module 前调用 before_first_mask；M4–M6 使用完整当前段特征与先前预测状态。P5 六个固定 CAL 长输入在共享冻结 backbone、旧状态及对齐 RNG 下干预，均为 CONTROLLED，原始预测指纹均改变。只证明受限内容敏感性，不能替代 AP 收益或重训练消融。

证据：[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 1 项、[SPARSE_CAL_SELECTION_VERIFICATION.json](SPARSE_CAL_SELECTION_VERIFICATION.json)、[P5_EXECUTION_VERIFICATION.json](P5_EXECUTION_VERIFICATION.json)、[MECHANISM.csv](MECHANISM.csv)。

## 2. 是否绕过当前 proposal 门槛并保留新对象发现

结论：调用与输入契约成立，不据此声称召回上升。M4–M6 检索不接收当前 proposal 或其置信度；最多 32 个历史提示查询，至少 68 个原 FPS 查询。空历史严格不修改查询和位置。M2 保持 proposal-read 适配对照含义。

证据：[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 2 项及其 models/revisit_prompt.py、models/revisit_prompt_modules.py 源码锚点。

## 3. UUID、时间、超点和坐标是否对应实际源码

结论：因果数据和索引契约、全部正式评测实际集合已核对。UUID 独立校验；TRAIN/CAL/SEL 使用 train 索引，PB 使用 validation；阶段 j 的共同真值仅使用 0..j 前缀。local-t、轨迹 absolute-t、superpoint 映射及参考坐标变换分开。全部 93 进程的 input_id/horizon/reference_id 实际集合及状态键已核对，不只比较计数。同类诊断以同一 native loader 读取 627 个最终当前 GT，逐项核对 18 份实际结果的可评价实体集合。

证据：[DATA_MANIFEST.json](DATA_MANIFEST.json)、[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 3 项、[FINAL_DELIVERY_VERIFICATION.json](FINAL_DELIVERY_VERIFICATION.json)、[SAME_CLASS_DIAGNOSTICS_VERIFICATION.json](SAME_CLASS_DIAGNOSTICS_VERIFICATION.json)。

## 4. 父参数、缓冲区与新旧身份是否完整保留

结论：真实训练加载、六个最佳 delta、两份 resume 及两个锁定端点的实际部署前向有证据。固定父任务包严格核对原 BASE 身份和任务参数/buffer 的名称、shape、dtype、值；编码器单独取得，不遗漏任务 decoder。delta 绑定父、编码器、taxonomy、训练数值源码及全部新增张量。

P5 的 M3-400/M4-200 实际结果均为 EXACT_PARENT_ENCODER_DELTA_COVERAGE_AND_REAL_FORWARD，加载 delta 与公开副本 SHA 相同。其余四个最佳 delta 为离线张量/身份核对及原训练/评测证据，不冒称额外进行了六次公开包 GPU 回载。M4 最佳 delta 位于 200，resume 位于 400，不混称相同 optimizer 边界。

证据：[PARAMETER_AND_LOAD_AUDIT.json](PARAMETER_AND_LOAD_AUDIT.json)、[MAIN_TRAINING_VERIFICATION.json](MAIN_TRAINING_VERIFICATION.json)、[MODEL_PAYLOAD_REVIEW.json](MODEL_PAYLOAD_REVIEW.json)、[P5_EXECUTION_VERIFICATION.json](P5_EXECUTION_VERIFICATION.json)、[MODEL_MANIFEST.json](MODEL_MANIFEST.json)、[DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)。

## 5. 梯度是否连通，M1/M4 是否真等参数且执行

结论：真实训练支持实现契约，未追加重复 GPU 梯度任务。父模块保持 eval；no_grad 仅包围冻结 backbone，提示后的冻结 decoder 保留输入梯度。M1/M4 均为 84224 个可训练参数且初始化哈希相同。六条正式轨迹的新增张量改变、父张量未变，P0 和正式训练证据分别保留其范围。

证据：[PARAMETER_AND_LOAD_AUDIT.json](PARAMETER_AND_LOAD_AUDIT.json)、[MAIN_TRAINING_VERIFICATION.json](MAIN_TRAINING_VERIFICATION.json)、[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 5 项。

## 6. 适配对照、全新 M0 和共同最终政策是否成立

结论：正式 CAL/SEL/PB/P5 使用同一冻结 V4。M2 是按本轮配方训练的旧 proposal-read 适配对照；M0 是新鲜三进程基线，没有借用旧 B4 数字。共同最终 route/commit、logical-ID/generation、commit0 及轨迹分数政策一致；M6 保留预定义的来源查询支持概率乘一次差异。

训练为 C3 `c472a4923c0418cd5ad02a47b6744b51e4e16922` 的 14 项数值源码；推理独立冻结 ordered-sparse-reference-v4 的 18 项源码，政策 SHA256 `4f1f7b6d1d1efdfbf73cbb6c56296b8d00b39dd5c1f2f6a7ead5a2a15dc309c6`。V4 含稳定 FP32 归约和保序稀疏参考计算，不称原论文推理实现完全不变。旧 Native 异常的精确低层原因仍未解决，旧失败、政策和成本保留。有限重复一致不证明全局确定性。

证据：[SPARSE_FORMAL_BASELINE_VERIFICATION.json](SPARSE_FORMAL_BASELINE_VERIFICATION.json)、[EVALUATION_M0_ADMISSION_V4.json](EVALUATION_M0_ADMISSION_V4.json)、[EVALUATION_POLICY_V4.json](EVALUATION_POLICY_V4.json)、[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 6 项。

## 7. GT 是否隔离，UNKNOWN 是否被当成负例

结论：预测检索不接收 GT，GT 只用于训练损失/账本和评价。训练在预测提交后的可靠 birth 才绑定 logical_id/generation/reference，不重写既有标签。检索无正段时跳过；support UNKNOWN=-1 不参与 BCE。显式负例要求可靠 anchor 明确属于另一可靠对象，缺席或未绑定身份不能自动变成负例。同类诊断也不把未知/歧义身份当可靠实体，不删除主评价输入。

证据：[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 7 项及 models/revisit_prompt_loss.py、models/revisit_prompt.py 锚点、[SAME_CLASS_DIAGNOSTICS_VERIFICATION.json](SAME_CLASS_DIAGNOSTICS_VERIFICATION.json)。

## 8. N、lr、种子、四点和六臂是否遵守锁

结论：六条 seed146 主训练均完成 N=400、lr=1e-4，100/200/300/400 共 24 个实际检查点逐份核对。seed146/147 是同一 seed45 父任务包的适配种子。RUN_LOCK SHA256 `c1978630045a611325a110653ecf9f983c4081d00705cf53dfc28342ee0d4ed5`；CAL_SELECTION_LOCK SHA256 `e182171047492635eb6b4b8efe9a502128a3ff639775df5b4c3cf0631fd09e02`，均未改写。

锁定对照 M3-400、描述性候选 M4-200。CAL 长期差 +0.776406pp，但 T3/T4 差 −0.804022pp/−1.402200pp，未满足逐长度 −0.2pp 门槛。seed147 如实为 NOT_EXECUTED_CAL_GATE_FAILED，不补训练、不伪造种子、不用确认集重选。

证据：[RUN_LOCK.json](RUN_LOCK.json)、[MAIN_TRAINING_VERIFICATION.json](MAIN_TRAINING_VERIFICATION.json)、[SPARSE_CAL_SELECTION_VERIFICATION.json](SPARSE_CAL_SELECTION_VERIFICATION.json)、[CAL_SELECTION_LOCK.json](CAL_SELECTION_LOCK.json)、[REPLICATION_LOCK.json](REPLICATION_LOCK.json)。

## 9. 重复、失败、分母、统计单位和历史暴露是否完整

结论：75 个 CAL、18 个 SEL/PB 正式进程完成，共 31 个三重复组、456 行指标、11943 行输入诊断。CAL T1–T5 输入为 31/19/14/14/6；SEL 为 39/21/21/15/15，每轮 111 输入/279 阶段；PB T2–T5 各 129，每轮 516 输入/1806 阶段，无 PB-T1。阶段数不冒充输入数。物理 reference 为 8/8/6，均 HISTORICALLY_EXPOSED；LOCAL 不加入，重叠输入和进程重复不增加独立场景数，不与 ReScene 完整测试主表直接比较。

SEL/PB 真实完成原规则 1000 次物理 reference 成对 bootstrap，seed9101，三进程差值分别计算后取均值，1000 有效/0 NA，不重抽。SEL 长期/current 95% 区间为 [−0.584956,+1.240682]pp/[−0.706283,+0.688935]pp；PB 为 [−0.064652,+0.598815]pp/[−0.407302,+0.296667]pp，全部跨零，少于十场景的警示保留。

CPU 加速只改变曲线计算和完整保序输入精确缓存，未改官方积分/抽样/NA。SEL 实际 36000 池化；PB 实际 36000 逻辑调用全部命中，仍读取六份真实结果并执行全部原抽样，预热不是正式结果。函数绑定最终恢复，CPU 未初始化 CUDA。安全交接仅在全部 GPU 评测结束且无 reservation/子进程后中断旧 CPU 父控制器；预期 KeyboardInterrupt 保留，新控制器完成 PB/P5 后正常退出。

原标量官方接口独立复算范围仍是 74 份总体状态及各自 reference 池化，不冒称所有重复又做一轮指标。最终字节核对覆盖 912 份总体、5244 份 reference 状态及 ZIP 内全部 6156 份实体；所有最终 CSV 字段与实际结果一致，TRAINING 共 2400 行。

证据：[FINAL_DELIVERY_VERIFICATION.json](FINAL_DELIVERY_VERIFICATION.json)、[SEL_BOOTSTRAP_EXECUTION_VERIFICATION.json](SEL_BOOTSTRAP_EXECUTION_VERIFICATION.json)、[PB_BOOTSTRAP_EXECUTION_VERIFICATION.json](PB_BOOTSTRAP_EXECUTION_VERIFICATION.json)、[PB_BOOTSTRAP_INDEPENDENT_VERIFICATION.json](PB_BOOTSTRAP_INDEPENDENT_VERIFICATION.json)、[CPU_CONTROLLER_HANDOFF_VERIFICATION.json](CPU_CONTROLLER_HANDOFF_VERIFICATION.json)、[DATASET_COMPARISON.md](DATASET_COMPARISON.md)。

## 10. 最佳候选、方法胜出和投稿就绪是否分开

结论：研究证据不足，收缩论断为 SCREEN_NO_GO，不声明 READY_FOR_CVPR。M4-200 是描述性最佳，不是验证成功的赢家。相对 M3，SEL 长期/current 为 +0.660414pp/−0.459685pp；PB 为 +0.288628pp/+0.055652pp，但 PB T2 为 −0.396755pp。CAL 失败门槛不由确认集改写，四个区间均跨零。

当前掩码一对一诊断每轮 SEL 恢复/丢失 29/26、PB 181/214（M4 相对 M0）；同类共现子集相对 M3 为 SEL 18/13、PB 91/125。PB 负结果保留，条件召回不是 AP，不能确立同类干扰因果。H1=NOT_ESTABLISHED；搬动子集无可靠变化标注绑定，H2=NOT_AVAILABLE_MOVING_SUBSET；六个受控输入的内容敏感性不足以证明增益，H3=NOT_ESTABLISHED。QUALITY_SCOPE=TEMPORAL_ONLY，不以梯度、代码测试或 query 改变替代科学有效性。

唯一停止判断：保留可复现基线与负结果，停止将此候选作为已验证赢家继续涨点或投稿，不追加训练或调整门槛。

证据：[SELECTION_DECISION.json](SELECTION_DECISION.json)、[REPORT.md](REPORT.md)、[CURRENT_MASK_DIAGNOSTICS.json](CURRENT_MASK_DIAGNOSTICS.json)、[SAME_CLASS_DIAGNOSTICS.json](SAME_CLASS_DIAGNOSTICS.json)、[MECHANISM.csv](MECHANISM.csv)。

## 11. 总账、预留和运行范围是否真实

结论：公开/私有账本 230 个唯一事件完全相同，每个只收费一次；历史消费、失败和诊断没有删除。正式主训练 9.12422078314583 GPUh，全部新增实际 45.853692703092854 GPUh，prior 1714.8140614156707 GPUh，累计 1760.6677541187635 GPUh。新增池上限 100、累计上限 1823、预留 8 GPUh；累计余额 62.332245881 GPUh，预留不是已结算消费，不重复计费。

全部 GPU 工作只使用本任务物理 GPU1，未占其他用户/设备。最终 ACTIVE_JOB 不存在，reservation 为零；正式控制器和训练/评测/P5/CPU 子任务退出。本任务只读 watcher 在收尾阶段单独检查并结束，实际状态见最终 STATUS，不把监控进程冒称训练仍运行。CPU bootstrap、缓存和同类诊断没有新增 GPU 费用，小试不冒充全量。

证据：[PROTOCOL_AMENDMENT.json](PROTOCOL_AMENDMENT.json)、[COST_LEDGER.jsonl](COST_LEDGER.jsonl)、[FINAL_DELIVERY_VERIFICATION.json](FINAL_DELIVERY_VERIFICATION.json)、[STATUS.json](STATUS.json)、[PROFILE.csv](PROFILE.csv)。

## 12. 可加载资产、交接、远端字节和 SHA 自引用

结论：本地资产及部署证据齐备；实际远端发布由外部回执验收，不能在 push 前填 PASS。MODEL_MANIFEST 有九项真实实体：六个 CAL 最佳 delta、M3/M4 的 N=400 resume（八份模型共 3724164 bytes）及 212215855 bytes 的充分状态 ZIP。实体 bytes/SHA 与原件核对，ZIP 内 6156 份状态逐份核对，无原始点云/逐点 GT。模型依赖固定父包、独立合法取得的编码器和 taxonomy，delta 不冒称可恢复 optimizer。

主代理已审阅发布脚本及 HANDOFF/COMMANDS 生成内容，涵盖原 §18.4。P5 两个锁定端点提供严格加载和有限真实前向，不反复加载几十遍。发布 A 冻结代码、配置、结果、模型和 REPORT；ARTIFACT_MANIFEST 排除自身、B 的 HANDOFF/COMMANDS/PROMPT_AUDIT 和外部回执。B 交接引用 A 完整 SHA，不写 B 自己的 SHA；旧分支/tag 不覆盖。

真实发布使用本轮显式分支/tag，上传九个选定 LFS 实体并推送引用；一次新临时目录和空 LFS 缓存实际取回全部新实体，逐项核对 bytes/SHA，同时核对所有 A manifest 文件、B 交接和远端引用。ls-remote 或指针列表不算实体证明。Git/LFS 成功但 Release 不可用时，依原指令记录 CODE_AND_LFS_PUSH_VERIFIED_RELEASE_RECEIPT_LOCAL_ONLY、实际错误和本地回执，不伪造 Release 链接；失败保留 A/B 和可重试命令。

证据：[MODEL_MANIFEST.json](MODEL_MANIFEST.json)、[DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)、[DEPLOYMENT_PROTOCOL.md](DEPLOYMENT_PROTOCOL.md)、[P5_EXECUTION_VERIFICATION.json](P5_EXECUTION_VERIFICATION.json)、[FINAL_DELIVERY_VERIFICATION.json](FINAL_DELIVERY_VERIFICATION.json)。远端事实不回写已冻结 A/B，保存在外部 PUBLICATION_RECEIPT。

本地审核结论：实现与锁定实验完成，条件复验按明确失败分支不执行；科学证据不支持胜出或 CVPR 就绪。实际发布是下一步操作，不由本页预先宣告成功。
