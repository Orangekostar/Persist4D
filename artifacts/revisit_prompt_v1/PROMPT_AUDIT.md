# MPR-V1 提示词执行审计

状态：`IN_PROGRESS`。本页是执行审计草稿，不是最终验收或发布成功声明。审阅基于截至 2026-10-16 07:19 UTC 已存在的源码、锁、正式结果与主代理核验证据；后续运行以实际进程和新增状态为准。`STATUS.json` 中的 `publication_ready` 与 `prompt_audit_execution_complete` 仍为 false。

原指令为 [EXECUTION_INSTRUCTION.md](EXECUTION_INSTRUCTION.md)，SHA256 `08cf2d2c83f1bd93d3120623f3ea9884fddd0f1fb13af843677b92160cc0070e`。原 ZIP 五项 checksum 和主指令字节已核对，见 [INPUT_PACKAGE_VERIFICATION.json](INPUT_PACKAGE_VERIFICATION.json)。

## 原提示词审阅与实际执行的边界

[PROMPT_REVIEW_SOURCE.md](PROMPT_REVIEW_SOURCE.md) 和 [INSTRUCTION_STATIC_AUDIT.json](INSTRUCTION_STATIC_AUDIT.json) 保留交付提示词时的静态审阅。其 23 项 PASS 不代表本轮模型、训练、部署或远端发布已完成。固定 BASE 的 29 个源码 blob 与 26 组索引的本地绑定见 [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json)。

本轮正式分支为 `research/persist4d-mpr-v1-formal-20261014`，固定 BASE 为 `6b1b779172afdeb51e2719a737cd05e7eeeca982`。旧预算阻塞轮及其已发布标签保留原身份，不用本轮记录改写它。最终 A/B 提交和 tag 尚未生成；发布时由 HANDOFF 引用 A，B 不嵌入自己的 SHA，远端核验事实另写外部回执。

## 实际修订、身份与科学边界

- 用户授权的预算续跑记录于 [PROTOCOL_AMENDMENT.json](PROTOCOL_AMENDMENT.json)。prior 为 1714.8140614156707 GPUh，新池上限 100 GPUh、累计上限 1823 GPUh、保留 8 GPUh；既有消费没有归零。冻结 RUN_LOCK 的 SHA256 为 `c1978630045a611325a110653ecf9f983c4081d00705cf53dfc28342ee0d4ed5`。
- 六条正式训练使用 C3 `c472a4923c0418cd5ad02a47b6744b51e4e16922` 及其 14 项训练数值源码。评测改用独立冻结的 [EVALUATION_POLICY_V4.json](EVALUATION_POLICY_V4.json)，政策 `ordered-sparse-reference-v4`，SHA256 `4f1f7b6d1d1efdfbf73cbb6c56296b8d00b39dd5c1f2f6a7ead5a2a15dc309c6`，绑定 18 项推理源码。两条身份链明确分开，没有为采用新推理政策重新训练模型。
- 早期 stable-mean、Native 与 centroid 诊断、失败和费用保留在相应报告及独立命名空间。V4 不是原论文推理程序完全不变的结果；正式 CAL/SEL/PB/P5 须使用同一 V4 上下文，不能混入旧策略结果。V4 的有限重复一致没有解决旧 Native 大幅异常的因果来源，也没有证明全局逐张量确定性。
- 唯一选点只使用完整 CAL 四点及每点三次独立进程。锁定对照为 M3-400，描述性候选为 M4-200；[CAL_SELECTION_LOCK.json](CAL_SELECTION_LOCK.json) 的 SHA256 为 `e182171047492635eb6b4b8efe9a502128a3ff639775df5b4c3cf0631fd09e02`。SEL/PB 不重选端点。
- CAL 长期均值差为 +0.7764pp，但 T3/T4 分别为 −0.8040pp/−1.4022pp，未满足逐长度 −0.2pp 门槛。[REPLICATION_LOCK.json](REPLICATION_LOCK.json) 如实为 `NOT_EXECUTED_CAL_GATE_FAILED`，不追加 seed147 或替换候选。完整交付仍应保留 `SCREEN_NO_GO`，不能因确认集某个正差值改写 CAL 晋级规则。
- 可选 CPU 统计入口仅加速原 bootstrap 内的 PR 曲线计算，保留官方积分、reference 成对抽样、1000 次重采样、seed9101、三进程差值均值与 NA 规则。56 份状态及 reference 池化和 220 条实际曲线已精确核对，23 项聚焦测试通过，见 [CPU_STATISTICS_ACCELERATION_VERIFICATION.json](CPU_STATISTICS_ACCELERATION_VERIFICATION.json)。截至本草稿，正式控制器接管和完整加速 bootstrap 均未执行；局部计时不代表完整统计耗时。

## §19 执行内容逐项审核

### 1. 首掩码前是否使用历史检索

结论：调用链已核对，机制收益尚未证实。`models/rescene.py` 在 initialize_queries 后、首次 mask module 前调用 `before_first_mask`；M4–M6 从完整当前段特征和先前预测状态检索。正式 CAL 前向与该冻结源码身份对应。实际干预尚待 P5，不能将 query 改变等同涨点机制有效。

证据：[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 1 项、[SPARSE_CAL_SELECTION_VERIFICATION.json](SPARSE_CAL_SELECTION_VERIFICATION.json)。

### 2. 是否绕过当前 proposal 门槛并保留新对象发现

结论：结构和输入契约已核对，召回收益未被本项证明。M4–M6 的检索路径不读取当前 proposal 或其分数；最多使用 32 个历史提示槽，至少保留 68 个原始 FPS 查询；空历史严格不修改查询与位置。M2 单独保留 proposal-read 适配对照的含义。

证据：[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 2 项及所列 `models/revisit_prompt.py`、`models/revisit_prompt_modules.py` 源码锚点。

### 3. UUID、时间、超点和坐标是否对应实际源码

结论：因果数据与索引契约已核对，锁定候选 PB 尚未全部核验。完整 UUID 独立校验；TRAIN/CAL/SEL 使用 train 索引，PB 使用 validation；阶段 j 共同真值只使用 0..j 因果前缀。局部 pair 的 local-t、轨迹 absolute-t、superpoint 映射与实际参考坐标变换分开。

SEL 三轮的实际 `(input_id, horizon, reference_id)` 集合及 reference 状态键与同角色 M0 完全相同。PB 候选完成后仍需同样检查，不能只凭计数相等通过。

证据：[DATA_MANIFEST.json](DATA_MANIFEST.json)、[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 3 项、[SPARSE_SEL_M4_VERIFICATION.json](SPARSE_SEL_M4_VERIFICATION.json)。

### 4. 父参数、缓冲区和新旧身份是否完整保留

结论：实际训练加载覆盖与六个最佳 delta 已核对；最终公开副本和真实部署回载待完成。父包按 BASE 原源码身份校验，所有任务参数及 buffer 的名称、shape、dtype、值严格覆盖；编码器仅按规定前缀单独取得，不能遗漏其任务 decoder。delta 校验父/编码器/taxonomy/训练数值身份及全部新增张量。

证据：[PARAMETER_AND_LOAD_AUDIT.json](PARAMETER_AND_LOAD_AUDIT.json)、[MAIN_TRAINING_VERIFICATION.json](MAIN_TRAINING_VERIFICATION.json)、[MODEL_PAYLOAD_REVIEW.json](MODEL_PAYLOAD_REVIEW.json)、[DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)。P5 实际前向标记、最终 MODEL_MANIFEST 和公开资产 SHA 仍需核对。

### 5. 冻结父模型后梯度是否连通、M1/M4 是否真等参数

结论：已有真实训练证据支持，不新增重复 GPU 梯度任务。父模块保持 eval，no_grad 仅包围冻结 backbone；提示后的冻结解码运算保留输入梯度。M1/M4 实际均为 84224 参数且初始化哈希相同；六条正式轨迹记录父张量未变、新增张量训练后改变。此前 P0 梯度核查和正式训练证据各按其实际范围表述。

证据：[PARAMETER_AND_LOAD_AUDIT.json](PARAMETER_AND_LOAD_AUDIT.json)、[MAIN_TRAINING_VERIFICATION.json](MAIN_TRAINING_VERIFICATION.json)、[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 5 项。

### 6. 适配对照、全新 M0 与共同最终政策是否成立

结论：CAL 和已完成确认组支持，M4 PB 尚待完成。M2 是按本轮训练配方适配的旧 proposal-read 对照；M0 是本轮 V4 的独立三进程基线，没有借用旧 B4 数字。各变体共用最终关联、一次 commit 与 logical-ID 绑定；M6 按预定义差异对来源 query 的官方候选乘支持概率一次。

证据：[SPARSE_FORMAL_BASELINE_VERIFICATION.json](SPARSE_FORMAL_BASELINE_VERIFICATION.json)、[EVALUATION_M0_ADMISSION_V4.json](EVALUATION_M0_ADMISSION_V4.json)、[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 6 项、已完成 SEL/PB 组核验文件。

### 7. GT 是否隔离、UNKNOWN 是否排除

结论：检索前向与训练标签用途已核对。预测检索不接收 GT；训练账本只在预测提交后、可靠 birth 时绑定 logical_id/generation/reference，已有标签不重写。检索没有正段时跳过；support 的 UNKNOWN=-1 不参与 BCE。可靠 anchor 明确属于另一可靠对象才是显式负例，不能将历史缺席或未绑定身份推断为负例。

证据：[PRIMARY_IMPLEMENTATION_REVIEW.json](PRIMARY_IMPLEMENTATION_REVIEW.json) 第 7 项及 `models/revisit_prompt_loss.py`、`models/revisit_prompt.py` 的冻结源码锚点。

### 8. N、学习率、种子、四点和六臂是否遵守锁

结论：正式主训练与 CAL 已完成，条件复验按失败分支不执行。M1–M6 seed146 均完成 N=400，检查点为 100/200/300/400，实际锁定优化配方学习率 1e-4。24 个训练检查点逐份核对。seed146/147 都是同一 seed45 父模型的适配种子；本轮没有执行 seed147，也没有以短跑或继续微调代替配对复验。

证据：[RUN_LOCK.json](RUN_LOCK.json)、[MAIN_TRAINING_VERIFICATION.json](MAIN_TRAINING_VERIFICATION.json)、[SPARSE_CAL_SELECTION_VERIFICATION.json](SPARSE_CAL_SELECTION_VERIFICATION.json)、[CAL_SELECTION_LOCK.json](CAL_SELECTION_LOCK.json)、[REPLICATION_LOCK.json](REPLICATION_LOCK.json)。

### 9. 重复、失败、分母、统计单位和暴露是否完整

结论：CAL 及部分确认已核对；完整确认、bootstrap 和报告待完成。CAL 共 75 个进程、25 个三重复组、375 行结果；M0/M3 的 SEL/PB、M4 的 SEL 均完成三重复。SEL 每轮 111 个不同输入、279 个阶段；PB 每轮应为 516 个输入、1806 个阶段。不能把阶段前向数冒充输入数。

固定输入分母：CAL T1–T5 为 31/19/14/14/6；SEL 为 39/21/21/15/15；PB T2–T5 各 129，没有 PB-T1。独立 reference 数 CAL/SEL/PB 为 8/8/6，均为 `HISTORICALLY_EXPOSED`，不能宣称新的外部独立测试集。

主代理使用原标量官方接口已独立 CPU 复算 66 份总体状态及各自 reference 池化，指标与 RESULTS 精确一致：原 [OFFLINE_METRIC_RECOMPUTATION.json](OFFLINE_METRIC_RECOMPUTATION.json) 56 份，另有 [SPARSE_SEL_M4_VERIFICATION.json](SPARSE_SEL_M4_VERIFICATION.json) 中 M4 SEL repeat0 的 10 份。该范围不包括 M4 PB、其他 repeat 的独立复算或最终压缩包。

bootstrap 必须沿用物理 reference 成对重采样、官方充分状态池化、三进程差值分别计算后取均值；NA 不重抽，少于 10 reference 保留警示。当前只有实现及局部一致性证据，完整区间尚未生成。

证据：[DATA_MANIFEST.json](DATA_MANIFEST.json)、各组核验文件、[CPU_STATISTICS_ACCELERATION_VERIFICATION.json](CPU_STATISTICS_ACCELERATION_VERIFICATION.json)、[FORMAL_DELIVERY_REVIEW_PROGRESS.md](FORMAL_DELIVERY_REVIEW_PROGRESS.md)。历史异常与失败不删除。

### 10. 最佳候选、方法胜出和投稿就绪是否分开

结论：CAL 筛选未通过，完整确认和诊断不改变该失败门槛。M4-200 是描述性最佳候选，不是已验证胜出的方法。M4 SEL 三轮 S_long=0.07959490269422531、S_current=0.271793266137441；相对锁定 M3-400 的描述性差值为 +0.6604pp/−0.4597pp，当前 AP 的下降必须同时保留。该证据不能支持“补回漏分割”的 H1。

H1 还需要当前掩码的一对一恢复与丢失诊断；仅长期正差而 current 无增益须保留 TEMPORAL_ONLY 范围。H2 搬动子集没有已验证变化标注绑定时为 NOT_AVAILABLE；H3 不凭参数梯度或 query 改变宣称成立。完整 PB、区间、机制和资源仍待实际完成。本轮不自动给出 READY_FOR_CVPR。

证据：[CAL_SELECTION_LOCK.json](CAL_SELECTION_LOCK.json)、[REPLICATION_LOCK.json](REPLICATION_LOCK.json)、[SPARSE_SEL_M4_VERIFICATION.json](SPARSE_SEL_M4_VERIFICATION.json)、[DATASET_COMPARISON.md](DATASET_COMPARISON.md)。

### 11. 总账、预留和运行范围是否真实

结论：截至已结算快照核对通过，最终总额与无活动进程待收尾核验。[SPARSE_SEL_M4_VERIFICATION.json](SPARSE_SEL_M4_VERIFICATION.json) 记录私有/公开账本 223 个唯一事件相同，三笔 SEL 费用均只结算一次；新增实际累计 41.784839852624266 GPUh，含历史累计 1756.598901268295 GPUh。活动 PB 的预留不是实际已结算消费，不能重复加到上述累计。

正式训练实际为 9.12422078314583 GPUh，额外费用还包括失败、诊断与评测。8 GPUh 预留保留，旧消费和失败不抹去。当前 GPU 子进程仍在运行，不能填写“无本任务活动进程”。原费用锁及去重仍由原控制器管理。

证据：[PROTOCOL_AMENDMENT.json](PROTOCOL_AMENDMENT.json)、[RUN_LOCK.json](RUN_LOCK.json)、[COST_LEDGER.jsonl](COST_LEDGER.jsonl)、[SPARSE_SEL_M4_VERIFICATION.json](SPARSE_SEL_M4_VERIFICATION.json)。最终需重核已结算总额、reservation 释放和进程终态。

### 12. 可加载资产、交接、远端字节和 SHA 自引用是否完成

结论：未完成。已有六个最佳 delta 和 M3/M4 的 N=400 resume 原件完成离线张量/身份审核；M4 最佳 delta 位于 step200，resume 位于 step400，两者不混称相同步数的 optimizer 恢复。最终公开副本、完整 MODEL_MANIFEST、官方状态压缩包及一次真实 P5 部署前向尚待生成或核验。

P5 profile 已实现严格父/编码器/delta 覆盖及固定 CAL panel 的实际 warmup/测量。只有真实输出中的 `EXACT_PARENT_ENCODER_DELTA_COVERAGE_AND_REAL_FORWARD` 与最终公开 delta 同 SHA，才满足该回载证据；源码字符串和加载示例不算实际完成。

发布脚本支持本轮显式 branch/tag、冻结推理源码进入 A manifest、选定 Git LFS 上传及一次空缓存远端下载核验。当前仅预检/dry-run，没有 A/B/tag 或实际 Git/LFS 上传；SSH 读取成功不证明写权限。Release API 凭据在预检时不可用，最终按实际尝试记录本地回执或 Release 回执，不能伪造成功链接。

证据：[MODEL_PAYLOAD_REVIEW.json](MODEL_PAYLOAD_REVIEW.json)、[DEPENDENCY_ACQUISITION.json](DEPENDENCY_ACQUISITION.json)、[DEPLOYMENT_PROTOCOL.md](DEPLOYMENT_PROTOCOL.md)、[PUBLICATION_IMPLEMENTATION_VERIFICATION.json](PUBLICATION_IMPLEMENTATION_VERIFICATION.json)、[PUBLICATION_PREFLIGHT_REVIEW.json](PUBLICATION_PREFLIGHT_REVIEW.json)。远端证据最终由 run root 外部 publication 回执负责，不能预填在 A/B。

## 最终验收仍需的实际证据

1. M4 PB 三次完整结果、分母/实际 ID 集合、官方总体/reference 状态实体 SHA 与一次结算；候选 PB 的独立 CPU 复算。
2. 原规则的 SEL/PB bootstrap 实际输出及 NA/少场景边界、P5 机制与 profile 的实际状态、共同上下文及部署回载证据。
3. 完整 TRAINING/CAL/SEL/PB/BY_REFERENCE/IDENTITY/MECHANISM/PROFILE、CURRENT_MASK_DIAGNOSTICS、SELECTION_DECISION、REPORT 和本页最终执行结论；失败及历史暴露完整保留。
4. 六个最佳 delta、实际候选/对照 resume、最终状态压缩包和 MODEL_MANIFEST 实体字节核验；依赖取得位置、合法来源和严格加载契约完整。最终公开 diff/压缩包不得含原始点云、逐点 GT、编码器、绝对运行路径或凭据。
5. 最终账本及无活动 GPU reservation/子进程；原规则与冻结源码身份仍一致；仅完成所需聚焦检查。
6. 冻结 A、仅交接的 B、tag、实际 Git/LFS push、一次空 LFS 缓存取回全部本轮资产及 B 文档；外部回执记录真实 SHA/时间/错误。四种 CODE/EXPERIMENT/SCIENCE/PUBLICATION 状态分别报告，远端未完成项不能用本地测试代替。

以上未完成项未被设置为 PASS。健康运行的评测属于等待实际输出；不得因观察等待超时重启同一任务或追加未约定的涨点实验。
