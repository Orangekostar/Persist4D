# 正式轮交付核对进度

状态：未完成，不能据此设置 publication_ready 或 goal complete。

权威要求为 EXECUTION_INSTRUCTION.md（与原 ZIP 主指令字节一致，SHA256 `08cf2d2c83f1bd93d3120623f3ea9884fddd0f1fb13af843677b92160cc0070e`）；预算更新以 PROTOCOL_AMENDMENT.json/RUN_LOCK.json 为准。原已发布的预算阻塞轮及其标签保持原记录。

| 要求 | 已核查的实际证据 | 当前结论 / 剩余工作 |
|---|---|---|
| §3 固定起点与源码绑定 | SOURCE_BINDINGS.json 在本地核对 BASE 的 29 个 Git blob；INPUT_PACKAGE_VERIFICATION.json 核对 ZIP 全部五个原始 checksum，主指令字节一致，原 registry 字段完整保留；原静态审阅另存 PROMPT_REVIEW_SOURCE/INSTRUCTION_STATIC_AUDIT | 绑定完成；源码阅读范围保留原审阅边界，原静态 PASS 不等于实际执行完成 |
| §5 父加载、冻结、新参数覆盖 | PARAMETER_AND_LOAD_AUDIT.json 核对六条真实正式训练状态、六个最佳 delta 张量及其身份；24 个训练检查点哈希另见 MAIN_TRAINING_VERIFICATION | M1/M4 均为 84224 参数且初始化一致；P5 profile 已实现严格回载及真实前向，待其实际完成后核对与打包副本的 SHA |
| §11 六臂与成本 | M1–M6 seed146 均 400 更新；四点 100/200/300/400；原 RUN_LOCK 与累计账本保留 | 正式主训练完成，不追加涨点搜索 |
| §12–14 全量 CAL 与选点 | SPARSE_CAL_SELECTION_VERIFICATION.json：75 个完整独立进程、25 个三重复组、全部指标及结构/原始预测记录相等；375 个 CSV 行与实际结果核对；用纯选模函数复算冻结 lock 完全相等 | CAL 完成；有限一致性不等于全局确定性或旧 Native 根因已解决 |
| §14 seed147 条件复验 | CAL_SELECTION_LOCK / REPLICATION_LOCK | 未执行是规定分支：M4-200 相对 M3-400，长期均值 +0.7764pp，但 T3 −0.8040pp、T4 −1.4022pp，未达到逐长度 −0.2pp 门槛；最终科学状态应保留 SCREEN_NO_GO |
| §13–14 锁定 SEL/PB | 原 V4 控制器仍执行 M0、M3-400、M4-200 的三次评测 | M0、M3 SEL/PB 及 M4 SEL 完成；M4-200 PB 进行中；其余以实时 STATUS/RESULTS 为准，CSV 在最终 collect 后更新 |
| §13 统计、机制与资源 | OFFLINE_METRIC_RECOMPUTATION.json 实际 CPU 复算两个锁定 CAL 端点及 M0/M3 SEL/PB repeat0 共 56 份状态；SPARSE_SEL_M4_VERIFICATION 另复算 M4 SEL repeat0 的 10 份状态，共 66 份，全部总体指标及 reference 池化均精确一致；原 run_campaign 后续包含 bootstrap 和 P5 | 尚未覆盖 M4 PB、其他 repeat 的独立复算与最终打包档案；继续等待真实 bootstrap/P5，并检查 NA、少场景边界、置换上下文及分母 |
| §17 模型与充分状态 | MODEL_PAYLOAD_REVIEW 已实际 CPU 读取六个选定 delta 与两个最终 resume，共 3724164 字节；身份、张量范围及候选/对照 optimizer/scheduler/RNG/400 步边界正确，无禁传字符串匹配；DEPENDENCY_ACQUISITION 核对 BASE 父 LFS 指针、106906589 字节父实体及 taxonomy；DEPLOYMENT_PROTOCOL 明确 V4 构造/前向上下文 | 原 summarize 尚待打包；须核对公开副本和真实部署前向。resume 保存最终 400 步边界，不冒充 CAL 最佳较早选点的 optimizer。部署文档两段 Python 仅 AST 检查，不冒充实际回载 |
| §17–19 报告与最终内容审核 | PRIMARY_IMPLEMENTATION_REVIEW.json 已记录主代理对十二问的源码和既有证据审阅；PROMPT_AUDIT.md 已逐题草拟执行结论，12 项编号、51 个本地证据链接及 4 项锁定身份核对通过，训练 14 个及推理 18 个源码身份再次相符 | 审计仍为 IN_PROGRESS；确认、统计、P5、全部交付文件及真实远端字节仍待核对；publication_ready 与 prompt_audit_execution_complete 均为 false；四种状态分开，不得自动宣布 CVPR 就绪 |
| §18 发布 | publish_revisit_prompt.py 已支持显式本轮 branch/tag、实际状态交接和冻结推理源码进入 A manifest；三项离线回归测试通过；PUBLICATION_PREFLIGHT_REVIEW 记录 93 份当前公开材料的定向文本检查无匹配，Git SSH 读取/LFS 可用，新分支/tag 尚未出现在远端，Release API 凭据当前不可用 | 当前仅 dry-run，缺最终 REPORT/MODEL_MANIFEST/PROMPT_AUDIT；尚未上传。最终须检查新二进制/zip 内容与最终 diff，完成 A/B 后实际取回实体字节；Release 回执能力按实际结果记录 |

发布修正验证命令：

```sh
python -m pytest tests/test_revisit_prompt_publication.py -q
python -m scripts.publish_revisit_prompt --dry-run --branch research/persist4d-mpr-v1-formal-20261014 --tag persist4d-mpr-v1-formal-20261014-v1
```

三项回归分别覆盖自定义发布身份、错误 origin 在写入前被拒绝，以及当前冻结推理源码的纳入与身份校验。该修正仅改发布脚本，训练 14 个数值文件和 V4 的 18 个推理文件身份保持原锁。

后台监控由 GPT-5.6 Luna 负责机械轮询；主代理保留错误判断、修复、科学结论和最终集成责任。2026-10-16 03:18 UTC Luna 实测 watcher/controller 均存活且无控制终端，M3-400 PB r0 为 332/516，无 error。实时状态以运行根的 monitor_luna/latest.json 与活进程核对为准。

主代理 2026-10-16 03:31 UTC 再次核对：watcher 3099815、控制器 2546831、评测子进程 3105331 均存活且无控制终端；03:30:53 心跳记录 M3-400 PB r0 为 405/516，较 03:27:53 的 391/516 持续推进，无 error。PRIMARY_IMPLEMENTATION_REVIEW.json 明确区分源码契约、既有实际执行证据与未完成项；没有启动新 GPU 作业或改写冻结的模型、数值源码、选点、预算与协议。最终部署验证可使用 P5 的 `EXACT_PARENT_ENCODER_DELTA_COVERAGE_AND_REAL_FORWARD` 实际完成记录，随后核对所打包 delta 与其源文件字节一致；目前不能把源码中的标记当作已完成的回载证据。

2026-10-16 04:01 UTC：M3-400 PB r0 已实际 COMPLETE。主代理核对 516 个不同输入、1806 个阶段、T2–T5 各 129 个输入、8 份总体及 48 份 reference 官方状态文件哈希，以及所加载 delta 身份；原收费事件仅结算一次，实际 1.273136423938332 GPUh。私有/公开账本 218 个唯一事件完全相同，新增累计 38.61483371920259 GPUh。控制器已自动切换 PB r1（PID 3141434），心跳进度 40/516；watcher/controller/child 存活且无控制终端。STATUS.json 保存该完成验证及实时快照；本轮已完成正式评测进程 85 个。单次 PB 完成不等于三重复确认完成，CAL 失败的复验门槛及全部未完成范围保持原判定。

## 2026-10-16 04:44 UTC — completed SEL/PB offline recomputation

CPU-only recomputation added all 36 official long/current states for M0 and locked M3, SEL/PB repeat0. Every six-metric dictionary exactly matches its original RESULTS record, and pooling the corresponding reference states gives the same dictionary. Together with the prior 20 locked CAL states, OFFLINE_METRIC_RECOMPUTATION.json now records 56 verified states. CUDA remained uninitialized and no GPU job was created. M4 confirmations, other repeats, and the final archive remain outside this completed scope.

A separate cProfile observation of the original M3 PB T5 pool reproduced all six metrics exactly: 26.198 seconds total, with 22.857 seconds in the official scalar precision/recall threshold loop. Any CPU acceleration must preserve the complete original bootstrap draws, formulas, NA handling and output values; the current controller and frozen sources remain unchanged.

2026-10-16 05:04 UTC：新增独立 CPU 统计模块及可选恢复入口；主代理核对原控制器与 V4 上下文的函数绑定顺序，限定加速仅在原 bootstrap 内生效。CPU_STATISTICS_ACCELERATION_VERIFICATION.json 记录 56 份总体状态及 reference 池化指标与原实现完全一致、220 条实际 PR 曲线逐元素精确一致，CUDA 未初始化。相同实际曲线参数的原循环耗时 26.086 秒、批处理 0.121 秒；这只是 PR 曲线局部计时，不是完整 bootstrap 加速比。新增入口及发布范围共 23 项测试通过；发布清单纳入两个新模块。未更改原训练/推理冻结源码、抽样规则、费用或锁，未启动新 GPU 作业，完整加速 bootstrap 与正式控制器接管均尚未执行。

2026-10-16 05:15 UTC：M3-400 PB r1 已真实完成并结算，516 个不同输入、1806 个阶段、T2–T5 各 129 个输入。SPARSE_PB_M3_VERIFICATION.json 核对前两次的指标、结构、原始预测指纹和 identity 诊断完全一致，并逐份检查 16 份总体及 96 份 reference 官方状态实体哈希；两个重复的对应状态字节相同。r1 实际费用 1.2452366792674971 GPUh，仅结算一次；公开账本与原账本 219 个唯一事件一致，新增实际累计 39.86007039847009 GPUh。本轮已完成正式评测 86 个；原控制器自动启动 PB r2（PID 3188365），STATUS 快照为 16/516，无错误，Luna 已续接只读监控。第三次 PB、M4 确认、bootstrap、P5、最终交付核对和真实发布仍待完成；CAL 复验门槛失败的科学判断保持原结论。

2026-10-16（M3 PB 三重复完成核查）：SPARSE_PB_M3_VERIFICATION.json 已扩展至全部三次独立进程，各 516 个不同输入、1806 个阶段；指标行、结构、原始预测指纹及 identity 诊断完全一致，24 份总体与 144 份 reference 官方状态实体哈希均已核对，对应状态字节完全相同。PB 的 S_long=0.06817758455872536，S_current=0.3445773224035899，单位 0–1，三次观察范围均为 0pp；有限重复一致不证明全局确定性。r2 实际费用 1.2620550535186097 GPUh，仅结算一次。公开/原账本 220 个唯一事件一致，新增实际累计 41.1221254519887 GPUh，含历史累计 1755.9361868676594 GPUh；原 RUN_LOCK、CAL 选点/快照及 V4 政策哈希保持原锁。本轮正式评测完成 87 个，原控制器自动进入 M4-200 SEL r0（PID 3228149），主代理写入快照时为 45/111、无错误，Luna 已续接 M4 的六个锁定确认任务。M4 确认、完整 bootstrap、P5、打包与最终核对/真实发布仍待完成；CPU 加速入口尚未接管控制器，科学门槛保持原负结果。

2026-10-16（M4 SEL 三重复完成核查）：SPARSE_SEL_M4_VERIFICATION.json 记录三次独立进程各 111 个不同输入、279 个阶段，实际 input_id/horizon/reference 集合及 reference 状态键与同角色 M0 完全相同；指标行、结构、原始预测指纹和 identity 诊断完全一致。主代理逐份核对 30 份总体和 204 份 reference 状态哈希；另用原标量官方池化接口独立 CPU 复算 repeat0 的 10 份总体状态及其 68 份 reference 状态，所有指标精确相同，耗时 31.305 秒，CUDA 未初始化。三次 S_long=0.07959490269422531、S_current=0.271793266137441，观察范围均为 0pp；相对锁定 M3-400 的描述性 SEL 差值分别为 +0.6604pp、−0.4597pp，不改 CAL 选点或失败的 seed147 门槛。三次分别收费 0.2257816890955534、0.21813176407611334、0.21880094746389658 GPUh，均仅结算一次；公开/原账本 223 个唯一事件一致，新增实际累计 41.784839852624266 GPUh，含历史累计 1756.598901268295 GPUh。正式评测已完成 90 个，原控制器进入 M4-200 PB r0（PID 3248718），写入 STATUS 快照时为 57/516、无错误，Luna 继续只读监控其余三个 PB 任务。完整 PB、bootstrap、P5、最终打包核对及真实发布仍待完成；CPU 加速控制器尚未部署。

2026-10-16 07:26 UTC：SEL 锁定对照 M3-400 与描述性候选 M4-200 的三轮结果均完成后，主代理启动独立 CPU bootstrap（PID 3257481，CUDA 不可见，2 个 CPU 线程），在原 V4 命名空间及已验证加速上下文内调用原 bootstrap_locked，保留 seed9101/1000 次成对物理 reference 抽样及 NA 规则。原 GPU 控制器 2546831 与 PB 子进程 3248718 继续运行，没有启动第二个 GPU 控制器。CPU 六份输入已验证，07:30 心跳记录 1083 次官方池化；完整区间仍未完成，PB bootstrap 尚未执行。CPU_STATISTICS_ACCELERATION_VERIFICATION/STATUS 仅记录 RUNNING，不据此关闭最终审计。Luna 已将该 CPU PID、心跳和池化计数纳入只读监控。
