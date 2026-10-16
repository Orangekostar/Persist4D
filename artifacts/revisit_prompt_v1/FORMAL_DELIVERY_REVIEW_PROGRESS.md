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
| §13–14 锁定 SEL/PB | 原 V4 控制器仍执行 M0、M3-400、M4-200 的三次评测 | M0 SEL/PB 完成；M3 SEL 完成、PB 进行中；其余以实时 STATUS/RESULTS 为准，CSV 在最终 collect 后更新 |
| §13 统计、机制与资源 | OFFLINE_METRIC_RECOMPUTATION.json 实际 CPU 复算两个锁定 CAL 端点 repeat0 的 long/current T1–T5 共 20 份状态，全部指标及分别按 reference 重新池化均精确一致；原 run_campaign 后续包含 bootstrap 和 P5 | CAL 离线复算范围已验证，未覆盖最终 SEL/PB 与打包档案；继续等待真实 bootstrap/P5，并检查 NA、少场景边界、置换上下文及分母 |
| §17 模型与充分状态 | MODEL_PAYLOAD_REVIEW 已实际 CPU 读取六个选定 delta 与两个最终 resume，共 3724164 字节；身份、张量范围及候选/对照 optimizer/scheduler/RNG/400 步边界正确，无禁传字符串匹配；DEPENDENCY_ACQUISITION 核对 BASE 父 LFS 指针、106906589 字节父实体及 taxonomy；DEPLOYMENT_PROTOCOL 明确 V4 构造/前向上下文 | 原 summarize 尚待打包；须核对公开副本和真实部署前向。resume 保存最终 400 步边界，不冒充 CAL 最佳较早选点的 optimizer。部署文档两段 Python 仅 AST 检查，不冒充实际回载 |
| §17–19 报告与最终内容审核 | PRIMARY_IMPLEMENTATION_REVIEW.json 已记录主代理对十二问的源码和既有证据审阅，重新校验训练 14 个及推理 18 个源码身份 | 尚非最终 PROMPT_AUDIT；确认、统计、P5、全部交付文件及真实远端字节仍待核对；四种状态分开；不得自动宣布 CVPR 就绪 |
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
