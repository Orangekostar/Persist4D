# 正式轮交付核对进度

状态：未完成，不能据此设置 publication_ready 或 goal complete。

权威要求为 EXECUTION_INSTRUCTION.md（与原 ZIP 主指令字节一致，SHA256 `08cf2d2c83f1bd93d3120623f3ea9884fddd0f1fb13af843677b92160cc0070e`）；预算更新以 PROTOCOL_AMENDMENT.json/RUN_LOCK.json 为准。原已发布的预算阻塞轮及其标签保持原记录。

| 要求 | 已核查的实际证据 | 当前结论 / 剩余工作 |
|---|---|---|
| §3 固定起点与源码绑定 | SOURCE_BINDINGS.json 在本地核对 BASE 的 29 个 Git blob；INPUT_PACKAGE_VERIFICATION.json 核对 ZIP 全部五个原始 checksum，主指令字节一致，原 registry 字段完整保留；原静态审阅另存 PROMPT_REVIEW_SOURCE/INSTRUCTION_STATIC_AUDIT | 绑定完成；源码阅读范围保留原审阅边界，原静态 PASS 不等于实际执行完成 |
| §5 父加载、冻结、新参数覆盖 | PARAMETER_AND_LOAD_AUDIT.json 核对六条真实正式训练状态、六个最佳 delta 张量及其身份；24 个训练检查点哈希另见 MAIN_TRAINING_VERIFICATION；MODEL_PAYLOAD_REVIEW/MODEL_MANIFEST 已核对六个 delta 和两份 resume 的本地公开副本，共 3724164 bytes | M1/M4 均为 84224 参数且初始化一致；P5 profile 已实现严格回载及真实前向，待其实际完成后核对与公开副本的 SHA；完整状态包与远端字节仍待完成 |
| §11 六臂与成本 | M1–M6 seed146 均 400 更新；四点 100/200/300/400；原 RUN_LOCK 与累计账本保留 | 正式主训练完成，不追加涨点搜索 |
| §12–14 全量 CAL 与选点 | SPARSE_CAL_SELECTION_VERIFICATION.json：75 个完整独立进程、25 个三重复组、全部指标及结构/原始预测记录相等；375 个 CSV 行与实际结果核对；用纯选模函数复算冻结 lock 完全相等 | CAL 完成；有限一致性不等于全局确定性或旧 Native 根因已解决 |
| §14 seed147 条件复验 | CAL_SELECTION_LOCK / REPLICATION_LOCK | 未执行是规定分支：M4-200 相对 M3-400，长期均值 +0.7764pp，但 T3 −0.8040pp、T4 −1.4022pp，未达到逐长度 −0.2pp 门槛；最终科学状态应保留 SCREEN_NO_GO |
| §13–14 锁定 SEL/PB | 原 V4 控制器仍执行 M0、M3-400、M4-200 的三次评测 | M0、M3 SEL/PB 及 M4 SEL 三轮完成；M4 PB repeat0/1 完成且有限重复逐项一致，repeat2 运行；其余以实时 STATUS/RESULTS 为准，CSV 在最终 collect 后更新 |
| §13 统计、机制与资源 | OFFLINE_METRIC_RECOMPUTATION.json 及 M4 SEL/PB 核验文件共独立 CPU 复算 74 份总体状态及各自 reference 池化，指标精确一致；SEL 原规则 bootstrap 已实际完成 1000 次抽样/36000 次池化，元数据、六份输入/源码/输出哈希、三进程差值及全部保存抽样百分位经独立核对 | SEL 的两个 95% 差值区间均跨零，8 个 reference 的少场景警示保留；PB bootstrap/P5/最终档案仍待完成，不声称已第二次执行全部抽样或其他 repeat 的独立复算 |
| §17 模型与充分状态 | MODEL_PAYLOAD_REVIEW 已实际 CPU 读取六个选定 delta 与两个最终 resume，共 3724164 字节；身份、张量范围及候选/对照 optimizer/scheduler/RNG/400 步边界正确，无禁传字符串匹配；DEPENDENCY_ACQUISITION 核对 BASE 父 LFS 指针、106906589 字节父实体及 taxonomy；DEPLOYMENT_PROTOCOL 明确 V4 构造/前向上下文 | 八份本地公开模型及 LFS 对象已核对；原 summarize 的最终状态包/完整 manifest 与真实部署前向仍待完成。resume 保存最终 400 步边界，不冒充 CAL 最佳较早选点的 optimizer。部署文档两段 Python 仅 AST 检查，不冒充实际回载 |
| §17–19 报告与最终内容审核 | PRIMARY_IMPLEMENTATION_REVIEW.json 已记录主代理对十二问的源码和既有证据审阅；PROMPT_AUDIT.md 已逐题草拟执行结论，12 项编号、本地证据链接及 4 项锁定身份核对通过，训练 14 个及推理 18 个源码身份再次相符 | 审计仍为 IN_PROGRESS；完整 PB 确认、PB 统计、P5、全部交付文件及真实远端字节仍待核对；publication_ready 与 prompt_audit_execution_complete 均为 false；四种状态分开，不得自动宣布 CVPR 就绪 |
| §18 发布 | publish_revisit_prompt.py 已支持显式本轮 branch/tag、实际状态交接和冻结推理源码进入 A manifest；三项离线回归测试通过；PUBLICATION_PREFLIGHT_REVIEW 记录 93 份当前公开材料的定向文本检查无匹配，Git SSH 读取/LFS 可用，新分支/tag 尚未出现在远端，Release API 凭据当前不可用 | 当前仅 dry-run；REPORT 尚未生成，MODEL_MANIFEST 与 PROMPT_AUDIT 已有明确未完成的草稿，最终验收待补齐；尚未上传。最终须检查新二进制/zip 内容与最终 diff，完成 A/B 后实际取回实体字节；Release 回执能力按实际结果记录 |

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

2026-10-16（本地模型副本）：六个 CAL 最佳 delta 与 M3/M4 的 N=400 resume 已复制到公开目录；八份共 3724164 bytes，逐份实体字节数及 SHA256 与已审阅原件一致，路径均具有 LFS 属性。MODEL_MANIFEST 只列实际八份模型并绑定 DEPENDENCY_ACQUISITION，不列尚不存在的状态压缩包；MODEL_PAYLOAD_REVIEW 记录本地副本核验完成。真实 P5 部署前向、最终指标包和远端上传/取回仍未完成，publication_ready 保持 false。最终 summarize 会重写 MODEL_MANIFEST，届时必须保留或重新补全依赖取得信息。

2026-10-16（M4 PB 首轮核查）：SPARSE_PB_M4_VERIFICATION.json 记录 repeat0 的 516 个不同输入、1806 个阶段，请求/状态身份一致，实际 input/horizon/reference 集合及 reference 状态键与 M0 同角色一致；8 份总体、48 份 reference 状态实体哈希全部核对。另使用原标量官方实现独立 CPU 复算全部 8 份总体状态及其 reference 池化，六项指标精确相同，耗时 166.375 秒，CUDA 未初始化。首轮 S_long=0.07106386745969455、S_current=0.34513384103775024，相对锁定 M3 的描述性差值为 +0.2886pp/+0.0557pp，但 T2 AP 为 −0.3968pp；未据首轮声称跨重复一致或修改 CAL 失败门槛。首轮实际费用 1.2812949446408473 GPUh，仅结算一次；公开/原账本 224 个唯一事件一致，新增累计 43.066134797265114 GPUh，含历史累计 1757.8801962129357 GPUh。正式评测完成 91 个，原控制器进入 M4 PB repeat1（PID 3292195），repeat2 仍待执行；SEL CPU bootstrap 继续运行。三轮确认、完整统计、P5、最终指标包与实际远端发布仍待完成。

2026-10-16 09:35 UTC（完整 SEL bootstrap）：独立 CPU 进程 3257481 于 09:34:57 UTC 正常完成并退出，原规则 seed9101/1000 次物理 reference 成对抽样全部有效，36000 次池化实际耗时 7729.626 秒。SEL_BOOTSTRAP_EXECUTION_VERIFICATION.json 绑定六份实际输入、原 bootstrap/加速器/入口源码、两个相同输出的字节数和 SHA；主代理独立检查全部保存抽样的百分位、三进程差值及范围、NA 计数、8 个 reference 的少场景警示。长期差值 95% 区间为 [−0.584956, +1.240682]pp，当前差值为 [−0.706283, +0.688935]pp，均跨零，不能确认稳定收益。函数绑定已恢复，CUDA 未初始化，没有新增 GPU 作业或收费事件；正式 GPU 控制器仍执行 M4 PB，没有接管或重启。完整 PB 确认、PB bootstrap、P5、最终打包审核与实际发布仍待完成，最终验收标记仍为 false。

2026-10-16 09:46 UTC（M4 PB 前两轮完成）：主代理核对 repeat1 的 516 个不同输入、1806 个阶段、T2–T5 各 129 输入及一次费用结算；SPARSE_PB_M4_VERIFICATION.json 已覆盖两轮共 16 份总体、96 份 reference 状态实体 SHA，对应状态字节、指标、结构、原始预测指纹和 identity 诊断完全一致，观察范围均为 0pp。保留 repeat0 原标量 CPU 复算的完整记录，不冒充 repeat1 另做一次独立复算。r1 实际费用 1.3310640143999868 GPUh；公开/原账本 225 个唯一事件相同，新增实际累计 44.3971988116651 GPUh，含历史累计 1759.2112602273357 GPUh。正式评测完成 92/93，原控制器自动启动最后的 M4 PB repeat2（PID 3327543）；SEL bootstrap 已完成，PB bootstrap、P5、最终打包审核和实际发布仍待完成，CAL 负结果及最终验收标记保持原结论。

2026-10-16 10:18 UTC（精确 CPU 池化缓存）：主代理实现独立模块及其测试，先观察新增行为测试失败，再验证全部 33 项相关测试通过。实际 M3/M4 PB repeat0、T3–T5 的 long/current 共 12 组完整池化和 12 组固定抽样池化，全部六项指标精确相同；修改状态但保留其旧嵌入 SHA 时仍被原校验拒绝。缓存按完整输入顺序、实际 taxonomy 字节、源码和运行环境身份匹配，损坏缓存报错，后续只读消费者对缺项计算原官方池化。实际 12 组完整池化的首次计算/命中平均耗时为 0.2641/0.0181 秒，只是局部计时。独立 CPU 预热 PID 3344563 已启动，CUDA 不可见、不产生 GPU 作业或正式统计结果；只枚举现有 repeat0 状态以复用相同输入，原正式 bootstrap 仍需三轮真实结果及全部 1000 次抽样。CPU_POOL_CACHE_VERIFICATION.json 记录实际证据、源码和边界，发布清单纳入新模块；32 项冻结训练/推理源码及四份锁再次核对未变。最后一轮 GPU PB 继续运行，控制器接管、PB bootstrap、P5、最终打包审核和远端发布均待完成，科学失败门槛保持原结论。
