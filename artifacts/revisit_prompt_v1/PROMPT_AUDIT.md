# MPR-V1 最终提示词执行审阅

审阅执行者：主执行者。审阅完成不代表所有研究阶段完成。最终执行分支：BLOCKED_FULL_MATRIX_BUDGET；科学结论 INCONCLUSIVE；正式训练、资产与部分证据未完成。
权威原文 EXECUTION_INSTRUCTION.md SHA256：08cf2d2c83f1bd93d3120623f3ea9884fddd0f1fb13af843677b92160cc0070e。固定BASE=6b1b779172afdeb51e2719a737cd05e7eeeca982；当前数值实现C2=c880ba7912a9ac8e416ba4ed3ca98474111409fc。包级审阅另见 PROMPT_PACKAGE_REVIEW.md。

## 实际接口修订与验证边界

完整 CPU 聚焦基线26例通过，只执行一次；之后受影响修复轮1为3例、轮2为5例。未增加第三轮pytest或重复完整套件。原失败、命令、耗时和每轮身份保留在 FOCUSED_CPU_TEST.json/私有修订档案；原23个CLI help/dry-run结果属于其当时身份，未假称当前源新跑了整套。
原生目标 Int32 导致CE要求Long：保持GT语义并转CPU long；state byte计数为属性而非方法：统一显式计数含geometry sidecar；parent原输出没有MPR诊断字段：比较真正共有 logits/masks/query features；CAL/SEL缺role导致0输入：仅增加195条显式role，扫描/索引/顺序/asset绑定保持，711个spec解析核验。CPU修复轮仅对应前两次受影响行为；后续诊断/role修订通过有范围静态核验和当前源实际 P0/P1验证，未作额外pytest轮。
旧P1 0输入失败、旧P0修订结果全部归档并收费，不覆盖为成功。当前源重新完成3 M6更新、12 P0 probes、三次全CAL及六pilots。普通CUDA seeding调用曾被CPU测试截获；初始化改为CPU default_generator私有fork，CUDA调用为零，未声称修复GPU数值不重复。
输入窗口独立recenter：适配器记录纯输入XYZ平移，旧几何在reference坐标系保存，PE使用当前window平移。canonical只作顺序运输标签，不代表已知采集时间。段majority并列为UNKNOWN，所有达membership阈值identity可作ret正目标。
选择锁只接受N集合与四季度点；seed147使用固定main候选/对照家族、不重选家族、不改主锁，原selector15例及CLI锁定夹具通过。bootstrap保留无定义重采样NA，不重抽；实际锁定bootstrap尚未执行。
CPU saved-state核验最初错误假设CAL也有P0专用input_structure，KeyError发生在该次额外pool调用前；改为实际保存的scan/shape描述后完成全部30指标核验。OFFICIAL_REPOOL_CHECK记录此离线核验修订；未改数值源码、未新增CPU修复轮。

## §19 十二项内容审核

### 1. 首掩码前历史检索

状态：IMPLEMENTED / P0_P1_CONNECTIVITY_VERIFIED。

models/rescene.py before_first_mask 在第一个 mask head 前调用；models/revisit_prompt.py M4–M6 的 normalize(Wq/Wk) 全局 anchor + 当前 nearest32 检索，M2 明确 after_decoder_stage。CPU T01/T03 与实际 P0/P1 接通已完成；尚无正式机制收益。

### 2. 置信度门槛与新对象

状态：IMPLEMENTED / EFFICACY_NOT_ESTABLISHED。

pre-mask 输入仅当前 features/centroids 和预测历史，不读取当前 proposal 置信度/GT。max32 历史提示，100 queries 至少68新查询，recent16/dormant16 填充及空历史父路径由 T01/T02/T03 验证。真实 normal/normal 非逐位重复，GPU empty-parent 严格逐张量相等不成立，保留数值限制。

### 3. 时空与物理身份

状态：VERIFIED_WITH_SCOPE_LIMIT。

UUID episode 与 native loader 的局部/绝对时刻、当前 superpoint 和输入平移相符，T02/T06/T08 覆盖滚动窗口。P0 TRAIN 面板完整输入哈希三次一致；CAL 三次84输入的扫描顺序/点数/段数描述一致，未保存 CAL 全点张量哈希，不能声称后者已核验。

### 4. 父状态与身份

状态：EXACT_LOAD_VERIFIED。

实际 CPU 7变体严格覆盖345任务状态条目、453 encoder参数、0 encoder缓冲区，总798父状态；旧父SHA、encoder/taxonomy 与新增源码身份分别保留。untrained M4 debug delta CPU 全状态回载准确，未当作正式部署资产。IMPLEMENTATION_COMMIT C2 对齐全部14数值文件。

### 5. 真实梯度和等参数对照

状态：REAL_P0_P1_VERIFIED / NOT_FORMAL。

当前源 M6 P0三步+M1–M6各八步，全部新增张量改变且父摘要不变。M1/M4同为84224参数，实际共享初始化SHA和抽样SHA，M5也一致。P1_PILOT_AUDIT 和 P0_P1_TRAINING 给出逐项证据；51试跑更新不构成完整N适配或性能收益。

### 6. 对照和共同输出

状态：IMPLEMENTED / M0_CAL_ACTUALLY_REMEASURED。

M2 是重新适配的旧首轮hlevels末读取对照，不直接套旧M3/B4成绩。M0以本轮流式政策重新完成三次CAL，非旧表搬运。最终 route/commit 一次、commit0/mean、query lineage与未绑定候选保留统一；M6预定义支持乘法各一次由T07覆盖。

### 7. GT与UNKNOWN

状态：CPU_ISOLATION_VERIFIED / GPU_CAUSAL_CHECK_LIMITED。

推理/state/association 不消费GT；原class/GT first-point前缀语义仅loss，birth ledger独立可靠匹配。UNKNOWN(-1) 从 BCE 排除，缺席无ret负例，ambiguous/tied segment保留UNKNOWN。T03/T04/T08通过。GPU标签互换时normal/normal已非逐位相同，不能将其当作隔离成功或证明标签泄漏。

### 8. 固定配方与完整矩阵

状态：BUDGET_BLOCKED / NOT_EXECUTED。

P1使用预设lr/seed146和八步计时，未调参。N∈{400,800,1200,1600}及四季度点、六臂+最贵两臂147完整复验由成本/选择CPU用例约束。实际最小400不满足预算，RUN_LOCK N=null；无正式四点/147，不短训、不只训候选、不改锁突破。

### 9. 重复、失败、分母与暴露

状态：ACTUAL_P0_P1_REPORT_COMPLETE / FORMAL_STATISTICS_MISSING。

保留3×84 CAL实际输入、15指标行、168reference行、252identity行、全部46成本事件含9失败。TRAIN361、CAL8、SEL8、PB6物理参考及历史暴露明确，输入顺序不当独立场景。30真实官方state重池化逐项一致；无锁定候选，1000次paired bootstrap未执行、CI未知，不伪造。

### 10. 最佳点、方法与投稿结论

状态：CLAIMS_RESTRICTED。

candidate/control为null，无实际赢家；科学INCONCLUSIVE，原因预算阻塞，不是算法负效果。H1/H3未建立，H2移动子集不可用；不把有梯度/代码通过/AP分数重排说成补回掩码。不宣称方法胜出或CVPR就绪。

### 11. 总账与试跑边界

状态：ACTUAL_ACCOUNTING_VERIFIED。

历史1713.8830547291614及原ledger SHA保持，新增46唯一event共0.9310066865094472 GPUh含失败；8小时reserve未用。P0/P1 cap与78新池保持。正式0更新，实际当前源51个试跑更新/714训练stage单列，未混进正式训练CSV。预算缺口15.450310993为预测。

### 12. 资产、交接与远端字节

状态：PARTIAL / PUBLICATION_PENDING / FORMAL_MODELS_BLOCKED。

真实生成198官方充分状态ZIP，MODEL_MANIFEST绑定实体bytes/SHA、固定父Git/LFS位置、encoder/data取得来源。正式M1–M6 delta/resume 未产生，正式部署reload/真实前向未执行；仅CPU debug重建通过。publisher以A冻结资产且manifest排除自身/B/receipt，B交接引用A、不嵌入B自身SHA。发布前状态待验证；最终Git/LFS新空缓存下载事实在外部receipt，Release若无CLI/API凭据则如实LOCAL_ONLY，不能全PASS。

## 阶段与交付总核对

A定位与B实现完成，代码已在当前源P0/P1实际接通，但后续phase逻辑仅有有范围CPU覆盖，不能称全量生产验证。C最小真实接通完成且根因未解决；D真实成本测量/唯一预算阻塞判定完成。E–I因D准入失败未执行，§15停止分支要求转结果/交接/发布，不能机械填研究完成。
§17全部要求的路径均应存在，缺失阶段CSV/锁/模型目录注明NOT_EXECUTED/NOT_PRODUCED和原因；未知指标null，实际0完成事件仅写计数。CAL实际重复行保留，正式N未准入没有伪造四点行。部署目录仅状态README，实际充分状态ZIP198文件。报告逐项回答方法、政策、数据/暴露、种子、收益未知、数值、资源、H1–H3与唯一后续方向。
J在本文件提交时为PUBLICATION_PENDING：主审阅完成后实际publisher建立A/B、tag、推送并在一个新目录空LFS缓存取回本次资产/代码/交接逐项比较bytes/SHA。最终成功/失败范围只以外部 PUBLICATION_RECEIPT 记事实；不为改STATUS追加自引用提交，不把未产生的正式模型说成已上传。
结论：预算阻塞分支的报告与内容审阅完成；全量研究证据没有完成，不称方法失败/胜出或投稿就绪。唯一后续方向为新的完整矩阵预算准入，保留本次冻结结果。

发布准备第一次 staged diff 检查因新CSV使用CRLF中断，发生在A提交前；仅将10个CSV换行转为LF，解析记录转换前后完全一致。未修改数值代码、未重跑GPU或pytest，准备失败及修复记录在 PUBLICATION_PRECHECK/PUBLISH_ENTRY_STATUS。最终发布事实仍由外部receipt确认。
