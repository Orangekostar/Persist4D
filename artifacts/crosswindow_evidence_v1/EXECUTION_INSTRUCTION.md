# Persist4D CrossWindow V1：最终 Codex 开发、选模、验证与发布指令

**版本：1.0（2026-09-20）｜性质：执行前方案，不是已完成实验或性能承诺。**

你是本任务的实现与实验负责人。按本文从头到尾完成：源码绑定、缓存同源重放、诊断、指定连接器、有限修订、条件性小模型、冻结评价、报告和 GitHub 发布。不要只输出计划，也不要在完成代码后停止而不执行已获授权的实验。性能没有提升时，仍需交付真实负结果。外部资产确实缺失或预算耗尽时，按本文状态机交付可恢复的部分结果，不编造数据，不换用弱基线填空。

本文完全替代 `Persist4D_CrossWindow_Scientific_Validation_Plan.md` 中尚未定量的执行条款；旧方案仅作来源。本文中的新算法、阈值、预算与选择规则均为**预注册工程设计**，不是论文已证明的最佳设置。实际旧分数仅作 provenance 检查，不能视为本轮重新测得的结果。

---

## 0. 任务核心与不可偷换的成功标准

### 0.1 先解决什么

主假设：冻结 R1 后，已有正确局部预测在跨窗口身份连接、分组评分和上一扫描修订过程中，可能损失一部分长时任务质量。先检验这部分损失是否可挽回，再决定是否需要训练。不能预先假定这就是唯一根因。

本轮的主要改动在 **R1 输出之后**，不改 ReScene backbone、decoder、loss、query 数或官方评价公式。保留当前网络已经输出的证据，统一比较：

1. 原 D-LAST 运行；
2. 同源、只修正确性的 D-LAST；
3. 同一身份决策贯穿状态与发布的 feature/class 对照；
4. overlap-priority；
5. feature/class/共享扫描 overlap 的共同分配；
6. 固定身份下的旧／新 mask 选择；
7. 仅在指定条件成立时训练小型关联纠错头。

### 0.2 三类完成状态分开

- **EXECUTION_COMPLETE**：所有必需步骤已执行，或符合本文的确定性条件并记作合法跳过，代码和结果已发布。
- **MECHANISM_SUPPORTED**：冻结候选下，可部署算法相对固定基线通过开发和冻结确认的增益检查；不是 GT-assisted 得分高，也不是冲突日志变少。
- **ALL_T_AND_RESOURCE_PASS**：同一最终系统在 T2/T3/T4/T5 的 t-mAP 均高于重新绑定的强本地 FH-R1-native，且通过第 14 节完整成本标准。

禁止把第一类写成第三类。纯连接在同源 T2 上可能只与一次双扫描原生候选等价；这时全 T 严格胜出未完成，不能更换缓存、分数规则或降低 T2 制造胜出。本轮**不自动追加新的 T2 分割网络**，未解决部分写入交接。

### 0.3 固定身份

```yaml
repository: Orangekostar/Persist4D
parent_commit: 96edca52d9d1cab7781bfa2a273baf9fe52b6a7d
parent_branch: research/persist4d-task-memory-retention-v2
new_branch: research/persist4d-crosswindow-evidence-v1
artifact_root: artifacts/crosswindow_evidence_v1
r1_checkpoint_sha256: 629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd
r1_checkpoint_bytes: 754813672
concerto_sha256: 845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07
window_scans: 2
query_budget: 100
resident_entity_capacity: 100
primary_policy: lag1
primary_reducer: mean
evaluation_seed: 45
horizons: [2, 3, 4, 5]
```

本地 FH-R1 不是作者未公开的 task checkpoint。原论文 34.8% 与本地 Protocol-B 分数不可直接比较。保留作者原生输出通道，不以 FH-CONT-lag1 的低分替换强基线。

### 0.4 禁止的扩张

不重训 R1；不恢复 V-CORE、Sonata、QCL 或新 backbone 搜索；不改 label255、EOS、objective；不下载大型外部权重；不移植完整 GTR/CLEAR/DEVA/DVIS++ 工程。新方法必须称为 inspired/本地适配，不声称完整复现。不得在最终 Protocol-B 结果上继续标定阈值或挑 checkpoint。

---

## 1. 证据与源码绑定：先读这些，不从历史聊天猜接口

### 1.1 已复核源码与具体任务

以下均绑定上述父提交。Git blob 是文件对象身份，不是仓库提交。读取时同时查看直接调用者；不要求遍历全仓。

| ID | 路径／符号 | 已确认行为 | 本轮行动 |
|---|---|---|---|
| R01 | `scripts/run_task_memory_controls.py::run_control_trajectory` | observation→route→commit，LAST/EMA；返回逐阶段 identity_map | 复现原 D 控制，作为 LEGACY 行 |
| R02 | 同文件 `_analyze` | supplement 提供 observation/prediction；base 提供 StageMeta/target；共享轨迹经两种 publisher | 建纯缓存入口；不因 analyze 强制加载 R1 和 pretrain |
| R03 | 同文件 `_prediction_overlap` | 比较新生成与 base 的候选来源、分数、mask | 不将两个 producer 的结果叫同源 |
| R04 | `scripts/task_memory_output.py::_resolve_identities` | routed ID 优先；overlap 冲突主要计数；部分候选用 fallback | 原逻辑只作基线；新逻辑只接受一个最终 AssignmentPlan |
| R05 | 同文件 `_iou_matrix/_align_masks` | 使用 source/target vertex IDs 重排 | E0 非恒等三循环测试；修正方向必须独立报告 |
| R06 | 同文件 `_archive_scan/_dense_from_archive/_materialize` | mask 打包，archive 恢复时使用 arange；按 PublishedIdentity 分列并对 occurrence scores 取 mean | 显式 canonical 点顺序；检查同 scan 同 key 的覆盖；不可静默丢候选 |
| R07 | 同文件 `LagOnePublisher` | 用当前窗口完整替换上一 buffer；更早 archive 冻结 | 保持 lag1 范围；E4 单独选择 mask |
| R08 | `models/task_memory_routing.py::PredictionObservation` | `[B,Q,D]` feature、class_prob、confidence、valid、current/previous_supported | DTO 必须保留这些实际字段 |
| R09 | 同文件 `EntityRoute/route_entities/commit_entities` | source-state commitment；匹配、出生、拒绝；只有 valid 且 current_supported 才更新 | 不偷偷修改已承诺 route；旧函数用于 LEGACY，新增小型统一状态器用于 A 系列 |
| R10 | `scripts/rescene_task_postprocess.py::OfficialTaskPrediction` | 完整窗口 bool mask、候选分数/类别、source_query/class IDs、latest mask | 不经 raw-query 简化重建候选；保留所有类别假设 |
| R11 | `scripts/analyze_persist4d_allt.py::AllTBaselineAccumulator` | 官方 temporal、published current、prefix legacy AP | 复用任务指标；raw-current AP 必须另算 |
| R12 | `scripts/system_comparison_metrics.py` | prediction 恰为三字段；target 为 masks/labels/ids/changes/temporal_stages；现通道 changes 要求为0 | 在适配边界剥离 lineage 元字段；遵守实际格式，不能捏造 ambiguity 字段 |
| R13 | `artifacts/task_memory_retention_v2/COMMANDS.md` | 原资产解析约定；PB 129 units 对应 115 unique cache files；dev 与 PB 目录不同 | 同一物理缓存可以被不同逻辑 unit 复用；每个 unit 独立重置状态 |
| R14 | `.../DATA_CONTRACT.json` | reference role 清单，包括 development/adaptation/additional_native_refs | 按 role 提取，先复核 exposure；名称独立不代表 R1-base 未见 |
| R15 | `.../FINAL_REPORT.md`、`.../HANDOFF.md`、`.../final/all_t_metrics.csv` | V2负结果；D额外645逻辑阶段前向未进D计时；单训练seed、历史PB曝光 | 旧结果只读；新结果必须绑定真实 producer 与成本范围 |
| R16 | `scripts/analyze_r1_downstream_validation.py::resolve_metric_dataset_spec` | 官方 dataset specification 解析 | 复用，不硬编码类别数或 IoU 忽略规则 |
| R17 | `scripts/run_task_memory_policy_baseline.py` 的 payload、meta、target helpers | controls 导入的真实缓存读写和 target 构造接口 | 运行时读其定义再绑定参数；共享 helper 但不写旧目录 |
| R18 | `scripts/profile_task_memory.py`、`scripts/r1_downstream_context.py` | 真实前向、R1加载、资源测量参考 | 抽取同硬件面板；不把缓存重放时间当模型更新速度 |

R01–R15 的重点行为已在本提示词准备中复核；R16–R18 是已确认调用的依赖，Codex 必须在实现相应步骤前读取定义。不要声称本文已逐行复核未展示的依赖全部实现。

### 1.2 本次审核发现的两个优先问题

**点重排方向：** 父版 `_resolve_identities` 把当前重复扫描的 IDs 作为 `source_vertex_ids`、旧 buffer IDs 作为 `target_vertex_ids` 传入 `_iou_matrix`；后者却用该重排索引重排 `prior_masks`。在非恒等三循环例子中会把真正相同的两个 mask 算成零重叠。已做的是对源码表达式的独立数值复核，不是导入原仓库跑完整 GPU 数据；真实缓存是否触发、影响多少分尚未知。

```text
old_ids = [10,20,30], old_mask = [1,0,0]
new_ids = [20,30,10], new_mask = [0,0,1]
按调用方向复现旧表达式：IoU=0
先把new_mask从new_ids对齐到old_ids：IoU=1
```

E0 必须在运行端导入真实函数重现，再检查真实重复扫描 IDs。若实际顺序一直相同，则该错误在该缓存上的影响为零；不能把所有历史低分归给它。

**指标名称：** controls 的 `direct_current_AP` 实际由已发布 prefix 的 current-stage 切片得到，候选分组与轨迹分数已参与。新结果同时报告 `raw_current_AP` 与 `published_current_AP`，不得继续混称。

### 1.3 文献角色

- GTR：轨迹层汇总关联证据。这里只用冻结 R1 的 feature/class/overlap 构造证据，不复现 GTR 的二维 transformer。
- CLEAR：共同实体与部分匹配约束。固定旧身份时本轮问题可能与一次分配等价；等价时跳过独立求解器，不制造新模块。
- DEVA：对齐后选择分割假设。W=2 只有两份假设且没有第三份可靠证据时，不虚构多数共识；本轮明确实现 old/new/score 三种选择。
- DVIS++：不可靠关联的去噪训练。只有 E1/E2 的固定条件成立时训练一个小型打分头，不训练整个分割网络。

来源与锁定文件见第 18 节。

---

## 2. 工作树、资产和预算

### 2.1 工作树

1. 找到真正含 `models/rescene.py`、`scripts/run_task_memory_controls.py` 的仓库根；忽略提示文本中的外层 `paper5/.worktrees/...` 展示前缀。
2. `git remote get-url origin` 必须指向 `Orangekostar/Persist4D`；记录完整 parent SHA。
3. 使用新 worktree，从固定 parent 创建新分支。旧工作树有未提交改动时不 stash、不 reset，保持不动。
4. 新分支已存在时：仅在它包含固定 parent，且本轮合同相同情况下恢复；否则停止写该分支，记录 `BRANCH_CONFLICT`。不得 force-push、覆盖主分支或擅自换 parent。
5. 启动后即把本 MD 原样复制到新 artifact 根下 `EXECUTION_INSTRUCTION.md`，写一次 SHA256。

建议命令，路径由已发现的仓库根计算，不硬编码用户目录：

```bash
BASE=96edca52d9d1cab7781bfa2a273baf9fe52b6a7d
BRANCH=research/persist4d-crosswindow-evidence-v1
ROOT=$(git rev-parse --show-toplevel)
WT="$(dirname "$ROOT")/persist4d-crosswindow-evidence-v1"
git fetch origin "$BASE"
git worktree add -b "$BRANCH" "$WT" "$BASE"
cd "$WT"
```

恢复已有 worktree 时跳过创建命令，不使用 `-B`。

### 2.2 资产解析顺序

依次使用：显式 CLI→环境变量→旧工作树未跟踪的 `artifacts/task_memory_retention_v2/external_assets.local.json`→旧 `COMMANDS.md` 和已存在 manifest 所引用的局部根目录。只在上述已知根及其一级子目录内定位，禁止 `find /` 或扫描全部共享盘。

在**外部 run root**生成 `assets.local.json`，不提交包含私有机器路径的此文件。标准键：

```json
{
  "data_root": null,
  "rio_metadata": null,
  "r1_checkpoint": null,
  "concerto_pretrained": null,
  "metric_dataset_spec": null,
  "dev_base_cache_root": null,
  "dev_supplement_root": null,
  "pb_base_cache_root": null,
  "pb_supplement_root": null,
  "fh_native_cache_root": null,
  "train_observation_cache_root": null,
  "external_run_root": null
}
```

这些 null 不是让执行者猜路径的占位符：bootstrap 必须按解析顺序填实际值，并将不能解析的字段列为资产状态。纯缓存 E0–E4 不要求 checkpoint/数据原件存在；真实补推理和 E5 profiling 才要求权重与数据。可从旧结果工作树读取资产映射，但不能把新输出写到旧 run root。

关键入口 manifest：

- `artifacts/task_memory_retention_v2/evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json`
- `artifacts/task_memory_retention_v2/evaluation/M5/protocol_b/baseline/control_observation_manifest.json`
- 开发集合使用旧 `baseline/` 与其对应缓存，不能误用上面的正式 PB 缓存进行调参。
- 原生 FH 优先复用 `artifacts/r1_downstream_validation_v1` 所绑定的 FullHistory task payload；必须核对 scan/order/point/class lineage，而不只是同一个 R1 SHA。

每个大文件首次实际使用时校验一次，后续凭路径、大小、mtime 与校验记录复用；文件发生改变才重算。缺失缓存仅补缺失 key；115 unique files 与129 logical units不一致是已知去重结果，不自动判错。首次打开日志记录 logical unit→physical file 一对多关系。

### 2.3 固定资源预算

| 类别 | 上限 | 行为 |
|---|---:|---|
| 全轮 GPU 总量 | 32 GPU-hours | 不是时间承诺；按设备活跃时长累计 |
| 补推理／训练观察生产 | 22 GPU-hours | 只补缺失；每 producer 最多一次完整逻辑覆盖 |
| E6 小头训练 | 6 GPU-hours | 两臂合计；未获 gate 不占用 |
| 最终 profiling | 4 GPU-hours | 给强基线与最终方法预留，不能被训练挪用 |
| CPU 计算 | 256 core-hours，最多8 worker | evaluator按CPU内存限制分块，禁止数十份大mask同时复制 |
| 分析工作RAM | 48 GiB | 每次处理一个或少量逻辑序列，超限降并发 |
| 新增缓存 | 8 GiB（不含复用旧缓存） | bit-pack masks；不重复保存全仓历史输出 |
| calibration 配置 | 12 个指定非学习配置 | 不增加隐藏网格，不逐T选参数 |

超过子预算后该分支 `BUDGET_EXHAUSTED`，保留完成部分并继续可执行的报告和发布。不得把预算缩短后重新称作充分训练；不得杀死不属于本轮的作业。不要重复扫描 GPU 或轮询其他任务；本轮作业分配一次，失败时再查。

---

## 3. 统一数据结构与边界

### 3.1 无 GT 的运行时输入

新增 `models/crosswindow_state.py` 与 `scripts/crosswindow_cache.py`。运行时函数只接收：

- `PredictionObservation`（原实际字段）；
- `OfficialTaskPrediction`；
- `StageMeta` 的输入 lineage，绝对阶段在代码中从0计数；
- 原始 point/vertex IDs；
- 当前固定大小状态和上一扫描 buffer。

GT 留在 evaluator/diagnostics 独立模块，不能传入 `associate`、`commit`、`select_mask`。

### 3.2 候选账本

唯一键为 `(producer_id, episode_id, order_id, absolute_stage, source_query_id, source_class_id, candidate_index)`。不同前向中的 query_id 不能直接相等；同一 query 的多类别候选共用一个几何 query-group，却仍是独立类别假设，原 class/score/mask 必须各自保留。

query-group 的集合为：`observation.valid=True` 的 query 与所有导出候选涉及的 query 的并集。无导出候选但符合原状态有效性规则的 query 仍可维持状态；无状态资格但导出过候选的 query 不能被删掉。若同 query 多类别 mask 相同可共享底层位图，不多计为多份 overlap 证据；若不同，取第8节明确的 max 聚合，不隐式合并 mask。

每条 candidate slice 记录其 scan_id、原始 vertex IDs、source_window、原始分数。E2中输出切片集合由 fixed lag1/M-new预先确定；所有A方法的每个原始切片必须出现一次，唯一区别是实体分组。既不根据GT过滤，也不因K不足删除输出。

### 3.3 规范化点顺序

内部 canonical scan order 定义为该scan原始vertex ID升序，而不是假设当前tensor行号就是原始ID。mask、target、所有新旧候选都映射到同一 canonical order 后才比较。不要只对预测重排不对GT重排。

`align(mask, from_ids, to_ids)` 唯一定义：输出第j行等于原mask中 `from_ids == to_ids[j]` 的那一行。禁止用方向不明的通用 `source/target` 再重排另一侧。

`ArchivedScan` 如果只存位图、不存ID，必须保证写入前已经canonical；否则同时存该scan的ID向量并计入归档，不允许用arange伪装已经对齐。

### 3.4 新统一状态与提交

A 系列使用新增状态器，不能原地修改旧 EntityRoute 哈希后调用旧 commit。旧模块只用于 LEGACY 重放。新状态器复用其数值规则与最小必要张量，不扩展复杂安全框架。

- resident slot 最多K个；每个含 normalized feature、class_prob、confidence、active/age/last_seen、logical_id/generation。
- 单调 `next_logical_id` 为标量；所有new/unmatched query-group获得独立ID，不能共用“null ID”。slot与public ID不是一回事。
- 所有方法共用一个上一scan buffer，最多Q个group及其类别切片。buffer记录其当时已发布ID、只用于前向连接的当时raw mask、feature/class probability，不读取更早archive。
- anchors 是resident slots与上一buffer的实体ID并集；重复ID合为同一anchor，最多K+Q个。resident存在时feature取resident；仅buffer存在时取buffer当时feature。generation必须匹配。
- 已有anchor被匹配后，读取／更新／发布使用同一 ID；无第二个publisher独立fallback。旧archive的ID不改，不能建立无界alias表去改历史。
- 有效且current_supported的关联query刷新LAST；没有current支持不刷新last_seen。新生有free slot按confidence降序、query稳定序分配；无slot仍输出独立ID，标记nonresident，不隐藏假阳性。
- buffer-only对象以后变为current-valid且有空slot时，沿用其public ID进入slot，不重新命名。无slot则继续buffer内传播；一旦从buffer与resident均消失，不能凭历史archive找回。
- 不淘汰旧slot：K压力实验观察拒绝出生；若以后研究淘汰，另立实验，本轮不临时添加LRU。
- 每stage一次commit。同一实体每个scan最多一个query-group；多类别假设分别挂在 `(logical_id,generation,class_id)` 下。

如果某个算法生成重复 `(entity,scan,class)`，不允许 `_materialize` 以最后一条覆盖前一条。运行时先拒绝造成重复的低优先级关联，让该group用独立ID保留；记录冲突。这是科学正确性规则，不是删除候选的NMS。

### 3.5 不夸大“同源”

共同R1权重并不等于共同候选。W2与FH输入不同，T≥3预测本来可以不同。“同源”在W2关联对照中要求同一物理前向；FH则要求同权重、同protocol、同预处理约定与可核查的scan/vertex对应，不强迫T≥3候选相同。

---

## 4. 状态机和开发划分

### 4.1 主执行顺序

```text
BOOTSTRAP → E0 → E1 → E2 → E3-equivalence → E4
                                     ↓
                       E6（仅gate允许，仍在开发阶段）
                                     ↓
                    SELECT_AND_LOCK → E5-final → REPORT → PUBLISH
```

E5正式PB评价在所有选择和E6训练之后。开发阶段可做小面板成本预检，但不得把它叫最终确认，也不得查看本轮候选的PB分数后继续开发。

### 4.2 数据划分规则

1. 从旧DATA_CONTRACT读取 `role=development` 的references，与真实dev缓存交集绑定。若少于4个reference，标记 `INSUFFICIENT_DEV_REFERENCES`；可执行固定配置诊断，但不进行全网格或E6，不声称选模稳定。
2. 对dev reference按 `SHA256('crosswindow-v1:'+reference_id)` 升序排列。前 `floor(n/2)` 为 `DEV-CAL`，其余为 `DEV-SEL`。同一reference的全部masters/order不能跨集合。
3. 使用这些缓存本来具备的顺序；不为dev临时生成最有利顺序。记录其实际数量，不强制47这个历史数字。
4. `DEV-CAL` 用于E1、阈值标定、E6 checkpoint选择；`DEV-SEL` 只在固定候选名单上选择与检验，不在其上重新开网格。
5. PB的6 refs/43 masters/129 units是已曝光历史benchmark。最终只读冻结名单，不做新独立测试声明。原生额外references按旧 `additional_native_refs` 或有效native集合单独列，复核R1训练曝光；不把合同里旧的“INDEPENDENT”字符串当事实。
6. E6训练只用 `role=adaptation` 且不在DEV/PB/native确认名单的reference；至少8个才允许训练。新数据不足时按条件跳过，不从PB借样本。

### 4.3 状态枚举

阶段状态只能为 `PASS/FAIL/INCONCLUSIVE/SKIPPED_CONDITION/SKIPPED_EQUIVALENT/SKIPPED_NO_INDEPENDENT_EVIDENCE/BLOCKED_ASSET/BUDGET_EXHAUSTED`。每个skip必须写命中公式、数值、依赖文件。技术故障最多重试一次相同命令；若是可定位代码错误可修复再重跑受影响单元，并记录，不覆盖原失败日志。

---

## 5. E0：基线、点序与T2同前向等价

### E0-A：LEGACY复现

从D supplement读取observation/prediction，从它绑定的base读取StageMeta/target。独立的纯缓存runner直接复用 `run_control_trajectory` 和父版publisher。不调用旧 `run_controls(mode=analyze)` 的权重校验/加载前置逻辑。

先在DEV-CAL复现，再在最终锁定后PB复现。旧PB表列为 `HISTORICAL_ONLY`，不得把它输入选择函数。baseline重放误差允许每指标绝对差 `<=1e-6`（0–1单位）；超差输出逐候选/来源定位，不通过改threshold补回分数。

### E0-B：正确性修复只作独立基线

依次产生：

- `D-LEGACY-LAST/EMA`：原版精确重放。
- `D-INDEXFIX-LAST/EMA`：只修点序方向与归档canonical化，其他相同。
- `D-CANONICAL-LAST/EMA`：在INDEXFIX上修已实际触发的候选覆盖/同键静默覆盖；没有触发则与INDEXFIX相同。

每个fix都有小反例和真实trigger_count；如果真实输入没有触发，该表对应差值应为0。所有后续方法采用同一套正确数据与输出基础。后续的 `D0` 指固定 `D-CANONICAL-LAST`，不是退步的适配模型。

不改父版文件与旧产物。通过小型复制/适配函数和明确来源构建新基线，避免全局monkey patch旧模块。

### E0-C：同一次T2前向的两个出口

`OfficialTaskPrediction.prediction(latest_only=False)` 送官方T2评价；同一候选送lossless-lag1出口。对跨scan联合候选仅一一重命名；empty mask按官方规则处理；不改score ties。核对mask/class/score全集与AP。

不要求未经修复的旧publisher必须相同；若不同则列出它做了哪项操作，并保留差异，不强行修改分数。

### E0-D：权重×输出矩阵

若完整FH-CONT native payload已缓存且可合法解析，自动补 `R1/FH-CONT × native/lag1`。只有裁切后W2 cache时，不能从中还原FH全部历史。缺完整预测时，仅在剩余22 GPU-hour推理预算内补；否则 `BLOCKED_ASSET`或`BUDGET_EXHAUSTED`，不影响D主线继续。

原生FH始终使用完整prefix，不进入需要W2的旧publisher。lag1系统对照另列。两种输出条件不同，不能把其差值称为纯模型／训练效应。

### E0最少单元检查

1. 非恒等3循环与往返point对齐；2. query行重排后直接AP不变，保留ties处理；3. ID一一改名AP不变；4. 同query多class不增几何证据；5. 两个unmatched不共用ID；6. 同scan同entity/class不静默覆盖。

输出：`e0/source_parity.csv`、`e0/index_trigger_counts.csv`、`e0/t2_same_forward_parity.json`、`e0/checkpoint_policy_matrix.csv`、`e0/correctness_fix_ledger.json`。

---

## 6. E1：可改善空间与GT辅助诊断

### 6.1 实际候选池，禁止偷偷用未来版本

在评价prefix T时，严格采用lag1可输出的候选版本：scan s<T用到达s+1时可用的新版本；scan T用当前版本；只有允许的一阶段旧／新对照可替换s=T-1。每个中间snapshot仅使用当时已到达的预测，不能先运行到T5再给T2选最好的mask。

同时可输出两种池的覆盖率，必须分列：

- `POLICY_POOL`：上述实际生产切片；用于主诊断。
- `RELAXED_OLD_NEW_POOL`：各scan在其合法一阶段窗口内的旧／新候选并集；只说明额外选择机会，不代表现方法已实现。

不融合mask，不修class，不删false positives；min_region_size、ignore及类别映射沿官方dataset specification。当前CausalPrefix target只有 `masks/labels/ids/changes/temporal_stages`；没有额外ambiguity信息时标 `AMBIGUITY_METADATA_UNAVAILABLE`，主表照旧官方路径，候选级唯一身份诊断仅覆盖可明确辨识部分，不能猜测歧义分组。

### 6.2 候选支持覆盖

对GT实体g及其可见阶段集合V，诊断tau∈{0.25,0.5,0.75}：

`C_g = all(exists same-class candidate with IoU > tau for every s in V)`。

C=0是固定池纯重连无法修复的必要条件失败；C=1不是TP保证。报告额外错误阶段支持、多人竞争同候选、类别失败、候选齐全但没有完整预测轨迹等重叠标签，不把这些占比加成100%原因分解。

逐事件主键：`reference,master,order,prefix_T,gt_id,tau`。记录first_failure_stage、可见阶段数、gap、每阶段bestIoU/候选数/所选source、当前published轨迹IoU。`diag_complete_candidate_failure` 定义为C=1但没有任何同类别的已发布轨迹与g在官方对应门槛下匹配；它是错误事件诊断，不是t-mAP。

### 6.3 两个GT辅助连接，范围写死

**GT-ID-RELAXED（乐观、离线、非部署）**：对POLICY_POOL的每个scan、每个class，按IoU>0.5的边做带unmatched的最大总IoU一对一分配，将匹配候选按GT ID分组；剩余候选保留为各自唯一轨迹。局部score/class/mask不变，mean reducer不变。它允许不同scan的同源query切片被重新组织，因而不一定能由当前在线状态器实现；不称“严格上界”。

**GT-ID-LAG1（动作受限）**：沿时间因果运行D0。旧archive与resident ID不重命名；仅在当前/上一扫描的既定query-group→anchor候选边上，把预测打分换成诊断用GT一致性分数。anchor的GT-tag仅为离线诊断账本：由过去可见slice的同类别IoU>0.5匹配确定；历史证据指向多个GT则标ambiguous，不能强制继承。每次仍一对一、K=100、同出生/更新/拒绝规则，且不改mask/class/score。无法连接旧anchor的正确新目标仍新生，不无限回填history。此账本类放在diagnostics，不得被production import。

辅助mask选择：冻结一个完整身份轨迹，pairable旧／新mask按GT IoU选择；仅用于诊断，false positives保留，不能作为正常算法主表行。GT为空时对应IoU按官方规则处理；未pairable的按统一M-new处理。

### 6.4 E1晋级规则（均在DEV-CAL，0–1单位）

令 `G = mean_T4,T5(AP_GT-ID-LAG1 - AP_D0)`；令 `P = count(C=1且published失败)/count(published失败)`，主tau=0.5。

- 若G≥0.005，或P≥0.10且至少20个此类事件覆盖≥3个reference：`ASSOCIATION_HEADROOM_SUPPORTED`，执行全部12个E2预定配置。
- 若不满足，但诊断求解/歧义不足导致INCONCLUSIVE：只运行A1默认与A2默认、无网格、E6不授权；结果可说明是否值得后续，不强称空间为零。
- 若G<0.005、P<0.10，且≥20个完整可诊断失败事件、覆盖≥3refs且诊断实现通过检查：`MASK_OR_OTHER_DOMINANT`。仅执行A1/A2默认，E3保持数学审查；可做E4便宜选择，不做E6。
- 有效事件不足时 `INSUFFICIENT_EVENTS`，行为同INCONCLUSIVE。

这些阈值是控制开发投入的预注册门槛，不是统计显著性标准。GT辅助没涨分不等于数学证明关联无改进空间；日志保留排序与求解局限。

输出：`e1/candidate_coverage.csv`、`e1/failure_ledger.parquet`（环境无parquet可用csv.gz）、`e1/gt_assisted.csv`、`e1/gate.json`。

---

## 7. E2：统一接口，防止新route被publisher第二次推翻

新增：

```python
build_evidence(frame, state, previous_buffer) -> EvidenceBundle
associate(bundle, method_config) -> AssignmentPlan
commit_observation(frame, state, plan) -> next_state
publish(frame, plan, mask_selection, history_boundary) -> PublishedPrefix
```

`AssignmentPlan`至少含group→entity、group→slot或nonresident、unmatched原因、score/margin、来源边、new_id预约、generation、stage/version。immutable plan只创建一次；`commit`和`publish`都不得重新匹配。新publisher不调用旧 `_resolve_identities`。任务没有神经读取时，不必创建假的read阶段；以后接入read必须消费同一plan。

生产与GT诊断不同模块，production单元调用签名不含target。训练模型打分接口也不得接收GT IDs。

### 7.1 桥接对照不可省

A系列改变了身份决策组织、unmatched分配及buffer-only锚点处理，因此额外报告：

- `D0`：正确性修复后的旧D-LAST；
- `A0-U-default`：下节共同新接口，只有feature/class，无overlap。

D0与A0-U的差值单列为 `interface_and_assignment_change`，不能算成GTR/CLEAR的收益。所有A1/A2与A0-U共用生命周期、节点资格、候选保留、输出规则与计时边界。

---

## 8. E2算法完全规定：最多12个配置

### 8.1 证据数值

- `cos(q,k)` 用L2-normalized query与anchor feature，float32计算；零范数相似度定义为0并记录。
- `C(q,k)=dot(class_prob_q,class_prob_k)`，使用原observation同一概率空间。先核对非负、sum≤1+1e-5；**不偷偷重新归一化**。不含最终输出类别修改。
- 原始feature/class分数 `L=cos+0.25*C`，范围[-1,1.25]。
- 统一到[0,1]：`B=(L+1)/2.25`，仅防浮点溢出clip。旧0.5 threshold对应B=2/3。
- 共享scan overlap：对当前group的previous-mask与旧buffer group的current-mask，用canonical vertex ID计算IoU；同query多class使用max，不加和。不要求预测类别完全相同，类别相容性已由C提供；最终class候选不改变。
- 对实体anchor k，`O(q,k)` 是该实体在同scan的合法旧group与当前group的最大IoU。缺少任何一侧有效mask时 `has_overlap=False`，O存0供日志，**不是负证据**。
- 有overlap且并集非零时IoU=0是实测不重叠，与missing不同。相同scan重复预测不计为多次独立观察；本轮每anchor只用resident最新特征或buffer特征，不追加全历史特征库。

### 8.2 通用带unmatched的分配

n个query-group，m个真实anchor，加n个私有dummy列。最大化总surplus，真实边收益 `S-tau`，每个q自己的dummy收益0，别人的dummy收益负无穷。只接受严格 `S>tau` 的实边，一对一；未匹配仍保留输出。

确定性并列：排序group按稳定ledger键、anchor按logical_id/generation，使用仓库稳定assignment helper或验证过的SciPy实现；禁止通过改变公开score打破并列。若solver tie可能影响身份，记录并用固定局部tie规则解决；其仅内部代价微扰不得改变非tie排序。

### 8.3 A0-U：feature/class-only

`S=B`。tau∈{0.60, 2/3, 0.73}，共3配置。默认tau=2/3。

### 8.4 A1：overlap-priority

优先锁定同时满足以下条件的(q,k)：O≥theta，q与k互为O的唯一最大值，双方top1-top2≥0.10（无第二项取0），B>tau默认2/3。锁定顺序O降序、稳定ledger键。

对剩余q/k，用A0-U-default分配。missing overlap不参与锁定，也不被惩罚。theta∈{0.50,0.65,0.80}，共3配置；默认0.65。

### 8.5 A2：共同证据分配

`S = (1-lambda)*B + lambda*O`（有overlap）；`S=B`（missing）。lambda∈{0.25,0.50}，tau∈{0.60,2/3,0.73}，共6配置；默认lambda=.25,tau=2/3。

A0-U的lambda=0作为必要对照，不再重复创建A2-lambda0。禁止额外类别硬门禁、距离门禁、GT置信度或按T不同参数。

### 8.6 标定与选择执行

E1充分时12配置都在DEV-CAL全部可用masters上跑，复用已算IoU与原始feature矩阵。每family选一个配置，按第13节排序，合计最多3个进入DEV-SEL。固定D0与A0-U-default始终报告。E1不足时仅两个默认A1/A2进入DEV-SEL，无额外参数搜索。

同时输出真正断链/误合并事件，不把route_conflicts减少直接当准确率提高。

输出：`e2/calibration_grid.csv`、`e2/family_candidates.json`、`e2/dev_selection.csv`、`e2/assignment_events.csv`、`e2/bridge_baselines.csv`。

---

## 9. E3：一致性审查；本轮不伪造一个重复求解器

本轮W=2、只有上一scan可改mask、旧实体对应作为边界固定，不新增第二层可修订身份变量。在这个范围，固定旧assignment U 后：

`J(Z)=sum(q,k) Z_qk B_qk + lambda*sum(q,i,k) Z_qk U_ik O_qi`

等价于：

`J(Z)=sum(q,k) Z_qk [B_qk + lambda*sum(i) U_ik O_qi]`。

后一式就是带固定加分项的一对一assignment；部分匹配与dummy也不改变该归约。不要把它换成谱分解或MILP再称为新机制。

本阶段的**确定任务**：

1. 在 `e3/CONSISTENCY_EQUIVALENCE.md` 写明固定变量、约束与代数等价；用2–3个group穷举合法分配验证目标值一致（数值容差1e-12）。
2. 运行统一AssignmentPlan的结构检查：唯一实体对应、generation、commit一次、发布不再改ID、早期archive不被重写。
3. 输出 `e3/status.json: SKIPPED_EQUIVALENT`，原因是当前合同下没有独立于E2的非平凡优化。**这属于完成E3，不是漏做实验。**
4. 不在本轮临时扩大身份修订范围、增加长期图或训练图网络。那会是新的研究干预，需另立方案；不会因想凑一张CLEAR消融表而执行。

共同身份空间是约束语言，不是原创声明。若A2实际违反这些约束，修实现错误并重跑受影响单元；正确性修复不能包装为另一个模型。

---

## 10. E4：有限修订选择，评分与身份严格拆开

### 10.1 先固定完整的身份决策轨迹

对D0与DEV-SEL选出的最佳A方法，各缓存逐阶段AssignmentPlan。E4复用它们，不让mask选择改变下一步的association evidence或memory update；状态仍由同一raw observation按LAST更新。这是用来隔离mask作用的受控系统，不是声称获得全部联合优化。

对可修订scan=t-1，pair键为 `(resolved_entity,generation,class_id)`；若旧或新存在多个同键candidate，按第3节先处理身份冲突，不能临时用GT或分数删除。只在旧、新都非空且class完全一致时pair。

无法pair的处理对所有M方法相同：采用M-new的候选集合，即新版本新增候选保留；只有旧版本存在、而新版本缺失的候选不自动补回。记录这类潜在损失；本轮不通过union放宽候选数量。t=1无修订直接原样发布。

### 10.2 三种明确选择

- **M-new**：pair选新mask。
- **M-old**：pair选旧mask，但使用已经固定的新实体对应，不退回旧tracker。
- **M-score**：若new.score > old.score+1e-6选新，否则选旧；完全相同mask直接复用新版本。阈值不标定。

只选已有mask，不求并集／交集，不复制geometry到没有观察的scan。更早t-2及以前不变。

### 10.3 两种评分通道

- `MASK_ONLY_FIXED_SCORE`：pair两种mask共用new.score；unpaired采用M-new分数。identity与class相同，用于解释mask选择作用。
- `SYSTEM_SELECTED_SCORE`：pair使用实际选中版本的score；这是最终可部署选择的真实输出。记录score变化，不能把增益全归给mask。

两通道的轨迹reducer均是mean。每个scan/entity/class只贡献一次occurrence，不重复加权。

### 10.4 不虚构“共识”

W=2可用的是同scan的两份预测，没有第三份独立mask或已经验证的质量估计器。本轮 `M-consensus` 明确记 `SKIPPED_NO_INDEPENDENT_EVIDENCE`。不能将两份预测的自IoU、重复均值或M-score改名为DEVA完整共识。DEVA启发体现在对齐并选择假设，以及old/new/score控制。

### 10.5 组合和统计

`{D0, selected-A} × {M-new,M-old,M-score}`，两评分通道都输出，最多12组固定组合，无mask阈值搜索。

报告各tau=.25/.5/.75的弱→强、强→弱、无候选→有候选、旧only消失、新假前景，以及每GT轨迹的minIoU变化。GT用于日志，不参与选择。

最终系统只能从 `SYSTEM_SELECTED_SCORE` 行选择；若分数通道独占增益而mask指标无改进，声明 `RANKING_GAIN_ONLY` 或mixed，不称mask更好。

输出：`e4/association_revision_factorial.csv`、`e4/revision_events.csv`、`e4/content_choice_status.json`。

---

## 11. E6：条件性的轻量关联学习（发生在最终E5之前）

### 11.1 是否授权，规则固定

仅当全部满足：

1. E1为 `ASSOCIATION_HEADROOM_SUPPORTED`；
2. DEV-CAL 的最佳非学习连接相对GT-ID-LAG1仍有T4/T5均值差≥0.010；
3. 可获得至少8个隔离的adaptation reference、≥200正pair和≥600负pair，且无DEV/PB混入；
4. GPU训练剩余额度≥6小时、生产观察可在22小时预算余量内完成；
5. E0/E2身份与指标检查通过。

否则列出具体未满足项 `SKIPPED_CONDITION`，不训练，不在PB上补监督。旧坏decoder不作为本轮初始化；R1始终冻结。

### 11.2 数据与标签

先复用训练观察缓存；缺失时只从 `role=adaptation` 的reference按固定SHA排序取前16个，每个最多一条最长native前缀（最大T5），不得复制scan，最多80次stage前向。保存压缩特征与pair表，不存全分辨率重复预测库。

用冻结A2默认或已选A2在训练序列上因果rollout。GT-tag只供监督：历史anchor已累计支持冲突GT时标ambiguous并排除该pair，报告覆盖；清楚的query/anchor对应同一GT为正，不同为负。query与GT IoU≤.5、ignore、无唯一标签的pair不伪装成负样本，另记unlabeled。候选生成和runtime状态都不用GT。

### 11.3 唯一网络与训练日程

十维输入：`[B, cos, C, O_or_0, has_overlap, clipped_log_age, q_conf, anchor_conf, overlap_top1_margin, B_top1_margin]`。age只来自到达阶段与last_seen；`clipped_log_age=min(1,log1p(age)/log(6))`；不存在second-best取0。特征归一化统计只用train；统计量、网络dtype=float32和clip方式写入head checkpoint。margin在全体同stage候选上先计算，禁止每对输入单独重估top1。

网络 `Linear(10,32) → ReLU → Linear(32,1)`，最后层零初始化。修正 `S_head=clip(S_A2 + 0.1*tanh(head(x)),0,1)`，沿用选定tau、assignment/lifecycle/publisher不变。不要添加位置硬门禁。

两臂：

- `L-PLAIN`：正常训练pair；
- `L-DENOISE`：同数据、网络、初始化、batch和步数；每batch确定性选25%样本，使用同类别另一真实实体的feature/class摘要替换anchor输入，保留该slot的真实身份标签，模拟旧状态污染。重算相关feature/class证据；overlap来源仍为原实际scan，不能为了伪造错误修改GT。仅行置换不算扰动。

两臂都使用1正:3负的256-pair batch（重复采样仅允许train），BCEWithLogits的logit为 `(S_head-tau)/0.1`；AdamW lr=1e-3、weight_decay=1e-4、grad_clip=1；1000 optimizer updates，前100 warmup，随后cosine到1e-5；seed45，共同初始化与采样顺序。保存0/250/500/750/1000和可恢复last。

DEV-CAL每250评价同一完整因果序列；选择规则第13节。每臂只有一个CAL选定checkpoint进入DEV-SEL。不得在不足1000步时因预算结果差而改称已收敛；预算耗尽标状态。训练过的头不获晋级时回退已冻结的非学习候选，仍上传负结果。

它是DVIS++-inspired错误扰动实验，不是DVIS++ tracker复现。

输出：`e6/train_pairs_manifest.json`、`e6/learning_curves.csv`、`e6/selection.json`、`e6/status.json`；权重外部存储、Git只放SHA/大小/逻辑路径。

---

## 12. 统一指标：原始感知与已发布结果分开

使用 `resolve_metric_dataset_spec` 与原 `AllTBaselineAccumulator`。生产prediction给validator只保留 `pred_masks/pred_scores/pred_classes`；lineage放旁表。target维度 `[G,N]`，prediction masks `[N,Candidates]`，不要转反。

必报任务指标：`t_mAP,t_mAP50,t_mAP25,t_REC,prefix_overall_mAP`。

另外必须同时报：

- `raw_current_AP`：从同一R1 OfficialTaskPrediction最新scan slice，保留原局部score/class，不经轨迹分组；用官方current-stage后端；相同生产器下所有连接器此值应一致。
- `published_current_AP`：已发布prefix切至当前scan，包含分组后的trajectory score；原AllT的`local_current_AP`映射到这里，不再叫direct raw。

官方主表指标仍按旧spec计算；E1用的tau和事件账本不代替它。若原始候选AP随纯关联方法改变，检查是否误从已发布对象读分数或候选。

所有结果行至少记录：population_id、data_role、method、config_id、source_commit、R1_SHA、head_SHA或null、producer_id、policy、mask_selector、score_mode、K、reference/master/order（汇总行注明all）、T、指标、status、reason。不可把不同范围/不同来源行无标签拼接。各指标以0–1存CSV，用户报告再乘100，百分点=差值×100。空分母写null并给denominator=0，不能用0伪装“失败率为零”。

---

## 13. 选模、保护固定起点与锁定

### 13.1 同一系统的定义

最终配置包含：R1_SHA、producer/config、关联family及参数、K、buffer上限、maskselector、scoremode/reducer、outputpolicy、小头SHA或null。T2–T5只能使用这一套。原R1 SHA不变仍要锁定新增模型head SHA。

### 13.2 排序规则

所有dev比较先计算相对**固定D0**的四维delta，不对比一个随训练退步的对照：

`S_min=min_T delta_T`；`S_long=mean(delta_T4,delta_T5)`；`S_mean=mean_T delta_T`。

CAL family挑选按 `(S_min,S_long,S_mean)` 降序，绝对差≤1e-6视为tie；tie取配置编号小者。固定A0/default与D0一直保留，不把每T最佳行拼成“虚构基线”。

DEV-SEL晋级需要：

- 每T相对D0下降不超过0.001（0.1pp）；
- T4/T5均值增益≥0.005（0.5pp）；
- 全T均值增益≥0；
- 至少3个DEV-SEL reference的T4/T5平均delta为正；若SEL少于3refs，则跳过这一稳定性门槛但候选只能记作PROVISIONAL_SMALL_REFERENCE_COUNT，不能记稳定PASS；
- 同比错误合并率与错误重激活率各不得增加超过0.01绝对率；分母为可核验GT事件，少于20个只标INCONCLUSIVE，不用这一项自动通过；由程序列出事件不足标志并按下段PROVISIONAL规则处理，不新增人工选参。

若事件分母不足或SEL reference少于3个，但数值增益与固定起点条件通过，分别标 `PROVISIONAL_SMALL_EVENT_COUNT` / `PROVISIONAL_SMALL_REFERENCE_COUNT`，仍可作为固定探索候选进入最终表，但最终机制稳定性保持未确认。

E4系统组合与E6最终head还要同时守住自己的固定父系统：所有T≥parent-0.001，S_long_vs_parent≥0；避免“比其他继续训练臂退得少”被当成新增能力。否则保留父方案。

满足者按 `(S_min,S_long,S_mean)` 排序，tie优先无训练、较少操作、固定配置ID小者。无候选晋级时，最终主体保持D0，同时把预注册排序最好的失败候选作为负结果陪同评价，不升级称best method。

### 13.3 锁定产物

在任何新方法PB打分之前写入 `selection/FINAL_LOCK.json`，含配置、输入来源、headSHA、数据roles、CAL/SEL结果、候选是否通过、计划最终对照列表和evaluator身份。产生git提交C并记录其SHA。后续若发现正确性bug，只能生成CORRECTION记录、受影响结果作废并全部重算；不能以PB分数不佳作为改配置理由。

最终表最少：native FH-R1、D-LEGACY、D0、A0-U-default、选定连接器（即使不晋级标明）、最终系统、必要父对照。只评价已锁定名单，不在PB评价全部12配置来挑最好。

---

## 14. E5：冻结后的完整确认、事件与成本

### 14.1 数据与精度

PB按真实manifest完整覆盖129逻辑units，每个从空state运行五次并输出T2–T5。共用unique物理cache不意味着共用state。缺任何关键unit时主表标INCOMPLETE并列覆盖，不能用不同子集分别算不同方法。

保留native FH，另外可列FH-lag1作为系统输出政策诊断，但主胜出判断对native。若相同T2前向尚未完成parity，禁用“纯关联超过原生T2”声明。

按gt可见gap=0/1/≥2、同类干扰数、首次出现、失联再现，报告events及delta。主曲线保持共同母序列cohort。新增native集合只在有真实缓存/预算时全覆盖其冻结名单，无则标NOT_RUN；不得换顺序增加“独立场景数”。

相关性单位是reference：主报逐reference差值、leave-one-reference-out的重算pooledAP。默认不做10000次bootstrap；若需要区间，固定1000次reference重采样且每次重算真正pooledAP，不把平均场景AP冒名。资源预算不足时只报LOO，不编区间。

### 14.2 容量

锁定方法与D0测试K={16,32,100}，保持其余配置。默认拒绝新resident但仍输出独立nonresident候选；不增加隐蔽secondary bank。报告peak occupied、拒绝出生、失联重新出现时仍可用的entity数及task质量。全未饱和就标capacity-uninformative，不宣称鲁棒淘汰。

### 14.3 Profiling

同一台A40，若无A40可用则选择现有同一GPU测试两侧并明确硬件差异，不与旧A40时间混比。固定面板为每PB reference按canonical master stable key选第一条。每方法完整T1→T5跑2个warmup sequence和5个measured sequence，重置状态，方法顺序固定交替。

至少测：FH-R1-native、D0、最终候选；若最终=D0，另测最佳失败候选供解释。必需实际网络前向，缓存replay速度单列不能替代。

四个scope：

1. `model_update_ms`：计实际forward、必须的device/host转换、关联、状态commit、上一scan修订、归档写入，排除GT指标；
2. `output_materialization_ms`：把已存prefix变成完整输出的代价；旧_materialize若每stage执行，全算进此scope，不隐藏；
3. `end_to_end_ms`：输入准备/传输＋1＋2，条件一致，IO缓存冷热明确；
4. `cumulative_ms[T]`：同一真实measured序列中从1到T的逐步时间和，不能拿不同序列的中位数拼接。

GPU同步在计时边界；记录absolute allocated/reserved peak、CPU RSS、resident字节、buffer字节、archive与materialized输出字节。只读archive为评估物化不等于runtime读历史；算法查询archive进行关联则计为违反bounded工作状态。

成本通过标准：T4/T5的面板median model_update与真实cumulative各比FH-R1-native低至少5%；absolute allocated peak不高于FH；端到端不高于FH；实体/短缓冲结构尺寸不随T隐性追加。任一未测则RESOURCE=UNCONFIRMED，不能因state小就PASS。短时变慢照实报告不隐藏。

### 14.4 最终目标

数值严格全T胜出：对FH-R1-native每T的t_mAP差>1e-6（0–1单位，仅去浮点噪声，不是显著性声明）。其他20任务格单独计数；raw_current_AP不得拿来代替prefix overall。

`RETENTION` 同时报绝对A(T)、A(T)/A(2)、A(2)-A(T)。不允许仅降低T2换平坦；长期质量保持没有预先指定90%之类成功线，本轮如实报告相对FH的保留率和绝对差，不能只选更好的一种。

`JOINT_GOAL_PASS = TMAP_ALL_T_PASS and RESOURCE_PASS`。所有统计稳定性/真正新reference结论另列；单seed与历史已曝光PB不能声称普适。

---

## 15. 代码文件、统一CLI与可恢复执行

### 15.1 必需新文件

| 新文件 | 职责 |
|---|---|
| `configs/crosswindow_evidence_v1.yaml` | 从第0/2/4/8/11/13/14节生成冻结参数 |
| `scripts/crosswindow_campaign.py` | 唯一主CLI、状态机、预算、恢复、报告与发布 |
| `scripts/crosswindow_cache.py` | 资产解析、base/supplement读取、canonical ledger、按unit迭代 |
| `models/crosswindow_state.py` | 新统一state、buffer、AssignmentPlan和单次commit |
| `models/overlap_entity_association.py` | A0-U/A1/A2分数和唯一分配 |
| `scripts/replay_crosswindow_association.py` | 旧D与新A重放；纯缓存不加载网络 |
| `scripts/diagnose_crosswindow_failures.py` | coverage、GT-assisted、事件日志；不得被production导入 |
| `scripts/evaluate_crosswindow_consensus.py` | E4三种选择与双评分通道；名称沿原计划，报告不称完整共识 |
| `scripts/profile_crosswindow.py` | 真实共同面板的四范围计时 |
| `models/crosswindow_score_head.py`、`scripts/train_crosswindow_score.py` | 只有E6授权才实现/执行；未授权可无此文件 |
| `tests/test_crosswindow_core.py`、`tests/test_crosswindow_metrics.py` | 必要功能与指标回归 |

可把小型私有helper放同文件，避免为了架构整洁再拆十几层。旧接口/旧结果不得改写。确需修改共享helper时必须保持原行为默认不变，并独立记录correctness patch，不能全仓无关重构。

### 15.2 CLI合同（这些命令需要你实现，不是声称父仓库已存在）

所有子命令均支持 `--config`、`--external-root`、`--resume`；`--external-root`省略时只能采用bootstrap已解析的assets.local.json值，无法解析即BLOCKED，不使用随意临时路径；bootstrap另支持`--assets-from`；preflight支持`--read-only`（仅禁止改动旧资产/旧结果，仍可写新审核日志）；某资源缺失输出structured状态而不是伪造成功。默认配置路径固定，无需用户再次填研究参数。

```bash
python -m scripts.crosswindow_campaign bootstrap --config configs/crosswindow_evidence_v1.yaml
python -m scripts.crosswindow_campaign preflight --config configs/crosswindow_evidence_v1.yaml --read-only
python -m scripts.crosswindow_campaign run --config configs/crosswindow_evidence_v1.yaml --through E4 --resume
python -m scripts.crosswindow_campaign conditional-train --config configs/crosswindow_evidence_v1.yaml --resume
python -m scripts.crosswindow_campaign lock --config configs/crosswindow_evidence_v1.yaml
python -m scripts.crosswindow_campaign confirm --config configs/crosswindow_evidence_v1.yaml --resume
python -m scripts.crosswindow_campaign report --config configs/crosswindow_evidence_v1.yaml
python -m scripts.crosswindow_campaign publish --config configs/crosswindow_evidence_v1.yaml
```

优先在已有`persist4d`环境执行；可用`conda run -n persist4d python`替换python。不得重新安装整套CUDA或降级用户主环境。需要轻量依赖时只在现环境可兼容范围安装并记录；不引入Gurobi/大型MILP依赖（本轮E3已等价跳过）。

阶段完成即原子保存 `RUN_STATE.json`：parent/instruction/config/data hash、已完成units、方法、已消耗资源、当前阶段、下一精确命令、失败说明。`--resume`只复用所有identity匹配的结果，不同配置创建新子目录，不能覆盖旧结果。

若`bootstrap`发现资产不足，先实现和执行可行的synthetic/unit部分，再生成BLOCKED报告并尝试GitHub发布；不要仅回复需要文件而停止全部交付。

---

## 16. 最小验证与执行前第二次审核

### 16.1 只验证八类直接风险

1. 原候选全集、class/score保留；query多class不多计；
2. 非恒等point三循环、往返与target共同对齐；
3. 原生T2同前向等价与raw_current_AP不随关联改变；
4. one-to-one、private null、generation、K满时不丢候选；
5. route→commit→publish一致，旧boundary不可改；
6. GT辅助数据与production隔离，合法prefix，不跨episode状态；
7. E3固定边界目标等价，E4只改指定mask/评分；
8. 恢复、选择锁定、预算记账、实际发布回读。

每类用一个或几个真实最小反例，复用相关旧测试一次；不追求测试数量。CPU小套件与ruff只跑改动路径；不跑全仓2000测试，不重复大checkpoint hash，不做无关权限/恶意输入/负载安全测试。

仅需要GPU时：2个真实DEV序列验证生产与cache字段、point lineage；E6需2个optimizer step梯度检查；E5按固定profile执行。不得“测试通过”替代科学结果。

### 16.2 Codex在正式实验前必须自行复审

输出 `PREFLIGHT_REVIEW.md`，逐项PASS/FAIL：源码对应、模型冻结、GT边界、候选保留、point方向、比较来源、缺失overlap、E3等价、E4评分、CAL/SEL/PB隔离、固定起点、预算、实际计时、发布循环。全部影响数值正确性的FAIL需先修复；纯外部资产项按BLOCKED，不无限重复审计。

首次preflight一次，代码发生相关改变只复审受影响项。正式过程中发现bug，记录旧结果INVALIDATED和精确受影响范围后重跑；不能静默修好后删去失败历史。

---

## 17. 产物、GitHub发布与最终回复（强制）

### 17.1 Git中的最小产物

artifact根必须包含：

- `EXECUTION_INSTRUCTION.md`、`EXPERIMENT_CONTRACT.json`、`DATA_ROLES.json`、`SOURCE_MANIFEST.json`；
- `PREFLIGHT_REVIEW.md`、`RUN_STATE.json`；
- 已执行各阶段的CSV/JSON与压缩小日志；大事件账本外部存储时提交汇总、SHA、大小和逻辑位置；
- `selection/FINAL_LOCK.json`；没有选模时填状态，不伪造文件缺省成功；
- `final/all_t_metrics.csv`、`final/identity_and_revision.csv`、`final/resources.csv`、`final/status.json`；无法计算的格写null和reason；
- `FINAL_REPORT.md`、`HANDOFF.md`、`FINAL_MANIFEST.json`、`COMMANDS.md`、`PUBLICATION.json`。

报告必须写：已做/没做、科学目标成败、最佳固定模型、所有负对照、同源与跨政策限制、数据曝光、预算和失败原因。GT-assisted单独表，不与可部署模型争冠。来源index记录本文已查的文件blob及本轮真正修改文件的commit；不把blob写成仓库commit。

`FINAL_MANIFEST`每项含repo相对path、bytes、sha256、role；**它自身不含自身hash**，receipt另记manifest hash。大ckpt/cache不入普通Git，表中的外部逻辑位置必须能由assets.local解析。

### 17.2 发布是必须执行的操作，不是建议

只stage本轮新代码/config/tests/artifacts及明确批准的最小shared patch。不得`git add .`掺入未检查的大文件/私有配置，不强推。对将提交的新增文本进行一次简短明显token/key检查即可，不做全盘secret扫描。

采用两段发布避免自身SHA循环：

**E：实验提交。** 包含代码、配置、冻结选择、数值结果和状态；生成并记录其SHA。正常`git push -u origin research/persist4d-crosswindow-evidence-v1`，用`git ls-remote`核验远端==E，再通过GitHub raw/API或已授权connector回读E上的主表。只在真实成功后写`RESULTS_PUSH_VERIFIED`。

**P：文档提交。** 生成FINAL_REPORT/HANDOFF/MANIFEST/COMMANDS/PUBLICATION，绑定E。文档中写“实验提交E已核验；本文档所在提交P由远端引用解析”，不写自己尚未发生的push成功。提交P，正常push，核验remote==P，回读P上的HANDOFF、MANIFEST、主表，比较实际字节SHA。

**外部receipt：** 在external root写 `PUBLICATION_RECEIPT.local.json`，记录E/P/remote SHA、回读hash、UTC时间、最终`PUSH_VERIFIED`，它不再提交，避免第三次无穷递归。最终用户回复给出P与回读摘要，即可真实声明发布完成。

如果接口回读无法访问但git推送成功：状态`PUSHED_READBACK_UNVERIFIED`，不能声称全部verified。鉴权失败最多一次重试；保存本轮可交付的git bundle（不含权重与cache）到外部root，并明确`PUBLISH_BLOCKED_AUTH`。不擅自更换GitHub远端或要求用户提供token明文。

### 17.3 交接文档必含

1. parent、code/results E、document P解析方法、分支；
2. 审核结论、实际修复及真实触发范围；
3. 输入缓存/权重/协议/数据角色；
4. 配置候选与CAL/SEL选择；
5. 已执行与合法跳过的E0–E6，每项证据；
6. 完整四T主表、20格与联合资源判断；
7. 失败假设、未解决T2或long-horizon问题；
8. 计算预算、外部资产逻辑路径、可恢复命令；
9. 必要测试与真实运行范围；
10. 发布状态与下一项**精确**工作，不默认复跑弱模型第二seed。

最终回复以表格概括结果，不长篇复制日志；必须报告真实P链接、是否全T胜出、是否成本胜出、有哪些步骤没运行及为什么。科学结果FAIL可以伴随EXECUTION_COMPLETE/PUSH_VERIFIED，但不能改写成项目成功。

---

## 18. 来源索引与适配边界

项目base：`https://github.com/Orangekostar/Persist4D/tree/96edca52d9d1cab7781bfa2a273baf9fe52b6a7d`

准备本指令实际读取的关键Git blobs：

| 文件 | Git blob |
|---|---|
| scripts/task_memory_output.py | 35341e1ebc08e8d4c4858b0b7481ac80def057b3 |
| models/task_memory_routing.py | 7a0e2dcb6d40fc81a58ff8d8c23bffaa71c5388b |
| scripts/run_task_memory_controls.py | ae43aa9540704af8b01da1962b55d8cae43e048c |
| scripts/rescene_task_postprocess.py | 2e068775f791e9684a78517bfb3aeb172f8b6c8b |
| scripts/analyze_persist4d_allt.py | 017effa10cc96cb931b7befeae05c4ff5e1b527f |
| scripts/system_comparison_metrics.py | 4512e6bfb9a45b6904742dbd69163556e163520d |
| artifacts/task_memory_retention_v2/COMMANDS.md | 0ffb0da003572a299392a8185c50ae72c17ff623 |
| artifacts/task_memory_retention_v2/DATA_CONTRACT.json | 1036484c6fdee70986312cd7ef3787cfe165b5ca |

文献：

- ReScene4D：`https://arxiv.org/html/2601.11508v2`。用于任务与t-IoU定义，不把跨数据集分数拼接。
- GTR：`https://arxiv.org/html/2203.13250` §4.3；代码 `https://github.com/xingyizhou/GTR/blob/master/gtr/modeling/meta_arch/gtr_rcnn.py`，曾核对blob `9a79d8d1939d09d4117e9317ef73c1f0ecd7dfb7`。本轮A2是证据汇总适配，不用其训练权重。
- CLEAR：`https://arxiv.org/html/1902.02256`；代码 `https://github.com/mit-acl/clear/blob/master/CLEAR_Python/CLEAR.py`，曾核对blob `3fff95f7032fd5c5f66a5de294f68a5f9ce605de`。本轮只使用部分匹配/固定边界推理，不复制其全图谱求解器。
- DEVA：`https://arxiv.org/html/2309.03903`；代码 `https://github.com/hkchengrex/Tracking-Anything-with-DEVA/blob/main/deva/inference/consensus_automatic.py`，曾核对blob `5bcd60df11a8822930b4c1fda315fe2782e6b905`。本轮只做旧/新假设选择，不声称实现原始半在线共识。
- DVIS++：`https://arxiv.org/html/2312.13305`；`https://github.com/zhang-tao-whu/DVIS_Plus`；曾核对 `DVIS_Plus/dvis_Plus/noiser.py` blob `b42f97f26480a3baadad106f51d30d7a83bacbef`。E6只借纠错扰动思想，不借用二维对象数量或连续运动假设。

默认分支地址会变化，真正复制源码时先取其实际commit、file blob与许可证并保留NOTICE。本文默认自行实现小型数学适配，不复制外部大工程，也不声称“首次提出共同身份/长期记忆”。

---

## 19. 最终执行检查表

- [ ] 新分支/worktree固定parent，旧产物未覆盖。
- [ ] 运行前合同、预算、数据roles、资产解析完成或明确BLOCKED。
- [ ] E0原版重放与正确性fix分表；point direction经过真实函数最小测试；raw/published-current分开。
- [ ] E1区分合法动作与放松诊断，未删除FP，未把GT-assisted写成上界或正式性能。
- [ ] E2仅执行指定配置；missing overlap不当负证据；query类别假设完整保留。
- [ ] E3等价证明与结构约束完成，无重复MILP/CLEAR大工程。
- [ ] E4固定identity、old/new/score与双评分通道明确；无假“共识”创新。
- [ ] E6按gate执行或跳过，发生在最终PB之前。
- [ ] 候选守住固定D0和父模型；FINAL_LOCK先于最终PB。
- [ ] 最终四T与成本按同一冻结系统报告，不拿弱FH-CONT替代native。
- [ ] 实际forward纳入成本；工作状态、输出归档各自计量。
- [ ] 所有负结果、合法skip、预算不足均可追溯。
- [ ] 代码、配置、必要测试、主表、FINAL_REPORT/HANDOFF同步到GitHub；P远端SHA和关键文件实际回读。

**最终原则：先验证已有正确信息是否被连接与修订破坏，再检验最小的预测驱动修复；不把模块数量、测试数量或GitHub推送次数当成科研进步。**
