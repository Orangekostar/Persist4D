# Persist4D Reviewer-Closure Design

## Objective

在不改变 Persist4D 或 ReScene4D 方法语义的前提下，依次排除三个审稿替代解释：

1. Full-History 的高 ID-switch 仅由缺少简单跨前缀关联造成；
2. Full-History 的长时域 task-quality 劣势仅由冻结 T2 checkpoint 造成；
3. 两个系统近似 t-mAP 的原因无法区分 association ceiling 与 perception ceiling。

实验必须遵循 Gate I、Gate II、Phase III 的顺序。LivingScenes/MORE2 只在 Phase III 得出
`ASSOCIATION_CEILING` 后进入兼容性实验。最终只允许 `FINAL_LOCK`、
`FINAL_PARETO_LOCK`、`ASSOCIATION_REOPEN` 或 `PERCEPTION_REOPEN`。

## Immutable Baseline

- 工作分支从 `b2414e3b2e89a990ee42a368caf6784eb27f8f01` 创建；其父提交
  `575acc12fbd63f38fc3c16578914b25c2fed8584` 是报告记录的 source commit。
- `artifacts/system_comparison/` 的基线 tree object 固定为
  `398fe87e1d40d67e61399fd893f02dc5f5f6b7ad`，本阶段不得修改。
- checkpoint 固定为 `repo:checkpoints/rescene4d_concerto_t2_repro.ckpt`，SHA256 为
  `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`。
- Protocol-B manifest SHA256 固定为
  `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`。
- 官方 ReScene4D source 固定为 `fb2fe42eb8f1e926567c48eea9acb874e608ee10`。
- 新产物统一写入 `artifacts/reviewer_closure/`；大张量 cache/sidecar entries 保持本地，
  只跟踪 manifest、哈希、统计、图表和报告。

## Verified Starting Facts

- Full-History 原 cache payload 只有 task/identity prediction、target、provenance 和
  observation fingerprints，不含 tracker 所需的 raw feature/class/mask observations。
- `postprocess_full_history_output` 已在内存中构造 feature、class probability、confidence、
  validity、当前 stage mask、mask support 和 local query ID，因此只需一次 sidecar inference，
  不重复生成或覆盖旧 task prediction cache。
- 官方 ReScene4D 的 non-parametric queries 从当前完整 forward input 通过 FPS 初始化；query
  index 是 per-forward namespace，不是失败的 tracker。
- Persist4D 是 local perception 之上的 persistent identity/state maintenance，memory 不反馈到
  ReScene query、attention、mask decoder 或 class decoder。

## Architecture

### Protocol And Provenance

`configs/reviewer_closure/protocol.yaml` 绑定 immutable baseline、统计设置、Gate 阈值和
sidecar schema。`scripts/reviewer_closure_protocol.py` 只复制既有 43 masters、6 个
reference-scene clusters、3 个 order 和 T2-T5 exact prefixes，不重新采样。所有 loader 都先
校验 source/content hash、prefix nesting 和 no-future 条件，再解码 payload。

### Full-History Observation Sidecar

`scripts/reviewer_closure_sidecar.py` 复用既有 dataset、collator、checkpoint loader 和
Full-History postprocess。schema `full_history_observations_v2` 每个 entry 包含：

- `features`, `class_prob`, `confidence`, `valid`；
- `current_stage_masks`, `mask_support`, `local_query_ids`；
- `reference_scene_id`, `master_sequence_id`, `order_id`, `horizon`, `scan_indices`；
- source prediction content SHA256、checkpoint/config/source commit hashes。

mask 的 query 轴必须与 feature/class/local-query 轴严格对齐；point 轴只覆盖 current stage。
写入使用内容寻址、临时文件加原子发布、拒绝覆盖不匹配条目。旧 Full-History cache 只读。

### Phase I: Reused Trivial Trackers

`scripts/reviewer_closure_tracking.py` 直接调用 `scripts/p6a_association.py` 中冻结的
`B1FeatureTracker`、`B2FeatureClassTracker`、`B3EmaTracker` 与既有 baseline runner，不复制
算法。论文产物仅使用 Pairwise Feature Association、Pairwise Feature-Class Association、
EMA Temporal Association 三个名称。

每个 master/order 独立 reset；每次 update 只消费该 prefix 的 current-stage observation；
tracker 不得读取 future prefix 或 GT。比较 Native Full-History、三个 tracker、Persist4D，
以及仅作诊断且明确标注的 Full-History + PersistentMemory。task metrics 继承冻结 cache；
deployment metrics 使用 tracker-issued IDs；任何重建的 stream task metric 单列报告。

Gate I 根据 T4/T5 identity-switch、gap recovery、bootstrap CI、order robustness 与 LOSO 输出：
`TRACKER_REJECTED` 或 `TRACKER_EXPLAINS_IDENTITY`。后一结果发布 challenge report，并停止
LivingScenes 之前的外部 matcher 论证；仍继续执行提示词要求的 Phase II/III 证据闭环。

### Phase II: One T3 Adaptation

先由 `scripts/reviewer_closure_training.py` 将官方与本地训练 recipe 的字段分为 known、
unknown、reconstructed 和 assumed，生成 `REScene_HORIZON_TRAINING_AUDIT.md`。仅当 dataset、
stage mapping、loss、初始化、optimizer、scheduler 和 update accounting 通过 smoke gate 后，
训练一个 T3 checkpoint。

优先 Level 1 同初始化/同 recipe T3；无法严格复现时使用显式命名的
`ReScene4D T2-to-T3 Horizon-Adapted`。训练数据 stage 必须恰为 `{0,1,2}`；T2 配置保持不变。
记录 batch/gradient accumulation、optimizer updates、scan exposures、GPU hours 和 checkpoint
hash。该 checkpoint 统一评估 T2-T5，并应用 Phase I 最强 tracker。

Gate II 输出 `HORIZON_ROBUST`、`FULL_HISTORY_DOMINANT` 或
`ACCURACY_ADVANTAGE_BUT_COSTLY`；不得因结果调整 recipe 或再训练 T4/T5 模型。

### Phase III: Performance Decomposition

`scripts/reviewer_closure_decomposition.py` 使用冻结预测和官方 `stmetrics` 语义完成：

- IoU thresholds 0.25 到 0.90、步长 0.05 的 temporal AP/recall sweep；
- 0.25/0.50/0.75 下的 no candidate、wrong class、insufficient IoU、associable coverage；
- 只替换 identity assignment 的 GT association Oracle；
- observation miss、class、high-IoU mask、fragmentation、merge、wrong gap、capacity、other 和
  unknown/unresolved 的可复算 failure decomposition。

Oracle 不改变 mask、class、feature、forward 或 observation population。综合 T2-T5 结果只输出
`ASSOCIATION_CEILING` 或 `PERCEPTION_CEILING`，并生成四面板解释图。

### Conditional Phase IV

只有 `ASSOCIATION_CEILING` 才 pin 官方 LivingScenes commit 并审计 sequential invariant、
`sim3_seq`、`eq_seq`、category scope、point support、coordinate frame 和 predicted-mask 输入。
先做官方 smoke，再做覆盖率；只在 supported categories 上用 predicted masks 运行外部 baseline，
不使用 GT masks，不自动接入 Persist4D。

## Statistics And Reporting

reference scene 是独立 cluster，数量固定为 6。每个核心差异报告 paired mean、relative
difference、95% paired cluster-bootstrap CI、order robustness 和六次 LOSO。主 task AP 继续用
pooled benchmark aggregation；cluster-macro 只作为 paired statistic，二者不得混写。

`FINAL_METHOD_LOCK_REPORT.md` 必须恰含提示词规定的 12 个章节。所有结果表、图和文字从
validated machine-readable artifacts 生成；未知、未运行或不适用值显式记录，不做挑选或补值。

## Failure Policy

- provenance、hash、schema、coverage、no-future 或 immutable-tree 校验失败即 fail closed；
- T3 smoke 不通过则不得开始正式训练；
- 单个样本失败必须保留 failure row，不得删除 horizon/order；
- baseline 环境缺失与实验回归分开记录；
- 任何 gate 外的新模块、超参搜索或结果导向复跑均禁止。

## Verification

按 TDD 增加 sidecar、tracker reuse、gap reappearance、T3 loader/forward-backward/reload、IoU
sweep、coverage 和 Oracle 三类 synthetic tests。每阶段运行新增测试和冻结 P6/system-comparison
回归；最终运行 full pytest、compile、diff/checksum、artifact verifier，并重新核对 immutable
system-comparison tree object。
