# Persist4D P6-A / P6-B Scientific Validation & Method Upgrade Prompt

## 0. 你的角色

你现在不是“为了把数字调高而写代码”的工程代理，而是 **Persist4D 项目的实验负责人 + 科学协议执行者**。

当前仓库：

```text
Orangekostar/Persist4D
```

当前已知阶段：

```text
P0/P1  长时 ReScene4D 数据与资源审计
P2     官方/现有 ReScene checkpoint 复现
P5     Persist4D single-memory MVP
P6-A   科学协议、强 baseline、严格评估、误差审计
P6-B   基于 P6-A 证据进行 association/memory 优化
P7     双时尺度 persistent memory
P8     可选 memory-conditioned query adapter
```

本轮首先只允许完成：

```text
P6-A
```

只有通过本提示词定义的门禁后，才能进入：

```text
P6-B
```

禁止未经证据直接实现 P7/P8。

---

# 1. 当前研究事实

当前 P5 已经得到如下结果：

| Metric | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|
| Persist4D t-mAP | 23.932 | 15.872 | 12.622 | 3.562 |
| Persist4D t-REC | 30.557 | 18.574 | 17.272 | 11.581 |
| ID switches | 70 | 133 | 122 | 121 |
| Reactivation Acc. | N/A | 81.58% | 82.24% | 81.06% |
| Internal baseline t-mAP | 0.816 | 0.278 | 0.160 | 0.139 |
| Internal baseline ID switches | 660 | 1088 | 1118 | 866 |

当前最可靠的机制结论仅为：

> Persist4D 在不修改基础 ReScene 局部输出的前提下，通过固定容量的 persistent identity state 显著降低跨窗口 identity fragmentation，并能实现约 81% 的 gap reactivation。

当前 **不得声称**：

- SOTA；
- 优于强 temporal baseline；
- 所有 horizon 均提高严格在线 t-mAP/t-REC；
- capacity=100 已经被证明充分；
- Persist4D 总计算量优于 full-history ReScene；
- 当前 per-stage AP 等价于 raw local perception quality；
- 当前 t-mAP/t-REC 是严格在线指标。

---

# 2. P5 必须冻结

首先创建独立工作分支或 worktree，例如：

```text
persist4d-p6a
```

必须满足：

```text
P5 source/results remain immutable.
```

禁止覆盖：

```text
artifacts/P5/
```

禁止通过修改旧工件来“修正”历史结论。

P6-A 所有代码、配置、工件单独保存，例如：

```text
artifacts/P6A/
configs/p6a/
scripts/p6a_*.py
```

如果需要重构 evaluator，应保证 P5 legacy evaluation 仍可以完整复现。

---

# 3. 固定基础感知模型

本阶段：

```text
DO NOT retrain ReScene4D.
DO NOT change the checkpoint.
DO NOT change local masks/classes for different association baselines.
```

所有 association/state methods 必须使用：

```text
the exact same frozen ReScene4D predictions
```

至少绑定：

```text
checkpoint path
checkpoint SHA256
git commit
config SHA256
dataset manifest/hash
```

所有结果报告必须明确记录这些信息。

研究变量必须被严格限制为：

```text
cross-stage association / state maintenance
```

而不是基础 segmentation quality。

---

# 4. P6-A 目标

P6-A 必须回答以下五个科学问题：

```text
RQ1:
在严格相同场景、相同 prefix 条件下，horizon 增长是否真的导致 identity fragmentation？

RQ2:
当前 Persist4D 相比合理的 simple/strong temporal association baseline，
是否仍然具有明显优势？

RQ3:
Persist4D 剩余的长时错误主要来自：
local perception、
association、
wrong reactivation、
merge、
fragmentation，
还是 semantic drift？

RQ4:
当前 t-mAP/t-REC 中有多少信息使用了 future-stage track aggregation？
真正 strict-online 指标是多少？

RQ5:
Persist4D 当前性能瓶颈主要需要：
更好的 assignment，
更好的 dormant reactivation，
更好的 memory representation，
还是需要修改基础 query representation？
```

---

# 5. 第一任务：实现严格 Common-Prefix Protocol B

当前不同 T：

```text
T2 = 154 sequences
T3 = 120
T4 = 75
T5 = 43
```

不是同一组场景，不能直接将 T2 → T5 指标下降解释为 horizon degradation。

必须新增：

```text
Protocol B: exact common-prefix evaluation
```

## 5.1 构造原则

从具有完整 5-stage revisit 的同一条 master sequence：

```text
S1 S2 S3 S4 S5
```

构造：

```text
T2 = S1 S2
T3 = S1 S2 S3
T4 = S1 S2 S3 S4
T5 = S1 S2 S3 S4 S5
```

不得分别使用互不保证 prefix 一致的：

```text
sequence_database_sliding_2
sequence_database_sliding_3
sequence_database_sliding_4
sequence_database_sliding_5
```

优先使用 repository 已有显式 scan-index loader，例如：

```text
load_scan_indices(...)
```

保证 T2/T3/T4/T5 是完全相同 master sequence 的严格 prefix。

## 5.2 必须保存

```text
protocol_b_manifest.json
```

至少包含：

```text
reference_scene_id
master_sequence_id
scan_indices
T2 prefix
T3 prefix
T4 prefix
T5 prefix
visit order
source manifest
```

必须人工/程序验证：

```text
T2 == prefix(T3)
T3 == prefix(T4)
T4 == prefix(T5)
```

否则 P6-A FAIL。

---

# 6. 第二任务：重构 Baseline 体系

当前：

```text
local_query_index
```

只能保留为：

```text
Sanity baseline
```

不能再作为论文 strong baseline。

需要实现以下 baselines。

---

## B0 — No Association / Stage-Unique IDs

每个 query 使用：

```text
track_id = (stage_id, local_query_index)
```

禁止跨 stage 强行复用 local slot ID。

目的：

> 测量没有 temporal association 时的性能，而不引入 local-query-index false merge。

---

## B0-Sanity — Local Query Index

保留当前 baseline：

```text
track_id = local_query_index
```

只用于证明：

> local query slot 并不是 persistent identity。

正文中不得把它包装为 competitive baseline。

---

## B1 — Previous-Stage Feature Hungarian

只允许关联：

```text
stage t-1 observations
        ↓
stage t observations
```

score：

\[
s_{ij}^{feat}
=
\cos(f_i,f_j)
\]

使用 Hungarian / global optimal one-to-one matching。

超过 threshold 才匹配，否则：

```text
new birth
```

不能使用更早历史 memory。

---

## B2 — Previous-Stage Feature + Class Hungarian

score：

\[
s_{ij}
=
s_{ij}^{feat}
+
\lambda_c s_{ij}^{class}
\]

class similarity 建议避免 background/no-object 主导。

应实现 foreground-normalized class probability 版本：

\[
p^{fg}
=
\frac{p_{1:C}}
{\sum_{c=1}^{C}p_c+\epsilon}
\]

然后：

\[
s_{ij}^{class}
=
p_i^{fg\top}p_j^{fg}
\]

---

## B3 — EMA Temporal Association

维护简单 prototype：

\[
m_i^t
=
(1-\alpha)m_i^{t-1}
+
\alpha f_t
\]

但：

- 不实现 dormant lifecycle；
- 不实现复杂 persistent state；
- 不使用 dual-timescale memory。

这是必须打败的 strong naive memory baseline。

---

## B4 — Current Persist4D P5

严格冻结当前 P5：

```text
single persistent embedding
class state
occupied/active state
EMA update
fixed capacity
gap reactivation
```

本阶段不要修改它。

---

## Oracle — GT Association Diagnostic

只用于推理后的性能上界分析。

GT 绝对禁止进入：

```text
association
memory update
threshold selection
inference
```

GT oracle 只能回答：

> 如果 association 完美，当前 local observations 理论上还能达到什么水平？

必须清楚标记：

```text
diagnostic upper bound
```

不得与真实方法混为一谈。

---

# 7. 第三任务：修正/新增评估指标

当前 evaluator 存在一个关键语义问题：

```text
track class probability
```

使用了全序列平均后再回填到各 stage。

因此：

```text
current per-stage AP
```

不是 strict raw-local AP。

必须新增三套分离指标。

---

## 7.1 Raw Local Perception Metrics

直接在 memory / tracking 前评价 ReScene 当前 stage prediction。

定义：

```text
raw_local_AP
raw_local_AP50
raw_local_AP25
raw_local_REC
```

不得使用：

```text
persistent ID
future stages
track averaging
historical class posterior
```

对于所有 association baselines：

```text
raw_local_AP
```

理论上必须一致。

如果不一致：

```text
P6-A FAIL
```

因为说明 baseline 没有真正共享相同 local predictions。

---

## 7.2 Strict-Online Track Metrics

在时间 \(t\)：

\[
Prediction_t
=
f(O_1,\dots,O_t)
\]

禁止使用：

\[
O_{t+1},O_{t+2},...
\]

对于 track class/confidence：

必须维护：

```text
prefix-only running state
```

而不是：

```text
full-sequence average
```

新增：

```text
online_t-mAP
online_t-mAP50
online_t-mAP25
online_t-REC
```

所有 headline task metric 必须来自 strict-online version。

原有 full-track reconstruction 可以保留，但名称改为：

```text
offline_reconstructed_t-mAP
offline_reconstructed_t-REC
```

只作为 diagnostic / supplementary。

---

## 7.3 Identity Metrics

当前 raw ID switch count 不够。

新增：

### Normalized ID Switch Rate

定义：

\[
IDSW\ Rate
=
\frac{N_{switch}}
{N_{valid\ identity\ transition\ opportunities}}
\]

必须同时输出：

```text
ID switches
transition opportunities
ID switch rate
```

避免不同 T / 不同序列数量无法公平比较。

---

## 7.4 Reactivation Metrics

必须输出：

```text
gap opportunities
reactivation attempts
correct reactivations
wrong reactivations
reactivation accuracy
reactivation precision
reactivation recall
```

明确区分：

```text
no attempt
wrong reactivation
correct reactivation
```

不能只报：

```text
correct / attempted
```

如果没有同时给出覆盖率。

---

## 7.5 Relative Temporal Retention

新增一个诊断指标：

\[
Retention
=
\frac{online\ t\text{-}mAP}
{mean(raw\ local\ AP)}
\]

以及：

\[
REC\ Retention
=
\frac{online\ t\text{-}REC}
{mean(raw\ local\ REC)}
\]

目的：

> 衡量局部 perception 中已经存在的能力，有多少能够被长期 tracking/state maintenance 保留下来。

这是 diagnostic metric，不能取代 absolute t-mAP。

---

# 8. 第四任务：GT/Prediction 诊断匹配改为全局最优

当前 diagnostic GT/pred matching 使用：

```text
class-compatible
IoU threshold
greedy matching
```

必须增加：

```text
global Hungarian diagnostic matching
```

以：

\[
IoU
\]

或：

\[
1-IoU
\]

作为全局 assignment cost。

明确记录：

```text
matching rule
IoU threshold
class compatibility
tie-breaking
```

保留旧 greedy evaluator 作为 regression comparison，但论文主 diagnostic 使用 global optimal matching。

需要增加 unit tests，构造 greedy 会产生非最优结果的反例。

---

# 9. 第五任务：逐 Observation / Track Error Logging

这是本阶段最重要的分析工件之一。

每一次 association 至少保存：

```text
scene_id
sequence_id
stage_id
query_id
candidate_slot_id
GT_entity_id              # diagnosis only
association_correct       # diagnosis only

feature_similarity
class_similarity
total_score

best_score
second_best_score
score_margin

observation_confidence
mask_support
predicted_class
class_entropy

slot_age
last_seen_stage
gap_length
slot_active
slot_occupied

association_result
new_birth
reactivation
reactivation_correct
```

输出建议：

```text
artifacts/P6A/association_events.parquet
```

或：

```text
.csv
```

必须能根据该表完全重建：

```text
correct active matches
wrong active matches
correct reactivations
wrong reactivations
births
false births
ID switches
```

---

# 10. 第六任务：Error Decomposition

针对所有 strict-online failures，至少分为：

```text
F1 local perception miss
F2 association miss
F3 identity fragmentation
F4 identity merge
F5 wrong reactivation
F6 semantic drift
F7 capacity/birth failure
```

优先使用可执行、可重复的规则，而不是人工主观分类。

输出：

```text
error_breakdown_T2.csv
error_breakdown_T3.csv
error_breakdown_T4.csv
error_breakdown_T5.csv
```

并生成：

```text
failure_share_by_T
failure_share_by_method
```

重点回答：

> Persist4D 剩余错误中，有多少已经不是 association 问题，而是 local perception 本身不存在？

如果 GT object 在 raw local predictions 中根本没有可匹配 observation：

```text
Persist4D cannot be blamed for association failure.
```

必须把 perception ceiling 与 association failure 分开。

---

# 11. 第七任务：Reactivation Error Audit

目前 T5 的错误 reactivation 已经可能解释较大比例剩余 ID switches。

必须分别分析：

```text
correct reactivation
wrong reactivation
```

的：

```text
best score
score margin
feature similarity
class similarity
gap length
slot age
observation confidence
mask support
class entropy
```

至少输出：

```text
reactivation_score_distribution.csv
reactivation_margin_distribution.csv
reactivation_by_gap.csv
```

并回答：

```text
Q1: 错误 reactivation 是否主要集中在低 score？
Q2: 是否主要集中在低 margin？
Q3: gap 越长是否越容易误激活？
Q4: observation confidence 是否能区分正确/错误？
Q5: class compatibility 是否有效？
```

这一阶段只分析，不修改正式 P5。

---

# 12. 第八任务：Capacity / State Audit

当前：

```text
capacity = 100
rejected_births = 0
```

不足以证明 capacity=100 sufficient。

新增：

```text
birth_count
occupied_count_per_stage
active_count_per_stage
dormant_count_per_stage
peak_occupied
peak_active
peak_dormant
occupancy_ratio
rejected_births
```

必须先解除 evaluator 中：

```text
memory capacity == local query count
```

等不合理耦合。

各 baseline 的 identity namespace 必须独立于 Persist4D capacity。

完成解耦之后，P6-A 仅做 capacity observability。

不要立刻宣称：

```text
K=25/50/100 robustness
```

正式 capacity ablation 留到 P6-B / P7。

---

# 13. 第九任务：Efficiency Protocol 重做

当前：

```text
T2 total sequence latency
...
T5 total sequence latency
```

不能直接证明 incremental efficiency。

必须新增：

### Per-New-Visit Update Latency

对于 Persist4D：

```text
[X_{t-1}, X_t] + M_{t-1}
```

记录：

```text
latency per update
GPU peak memory per update
association overhead
memory update overhead
```

对于 ReScene full-history：

```text
[X_1, ..., X_t]
```

记录：

```text
latency per new visit
GPU peak memory
```

最终主要比较：

\[
Cost(t)
\]

随 history length 的增长趋势。

需要区分：

```text
total sequence runtime
incremental update runtime
persistent-state memory
GPU working memory
```

禁止用：

```text
63,808 bytes state
```

误导性地声称整个系统 memory constant。

正确表述只能是：

> persistent historical state size is bounded.

---

# 14. 第十任务：统计协议

不同 sequence 可能来自同一 reference scene，因此不能把所有 windows 当成独立样本。

必须保存：

```text
reference_scene_id
```

统计优先使用：

```text
cluster bootstrap by reference scene
```

而不是 naive window bootstrap。

至少输出：

```text
mean
std
95% cluster-bootstrap CI
```

对于 Persist4D vs baseline：

优先采用 paired analysis：

\[
\Delta_i
=
Metric_i^{Persist4D}
-
Metric_i^{Baseline}
\]

其中 pair 必须来自：

```text
same master sequence
same prefix
same frozen ReScene predictions
```

---

# 15. 顺序鲁棒性 / Seeds

注意：

如果模型本身完全 deterministic：

```text
random seed
```

不能伪装成模型不确定性。

应首先识别随机性来源。

如果存在：

```text
visit-order ambiguity
FPS randomness
sampling randomness
tie-breaking
```

则明确区分：

```text
algorithmic seed
sequence-order seed
```

如果真实数据不存在可靠 chronology，不得虚构真实时间顺序。

可以测试：

```text
3 deterministic admissible orderings
```

例如：

```text
canonical
reverse
seeded permutation
```

但必须明确说明它们是：

> order-robustness stress tests

而不是 ground-truth chronology。

---

# 16. P6-A 必须生成的主表

## Table A — Protocol-B Common Prefix

至少：

| Method | T | Raw AP | Online t-mAP | Online t-REC | IDSW Rate | React Acc |
|---|---:|---:|---:|---:|---:|---:|
| No Association | 2–5 | | | | | |
| Feature Hungarian | 2–5 | | | | | |
| Feature+Class | 2–5 | | | | | |
| EMA | 2–5 | | | | | |
| Persist4D | 2–5 | | | | | |

---

## Table B — Identity Mechanism Comparison

至少：

```text
ID switch rate
reactivation precision
reactivation recall
reactivation accuracy
fragmentation count
merge count
```

---

## Table C — Efficiency

至少：

```text
method
T
per-new-visit latency
GPU peak memory
persistent-state bytes
```

---

# 17. P6-A 必须生成的图

至少：

### Fig A

```text
x = T
y = normalized ID switch rate
```

所有 baseline + Persist4D。

---

### Fig B

```text
x = T
y = strict-online t-mAP
```

---

### Fig C

```text
reactivation correct vs wrong
score / margin distribution
```

---

### Fig D

```text
failure decomposition
```

---

### Fig E

```text
per-update latency vs T
```

---

# 18. P6-A 门禁

完成全部实验后必须自动生成：

```text
artifacts/P6A/P6A_GO_NOGO_REPORT.md
```

严禁只给“总体感觉不错”。

必须逐项判定。

---

## Gate G6A-1 — Strong-baseline identity advantage

至少对最强 simple baseline：

```text
EMA / Feature+Class temporal baseline
```

在 common-prefix T4/T5 上：

\[
IDSWRate_{Persist4D}
<
IDSWRate_{StrongBaseline}
\]

要求：

```text
paired relative reduction >= 20%
```

并且 cluster-bootstrap：

```text
95% CI 不应明显支持 baseline 更好
```

若无法达到：

```text
P6-A FAIL for persistent identity claim
```

必须先分析而不是直接写论文。

---

## Gate G6A-2 — Reactivation value

T3–T5：

```text
reactivation accuracy >= 70%
```

且相对 strong baseline 有明确提升。

同时必须报告：

```text
reactivation coverage/recall
```

如果通过极端保守阈值达到高 accuracy、但几乎从不 re-activate：

```text
FAIL
```

---

## Gate G6A-3 — Local perception invariance

所有 association methods：

```text
Raw Local AP
```

必须一致到数值误差范围。

如果不一致：

```text
FAIL
```

说明实验没有真正控制 perception。

---

## Gate G6A-4 — Strict-online task retention

不要求所有 T 都显著提升。

但至少要求：

```text
T2 不出现明显灾难性退化
```

以及：

```text
T4 或 T5 至少一个 long horizon
```

相对 strong baseline：

```text
online t-REC or online t-mAP > baseline
```

如果 Persist4D 只降低 ID switches，却 strict-online task metric 全面恶化：

```text
需要重新定位方法或进入 P6-B 修正
```

不能直接宣称整体任务性能更好。

---

## Gate G6A-5 — Error explainability

必须能将：

```text
>= 90%
```

的主要 identity/task failures 归入明确 error categories。

否则：

```text
P6-A analysis incomplete
```

---

# 19. 只有 P6-A GO 后，才进入 P6-B

P6-B 的目标不是堆大模型，而是：

> 根据 P6-A 诊断，优先修复最明确的 association / reactivation failure。

推荐顺序严格如下。

---

# 20. P6-B-1：Threshold-Aware Global Assignment

当前逻辑：

```text
global assignment
↓
threshold
```

存在 assignment optimum 与 accepted-match optimum 不一致的问题。

实现至少一种正确方案：

### Option A

在 assignment 前：

```text
score < threshold
```

设置为 forbidden edge。

### Option B

增加 dummy unmatched nodes。

目标：

> assignment 本身同时优化 matching 与 unmatched decision。

必须添加反例 unit test，例如：

\[
S=
\begin{bmatrix}
0.99 & 0.74\\
0.73 & 0.49
\end{bmatrix}
\]

\[
\tau=0.50
\]

要求新算法不会因为 0.49 占据 assignment 而损失：

```text
0.74 + 0.73
```

两个合法匹配。

---

# 21. P6-B-2：Dormant-Specific Reactivation Threshold

当前 active/dormant 基本使用同一匹配规则。

新增：

\[
\tau_{active}
\]

和：

\[
\tau_{react}
\]

要求：

\[
\tau_{react}\geq\tau_{active}
\]

并允许：

\[
score_{best}
-
score_{second}
>
\delta_{react}
\]

validation sweep：

```text
tau_active
tau_react
margin
```

禁止在 final test subset 上循环选择最优参数。

---

# 22. P6-B-3：Foreground Class Compatibility

将 background/no-object 从 class compatibility 中剔除。

比较：

```text
full-class dot product
foreground-normalized dot product
```

只在 validation 选择。

---

# 23. P6-B-4：Confidence-Gated Consolidation

不要每次 association 都永久更新 memory。

至少比较：

### Full update

当前方法。

### Confidence gated

只有：

\[
confidence>\tau_c
\]

且：

\[
association\ margin>\delta
\]

时：

\[
memory \leftarrow update
\]

否则：

```text
track matched
but persistent anchor not consolidated
```

目标：

> 防止一次低质量 observation 污染长期 memory。

---

# 24. P6-B-5：Birth Quality Gate

当前：

```text
minimum_mask_support = 1
```

可能过宽。

根据 P6-A 诊断只在 validation sweep：

```text
confidence threshold
minimum mask support
class entropy
```

目标：

```text
reduce false persistent births
```

但必须同时观察 recall，不能通过简单拒绝大量 observations 让 IDSW 看起来更漂亮。

---

# 25. 参数调优纪律

允许：

```text
aggressive validation tuning
grid search
random search
multi-objective tuning
```

这是合法的。

可以优化：

\[
Objective
=
w_1\cdot IDSWRate
-w_2\cdot ReactAcc
-w_3\cdot OnlineREC
\]

或 Pareto analysis。

但必须：

```text
validation -> tune
freeze -> final evaluation
```

禁止：

```text
test -> inspect -> tune -> test -> inspect -> tune
```

禁止：

```text
挑 seed
删 failure scenes
只报告最好 run
```

必须保存所有 sweep：

```text
hyperparameter_sweep.csv
```

并记录：

```text
selection rule
selected configuration
selection split
```

---

# 26. P7 何时允许开始

只有在以下事实成立后：

```text
single-memory Persist4D
```

确实相对强 baseline 有价值，但错误分析表明：

```text
memory contamination
appearance adaptation
long-gap drift
```

仍是核心问题，才能进入 P7。

P7 计划：

```text
Dual-Timescale Persistent Entity Memory
```

---

## Identity Anchor

慢更新：

\[
A_t
\]

负责：

```text
stable long-term identity
```

---

## Working Memory

快更新：

\[
H_t
\]

负责：

```text
recent appearance adaptation
```

---

## Association

例如：

\[
S
=
\lambda_A S(A,q)
+
\lambda_H S(H,q)
+
\lambda_C S_{class}
\]

---

## Consolidation

只有高可信 observation 才更新 anchor：

\[
A_t
=
(1-\alpha_A)A_{t-1}
+
\alpha_A q_t
\]

工作 memory：

\[
H_t
=
(1-\alpha_H)H_{t-1}
+
\alpha_H q_t
\]

其中：

\[
\alpha_H>\alpha_A
\]

P7 必须做 ablation：

```text
single EMA
anchor only
working only
anchor + working
anchor + working + gated consolidation
```

---

# 27. Query Adapter 暂时禁止

当前阶段不要实现：

```text
memory-conditioned query adapter
cross-attention into ReScene decoder
query refinement
```

除非：

1. P6/P7 已证明 association/state mechanism 本身成立；
2. strict-online t-mAP 仍明显受到 local representation ceiling 限制；
3. Oracle association 证明即使 perfect association，task performance 仍有明显可提升空间；
4. 已有证据表明 post-hoc memory 达到上限。

否则 Query Adapter 会破坏当前最干净的科学控制变量：

> identical frozen local predictions.

---

# 28. 不允许出现的科研行为

严禁：

```text
挑最好 seed 作为主结果
删除不利 scene
根据 test 调阈值
只报告有利 horizon
把 offline reconstruction 当 strict online
把 local_query_index 称为 strong baseline
把 63,808 bytes 说成整个系统固定内存
把不同 scene set T2→T5 直接解释为 horizon degradation
把 GT diagnostic 信息用于 inference
```

允许且鼓励：

```text
validation tuning
paired evaluation
common-scene protocol
normalized metrics
relative retention
careful metric design
error decomposition
good visualization
confidence interval
ablation
failure analysis
```

原则：

> 可以尽最大努力让方法真正更强，也可以选择最能表达贡献的合理 primary metrics，但所有结论必须能够由完整可复现的实验协议支持。

---

# 29. 工件要求

P6-A 至少生成：

```text
artifacts/P6A/
├── P6A_GO_NOGO_REPORT.md
├── protocol_b_manifest.json
├── baseline_results.csv
├── strict_online_results.csv
├── raw_local_results.csv
├── per_sequence_results.csv
├── association_events.csv/parquet
├── error_breakdown.csv
├── reactivation_audit.csv
├── capacity_audit.csv
├── efficiency_results.csv
├── statistical_analysis.md
├── figures/
└── configs/
```

P6-B：

```text
artifacts/P6B/
├── P6B_GO_NOGO_REPORT.md
├── assignment_ablation.csv
├── reactivation_threshold_sweep.csv
├── consolidation_ablation.csv
├── birth_gate_sweep.csv
├── selected_config.yaml
└── figures/
```

---

# 30. 最终报告格式

每阶段报告必须使用：

```text
1. What was changed
2. Why it was changed
3. Experimental protocol
4. Reproducibility binding
5. Main results
6. Statistical evidence
7. Failure analysis
8. What claims are supported
9. What claims are NOT supported
10. GO / NO-GO decision
11. Exact next action
```

不要只给：

```text
“效果不错”
“基本成功”
“可以继续”
```

所有 GO 必须有 quantitative reason。

---

# 31. 当前执行顺序

严格按照：

```text
Step 1
Freeze P5 and create P6-A branch/worktree

Step 2
Implement exact common-prefix Protocol B

Step 3
Decouple evaluator identity namespace from memory capacity

Step 4
Implement Raw Local metrics

Step 5
Implement strict-online metrics

Step 6
Implement B0/B1/B2/B3/B4 + Oracle

Step 7
Replace/augment diagnostic greedy matching with Hungarian

Step 8
Add normalized ID metrics and full reactivation diagnostics

Step 9
Add association event logging

Step 10
Run Protocol B T2/T3/T4/T5

Step 11
Run clustered paired statistics

Step 12
Perform error decomposition

Step 13
Perform capacity/state audit

Step 14
Perform per-update efficiency audit

Step 15
Generate P6A_GO_NOGO_REPORT.md
```

只有 P6-A GO：

```text
Step 16
Implement threshold-aware assignment

Step 17
Dormant-aware reactivation

Step 18
Foreground class compatibility

Step 19
Confidence-gated consolidation

Step 20
Validation sweeps + frozen final evaluation

Step 21
Generate P6B_GO_NOGO_REPORT.md
```

---

# 32. 你需要主动发现问题

不要机械执行。

如果发现：

```text
metric semantic mismatch
dataset leakage
scene duplication
invalid independence assumption
non-causal feature use
baseline unfairness
unintended future information
P5 regression
```

必须：

1. 暂停受影响结论；
2. 写入报告；
3. 添加最小 reproducible test；
4. 修复协议；
5. 重跑受影响实验。

不要因为“数字已经很好看”而忽略 protocol bug。

---

# 33. 本轮最重要的科学目标

最终不是为了证明：

> Persist4D beats a broken local-query baseline.

而是证明：

> Given identical frozen ReScene4D observations, simple pairwise association and single-timescale temporal smoothing are insufficient for maintaining long-horizon entity identity. A bounded persistent entity state provides measurable benefits in identity continuity and dormant-object reactivation, and the remaining errors can be separated into perception failures and memory/association failures.

如果 P6-A/P6-B 数据不支持这句话：

```text
do not force the claim.
```

重新分析。

如果支持：

后续 P7 才正式发展为：

> dual-timescale bounded persistent entity memory with confidence-gated consolidation.

---

# 34. 最后要求

执行过程中：

```text
Do not ask for confirmation for routine implementation decisions.
```

优先：

```text
inspect repository
inspect tests
inspect artifacts
make the smallest scientifically correct change
run tests
run controlled experiment
report evidence
```

所有已有测试必须保持通过。

任何新 evaluator / association 行为都必须补 unit tests。

最终请首先交付：

```text
P6A_GO_NOGO_REPORT.md
```

而不是直接进入新的模型设计。

**P6-A 的目标不是让数字漂亮，而是让我们第一次知道“哪些数字是真的、Persist4D 到底强在哪里、下一步该优化什么”。**