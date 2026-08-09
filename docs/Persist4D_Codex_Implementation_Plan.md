# Persist4D / ReScene4D Long-Horizon Memory — Codex 执行任务书

> 目标：在 **ReScene4D 官方代码**基础上，验证并解决 temporally sparse full-scene 4DSIS 在历史长度增长时的可扩展性与身份稳定性问题。
> 核心原则：**不要重写 spatial backbone，不要先造复杂模型；先验证 limitation，再逐步增加 persistent memory。**

---

## 0. 项目最终研究命题

本项目不做“ReScene4D 把 T=2 改成 T=5”这种简单扩展。

目标问题是：

> ReScene4D 通过联合处理多个 temporal stages 获得短窗口的一致实例分割，但随着历史 observation 数量增加，完整历史 3D context 需要持续参与 joint reasoning。我们研究能否将历史压缩为固定容量的多实例 persistent memory，并在不显著损伤单次 3D instance segmentation 的前提下，在更长的 revisit horizon 上保持稳定 identity。

推荐论文暂定题目：

**Persist4D: Bounded Persistent Instance Memory for Long-Horizon 4D Scene Understanding**

禁止使用以下未经验证的 claim：

- “first causal 4D segmentation”
- “first persistent query memory”
- “ReScene4D memory complexity is exactly O(T)”（除非 profiling 后严格支持）
- “first to separate identity and state”
- “ReScene4D cannot run T>2”
- “3RScan is chronologically ordered”

更安全的关键词：

- streaming
- recurrent
- bounded-history
- persistent multi-instance memory
- temporally sparse revisits
- long-horizon identity consistency
- full-scene 4D instance segmentation

---

# 1. 不可更改的研究边界

## 1.1 保留并冻结的部分

优先复用 ReScene4D-C / Concerto 路线：

- 官方 3RScan preprocessing
- Concerto / PTv3 pretrained encoder
- 原始 ReScene4D Mask3D-style decoder
- 原始 mask / class prediction
- 原始 Hungarian matching / segmentation losses
- 官方 `stmetrics`
- 原始 t-mAP / t-mREC / per-stage AP evaluation

不要在第一阶段改：

- Concerto encoder
- voxel size
- spatial backbone architecture
- 2D/RGB modules
- open-vocabulary modules
- robotics/navigation modules

本项目只研究 temporal memory。

---

# 2. 第一阶段：先验证 ReScene4D 的真实限制

**任何新 Method 实现之前必须完成本阶段。**

如果本阶段无法证明存在可研究的 long-horizon bottleneck，则暂停整个 Persist4D 方法开发。

---

## 2.1 数据统计

现有本地文件：

```text
vv/dataset/3RScan/3RScan.json
```

已知全量 478 reference scenes 的访问长度统计：

```text
T = 2       : 194 scenes
T >= 3      : 284 scenes
T >= 4      : 124 scenes
T >= 5      : 56 scenes
T >= 6      : 25 scenes
```

Codex 新建：

```text
scripts/analyze_3rscan_temporal_distribution.py
```

要求输出：

### A. 按 split 统计

```text
train:
  T>=2
  T>=3
  T>=4
  T>=5
  T>=6

val:
  T>=2
  T>=3
  T>=4
  T>=5
  T>=6
```

### B. 每个 scene 输出

```json
{
  "reference_id": "...",
  "split": "train/val",
  "T": 4,
  "scan_ids": [...]
}
```

### C. 如果 metadata 中能可靠解析 change 信息，再统计

- static trajectories
- rigid changed trajectories
- non-rigid changed trajectories
- added/removed trajectories

**如果 metadata 不能可靠支持，不要猜。**

保存：

```text
artifacts/data_audit/3rscan_temporal_stats.json
artifacts/data_audit/3rscan_temporal_stats.md
```

---

## 2.2 使用官方 sequence builder 生成长窗口数据库

优先使用官方：

```text
datasets/preprocessing/build_rscan_sequence_db.py
```

分别构造：

```text
T=2
T=3
T=4
T=5
```

不要自己重新实现 sequence construction。

注意：

- 官方代码会随机排列 reference + rescans 后构建 sliding/cyclic windows。
- 因此当前项目称作 `streaming/recurrent`，不要把 stage index 解释成真实 chronological time。
- 固定官方 seed 作为主实验。
- 后续可增加多 permutation seed robustness，但不是 MVP。

检查实际生成文件名并记录到：

```text
artifacts/data_audit/sequence_db_manifest.json
```

---

## 2.3 修复 / 审计 T>2 change label

当前官方 loader 对 T>2 的 transition change labels 可能只保留第一列。

检查：

```text
datasets/semseg.py
```

针对类似：

```python
if changes.ndim == 2:
    changes = changes[:, 0]
```

的逻辑。

要求：

1. **主 t-mAP pipeline 不依赖 transition change label。**
2. 在未验证 annotation semantics 之前，不修改官方 change-conditioned evaluation。
3. 如果后续确实需要 T>2 per-transition change：
   - 新增单独的数据字段；
   - 不破坏 T=2 官方行为；
   - 写单元测试验证 shape `(N, T-1)`；
   - 不把“缺失于 scan”直接等同于物理 removal。

---

# 3. 第二阶段：ReScene4D temporal scaling profiling

新建：

```text
scripts/profile_temporal_scaling.py
```

目标：测量 ReScene4D 原始 all-history joint processing 在 T 增长时的实际成本。

测试：

```text
T = 2, 3, 4, 5
```

固定：

- 相同 voxel setting
- 相同 backbone
- 相同 precision
- 相同 GPU
- 尽量相同 scene-size bucket
- inference mode 与 training forward 分开测试

记录：

```text
peak_gpu_memory_mb
wall_time_ms
samples_per_second
num_points
num_voxels
T
batch_size
```

training forward 还记录：

```text
max_batch_size_without_oom
forward_backward_ms
```

输出：

```text
artifacts/profiling/re_scene4d_scaling.csv
artifacts/profiling/re_scene4d_scaling.md
```

必须生成图：

```text
Peak VRAM vs T
Latency vs T
Throughput vs T
```

### Go / No-Go G1

如果从 T=2 到 T=4/5：

- peak VRAM / latency 几乎没有有意义的恶化，
- 且长序列训练也很容易，

则不要再把 `scalability` 当论文第一 claim。

此时项目只能转向 `identity anti-drift`，或者停止。

---

# 4. 第三阶段：复现可信的 ReScene4D-C T=2 baseline

官方 checkpoint 若仍不可用：

- 使用官方 Concerto pretrained encoder
- encoder frozen
- 训练 ReScene4D decoder / heads
- 优先完全遵循官方 preprocessing 和 training config
- 不需要复现所有 Minkowski / Sonata variants

目标 checkpoint：

```text
checkpoints/rescene4d_concerto_t2_repro.ckpt
```

记录：

```text
paper_reported_tmap = 34.8
our_reproduced_tmap = ...
```

务必保留：

```text
artifacts/baseline/reproduction_config.yaml
artifacts/baseline/reproduction_metrics.json
artifacts/baseline/reproduction_log.txt
```

### A40 训练原则

不要因为 physical batch 小就改变 effective batch。

使用 gradient accumulation 尽量保持：

```text
effective_batch ~= official effective batch
```

所有后续方法与 baseline 使用完全相同的 compute / optimizer budget。

### Go / No-Go G2

如果复现结果与论文相差过大：

```text
例如 34.8 -> 25 左右
```

停止 Method 开发，先修 baseline。

经验目标（不是论文规则）：

```text
尽量 >= 32~33 t-mAP
```

或者至少趋势与官方结果高度一致。

---

# 5. 第四阶段：构造强 long-window ReScene4D baseline

必须避免不公平地只比较：

```text
train T=2 -> test T=5
```

至少训练：

### Baseline B0

```text
ReScene4D
train T=2
```

### Baseline B1

```text
ReScene4D
variable horizon train T ∈ {2,3}
```

如果资源允许：

### Baseline B2

```text
ReScene4D
variable horizon train T ∈ {2,3,4}
```

测试统一：

```text
T=2
T=3
T=4
T=5
```

记录：

- t-mAP
- per-stage mAP
- t-SIM
- peak VRAM
- latency
- throughput

### Go / No-Go G3

如果 `ReScene4D train T<=4`：

- 已经解决 T=4/5 identity performance，
- compute 也完全可接受，

则 persistent memory 的论文价值显著下降。

只有满足至少一项才继续：

1. all-history compute / memory 成本明显增加；
2. long-window training 后 identity consistency 仍明显下降；
3. T>training-horizon generalization 很差。

---

# 6. Method MVP：Bounded Persistent Multi-Instance Memory

**只有 G1~G3 支持 limitation 后开始实现。**

第一版不要修改 Concerto。

第一版不要双 Entity/State query bank。

第一版不要做 robotics / TSDF / open-vocabulary / occlusion reasoning。

---

## 6.1 新增文件

建议：

```text
models/
  persistent_memory.py
  streaming_rescene.py

conf/model/
  persist4d.yaml

trainer/
  persistent_trainer.py

scripts/
  evaluate_persist4d.py
  profile_persist4d_scaling.py

tests/
  test_persistent_memory.py
  test_memory_association.py
  test_streaming_sequence.py
```

尽量避免大面积修改官方代码。

---

# 7. 修改 ReScene4D：暴露最终 query feature

当前 ReScene4D 默认使用 non-parametric FPS queries，每次 forward 从当前输入重新初始化 query。

因此不能简单：

```text
previous_query -> replace current FPS query
```

第一版只要求暴露最终 decoder query feature。

给 `ReScene.forward()` 增加可选配置：

```yaml
return_query_features: true
```

输出新增：

```python
output["query_features"]
```

shape：

```text
[B, Q, D]
```

注意：

- 必须返回最终 decoder normalization 后、mask/class head 前后语义清晰的 query feature。
- 保持默认行为完全兼容官方 checkpoint/config。
- 若 `return_query_features=false`，输出必须和官方一致。

写测试确认：

```text
官方模式输出字段不变
新模式只多 query_features
```

---

# 8. Local Observation 定义

定义：

```python
@dataclass
class LocalInstanceObservation:
    query_features: Tensor       # [B,Q,D]
    class_logits: Tensor         # [B,Q,C]
    mask_logits: Any
    confidence: Tensor           # [B,Q]
    stage_index: Tensor
    gt_instance_ids: Optional[Tensor] = None
```

MVP 支持两种 local observation：

### Local-1

```text
X_t -> ReScene4D / single-stage decoder -> q_t
```

用于快速 debug。

### Local-2（主版本）

```text
[X_{t-1}, X_t] -> original ReScene4D short-window -> q_t
```

只提取 current/latest stage 的 observation 用于更新 long-term memory。

目标：

> 保留 ReScene4D 的短窗口 temporal perception，但不让 full history 进入当前 GPU inference。

---

# 9. Persistent Memory State

实现：

```python
@dataclass
class PersistentMemoryState:
    anchor: Tensor          # [B,K,D]
    working: Tensor         # [B,K,D]
    occupied: Tensor        # [B,K] bool
    class_prob: Tensor      # [B,K,C]
    confidence: Tensor      # [B,K]
    age: Tensor             # [B,K]
    last_seen: Tensor       # [B,K]
```

初始：

```text
K = 100
D = ReScene query dim
```

禁止保存完整历史 point cloud。

每个 slot 表示一个 persistent entity hypothesis。

---

# 10. Dual-timescale memory

## 10.1 Identity Anchor

```text
anchor[i]
```

目的：

> 稳定长期 identity，不因单个低质量 revisit 被完全覆盖。

更新必须慢。

---

## 10.2 Working Memory

```text
working[i]
```

目的：

> 快速吸收当前/最近几次 observation 的几何和 appearance variation。

更新可以快。

---

# 11. Memory Association

实现模块：

```python
class MemoryAssociation(nn.Module):
    ...
```

输入：

```text
current local queries Q_t
occupied memory slots M_{t-1}
```

score 初版：

```math
s_ij =
w_a cos(W_q q_j, W_a a_i)
+
w_h cos(W'_q q_j, W_h h_i)
+
w_c class_compatibility(i,j)
```

不要第一版依赖 absolute 3D spatial proximity：

- 物体在 rescans 间可能移动；
- ReScene4D 的任务本身就是大变化重访。

可把 geometry 作为可选 soft feature，但禁止 hard spatial gate。

assignment：

```text
Hungarian
```

或可 differentiable Sinkhorn 作为后续 ablation。

要求：

- one-to-one competition
- unmatched current query -> birth candidate
- unmatched memory slot -> dormant, 不立即释放 slot
- low-confidence query 不允许轻易污染 memory

---

# 12. Gated Memory Consolidation

实现：

```python
class GatedMemoryConsolidation(nn.Module):
    ...
```

### Working update

```math
\tilde h_i^t = GRU(q_i^t, h_i^{t-1})
```

gate：

```math
g_i^t =
sigmoid(
MLP([
q_i^t,
h_i^{t-1},
a_i^{t-1},
obs_confidence,
association_score
])
)
```

更新：

```math
h_i^t =
(1-g_i^t) h_i^{t-1}
+
g_i^t \tilde h_i^t
```

### Anchor update

anchor 必须比 working 慢。

```math
r_i^t =
clip(
eta * confidence_i * sigmoid(MLP_a(...)),
0,
r_max
)
```

```math
a_i^t =
normalize(
(1-r_i^t)a_i^{t-1}
+
r_i^t q_i^t
)
```

推荐：

```text
eta small
r_max <= 0.1~0.2
```

这些是初始值，需要配置化，不要硬编码。

---

# 13. Dormancy / Birth

MVP 状态只需要：

```text
FREE
ACTIVE
DORMANT
```

不要第一版加入：

```text
RELEASED
GHOST
UNCERTAIN
REMOVED
REACTIVATED
```

规则：

### ACTIVE

本 stage 匹配到 current observation。

### DORMANT

本 stage 未匹配，但 memory 保留。

### FREE

未分配 slot。

新 observation：

```text
如果与所有 occupied slot 匹配分数均低
且 observation confidence 高
=> allocate FREE slot
```

DORMANT slot 后续重新匹配到：

```text
=> ACTIVE
```

这已经可以测试 long-term re-identification。

---

# 14. Memory-conditioned Query Refinement（第二阶段，重要）

**纯 post-hoc memory association 可能被审稿人认为只是 learned matcher。**

因此 MVP memory 有效后，增加：

```python
class MemoryQueryAdapter(nn.Module):
    ...
```

思路：

当前 ReScene4D 仍使用 fresh FPS query 初始化。

不要替换 FPS query。

在 final / late decoder query 上增加：

```math
Q'_t =
Q_t +
gamma * CrossAttention(
query=Q_t,
key=A_{t-1},
value=[A_{t-1}, H_{t-1}]
)
```

其中：

```text
A = occupied anchor memory
H = occupied working memory
```

mask 掉 FREE slots。

再使用：

```text
Q'_t
```

进入最终 mask/class prediction和 memory association。

目的：

> persistent memory 不只是后处理 ID，而是实际帮助当前 instance representation。

建议配置：

```yaml
memory_adapter:
  enabled: true
  num_heads: 4
  num_layers: 1
  residual_scale_init: 0.0
```

`residual_scale_init=0` 可保证初始化时等价于原 ReScene4D。

---

# 15. Loss

保留官方：

```text
L_seg
```

新增尽量少。

## 15.1 Association / Identity loss

根据 3RScan 跨 scan GT instance ID：

```math
L_id
```

要求：

- same entity -> feature close
- different entity -> separated
- 优先采样 same-class negatives

可以使用 supervised contrastive / InfoNCE。

---

## 15.2 Anchor consistency

```math
L_anchor =
1 - cos(
stopgrad(a_i^{t-1}),
q_i^t
)
```

仅对：

- GT identity 确定
- 高置信 observation
- 匹配正确

的 pair 计算。

不要对所有 observation 无条件强拉，防止错误监督。

---

## 15.3 可选 diversity loss

防止多个 memory slots collapse：

```math
L_sep
```

只在真实实验观察到 collapse 后加入。

第一版不要预先堆 loss。

最终：

```math
L =
L_seg
+
lambda_id L_id
+
lambda_anchor L_anchor
```

先从这三个开始。

---

# 16. Long-Horizon Training

训练 sequence：

```text
T sampled from {2,3}
```

验证：

```text
T=2,3,4,5
```

如果资源允许再训练：

```text
T sampled from {2,3,4}
```

使用 recurrent memory：

```math
M_t = F(M_{t-1}, O_t)
```

不要把所有历史 memory feature 拼回一个巨大 tensor。

---

## 16.1 Truncated BPTT

为避免 computation graph 随 T 无限增长：

配置：

```yaml
memory:
  detach_every: 2
```

例如：

```python
if t % detach_every == 0:
    memory = memory.detach()
```

注意：

- memory 数值继续传递
- gradient graph 截断

同时报告：

```text
detach_every = 1 / 2 / full
```

至少做一次敏感性验证。

---

# 17. 可选 Feature / Observation Cache

因为 Concerto encoder frozen，可考虑缓存 local observation。

新建：

```text
scripts/cache_local_observations.py
```

缓存：

```text
query_features
class_logits
mask confidence
GT temporal identity metadata
```

路径例如：

```text
cache/rescene_observations/{scene_id}/{window_id}.pt
```

用途：

- 快速训练 memory module
- 快速 ablation
- 不需要每次重新跑 frozen spatial backbone

注意：

> 这只能用于 memory-only prototype / controlled training。

如果官方 augmentation 会改变 point cloud 和 local query features，则最终 end-to-end finetuning 仍需要 live pipeline。

---

# 18. Evaluation

## 18.1 核心指标

使用官方 stmetrics：

```text
t-mAP@T=2
t-mAP@T=3
t-mAP@T=4
t-mAP@T=5

per-stage mAP
t-SIM
```

对 T=2 保留官方：

```text
change-conditioned t-mREC
```

如果 T>2 transition change labels 未可靠修复，不在主文声称 T>2 change-conditioned结果。

---

## 18.2 Scalability 指标

必须报告：

```text
peak GPU memory
latency per new visit
throughput
```

区分两种使用方式：

### ReScene4D all-history

到第 t 次 revisit：

```text
process X_1 ... X_t jointly
```

### Persist4D

到第 t 次 revisit：

```text
process local window + M_{t-1}
```

报告：

```text
history length = 2,3,4,5
```

不要在没有测量的情况下写 asymptotic conclusion。

---

# 19. Baselines

主表至少：

```text
1. ReScene4D-C paper reported T=2        [reference only]
2. ReScene4D-C our reproduction T=2
3. ReScene4D train-T2, test T=2..5
4. ReScene4D long-window train T<=3/4
5. Naive recurrent query / EMA memory
6. Single-timescale persistent memory
7. Dual-timescale memory
8. Full Persist4D + memory-conditioned adapter
```

外部：

```text
AutoSeg3D
ChronoTrack
Rescan
```

主要放 Related Work。

除非输入/任务/metric 可以公平对齐，否则不要强行塞进同一主表。

---

# 20. Ablation

只做与核心 claim 有关的：

```text
A. no persistent memory
B. single recurrent memory
C. + identity anchor
D. + gated consolidation
E. + memory-conditioned query adapter
F. full
```

指标：

```text
t-mAP @ T=2/4/5
t-SIM
peak VRAM @ T=5
```

不要做大量 backbone ablation。

---

# 21. 必须输出的诊断

新建：

```text
scripts/analyze_identity_drift.py
```

对于跨 stage 同一个 GT instance：

计算：

```math
cos(q_i^1, q_i^t)
cos(a_i^1, a_i^t)
cos(h_i^1, h_i^t)
```

按：

```text
same class / different class
static / changed
T
```

统计。

输出：

```text
artifacts/analysis/identity_drift.csv
artifacts/analysis/identity_drift.png
```

目的：

> 证明 dual-timescale memory 解决的确实是长期 identity representation 问题，而不是单纯加参数。

---

# 22. 实验主表模板

最终目标表：

| Method | T=2 t-mAP | T=3 | T=4 | T=5 | Per-stage AP | t-SIM | VRAM@T5 | Update ms@T5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReScene4D paper | 34.8 | - | - | - | - | - | - | - |
| ReScene4D repro | | | | | | | | |
| ReScene4D long-trained | | | | | | | | |
| Naive recurrent | | | | | | | | |
| Single memory | | | | | | | | |
| Dual memory | | | | | | | | |
| Persist4D | | | | | | | | |

注意：

> 不要把不同 T 的绝对 t-mAP 当成同难度任务直接横向比较。

重点比较：

```text
同一个 T 下不同 method
```

以及：

```text
performance degradation slope vs T
```

---

# 23. 代码质量要求

Codex 必须遵守：

1. 不破坏官方 ReScene4D 默认 config。
2. 新方法全部 feature-flag 控制。
3. 新增 unit tests。
4. 所有 tensor shape 用 assert。
5. 所有 temporal stage indices 明确记录。
6. 所有 experiment config 保存。
7. 所有随机 seed 保存。
8. 所有 profiling 硬件信息保存。
9. 不在代码中硬编码本机路径。
10. 不使用 GT identity 作为 inference 输入；GT ID 只能用于 loss / evaluation。
11. memory slot assignment inference 时必须由模型 score 决定。
12. 不允许未来 stage feature 泄漏到当前 persistent memory update。

---

# 24. 实施顺序

严格按以下顺序：

```text
P0 数据审计
↓
P1 T=2/3/4/5 profiling
↓
G1: limitation 是否真实？
↓
P2 T=2 baseline reproduction
↓
G2: baseline 是否可信？
↓
P3 ReScene4D long-window strong baseline
↓
G3: longer training 是否已经解决问题？
↓
P4 query feature export
↓
P5 naive recurrent memory
↓
P6 single-timescale persistent memory
↓
P7 dual-timescale anchor + working memory
↓
P8 gated consolidation
↓
P9 memory-conditioned query adapter
↓
P10 final variable-horizon training
↓
P11 complete evaluation / ablation
```

---

# 25. 每阶段验收条件

## P0

必须有：

```text
train/val T distribution
sequence DB manifest
```

## P1

必须有：

```text
VRAM / latency / throughput vs T
```

如果 scaling limitation 不明显，停止或转方向。

## P2

必须有可信 T=2 checkpoint。

如果 baseline 严重低于官方，停止新方法开发。

## P3

必须有 long-trained ReScene4D strong baseline。

如果它已在计算可接受条件下解决 T=4/5，重新评估论文 novelty。

## P5

naive recurrent memory 必须能完整跑通 T=5。

## P7

dual-timescale memory 应至少在 identity diagnostic / t-SIM / T=4-5 t-mAP 中体现增益，否则删除该模块。

## P9

memory-conditioned adapter 必须证明提升来自“memory 参与当前 representation”，否则不要保留复杂度。

---

# 26. 最终论文可接受的核心贡献表述

如果实验全部成立，论文贡献写成：

### Contribution 1

A streaming formulation for temporally sparse full-scene 4D instance segmentation that replaces growing all-history context with a fixed-capacity multi-instance memory.

### Contribution 2

A competition-aware dual-timescale entity memory with stable identity anchors, fast working representations, and confidence-gated consolidation to limit long-horizon identity drift.

### Contribution 3

A variable-horizon training and evaluation protocol on native multi-rescan 3RScan sequences, measuring accuracy, temporal consistency, and compute as revisit history grows.

不要声称：

```text
first online
first query memory
first identity-state separation
first long-term 4D segmentation
```

除非最终再次系统检索后可以严格证明。

---

# 27. Codex 首轮任务

**现在不要实现 Persist4D。**

首先只完成：

```text
P0 + P1
```

即：

1. 数据 split / T 分布审计；
2. 生成 T=2/3/4/5 sequence DB；
3. 验证官方 loader 能正确读取；
4. ReScene4D T=2/3/4/5 forward profiling；
5. 输出 GPU memory / latency / throughput 图表；
6. 检查 T>2 是否存在 tensor-shape / change-label 问题；
7. 写一份：

artifacts/P0_P1_GO_NOGO_REPORT.md

报告必须明确回答：

```text
A. T=3/4/5 是否可直接运行？
B. ReScene4D resource cost 随 T 的真实曲线是什么？
C. T=4/5 的最大可用 batch 是多少？
D. 当前官方代码中有哪些 T>2 bug / assumption？
E. 是否存在足够强的 scalability limitation 支撑下一阶段？
F. 推荐 GO / NO-GO。
```

只有报告结论为 GO，才开始训练正式 baseline 和实现 memory module。
