# Persist4D Master Plan — Codex Execution Prompt
## From ReScene4D Reproduction (P2) to Long-Horizon Persistent Memory

> **Repository state:** P0/P1 completed and G1=GO at commit `705e0ee`.
>
> **Current authorization:** begin **P2 only**.
> Persist4D implementation remains gated behind **G2** and **G3**.
>
> This document defines the complete research logic so later stages do not drift, but Codex MUST obey the stage gates and MUST NOT implement future stages early.

---

# 0. Executive Research Decision

We are **not** writing “ReScene4D with a larger T”.

The paper hypothesis is:

> ReScene4D obtains strong temporally consistent 4D instance segmentation by jointly reasoning over the temporal observations available in a sequence. This is effective for short windows, but growing revisit history increases resource cost and makes long-horizon identity consistency increasingly difficult. We ask whether complete historical 3D observations can be replaced by a **fixed-capacity persistent multi-instance memory** that preserves identity over longer revisit horizons while processing only a bounded local window.

The intended paper identity is:

> **Persist4D: Bounded Persistent Instance Memory for Long-Horizon 4D Scene Understanding**

The project is about:

- temporally sparse full-scene 4D semantic instance segmentation;
- progressive/streaming processing of revisits;
- bounded-history computation;
- persistent instance identity;
- long-horizon anti-drift representation.

It is **not** about:

- dynamic SLAM;
- online RGB-D frame tracking;
- robot navigation;
- open-vocabulary mapping;
- TSDF;
- SAM;
- free-space reasoning;
- generic object memory;
- simply increasing sequence length.

---

# 1. Verified Starting Point

P0/P1 has already established a real scaling effect on the current codebase.

Measured T2 -> T5 forward+backward behavior:

```text
Peak memory:   5575.5 -> 12045.7 MiB   = 2.16x
Throughput:    1.5836 -> 0.8864 sample/s = -44.0%
Max batch:     8 -> 3
```

Official T=3/4/5 data/model paths execute successfully.

Known semantic risks already recorded:

- chronology/order semantics;
- T>2 change projection;
- `prev_transform`.

Existing artifacts MUST be treated as immutable evidence:

```text
paper5/artifacts/P0_P1_GO_NOGO_REPORT.md
paper5/artifacts/profiling/re_scene4d_scaling.md
```

Do not rewrite those conclusions unless a reproducible bug is discovered.

---

# 2. Scientific Logic Audit

## 2.1 What ReScene4D already solves

ReScene4D is already strong at:

- temporally sparse 4DSIS;
- shared spatio-temporal queries;
- joint mask prediction across visits;
- temporal contrastive learning;
- spatio-temporal decoder serialization;
- short-window identity consistency;
- 3RScan t-mAP evaluation.

The paper reports for ReScene4D-C / Concerto:

```text
t-mAP       34.8
t-mAP50     52.5
t-mAP25     66.8
overall mAP 43.3

per-stage:
stage 1 mAP 47.8
stage 2 mAP 48.3
```

The best Concerto temporal-sharing combination is:

```text
cross-time contrastive loss = ON
ST serialization            = ON
ST masking                  = OFF
```

Do not “improve” this configuration during baseline reproduction.

---

## 2.2 Why this follow-up is logically justified

The official architecture uses:

- 100 fixed spatio-temporal queries;
- FPS non-parametric query initialization;
- joint query refinement over features from all temporal stages;
- temporal decoder serialization across the sequence.

The official paper explicitly notes:

- joint 4DSIS has computational cost;
- temporally paired scans increase memory/processing demand;
- experiments are restricted to short temporal windows / moderate scene sizes;
- 3RScan training is constructed primarily from length-2 sequences.

Therefore the valid research question is:

> Can a full-scene 4DSIS model preserve long-term instance identity without retaining and jointly reprocessing a growing set of full historical 3D observations?

This question is stronger than:

```text
T=2 -> T=5
```

because it targets the **history representation** rather than a dataloader setting.

---

## 2.3 What is NOT novel

Do not claim any of these as novelty:

```text
query propagation
recurrent query memory
online/streaming instance tracking
long-short memory in general
identity/state separation in general
long-term 3D memory in general
T > 2
```

Nearby literature already includes query propagation / online instance tracking and compact long-term memory in other 3D temporal tasks.

Our intended differentiation is:

> **fixed-capacity, competition-aware, multi-instance persistent memory for temporally sparse full-scene 4DSIS**, retaining ReScene4D-quality local segmentation while replacing growing all-history context and explicitly controlling long-horizon identity drift.

---

# 3. Paper-Level Novelty — Provisional, Gated

These contributions are **provisional until G2/G3 pass**.

## N1. Bounded-History Progressive 4DSIS

Replace:

```text
all historical visits X_1 ... X_T
        ->
joint current inference
```

with:

```text
bounded local window
+
fixed-capacity persistent memory M_(t-1)
        ->
current prediction + M_t
```

The resource footprint of the historical representation should not grow with the number of revisits.

Do NOT claim exact asymptotic complexity until measured and justified.

---

## N2. Anti-Drift Dual-Timescale Multi-Instance Memory

Each persistent entity slot maintains:

```text
Identity Anchor A_i
    slow, stable representation

Working Memory H_i
    fast, adaptable representation
```

New observations are not blindly written into memory.

A confidence-gated consolidation mechanism controls updates so partial/noisy revisits do not progressively corrupt long-term identity.

---

## N3. Memory-Conditioned Current Query Refinement

Persistent memory must not remain only a post-hoc tracker.

After the memory MVP works, a lightweight memory-to-query adapter allows historical entity memory to refine the current ReScene query representation before final mask/class prediction.

This is needed to distinguish the full method from:

```text
ReScene4D + learned Hungarian matcher
```

---

## N4. Native Long-Horizon Scaling Protocol

Use the real multi-rescan structure of 3RScan.

Evaluate:

```text
T = 2, 3, 4, 5
```

along both:

- accuracy / identity consistency;
- compute / memory scaling.

Use both:

1. all eligible scenes at each T;
2. a **common-scene controlled subset** so changes in dataset composition do not masquerade as horizon effects.

---

# 4. Important Scope Correction: Do NOT Center the Paper on Existence/Removal

Earlier planning considered an explicit:

```text
entity identity E
vs.
current state S_t
```

formulation.

Do **not** make that the main paper mechanism at this stage.

Reason:

- ReScene4D reports only a very small fraction of added/removed validation instances;
- the paper itself notes this signal is noisy and insufficient for strong removal learning;
- the current T>2 change-label loader also has known limitations.

Therefore:

```text
existence / removal / reactivation head
```

is OPTIONAL future work / secondary ablation only.

The core problem is:

```text
long-horizon identity consistency
+
bounded history
```

which is much better supported by 3RScan.

---

# 5. Dataset Protocol

## 5.1 Existing 3RScan T distribution

Already measured from:

```text
vv/dataset/3RScan/3RScan.json
```

Full 478 reference scenes:

```text
T = 2       : 194
T >= 3      : 284
T >= 4      : 124
T >= 5      : 56
T >= 6      : 25
```

P0 should already contain split-aware statistics. Reuse them.

---

## 5.2 Terminology

Do NOT assume 3RScan ordering is physical chronology.

ReScene4D itself forms temporal sequences by ordering/reordering primary scans and rescans, and stage time is represented by stage index.

Use the terms:

```text
revisit order
progressive revisit sequence
streaming revisit order
temporal stage
```

Avoid:

```text
true chronological causal history
```

unless timestamp chronology is independently verified.

“Causal” may be used only in the narrow computational sense:

```text
prediction at stage t does not use stages > t
```

Prefer `streaming` or `progressive` in the paper title/text.

---

## 5.3 Two evaluation protocols

### Protocol A — Standard Eligible-Scene Evaluation

For each T:

```text
T=2: all eligible val scenes
T=3: all eligible val scenes
T=4: all eligible val scenes
T=5: all eligible val scenes
```

Purpose:

> maximum statistical coverage at each horizon.

Do not compare absolute t-mAP across different T as though they were the same test set.

Compare methods **within the same T**.

---

### Protocol B — Controlled Common-Scene Horizon Evaluation

Construct a deterministic subset:

```text
S5 = validation scenes with at least T>=5
```

For each scene, choose one fixed deterministic revisit permutation.

Evaluate prefixes:

```text
prefix length 2
prefix length 3
prefix length 4
prefix length 5
```

on the **same scenes**.

Purpose:

> isolate actual horizon degradation from changing scene composition.

Report:

```text
t-mAP(T)
per-stage mAP(T)
t-SIM(T)
relative retention = t-mAP(T) / t-mAP(T=2)
```

`relative retention` is a diagnostic, not a replacement benchmark metric.

---

## 5.4 Order Robustness

Because recurrent memory is order-sensitive and 3RScan chronology is not guaranteed:

On S5, evaluate at least 3 deterministic revisit permutations if computationally affordable.

Report mean/std for the final method and strongest baseline.

Do not use order robustness as the primary metric; it is a validity check.

---

# 6. P2 — ReScene4D-C T=2 Official Baseline Reproduction

## Authorization

**P2 is the only currently authorized implementation stage.**

Starting commit:

```text
705e0ee
```

Before edits:

```bash
git status --short
git rev-parse HEAD
```

Require:

```text
clean worktree
HEAD == 705e0ee
```

If not, record why and stop before destructive operations.

Suggested branch:

```text
research/p2-rescene4d-t2-repro
```

---

# 7. P2 Goal

Produce one trustworthy checkpoint:

```text
paper5/checkpoints/rescene4d_concerto_t2_repro.ckpt
```

that is close enough to the official ReScene4D-C T=2 result to serve as:

- G2 evidence;
- P3/G3 initialization;
- the common baseline for future Persist4D experiments.

We do NOT need to reproduce Minkowski or Sonata.

---

# 8. P2 Preflight Audit

Create:

```text
paper5/artifacts/P2/config_audit.md
paper5/artifacts/P2/environment_manifest.json
paper5/artifacts/P2/reproduction_target.yaml
```

The audit must identify the exact official/current settings for:

```text
backbone
Concerto checkpoint/repo
frozen encoder behavior
decoder trainability
num_queries
non_parametric_queries
query initialization
temporal_window
contrastive loss
ST serialization
ST masking
voxel size
loss weights
no-object/empty weight
optimizer
scheduler
max LR
epochs
effective batch size
precision
3RScan/ScanNet mixing
augmentations
evaluation class taxonomy
sequence DB
metric config
```

Official paper target:

```text
Concerto
100 queries
FPS non-parametric query initialization
2 cm voxel
450 epochs
global/effective batch 32
AdamW
OneCycle
max LR 5e-4
PTv3 encoder frozen
contrastive ON
ST serialization ON
ST masking OFF
```

Do not assume the current repository defaults exactly reproduce the paper.

Generate a machine-readable config diff:

```text
paper5/artifacts/P2/official_vs_repro_config_diff.json
```

---

# 9. ScanNet Is a Reproduction Requirement

The official ReScene4D training setup mixes:

```text
3RScan T=2
+
ScanNet single-stage
```

approximately at the published ratio.

Before formal P2 training:

1. verify ScanNet is available and correctly preprocessed;
2. verify class mapping matches the 18-class ReScene4D evaluation setup;
3. verify the mixed dataset sampler is actually active.

If ScanNet is unavailable:

**STOP formal reproduction.**

Do NOT train 3RScan-only and call it an official reproduction.

Instead write:

```text
paper5/artifacts/P2/BLOCKED_MISSING_SCANNET.md
```

with the exact missing prerequisite.

---

# 10. A40 Compute Strategy

Do not assume “more GPUs = faster”.

The user has a multi-server A40 cluster; inter-node communication may dominate.

Run a short topology benchmark if feasible:

```text
2 GPU
4 GPU
8 GPU
```

or available practical combinations.

Measure:

```text
samples/s
optimizer steps/s
communication overhead
peak VRAM/GPU
```

Select the topology with the best useful throughput.

Record:

```text
paper5/artifacts/P2/hardware_topology_profile.csv
```

---

# 11. Effective Batch and Gradient Accumulation

Official effective batch:

```text
32
```

If A40 physical batch is smaller, use gradient accumulation.

Example only:

```text
physical global batch = 8
accumulate = 4
effective batch = 32
```

Do not blindly change learning rate if effective batch remains 32.

### Mandatory scheduler audit

Because OneCycle scheduling is step-sensitive:

log:

```text
micro_step
optimizer_step
global_step
LR
```

Confirm LR scheduler steps on the intended optimizer-step semantics under accumulation.

If PyTorch Lightning already handles it correctly, document that from runtime behavior.

Do not patch scheduler code without evidence.

Output:

```text
paper5/artifacts/P2/lr_schedule_audit.csv
paper5/artifacts/P2/lr_schedule_audit.md
```

---

# 12. P2 Smoke Tests Before Full Training

## 12.1 Frozen Encoder Test

Assert:

```text
Concerto encoder parameters require_grad == False
encoder gradient == None / zero
decoder/head gradients exist
```

---

## 12.2 Short Training Smoke Test

Run enough iterations to confirm:

```text
no NaN/Inf
segmentation losses finite
contrastive loss finite
optimizer steps occur
LR follows expected OneCycle curve
validation/evaluation pipeline executes
checkpoint save/resume works
```

---

## 12.3 Tiny-Subset Overfit Test

Use a tiny deterministic subset.

The model should materially overfit it.

If it cannot:

```text
DO NOT start 450-epoch training.
```

Investigate data, target, matcher, optimizer, or model wiring first.

Save:

```text
paper5/artifacts/P2/tiny_overfit_report.md
```

---

# 13. Formal P2 Training

Only after all preflight tests pass.

Train the exact ReScene4D-C target.

Save:

```text
last.ckpt
best_tmap.ckpt
selected periodic checkpoints
```

Log:

```text
train loss
dice/BCE/class components
contrastive loss
validation t-mAP
validation mAP
per-stage mAP
LR
VRAM
throughput
```

Do not implement Persist4D code in the same branch.

---

# 14. G2 Evaluation

Official target values:

```text
t-mAP        = 34.8
t-mAP50      = 52.5
t-mAP25      = 66.8
overall mAP  = 43.3
stage1 mAP   = 47.8
stage2 mAP   = 48.3
```

Internal engineering gate:

### GREEN

Prefer:

```text
t-mAP >= ~32.7
```

and spatial/per-stage metrics show the same general quality regime as the paper.

This corresponds roughly to >=94% of the reported t-mAP and is an internal diagnostic threshold, NOT a publication rule.

### YELLOW

```text
t-mAP roughly 30.5–32.7
```

with good per-stage mAP.

Action:

- audit temporal loss / serialization / sequence DB / sampler;
- optionally run a second seed;
- do not immediately reject.

### RED

Examples:

```text
t-mAP << 30
or
per-stage mAP also severely degraded
```

Action:

```text
G2 = NO-GO
```

Do not implement Persist4D.

---

# 15. Diagnose Reproduction Failure by Spatial vs Temporal Performance

Case A:

```text
stage mAP ~ paper
t-mAP much lower
```

Likely issue:

```text
temporal identity / sequence / contrastive / serialization / metric
```

Case B:

```text
stage mAP also much lower
```

Likely issue:

```text
backbone/decoder/data/optimization/general reproduction
```

Do not treat both cases the same.

---

# 16. P2 Required Output

Create:

```text
paper5/artifacts/P2_G2_REPRODUCTION_REPORT.md
```

Must include:

1. official paper targets;
2. reproduced metrics;
3. exact checkpoint path + SHA256;
4. git commit;
5. complete config diff;
6. environment versions;
7. effective batch details;
8. LR audit;
9. best epoch;
10. per-stage metrics;
11. training wall time;
12. GPU topology;
13. peak memory / throughput;
14. all deviations from official recipe;
15. G2 verdict: GREEN / YELLOW / RED;
16. recommendation for P3.

Commit P2 separately.

Do not squash into P0/P1.

---

# 17. P3 / G3 — Strong Long-Window ReScene4D Baselines

**Do not start until G2 is GREEN or explicitly accepted YELLOW.**

Purpose:

> determine whether the observed long-horizon problem is merely `train T=2 -> test longer T`, or whether the all-history architecture remains problematic after fair long-window training.

---

# 18. P3 Baselines

## B0 — Official-Style ReScene4D

```text
train: T=2
test:  T=2,3,4,5
```

Use P2 checkpoint.

---

## B1 — Variable-Horizon ReScene4D

Train:

```text
T sampled from {2,3}
```

Balance horizon sampling so T=2 does not dominate solely because more sequences exist.

Test:

```text
T=2,3,4,5
```

---

## B2 — Stronger Long-Window Baseline

If feasible:

```text
T sampled from {2,3,4}
```

or fixed T=4 long-window training.

Do not force B2 if the resource requirement makes it impossible; document the practical limit.

The point is to provide ReScene4D the strongest reasonable long-horizon training, not to sabotage it.

---

# 19. P3 Training Fairness

For B0/B1/B2 record separately:

```text
optimizer steps
total scan-stages seen
training wall-clock
peak memory
effective batch
```

Accuracy comparison should use a converged/credible baseline.

Efficiency comparison should be explicit about compute budget.

Do not claim “same training budget” unless it is defined.

At minimum report:

```text
same optimizer updates?
same number of scan-stages?
same wall-clock?
```

---

# 20. P3 Controlled Horizon Evaluation

For every baseline run both:

```text
Protocol A: all eligible scenes per T
Protocol B: common S5 subset prefixes
```

Metrics:

```text
t-mAP
t-mAP50
t-mAP25
per-stage mAP
t-SIM
peak VRAM
latency
throughput
```

T>2 change-conditioned t-mREC is NOT primary until the loader/annotation semantics are fixed and validated.

---

# 21. G3 Decision

G3 asks:

> Does strong longer-window training remove the need for a bounded persistent memory?

### G3 = GO if at least one major structural problem remains:

#### Accuracy/identity problem

Long-window-trained ReScene4D still suffers meaningful horizon degradation on the common-scene protocol.

OR

#### Resource problem

Accuracy recovers, but doing so still requires materially growing all-history memory/compute, preserving a strong efficiency/scalability motivation.

OR

#### Generalization problem

Model trained on shorter horizons degrades substantially when evaluated beyond its training horizon.

### G3 = NO-GO / REPOSITION if:

- long-window ReScene4D achieves stable T=4/5 identity;
- resource cost is practically modest;
- and there is no meaningful horizon-generalization failure.

Do not invent a method if the strong baseline solves the problem.

Create:

```text
paper5/artifacts/P3_G3_LONG_HORIZON_REPORT.md
```

---

# 22. Method Authorization After G3=GO

Only then create the Persist4D implementation.

The initial implementation must be incremental.

Order:

```text
M0 naive recurrent/global-ID baseline
M1 fixed-capacity single-timescale memory
M2 dual-timescale anchor + working memory
M3 gated consolidation
M4 memory-conditioned query adapter
M5 final variable-horizon training
```

Each module must justify its existence empirically.

If a simpler variant performs equally well, remove complexity.

---

# 23. Persist4D Primary Inference Protocol

At temporal stage `t`:

```text
local input window W_t
+
persistent memory M_(t-1)
    ->
current-stage predictions
+
M_t
```

Primary bounded window:

```text
W = 2
```

For `t >= 2`:

```text
[X_(t-1), X_t]
```

For the first stage:

```text
[X_1]
```

Important:

- process only the bounded local window;
- do not reload X_1...X_(t-2);
- do not use future X_(t+1...);
- commit only the **current/latest-stage prediction** in the primary streaming evaluation;
- do not retroactively revise earlier committed predictions using future visits.

A W=1 variant should be retained as an ablation.

This preserves ReScene4D’s useful short-range temporal context without allowing history cost to grow with T.

---

# 24. Query Interface Design

Official ReScene4D uses:

```text
non_parametric_queries: true
FPS initialization
100 queries
```

Therefore DO NOT replace fresh FPS queries with previous memory slots.

Instead:

1. keep fresh local ReScene queries;
2. expose the final/late-stage query embedding;
3. retrieve/associate persistent memory;
4. optionally inject memory through a residual adapter.

First compatibility modification:

```python
output["query_features"]
```

shape:

```text
[B, Q, D]
```

must be optional under config:

```yaml
return_query_features: false
```

Official behavior must remain unchanged when disabled.

---

# 25. Memory Capacity Audit Before Choosing K

Do not blindly set persistent memory capacity to 100.

Before method training, compute for train/val:

```text
number of unique GT instance trajectories
per scene
for T=2/3/4/5
```

Report:

```text
median
p90
p95
p99
max
```

Choose fixed memory capacity `K_mem` based on the training distribution plus a documented safety margin.

Candidate values may include:

```text
100 / 128 / 160
```

but the final value must be data-driven.

Report memory-slot saturation rate.

---

# 26. Persistent Memory State

Suggested structure:

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

No historical point cloud is stored.

No GT ID is stored or used during inference.

---

# 27. Memory States

MVP requires only:

```text
FREE
ACTIVE
DORMANT
```

Do not build a complex lifecycle.

### ACTIVE
matched to a current local instance.

### DORMANT
not matched this stage; identity memory remains.

### FREE
available for a new entity.

A dormant slot may later match a new local observation again.

This is sufficient for persistent identity experiments.

---

# 28. Memory Association

Inputs:

```text
fresh/current local query q_j
persistent anchor a_i
persistent working state h_i
class distributions
observation confidence
```

Initial similarity:

```math
s_ij =
w_a * cosine(Wq(q_j), Wa(a_i))
+
w_h * cosine(Wq2(q_j), Wh(h_i))
+
w_c * class_compatibility(i,j)
```

Do NOT use a hard absolute-position gate.

3RScan entities may move substantially between visits.

Optional geometry/shape cues may be introduced only as soft features.

Use:

```text
Hungarian assignment
```

for one-to-one competition.

Inference behavior:

```text
matched query + slot -> ACTIVE/update
unmatched occupied slot -> DORMANT
high-confidence unmatched query -> allocate FREE slot
```

GT temporal instance ID is allowed only for:

```text
training supervision
evaluation
```

never inference association.

---

# 29. M0 — Naive Recurrent Baseline

Before novel memory:

```text
single embedding per memory slot
simple EMA or GRU update
```

Purpose:

> establish whether plain recurrent propagation already solves the problem.

If M0 performs nearly as well as all later modules:

**do not claim complex dual-memory novelty.**

---

# 30. M1/M2 — Dual-Timescale Memory

For entity slot i:

```text
A_i = identity anchor
H_i = working memory
```

Interpretation:

```text
A_i: stable identity representation
H_i: recent adaptive representation
```

Working update:

```math
\tilde H_i^t = GRU(q_i^t, H_i^{t-1})
```

or a small MLP-gated update.

Anchor update must be substantially slower.

---

# 31. M3 — Confidence-Gated Consolidation

Do not overwrite memory on every match.

Working gate:

```math
g_i^t =
sigmoid(
MLP[
q_i^t,
H_i^(t-1),
A_i^(t-1),
association_score,
observation_confidence
]
)
```

Update:

```math
H_i^t =
(1-g_i^t) H_i^(t-1)
+
g_i^t * \tilde H_i^t
```

Anchor write rate:

```math
r_i^t =
eta * sigmoid(MLP_a(...)) * confidence
```

with small configurable `eta`.

Update:

```math
A_i^t =
normalize(
(1-r_i^t) A_i^(t-1)
+
r_i^t q_i^t
)
```

All gates/eta values must be configurable.

No hard-coded magic thresholds inside module code.

---

# 32. M4 — Memory-Conditioned Query Adapter

This is important for the full paper.

Without it, the system can be criticized as:

```text
ReScene4D + post-hoc temporal matcher
```

Keep FPS fresh queries.

At a late decoder stage:

```math
Q'_t =
Q_t
+
gamma * CrossAttention(
    query = Q_t,
    key   = occupied memory anchors,
    value = [anchor, working]
)
```

Requirements:

```text
1 lightweight layer initially
masked FREE slots
residual connection
gamma initialized to 0 or near 0
```

Then use `Q'_t` for:

```text
current mask/class prediction
memory association
memory update
```

This ensures history affects the current representation, not only its output ID.

---

# 33. Losses

Keep official ReScene segmentation losses.

Start with only:

```math
L =
L_seg
+
lambda_id * L_id
+
lambda_anchor * L_anchor
```

## Identity association loss

Use 3RScan temporally consistent GT identities.

Same instance across visits:

```text
positive
```

Different instances:

```text
negative
```

Prioritize hard same-class negatives.

Use supervised contrastive / InfoNCE or equivalent.

---

## Anchor consistency loss

For reliable matched identities:

```math
L_anchor =
1 - cosine(stopgrad(A_i^(t-1)), q_i^t)
```

Only compute when GT identity is known and the observation is valid.

Do not indiscriminately force every revisit embedding to the anchor.

---

## Optional later loss

Only if experiments reveal slot/representation collapse:

```text
memory diversity / cycle consistency
```

Do not add it pre-emptively.

---

# 34. Long-Horizon Training

Train persistent memory with variable horizon.

Initial:

```text
T sampled from {2,3}
```

Then, if feasible:

```text
T sampled from {2,3,4}
```

Use balanced horizon sampling.

---

# 35. Truncated BPTT

The method must not “solve inference memory but recreate O(T) training activation memory”.

Config:

```yaml
memory:
  detach_every: 2
```

Example:

```python
if stage_idx % detach_every == 0:
    memory = detach_memory(memory)
```

The memory values persist; the computation graph is truncated.

Evaluate at least:

```text
detach_every = 1
detach_every = 2
full BPTT on feasible short T
```

on a small controlled experiment.

---

# 36. Optional Frozen-Backbone Caching

Because the PTv3 encoder is frozen, a feature/query cache may be useful for memory-only prototyping.

However:

- do NOT use caching to claim official P2 reproduction;
- do NOT silently remove training augmentations;
- distinguish “cached memory prototype” from “live end-to-end decoder training”.

Possible tool:

```text
scripts/cache_local_observations.py
```

Cache only when semantics are verified.

---

# 37. Final Evaluation Matrix

Main methods:

```text
ReScene4D paper reported T=2               [reference only]
ReScene4D P2 reproduction
ReScene4D long-window strong baseline
Naive recurrent memory
Single-timescale fixed memory
Dual-timescale memory
+ gated consolidation
Full Persist4D + memory-conditioned adapter
```

External methods such as AutoSeg3D / ChronoTrack are Related Work unless the input/output/metric protocol can be made genuinely comparable.

Do not force incompatible methods into the main table.

---

# 38. Main Result Table Template

```text
Method
T=2 t-mAP
T=3 t-mAP
T=4 t-mAP
T=5 t-mAP
per-stage mAP
t-SIM
Peak VRAM @ T5
Latency/update @ T5
Throughput @ T5
```

For horizon claims also provide the common S5 subset result.

Never compare a T=2 number from one test set directly against a T=5 number from another as “absolute degradation” without the common-scene protocol.

---

# 39. Core Ablation

Keep ablations tight:

```text
A. ReScene4D strong long-window baseline
B. naive recurrent single memory
C. + identity anchor
D. + gated consolidation
E. + memory-conditioned query adapter = Full
```

Metrics:

```text
T2 t-mAP
T4 t-mAP
T5 t-mAP
t-SIM
VRAM T5
```

No backbone ablation unless reviewers require it.

---

# 40. Identity Drift Diagnostics

Create:

```text
scripts/analyze_long_horizon_identity.py
```

Use architecture-neutral diagnostics wherever possible.

Primary:

```text
t-SIM
trajectory t-IoU / t-mAP
same-GT-instance retrieval similarity
same-class wrong-identity confusion
ID fragmentation rate if definable without changing benchmark semantics
```

Do not compare raw query drift between architectures unless query semantics are demonstrably aligned.

For Persist4D internal analysis, additionally visualize:

```text
anchor similarity vs stage
working-memory similarity vs stage
gate/write magnitude vs stage
```

These are diagnostics, not benchmark metrics.

---

# 41. Compute Scaling Claim

The final paper should empirically report:

```text
historical T
VRAM
latency
throughput
```

for:

```text
ReScene4D all-history
Persist4D bounded-window + memory
```

Do not write:

```text
O(T) vs O(1)
```

as a formal theorem unless exact architecture-level accounting is proven.

Safer language:

> Persist4D keeps the stored temporal representation fixed-capacity and processes a bounded local revisit window, so historical representation size does not grow with the number of processed revisits.

Then support it with measured curves.

---

# 42. Final Paper Story

## Problem

ReScene4D shows that cross-visit temporal context is valuable, but its joint sequence representation is designed around short temporal windows.

P0/P1 confirms on our implementation that increasing T materially increases memory cost and reduces throughput.

Longer revisit horizons also stress temporal identity consistency.

---

## Core idea

Do not keep all past 3D observations active.

Compress historical identity information into a fixed-capacity persistent multi-instance memory.

Use:

```text
bounded local ReScene window
+
persistent identity memory
```

so the current visit can be segmented using short-term geometry and long-term identity context.

---

## Why dual-timescale memory

Plain recurrent updating can solve storage growth but creates a different risk:

```text
progressive identity drift
```

Therefore separate:

```text
slow identity anchor
fast working representation
```

and learn confidence-gated consolidation.

---

## Why memory-conditioned queries

Long-term memory should improve the current representation, not merely assign an ID after segmentation.

Hence the late memory-to-query residual adapter.

---

# 43. Related Work Structure

Use only three focused subsections.

## 43.1 Temporally Sparse 4D Instance Segmentation

Cover:

```text
Mask3D / query-based 3DSIS
ReScan / sparse revisits
ReScene4D
```

Position:

> ReScene4D is the closest task/backbone and the direct starting point.

---

## 43.2 Streaming / Query-Based 3D Instance Tracking

Cover relevant online query-propagation work such as AutoSeg3D / embodied online segmentation.

Position:

> Query propagation itself is not new. These works operate in different dense/streaming observation regimes. Persist4D targets sparse whole-scene revisits and full-scene 4DSIS under long revisit horizons.

---

## 43.3 Compact Long-Term 3D Memory

Cover ChronoTrack and related temporal-memory methods.

Position:

> Compact anti-drift memory has proven useful in single-target 3D tracking, but full-scene sparse 4DSIS requires simultaneous memory competition, entity birth/dormancy, dense instance masks, and many persistent identities.

Do not overstate firstness.

---

# 44. Final Provisional Contributions

Only use these after experiments support them.

### Contribution 1 — Bounded-history 4DSIS

A progressive formulation for temporally sparse full-scene 4D instance segmentation that replaces growing active historical context with a fixed-capacity multi-instance memory plus a bounded local revisit window.

### Contribution 2 — Anti-drift persistent instance memory

A competition-aware dual-timescale memory with stable identity anchors, adaptive working states, and confidence-gated consolidation.

### Contribution 3 — Memory-conditioned long-horizon learning

A lightweight memory-to-query adapter and variable-horizon recurrent training that maintain current segmentation quality while improving identity retention as revisit history grows.

### Contribution 4 — Long-horizon protocol

A controlled native 3RScan evaluation across T=2/3/4/5 that jointly measures 4DSIS accuracy and resource scaling, including a common-scene horizon protocol.

---

# 45. Success Criteria for the Full Project

A publishable outcome should ideally satisfy:

## Short-window preservation

At T=2:

```text
Persist4D close to strong ReScene4D baseline
```

Do not trade away large short-window quality merely for scalability.

---

## Long-horizon gain

At T=4/5 on the controlled common-scene protocol:

```text
Persist4D materially better t-mAP/t-SIM retention
```

than the strongest feasible long-window ReScene4D baseline.

---

## Bounded resource behavior

Measured VRAM/latency growth should be substantially flatter than all-history ReScene4D as T grows.

---

## Mechanistic evidence

Ablation should show:

```text
naive recurrence
<
dual-timescale memory
<
gated + memory-conditioned full method
```

at longer T.

If not:

remove unnecessary modules and simplify claims.

---

# 46. Failure / Pivot Rules

## If G2 fails

Do not implement Persist4D.

Fix reproduction.

---

## If G3 shows long-window training completely fixes performance

Keep only the efficiency/scalability hypothesis if resource scaling remains meaningful.

Reassess whether that is strong enough for the intended venue before continuing.

---

## If naive recurrent memory is already best

Do not force dual-memory complexity.

Reframe around fixed-capacity recurrent 4DSIS.

---

## If T=5 validation sample count is too small

Use T=4 as primary long-horizon endpoint.

Keep T=5 as stress test.

---

## If revisit-order variance is large

Report it and either:

- make training permutation-robust;
- or narrow the claim to a defined revisit-order protocol.

Do not hide order sensitivity.

---

# 47. Engineering Rules

Codex must:

1. preserve official ReScene4D behavior behind feature flags;
2. avoid large rewrites;
3. add unit tests for every new memory component;
4. keep tensor-shape assertions;
5. save all configs and seeds;
6. hash important artifacts/checkpoints;
7. keep the worktree clean at stage boundaries;
8. create one commit per accepted stage;
9. never use GT identity at inference;
10. never leak future stages into current-stage streaming prediction;
11. never silently alter dataset split/order;
12. never silently change evaluation metrics;
13. never claim results not present in generated artifacts;
14. treat P0/P1 measurements as evidence, not target values to fabricate;
15. document every deviation from official ReScene4D.

---

# 48. Stage Order — Non-Negotiable

```text
P0/P1  DONE
  |
  v
G1 = GO
  |
  v
P2  ReScene4D-C T=2 reproduction
  |
  v
G2
  |
  +---- NO -> fix baseline
  |
  v
P3  strong long-window ReScene4D
  |
  v
G3
  |
  +---- NO -> stop/reposition
  |
  v
M0 naive recurrent
  |
M1 fixed memory
  |
M2 dual-timescale
  |
M3 gated consolidation
  |
M4 memory-conditioned query adapter
  |
M5 variable-horizon final training
  |
final evaluation + ablation + paper artifacts
```

---

# 49. Codex Immediate Task — Execute P2 Only

Start now with **P2**.

Do NOT create:

```text
persistent_memory.py
Persist4D
memory adapter
dual memory
```

yet.

P2 completion requires:

```text
P2 config audit
environment manifest
ScanNet prerequisite verification
hardware topology profile
gradient accumulation + LR audit
frozen encoder test
short smoke training
tiny overfit test
formal ReScene4D-C T=2 training
official-vs-reproduced evaluation
P2_G2_REPRODUCTION_REPORT.md
tests
clean worktree
separate commit
```

Final P2 report must end with exactly one of:

```text
G2 = GREEN — authorize P3
G2 = YELLOW — audit/second seed before P3
G2 = RED — do not proceed
```

Do not execute P3 automatically in the same task.

Stop after P2, report the evidence, and wait for explicit authorization.

---

# 50. Expected P2 Response to the Researcher

At completion, summarize:

```text
1. exact baseline reproduced
2. official vs reproduced metrics
3. training hardware/effective batch
4. deviations from paper recipe
5. checkpoint path/hash
6. major reproduction risks
7. G2 verdict
8. git commit
```

Do not discuss Persist4D implementation progress because none should exist yet.
