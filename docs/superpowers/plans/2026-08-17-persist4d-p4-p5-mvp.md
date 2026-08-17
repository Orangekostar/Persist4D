# Persist4D P4-P5 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backward-compatible ReScene query export and a fixed-capacity recurrent instance memory that runs causally over T=2/3/4/5 sequences.

**Architecture:** ReScene remains the segmentation model and optionally exposes its final normalized query tensor. A separate observation builder restricts predictions to the newest stage, a deterministic memory module assigns persistent slot IDs, and a stateless streaming wrapper passes memory explicitly between adjacent local windows. Evaluation reconstructs sequence predictions from slot IDs without retaining historical point clouds.

**Tech Stack:** Python 3.10, PyTorch 2.6, Hydra/OmegaConf, SciPy Hungarian assignment, PyTorch Lightning checkpoint loading, pytest, existing Pointcept/Concerto data pipeline.

---

## File Map

| File | Responsibility |
| --- | --- |
| `models/rescene.py` | Optional final normalized query export |
| `conf/model/rescene.yaml` | Default-off `return_query_features` switch |
| `models/persistent_memory.py` | Observation/state dataclasses, validation, association, and recurrent update |
| `models/streaming_rescene.py` | Stateless orchestration of ReScene and persistent memory |
| `datasets/semseg.py` | Public loading path for an explicit scan-index subsequence |
| `datasets/streaming_sequence.py` | Causal T1 bootstrap and adjacent T2 window view |
| `conf/model/persist4d.yaml` | Packaged P5 memory and observation settings |
| `scripts/evaluate_persist4d.py` | Real T=2/3/4/5 evaluation, identity diagnostics, and profiling |
| `tests/test_rescene_query_features.py` | P4 output and autograd compatibility |
| `tests/test_persistent_memory.py` | State construction, validation, detach, and boundedness |
| `tests/test_memory_association.py` | One-to-one matching and deterministic ties |
| `tests/test_streaming_sequence.py` | State transitions, causal windows, reset, and wrapper behavior |
| `tests/test_persist4d_evaluator.py` | Sequence accumulation, identity metrics, artifact schema, and CLI faults |

## Task 1: Export Final ReScene Query Features

**Files:**
- Modify: `models/rescene.py:11-176`
- Modify: `models/rescene.py:422-541`
- Modify: `conf/model/rescene.yaml`
- Create: `tests/test_rescene_query_features.py`

- [ ] **Step 1: Write a lightweight ReScene forward fixture and failing compatibility tests**

Create a fixture with zero decoder iterations so it exercises the real
`ReScene.forward()` output assembly without the Concerto backbone:

```python
from types import SimpleNamespace

import torch
from torch import nn

from models.rescene import ReScene


class _Backbone(nn.Module):
    def forward(self, x):
        return object(), [], [[torch.zeros(3, 4)]]


def _stub_rescene(return_query_features: bool) -> ReScene:
    model = ReScene.__new__(ReScene)
    nn.Module.__init__(model)
    model.train_on_segments = False
    model.num_decoders = 0
    model.shared_decoder = True
    model.return_query_features = return_query_features
    model.backbone = _Backbone()
    model.decoder_norm = nn.LayerNorm(4)
    model.get_pos_encs = lambda coords: []
    model.aggregate_features = lambda features, point2segment: ([torch.ones(3, 4)], None)
    model.sample_and_batch_features = lambda features: (
        torch.ones(1, 3, 4),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    model.initialize_queries = lambda **kwargs: (
        torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
        torch.zeros(1, 2, 4),
        None,
    )
    model.mask_module = lambda queries, features: (
        torch.ones(1, 2, 3),
        None,
        torch.ones(1, 3, 2),
    )
    model.unstack_batched = lambda logits, mapping: [logits[0]]
    model._set_aux_loss = lambda *args: []
    return model
```

Add tests that assert disabled keys are exactly the legacy key set, enabled
output adds only `query_features`, shape is `[1,2,4]`, values equal
`decoder_norm(queries)`, and `sum().backward()` reaches the input query tensor.

- [ ] **Step 2: Run the P4 tests and verify RED**

Run:

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_rescene_query_features.py -q
```

Expected: constructor/config assertions fail because `return_query_features`
does not exist and enabled output lacks `query_features`.

- [ ] **Step 3: Add the default-off constructor field and conditional output**

Add the constructor argument and state:

```python
def __init__(
    self,
    config,
    hidden_dim,
    num_queries,
    num_heads,
    dim_feedforward,
    sample_sizes,
    shared_decoder,
    num_classes,
    num_decoders,
    dropout,
    pre_norm,
    positional_encoding_type,
    non_parametric_queries,
    train_on_segments,
    normalize_pos_enc,
    use_level_embed,
    scatter_type,
    hlevels,
    use_np_features,
    voxel_size,
    max_sample_size,
    random_queries,
    gauss_scale,
    random_query_both,
    random_normal,
    D,
    num_changes,
    temporal_masking,
    use_changes_loss,
    save_segment_info,
    return_query_features=False,
):
    super().__init__()
    self.return_query_features = bool(return_query_features)
```

After the unchanged final `mask_module` call and output dictionary assembly,
add:

```python
if self.return_query_features:
    output_dict["query_features"] = self.decoder_norm(queries)
```

Add to `conf/model/rescene.yaml`:

```yaml
return_query_features: false
```

- [ ] **Step 4: Run focused and existing model-contract tests**

Run:

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_rescene_query_features.py \
  tests/test_p2_training_contract.py \
  tests/test_p2_native_training_smoke.py
```

Expected: all CPU tests pass; existing opt-in GPU artifact tests remain skipped.

- [ ] **Step 5: Commit P4**

```bash
git add models/rescene.py conf/model/rescene.yaml tests/test_rescene_query_features.py
git commit -m "feat: expose ReScene query features"
```

## Task 2: Define Observation And Persistent State Contracts

**Files:**
- Create: `models/persistent_memory.py`
- Create: `tests/test_persistent_memory.py`

- [ ] **Step 1: Write failing state and observation tests**

Test these exact contracts:

```python
def test_empty_state_has_fixed_shapes_and_sentinels():
    state = PersistentMemoryState.empty(
        batch_size=2,
        capacity=3,
        feature_dim=4,
        class_count=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert state.embedding.shape == (2, 3, 4)
    assert state.class_prob.shape == (2, 3, 5)
    assert not state.occupied.any()
    assert not state.active.any()
    assert torch.equal(state.last_seen, torch.full((2, 3), -1))


def test_detach_returns_new_state_without_changing_values():
    state = _state_with_gradients()
    detached = state.detach()
    assert detached is not state
    assert torch.equal(detached.embedding, state.embedding)
    assert detached.embedding.grad_fn is None
```

Also test rejection of negative capacity, wrong tensor ranks, inconsistent batch
sizes, non-finite floating values, `active & ~occupied`, negative ages, and
`last_seen < -1`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_persistent_memory.py -q
```

Expected: collection fails with `ModuleNotFoundError: models.persistent_memory`.

- [ ] **Step 3: Implement typed dataclasses and validation**

Create these public types:

```python
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LocalInstanceObservation:
    features: Tensor
    class_prob: Tensor
    confidence: Tensor
    latest_mask: list[Tensor]
    valid: Tensor

    def validate(self) -> None:
        batch_size, query_count, feature_dim = self.features.shape
        if feature_dim <= 0 or self.class_prob.shape[:2] != (batch_size, query_count):
            raise ValueError("observation tensor shapes are inconsistent")
        if self.confidence.shape != (batch_size, query_count):
            raise ValueError("observation confidence shape is inconsistent")
        if self.valid.shape != (batch_size, query_count) or self.valid.dtype != torch.bool:
            raise ValueError("observation valid mask is inconsistent")
        if len(self.latest_mask) != batch_size:
            raise ValueError("observation mask batch is inconsistent")
        if any(mask.ndim != 2 or mask.shape[0] != query_count for mask in self.latest_mask):
            raise ValueError("observation masks must have shape [Q,S_latest]")
        for tensor in (self.features, self.class_prob, self.confidence):
            if not torch.isfinite(tensor).all():
                raise ValueError("observation contains non-finite values")


@dataclass(frozen=True)
class PersistentMemoryState:
    embedding: Tensor
    class_prob: Tensor
    confidence: Tensor
    occupied: Tensor
    active: Tensor
    age: Tensor
    last_seen: Tensor

    @classmethod
    def empty(cls, *, batch_size, capacity, feature_dim, class_count, device, dtype):
        if min(batch_size, capacity, feature_dim, class_count) <= 0:
            raise ValueError("state dimensions must be positive")
        return cls(
            embedding=torch.zeros(batch_size, capacity, feature_dim, device=device, dtype=dtype),
            class_prob=torch.zeros(batch_size, capacity, class_count, device=device, dtype=dtype),
            confidence=torch.zeros(batch_size, capacity, device=device, dtype=dtype),
            occupied=torch.zeros(batch_size, capacity, device=device, dtype=torch.bool),
            active=torch.zeros(batch_size, capacity, device=device, dtype=torch.bool),
            age=torch.zeros(batch_size, capacity, device=device, dtype=torch.long),
            last_seen=torch.full((batch_size, capacity), -1, device=device, dtype=torch.long),
        )

    def detach(self):
        return type(self)(
            embedding=self.embedding.detach(),
            class_prob=self.class_prob.detach(),
            confidence=self.confidence.detach(),
            occupied=self.occupied.detach(),
            active=self.active.detach(),
            age=self.age.detach(),
            last_seen=self.last_seen.detach(),
        )

    def tensors(self):
        return (
            self.embedding,
            self.class_prob,
            self.confidence,
            self.occupied,
            self.active,
            self.age,
            self.last_seen,
        )
```

Add `validate()` properties for `batch_size`, `capacity`, `feature_dim`, and
`class_count`. Validation must run before every public memory step.

- [ ] **Step 4: Verify contracts and static checks**

Run:

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_persistent_memory.py -q
/home/ww/miniconda3/bin/ruff check models/persistent_memory.py tests/test_persistent_memory.py
```

Expected: all state tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit state contracts**

```bash
git add models/persistent_memory.py tests/test_persistent_memory.py
git commit -m "feat: define persistent memory state"
```

## Task 3: Build Latest-Stage Observations

**Files:**
- Modify: `models/persistent_memory.py`
- Modify: `tests/test_persistent_memory.py`

- [ ] **Step 1: Write failing observation-builder tests**

Use two samples with different segment counts and verify:

```python
outputs = {
    "query_features": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    "pred_logits": torch.tensor([[[4.0, 1.0, -2.0], [0.1, 0.2, 3.0]]]),
    "pred_masks": [torch.tensor([[5.0, -5.0], [-5.0, 5.0], [5.0, -5.0]])],
}
segment_stages = [torch.tensor([0, 1, 1])]
observation = build_local_observation(
    outputs,
    segment_stages,
    latest_stage=1,
    background_class=2,
    confidence_threshold=0.5,
    mask_threshold=0.5,
    minimum_mask_support=1,
)
assert observation.latest_mask[0].shape == (2, 2)
assert observation.valid.tolist() == [[True, False]]
```

Add failures for a missing query key, stage/mask length mismatch, absent latest
stage, invalid background index, and non-finite logits.

- [ ] **Step 2: Run the builder tests and verify RED**

Run:

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_persistent_memory.py -q -k local_observation
```

Expected: import fails because `build_local_observation` is not defined.

- [ ] **Step 3: Implement the observation builder**

Add:

```python
def build_local_observation(
    outputs,
    segment_stages,
    *,
    latest_stage,
    background_class,
    confidence_threshold,
    mask_threshold,
    minimum_mask_support,
):
    features = outputs["query_features"]
    logits = outputs["pred_logits"]
    probabilities = logits.softmax(dim=-1)
    foreground = torch.cat(
        (probabilities[..., :background_class], probabilities[..., background_class + 1 :]),
        dim=-1,
    )
    confidence = foreground.max(dim=-1).values
    latest_masks = []
    support = []
    for mask_logits, stages in zip(outputs["pred_masks"], segment_stages):
        if mask_logits.shape[0] != stages.numel():
            raise ValueError("segment stage count does not match prediction masks")
        selected = mask_logits[stages == latest_stage].transpose(0, 1)
        if selected.shape[1] == 0:
            raise ValueError("latest stage has no predicted segments")
        latest_masks.append(selected)
        support.append((selected.sigmoid() >= mask_threshold).sum(dim=1))
    valid = confidence.ge(confidence_threshold) & torch.stack(support).ge(
        minimum_mask_support
    )
    observation = LocalInstanceObservation(
        features=features,
        class_prob=probabilities,
        confidence=confidence,
        latest_mask=latest_masks,
        valid=valid,
    )
    observation.validate()
    return observation
```

Preserve the original class dimension in `class_prob`; only confidence excludes
the no-object class.

- [ ] **Step 4: Run all persistent-memory tests**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_persistent_memory.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit observation construction**

```bash
git add models/persistent_memory.py tests/test_persistent_memory.py
git commit -m "feat: build local instance observations"
```

## Task 4: Implement Deterministic One-To-One Association

**Files:**
- Modify: `models/persistent_memory.py`
- Create: `tests/test_memory_association.py`

- [ ] **Step 1: Write failing association tests**

Cover exact matches, class conflicts, below-threshold pairs, empty memory,
invalid observations, and ties:

```python
def test_association_is_one_to_one_and_uses_lowest_indices_for_ties():
    observation = _observation(
        features=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        valid=torch.tensor([[True, True]]),
    )
    state = _occupied_state(
        embeddings=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    )
    result = associate_observations(
        observation,
        state,
        class_weight=0.0,
        association_threshold=0.5,
    )
    assert result.slot_for_query.tolist() == [[0, 1]]
    assert result.query_for_slot.tolist() == [[0, 1]]
```

Assert all accepted slot IDs and query IDs are unique per sample.

- [ ] **Step 2: Run association tests and verify RED**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_memory_association.py -q
```

Expected: import fails because `AssociationResult` and
`associate_observations` do not exist.

- [ ] **Step 3: Implement score calculation and Hungarian matching**

Add:

```python
@dataclass(frozen=True)
class AssociationResult:
    slot_for_query: Tensor
    query_for_slot: Tensor
    score_for_query: Tensor


def associate_observations(
    observation,
    state,
    *,
    class_weight,
    association_threshold,
):
    observation.validate()
    state.validate()
    query = torch.nn.functional.normalize(observation.features.float(), dim=-1)
    memory = torch.nn.functional.normalize(state.embedding.float(), dim=-1)
    cosine = torch.einsum("bkd,bqd->bkq", memory, query)
    class_score = torch.einsum(
        "bkc,bqc->bkq", state.class_prob.float(), observation.class_prob.float()
    )
    score = cosine + float(class_weight) * class_score
    batch_size, capacity, query_count = score.shape
    slot_for_query = torch.full(
        (batch_size, query_count), -1, device=score.device, dtype=torch.long
    )
    query_for_slot = torch.full(
        (batch_size, capacity), -1, device=score.device, dtype=torch.long
    )
    score_for_query = torch.full(
        (batch_size, query_count), float("-inf"), device=score.device
    )
    for batch_index in range(batch_size):
        slots = state.occupied[batch_index].nonzero(as_tuple=False).flatten()
        queries = observation.valid[batch_index].nonzero(as_tuple=False).flatten()
        if slots.numel() == 0 or queries.numel() == 0:
            continue
        selected = score[batch_index][slots][:, queries]
        cost = -selected.detach().double().cpu().numpy()
        row_rank = numpy.arange(cost.shape[0], dtype=numpy.float64)[:, None]
        column_rank = numpy.arange(cost.shape[1], dtype=numpy.float64)[None, :]
        tie_rank = (
            numpy.abs(row_rank - column_rank) * (cost.shape[1] + 1)
            + column_rank
        )
        tie_scale = numpy.spacing(max(1.0, float(numpy.abs(cost).max())))
        cost = cost + tie_rank * tie_scale
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows.tolist(), columns.tolist()):
            slot = int(slots[row])
            query = int(queries[column])
            pair_score = score[batch_index, slot, query]
            if float(pair_score) < association_threshold:
                continue
            slot_for_query[batch_index, query] = slot
            query_for_slot[batch_index, slot] = query
            score_for_query[batch_index, query] = pair_score
    return AssociationResult(slot_for_query, query_for_slot, score_for_query)
```

Use the float64 rank perturbation only for Hungarian ordering. It favors aligned
low-index rows/columns when the original scores are numerically tied. Threshold
the unmodified score tensor and return maps on the observation device.

- [ ] **Step 4: Verify association and state tests**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_memory_association.py tests/test_persistent_memory.py
```

Expected: all tests pass, including repeated tie runs.

- [ ] **Step 5: Commit association**

```bash
git add models/persistent_memory.py tests/test_memory_association.py
git commit -m "feat: associate observations with memory slots"
```

## Task 5: Implement Recurrent Memory Transitions

**Files:**
- Modify: `models/persistent_memory.py`
- Modify: `tests/test_persistent_memory.py`
- Modify: `tests/test_memory_association.py`
- Create: `conf/model/persist4d.yaml`

- [ ] **Step 1: Write failing lifecycle and boundedness tests**

Add tests for:

```python
def test_matched_dormant_slot_reactivates_without_changing_slot_id():
    memory = PersistentMemory(capacity=2, update_rate=0.5, max_update_rate=0.5)
    state = memory.empty_state(_observation_one_query())
    born = memory.step(_observation_one_query(), state, stage_index=0)
    dormant = memory.step(_observation_no_valid_queries(), born.state, stage_index=1)
    active = memory.step(_observation_one_query(), dormant.state, stage_index=2)
    assert born.slot_ids.item() == 0
    assert not dormant.state.active[0, 0]
    assert active.slot_ids.item() == 0
    assert active.state.active[0, 0]
    assert active.state.last_seen[0, 0].item() == 2


def test_state_storage_is_constant_for_one_hundred_stages():
    state = memory.empty_state(observation)
    initial_shapes = tuple(t.shape for t in state.tensors())
    for stage in range(100):
        state = memory.step(observation, state, stage_index=stage).state
    assert tuple(t.shape for t in state.tensors()) == initial_shapes
```

Also cover lowest-free-slot birth, invalid-query rejection, full-capacity birth
rejection, decreasing stage rejection, batch mismatch, normalized embedding
updates, and no in-place mutation of the input state.

- [ ] **Step 2: Run lifecycle tests and verify RED**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_persistent_memory.py tests/test_memory_association.py
```

Expected: failures because `PersistentMemory` and `MemoryStepResult` are absent.

- [ ] **Step 3: Implement `PersistentMemory`**

Expose:

```python
@dataclass(frozen=True)
class MemoryStepResult:
    state: PersistentMemoryState
    slot_ids: Tensor
    association_scores: Tensor
    rejected_births: Tensor


class PersistentMemory(torch.nn.Module):
    def __init__(
        self,
        *,
        capacity=100,
        class_weight=0.25,
        association_threshold=0.5,
        update_rate=0.2,
        max_update_rate=0.2,
    ):
        super().__init__()
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 <= update_rate <= max_update_rate <= 1.0:
            raise ValueError("memory update rates must satisfy 0 <= rate <= max <= 1")
        self.capacity = int(capacity)
        self.class_weight = float(class_weight)
        self.association_threshold = float(association_threshold)
        self.update_rate = float(update_rate)
        self.max_update_rate = float(max_update_rate)

    def empty_state(self, observation):
        observation.validate()
        return PersistentMemoryState.empty(
            batch_size=observation.features.shape[0],
            capacity=self.capacity,
            feature_dim=observation.features.shape[2],
            class_count=observation.class_prob.shape[2],
            device=observation.features.device,
            dtype=observation.features.dtype,
        )
```

`step()` clones every state tensor, marks all occupied slots dormant, applies
accepted matches, then allocates valid unmatched observations in ascending query
order to ascending free-slot order. It computes EMA updates out of place and
normalizes updated embeddings with `torch.nn.functional.normalize`.

Create `conf/model/persist4d.yaml` with a package override rather than replacing
the ReScene model:

```yaml
# @package persist4d
capacity: 100
class_weight: 0.25
association_threshold: 0.5
update_rate: 0.2
max_update_rate: 0.2
confidence_threshold: 0.5
mask_threshold: 0.5
minimum_mask_support: 1
background_class: 18
```

- [ ] **Step 4: Verify lifecycle and configuration**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_persistent_memory.py tests/test_memory_association.py
/home/ww/miniconda3/bin/ruff check \
  models/persistent_memory.py tests/test_persistent_memory.py \
  tests/test_memory_association.py
```

Expected: all tests and Ruff pass.

- [ ] **Step 5: Commit recurrent memory**

```bash
git add models/persistent_memory.py conf/model/persist4d.yaml \
  tests/test_persistent_memory.py tests/test_memory_association.py
git commit -m "feat: update fixed-capacity persistent memory"
```

## Task 6: Add The Stateless Streaming Wrapper

**Files:**
- Create: `models/streaming_rescene.py`
- Create: `tests/test_streaming_sequence.py`

- [ ] **Step 1: Write failing wrapper tests with a fake base model**

Define a fake base model that records calls and returns fixed query/logit/mask
outputs. Assert:

```python
def test_forward_step_passes_state_explicitly_and_preserves_base_predictions():
    wrapper = StreamingReScene(_FakeReScene(), PersistentMemory(capacity=3), settings)
    result, state = wrapper.forward_step(
        x=object(),
        point2segment=[torch.tensor([0, 1])],
        raw_coordinates=None,
        segment_stages=[torch.tensor([0, 1])],
        state=None,
        stage_index=1,
        is_eval=True,
    )
    assert torch.equal(result["pred_logits"], wrapper.base_model.output["pred_logits"])
    assert "persistent_slot_ids" in result
    assert state.occupied.any()
    assert not hasattr(wrapper, "state")
```

Add tests for reset on `state=None`, batch mismatch, decreasing stage index,
missing query export, and non-finite model outputs.

- [ ] **Step 2: Run wrapper tests and verify RED**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_streaming_sequence.py -q
```

Expected: collection fails with `ModuleNotFoundError: models.streaming_rescene`.

- [ ] **Step 3: Implement `StreamingReScene`**

Create:

```python
class StreamingReScene(torch.nn.Module):
    def __init__(self, base_model, memory, observation_settings):
        super().__init__()
        if not getattr(base_model, "return_query_features", False):
            raise ValueError("base ReScene must enable return_query_features")
        self.base_model = base_model
        self.memory = memory
        self.observation_settings = dict(observation_settings)

    def forward_step(
        self,
        *,
        x,
        point2segment,
        raw_coordinates,
        segment_stages,
        state,
        stage_index,
        is_eval=True,
    ):
        outputs = self.base_model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
        )
        latest_local_stages = [
            int(stages.max().item()) for stages in segment_stages if stages.numel()
        ]
        if len(latest_local_stages) != len(segment_stages):
            raise ValueError("every sample must contain temporal stage metadata")
        if len(set(latest_local_stages)) != 1:
            raise ValueError("all samples must share the latest local stage")
        observation = build_local_observation(
            outputs,
            segment_stages,
            latest_stage=latest_local_stages[0],
            **self.observation_settings,
        )
        if state is None:
            state = self.memory.empty_state(observation)
        step = self.memory.step(observation, state, stage_index=stage_index)
        result = dict(outputs)
        result["persistent_slot_ids"] = step.slot_ids
        result["persistent_association_scores"] = step.association_scores
        return result, step.state
```

Do not retain `outputs`, `observation`, or state on `self`.

- [ ] **Step 4: Verify wrapper and all P4-P5 unit tests**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_rescene_query_features.py tests/test_persistent_memory.py \
  tests/test_memory_association.py tests/test_streaming_sequence.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the streaming wrapper**

```bash
git add models/streaming_rescene.py tests/test_streaming_sequence.py
git commit -m "feat: stream ReScene through persistent memory"
```

## Task 7: Expose Causal Adjacent Windows

**Files:**
- Modify: `datasets/semseg.py:388-520`
- Create: `datasets/streaming_sequence.py`
- Modify: `tests/test_streaming_sequence.py`
- Modify: `tests/test_temporal_loader.py`

- [ ] **Step 1: Write RED tests for explicit subsequences and causal windows**

Extend the existing three-scan fixture and assert:

```python
def test_explicit_scan_indices_preserve_default_sample_and_make_t2_pair(tmp_path):
    dataset = _make_dataset(_make_three_scan_fixture(tmp_path)[0], temporal_window=3)
    default = dataset[0]
    explicit = dataset.load_scan_indices(0, dataset.sequence_indices[0])
    for default_item, explicit_item in zip(default, explicit):
        np.testing.assert_array_equal(default_item, explicit_item)
    pair = dataset.load_scan_indices(
        0,
        dataset.sequence_indices[0][1:3],
        change_file=None,
    )
    assert set(pair[0][:, 3].astype(int)) == {0, 1}
```

Test `causal_windows([10,11,12,13]) == [(10,), (10,11), (11,12),
(12,13)]`, where the singleton is the T1 bootstrap and every later window is
T2. Reject fewer than two scans and duplicate/non-integral indices.

- [ ] **Step 2: Run loader/window tests and verify RED**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_temporal_loader.py tests/test_streaming_sequence.py
```

Expected: failures because `load_scan_indices` and `causal_windows` are absent.

- [ ] **Step 3: Refactor the loader without changing default behavior**

In `SemanticSegmentationDataset`, make `__getitem__` delegate:

```python
_USE_SEQUENCE_CHANGE_FILE = object()


def __getitem__(self, idx):
    idx = idx % len(self.sequence_indices)
    return self.load_scan_indices(
        idx,
        self.sequence_indices[idx],
    )


def load_scan_indices(
    self,
    context_idx,
    scan_indices,
    *,
    change_file=_USE_SEQUENCE_CHANGE_FILE,
):
    scan_indices = np.asarray(scan_indices, dtype=int)
    if scan_indices.ndim != 1 or scan_indices.size == 0:
        raise ValueError("scan_indices must be a non-empty rank-1 sequence")
    return self._load_scan_sequence(
        context_idx=context_idx,
        scan_indices=scan_indices,
        change_file=(
            self.change_files[context_idx]
            if change_file is _USE_SEQUENCE_CHANGE_FILE
            else change_file
        ),
    )
```

Rename the current `__getitem__` implementation to `_load_scan_sequence`, give
it the exact keyword-only arguments `context_idx`, `scan_indices`, and
`change_file`, replace its local `idx` references with `context_idx`, remove its
first two index-resolution lines, and replace `self.change_files[idx]` with the
argument `change_file`. Preserve all augmentation, known-empty, label,
segment-offset, and return-value statements unchanged. This is a mechanical
extraction; the array-equality test is the acceptance condition.

Create `datasets/streaming_sequence.py`:

```python
def causal_windows(scan_indices):
    indices = tuple(int(index) for index in scan_indices)
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError("a streaming sequence needs at least two unique scans")
    return ((indices[0],),) + tuple(
        (indices[index - 1], indices[index])
        for index in range(1, len(indices))
    )
```

- [ ] **Step 4: Verify exact default-loader parity**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_temporal_loader.py tests/test_p2_data_failure.py \
  tests/test_p2_unsupervised_sequence_filter.py tests/test_streaming_sequence.py
```

Expected: all tests pass; existing loader samples remain array-identical.

- [ ] **Step 5: Commit causal window loading**

```bash
git add datasets/semseg.py datasets/streaming_sequence.py \
  tests/test_temporal_loader.py tests/test_streaming_sequence.py
git commit -m "feat: load causal ReScene windows"
```

## Task 8: Accumulate Persistent Sequence Predictions And Metrics

**Files:**
- Create: `scripts/evaluate_persist4d.py`
- Create: `tests/test_persist4d_evaluator.py`

- [ ] **Step 1: Write failing pure-function evaluator tests**

Define synthetic stages where GT identity 7 receives local query IDs 0, 4, 2
but persistent slot 3 throughout. Assert:

```python
def test_identity_diagnostics_count_switches_and_reactivation():
    result = identity_diagnostics(
        gt_ids_by_stage=[[7], [7], [], [7]],
        predicted_ids_by_stage=[[3], [3], [], [3]],
    )
    assert result == {
        "matched_identity_observations": 3,
        "identity_switches": 0,
        "reactivation_events": 1,
        "correct_reactivations": 1,
        "reactivation_accuracy": 1.0,
    }
```

Test a changed slot produces one switch and one incorrect reactivation. Test
sequence accumulation places each stage mask into one persistent slot, keeps
class-score averages, and never concatenates query features across stages.

Add a CLI fault test that requires an output path, rejects an existing output,
and writes a `status=failed` JSON artifact on malformed input.

- [ ] **Step 2: Run evaluator tests and verify RED**

```bash
env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest \
  tests/test_persist4d_evaluator.py -q
```

Expected: collection fails because `scripts.evaluate_persist4d` is absent.

- [ ] **Step 3: Implement pure accumulation and identity diagnostics first**

Expose:

```python
@dataclass
class SequenceAccumulator:
    capacity: int
    stage_masks: list[dict[int, torch.Tensor]]
    class_prob_sum: torch.Tensor
    class_prob_count: torch.Tensor

    def add_stage(self, masks, class_prob, slot_ids):
        valid_slots = slot_ids[slot_ids >= 0]
        if valid_slots.unique().numel() != valid_slots.numel():
            raise ValueError("a stage cannot assign one slot more than once")
        stage = {}
        for query_index, slot in enumerate(slot_ids.tolist()):
            if slot < 0:
                continue
            stage[int(slot)] = masks[query_index].detach().bool().cpu()
            self.class_prob_sum[slot] += class_prob[query_index].detach().cpu()
            self.class_prob_count[slot] += 1
        self.stage_masks.append(stage)


def identity_diagnostics(gt_ids_by_stage, predicted_ids_by_stage):
    previous = {}
    switches = 0
    reactivations = 0
    correct_reactivations = 0
    observations = 0
    for stage_index, (gt_ids, predicted_ids) in enumerate(
        zip(gt_ids_by_stage, predicted_ids_by_stage)
    ):
        if len(gt_ids) != len(predicted_ids):
            raise ValueError("GT and predicted identity lists must align")
        for gt_id, predicted_id in zip(gt_ids, predicted_ids):
            observations += 1
            if gt_id in previous:
                prior_id, prior_stage = previous[gt_id]
                switches += int(prior_id != predicted_id)
                if stage_index - prior_stage > 1:
                    reactivations += 1
                    correct_reactivations += int(prior_id == predicted_id)
            previous[gt_id] = (predicted_id, stage_index)
    return {
        "matched_identity_observations": observations,
        "identity_switches": switches,
        "reactivation_events": reactivations,
        "correct_reactivations": correct_reactivations,
        "reactivation_accuracy": (
            correct_reactivations / reactivations if reactivations else None
        ),
    }
```

Unit tests must pass before importing Hydra, Lightning, Concerto, or CUDA.

- [ ] **Step 4: Implement the real evaluation CLI**

The CLI must:

1. Compose `config_p2_rescene4d_concerto_t2` with local CSV logging and
   `model.return_query_features=true`.
2. Instantiate `InstanceSegmentation`, load the canonical checkpoint strictly
   with `weights_only=False`, move to one selected CUDA device, and call eval.
3. Instantiate validation datasets for T=2/3/4/5 and reuse the official
   validation collator.
4. For each sequence, call the T1 bootstrap then adjacent T2 windows, reset
   state at the scene boundary, and release local tensors after each step.
5. Match current-stage predicted masks to GT masks by class-compatible IoU for
   diagnostics only; GT never enters memory association.
6. Accumulate persistent slot masks and feed the existing metric adapter where
   its input contract is satisfied.
7. Record for each horizon: sample count, t-mAP, t-REC, per-stage AP, identity
   switches, reactivation accuracy, rejected births, peak allocated CUDA bytes,
   latency, throughput, and serialized state bytes.
8. Write atomically to `artifacts/P5/persist4d_mvp_eval.json` with repo-relative
   or `external:` references and no GPU UUID.

The artifact root schema is:

```python
{
    "schema_version": 1,
    "status": "pass",
    "method": "persist4d_p5_single_memory",
    "source_commit": git_commit(repo_root),
    "checkpoint": {
        "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
        "sha256": sha256_file(checkpoint_path),
    },
    "settings": {"capacity": 100, "local_window": 2},
    "horizons": [{"T": 2}, {"T": 3}, {"T": 4}, {"T": 5}],
    "bounded_state": {"constant_shape": True, "maximum_state_bytes": 0},
    "errors": [],
}
```

Populate numeric values only from the completed run; do not emit fabricated or
synthetic measurements.

- [ ] **Step 5: Verify evaluator unit and CLI fault contracts**

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_persist4d_evaluator.py tests/test_streaming_sequence.py
/home/ww/miniconda3/bin/ruff check \
  scripts/evaluate_persist4d.py tests/test_persist4d_evaluator.py
```

Expected: all CPU tests and Ruff pass.

- [ ] **Step 6: Commit evaluator code before generating artifacts**

```bash
git add scripts/evaluate_persist4d.py tests/test_persist4d_evaluator.py
git commit -m "feat: evaluate persistent sequence identities"
```

## Task 9: Real Checkpoint And A40 Gates

**Files:**
- Modify: `tests/test_rescene_query_features.py`
- Modify: `tests/test_streaming_sequence.py`
- Create: `artifacts/P5/persist4d_mvp_eval.json`
- Create: `artifacts/P5/persist4d_mvp_eval.md`

- [ ] **Step 1: Add opt-in real checkpoint tests before running the CLI**

Add tests gated by `P5_VERIFY_GPU_ARTIFACTS=1` that assert:

```python
assert report["status"] == "pass"
assert [item["T"] for item in report["horizons"]] == [2, 3, 4, 5]
assert all(item["loaded_sequences"] > 0 for item in report["horizons"])
assert report["bounded_state"]["constant_shape"] is True
assert report["bounded_state"]["maximum_state_bytes"] > 0
assert report["checkpoint"]["sha256"] == canonical_checkpoint_sha256()
```

The real P4 test must strict-load the canonical checkpoint, run one T=2
validation sample with export disabled and enabled under fixed seeds, and assert
all legacy prediction tensors are equal while enabled output adds a finite
`[1,100,128]` query tensor.

- [ ] **Step 2: Run the opt-in tests and verify RED**

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 P5_VERIFY_GPU_ARTIFACTS=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_rescene_query_features.py tests/test_streaming_sequence.py
```

Expected: artifact tests fail because the real P5 report does not exist.

- [ ] **Step 3: Run the real evaluator on GPU0**

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CONCERTO_CHECKPOINT=/home/ww/.cache/persist4d/concerto/concerto_base.pth \
  /home/ww/miniconda3/envs/persist4d/bin/python \
  scripts/evaluate_persist4d.py \
  --checkpoint checkpoints/rescene4d_concerto_t2_repro.ckpt \
  --output artifacts/P5/persist4d_mvp_eval.json \
  --markdown artifacts/P5/persist4d_mvp_eval.md \
  --horizons 2 3 4 5 \
  --device cuda:0
```

Expected: exit 0, all four horizons have real samples, no OOM/non-finite error,
and state shapes/bytes are constant with stage count.

- [ ] **Step 4: Run artifact, privacy, regression, and diff gates**

```bash
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 P5_VERIFY_GPU_ARTIFACTS=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_rescene_query_features.py tests/test_persistent_memory.py \
  tests/test_memory_association.py tests/test_streaming_sequence.py \
  tests/test_persist4d_evaluator.py

env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q

/home/ww/miniconda3/envs/persist4d/bin/python \
  /home/ww/CCFA-Skills/ccf-common/scripts/check_path_privacy.py artifacts/P5

git diff --check
```

Expected: focused GPU gates pass; full CPU suite passes with only documented
opt-in GPU skips; privacy and diff checks pass.

- [ ] **Step 5: Review the P5 method gate without using official AP as a target**

The Markdown report must state:

- whether default T=2 segmentation predictions are unchanged;
- whether T=2/3/4/5 execute with fixed state size;
- internal baseline versus persistent identity switches and t-REC;
- T=4/T=5 reactivation accuracy and association failure counts;
- VRAM, latency, throughput, and state bytes;
- one of `P5_MVP_PASS`, `P5_ASSOCIATION_DIAGNOSIS`, or `P5_STREAMING_BLOCKED`.

`P5_MVP_PASS` requires bounded execution plus an improvement in at least one
T=4/T=5 identity metric without changing T=2 segmentation predictions.

- [ ] **Step 6: Commit verified P5 evidence**

```bash
git add tests/test_rescene_query_features.py tests/test_streaming_sequence.py \
  artifacts/P5/persist4d_mvp_eval.json artifacts/P5/persist4d_mvp_eval.md
git commit -m "test: verify Persist4D P5 MVP"
```

## Final Plan Self-Review Checklist

- [ ] Every included design component maps to one task and one focused test.
- [ ] ReScene default output compatibility is verified before memory work.
- [ ] GT instance IDs are confined to evaluation and never association.
- [ ] State is explicit, fixed-capacity, scene-reset, and never stored on the wrapper.
- [ ] T1 bootstrap plus adjacent T2 windows preserves causal execution.
- [ ] Real metrics are generated only by the committed evaluator.
- [ ] No step optimizes for the paper-reported absolute AP.
- [ ] P7/P8/P9 remain outside this implementation plan.
