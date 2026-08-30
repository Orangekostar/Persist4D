# Sonata Second SS5 Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the preregistered Sonata checkpoint and qualify it against the frozen Concerto checkpoint with one matched three-seed official-like T2 validation harness.

**Architecture:** Add one SS5-specific command with `prepare`, `evaluate`, and `finalize` stages. Both model identities use fresh Hydra composition, the same Lightning validation path and runtime controls, while identity-specific config, pretrained weight, and task checkpoint values are explicit inputs; aggregation validates all six run bindings before applying the preregistered gate.

**Tech Stack:** Python 3.10, Hydra/OmegaConf, PyTorch, PyTorch Lightning, pytest, CSV/JSON/Markdown artifacts.

---

### Task 1: Lock the evaluation and gate contracts

**Files:**
- Create: `tests/test_sonata_second_qualification.py`
- Create: `scripts/evaluate_sonata_second_checkpoint.py`

- [ ] **Step 1: Write failing tests for the fixed seeds, shared runtime and metric schema**

```python
def test_evaluation_contract_is_matched_and_preregistered():
    assert qualification.EVALUATION_SEEDS == (45, 46, 47)
    assert qualification.METRIC_KEYS == {
        "t_mAP": "val_mean_t-AP",
        "t_mAP50": "val_mean_t-AP_50",
        "t_mAP25": "val_mean_t-AP_25",
        "overall_mAP": "val_mean_AP",
        "stage1_mAP": "val_mean_stage1-AP",
        "stage2_mAP": "val_mean_stage2-AP",
    }
    assert qualification.RUNTIME_CONTRACT == {
        "accelerator": "gpu",
        "devices": 1,
        "batch_size": 1,
        "num_workers": 4,
        "precision": "32-true",
    }
```

- [ ] **Step 2: Run the contract test and verify RED**

Run: `pytest -q tests/test_sonata_second_qualification.py::test_evaluation_contract_is_matched_and_preregistered`

Expected: FAIL because `scripts.evaluate_sonata_second_checkpoint` does not exist.

- [ ] **Step 3: Implement the constants, model specifications and Hydra overrides**

```python
EVALUATION_SEEDS = (45, 46, 47)
RUNTIME_CONTRACT = {
    "accelerator": "gpu",
    "devices": 1,
    "batch_size": 1,
    "num_workers": 4,
    "precision": "32-true",
}
METRIC_KEYS = {
    "t_mAP": "val_mean_t-AP",
    "t_mAP50": "val_mean_t-AP_50",
    "t_mAP25": "val_mean_t-AP_25",
    "overall_mAP": "val_mean_AP",
    "stage1_mAP": "val_mean_stage1-AP",
    "stage2_mAP": "val_mean_stage2-AP",
}
```

The two model specifications must bind `config_rescene4d_sonata_second` to `SONATA_CHECKPOINT` and `config_p2_rescene4d_concerto_t2` to `CONCERTO_CHECKPOINT`. Both compositions must override `general.gpus=1`, `general.train_mode=false`, `data.batch_size=1`, `data.test_batch_size=1`, `data.num_workers=4`, and `trainer.precision=32-true`.

- [ ] **Step 4: Run the contract test and verify GREEN**

Run: `pytest -q tests/test_sonata_second_qualification.py::test_evaluation_contract_is_matched_and_preregistered`

Expected: PASS.

### Task 2: Enforce full strict checkpoint loading

**Files:**
- Modify: `tests/test_sonata_second_qualification.py`
- Modify: `scripts/evaluate_sonata_second_checkpoint.py`

- [ ] **Step 1: Write a failing strict-load test**

```python
def test_strict_load_rejects_incomplete_state_dict(tmp_path):
    system = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    state = system.state_dict()
    state.pop(next(iter(state)))
    path = tmp_path / "incomplete.ckpt"
    torch.save({"state_dict": state}, path)
    with pytest.raises(RuntimeError, match="strict task checkpoint load failed"):
        qualification.strict_load_task_checkpoint(system, path)
```

- [ ] **Step 2: Run the strict-load test and verify RED**

Run: `pytest -q tests/test_sonata_second_qualification.py::test_strict_load_rejects_incomplete_state_dict`

Expected: FAIL because `strict_load_task_checkpoint` is absent.

- [ ] **Step 3: Implement full Lightning mapping validation and strict loading**

```python
def strict_load_task_checkpoint(system, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("task checkpoint must contain a full Lightning state_dict")
    try:
        incompatible = system.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise RuntimeError("strict task checkpoint load failed") from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict task checkpoint load failed")
    return {"state_dict_entry_count": len(payload["state_dict"]), "strict": True}
```

- [ ] **Step 4: Run strict-load tests and verify GREEN**

Run: `pytest -q tests/test_sonata_second_qualification.py -k strict_load`

Expected: PASS.

### Task 3: Implement the shared official-like evaluator

**Files:**
- Modify: `tests/test_sonata_second_qualification.py`
- Modify: `scripts/evaluate_sonata_second_checkpoint.py`

- [ ] **Step 1: Write failing tests for result validation and per-run bindings**

```python
def test_normalize_metrics_requires_all_finite_unit_interval_values():
    source = {source: torch.tensor(0.25) for source in qualification.METRIC_KEYS.values()}
    assert qualification.normalize_metrics(source)["t_mAP"] == pytest.approx(0.25)
    source.pop("val_mean_AP")
    with pytest.raises(ValueError, match="missing evaluation metric"):
        qualification.normalize_metrics(source)
```

- [ ] **Step 2: Run the result-validation test and verify RED**

Run: `pytest -q tests/test_sonata_second_qualification.py::test_normalize_metrics_requires_all_finite_unit_interval_values`

Expected: FAIL because `normalize_metrics` is absent.

- [ ] **Step 3: Implement one shared Lightning validation function**

The function must call `seed_everything(seed, workers=True)`, instantiate `InstanceSegmentation` from a freshly composed identity config, call `strict_load_task_checkpoint`, instantiate the configured validation dataset and dataloader, then call `Trainer.validate` with no logger or callbacks. It must reject non-preregistered seeds and emit one immutable JSON record containing checkpoint/config hashes, sequence count, GPU, runtime settings, strict-load result, elapsed time, and the six normalized metrics.

- [ ] **Step 4: Run the unit tests and a real one-sequence load smoke for each identity**

Run: `pytest -q tests/test_sonata_second_qualification.py`

Expected: PASS.

Run: `/home/ww/miniconda3/envs/persist4d/bin/python scripts/evaluate_sonata_second_checkpoint.py smoke --model sonata --device 0`

Expected: JSON status `pass`, strict load `true`, one batch evaluated.

Run: `/home/ww/miniconda3/envs/persist4d/bin/python scripts/evaluate_sonata_second_checkpoint.py smoke --model concerto --device 0`

Expected: JSON status `pass`, strict load `true`, one batch evaluated.

### Task 4: Freeze checkpoint provenance artifacts

**Files:**
- Modify: `tests/test_sonata_second_qualification.py`
- Modify: `scripts/evaluate_sonata_second_checkpoint.py`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/CHECKPOINT_MANIFEST.json`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/CHECKPOINT_SELECTION.md`

- [ ] **Step 1: Write failing tests for selected epoch, step, hashes and mode**

```python
def test_checkpoint_manifest_requires_preregistered_top1_and_full_budget(training_manifest):
    manifest = qualification.build_checkpoint_manifest(...)
    assert manifest["sonata"]["epoch"] == 449
    assert manifest["sonata"]["global_step"] == 29700
    assert manifest["sonata"]["selection"] == "highest val_mean_t-AP"
    assert manifest["sonata"]["mode"] == "0444"
```

- [ ] **Step 2: Run the manifest test and verify RED**

Run: `pytest -q tests/test_sonata_second_qualification.py -k checkpoint_manifest`

Expected: FAIL because manifest generation is absent.

- [ ] **Step 3: Implement and run immutable checkpoint preparation**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python scripts/evaluate_sonata_second_checkpoint.py prepare`

Expected: Sonata SHA256 `3d6432711dd9639d9e9203134d846d9a1a29f09b7fb3fbb85375e2127945a199`, epoch `449`, step `29700`; Concerto SHA256 `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`; outputs contain no host-private absolute paths.

- [ ] **Step 4: Verify unit tests and artifact privacy**

Run: `pytest -q tests/test_sonata_second_qualification.py`

Expected: PASS.

Run: `rg -n '/home/|/mnt/shared/|192\\.168\\.' artifacts/sonata_second_perception_v1/checkpoint`

Expected: no matches.

### Task 5: Run the matched six evaluations and aggregate the gate

**Files:**
- Create: `artifacts/sonata_second_perception_v1/checkpoint/runs/sonata_seed45.json`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/runs/sonata_seed46.json`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/runs/sonata_seed47.json`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/runs/concerto_seed45.json`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/runs/concerto_seed46.json`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/runs/concerto_seed47.json`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/official_like_per_seed.csv`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/official_like_summary.csv`
- Create: `artifacts/sonata_second_perception_v1/checkpoint/SONATA_QUALIFICATION_REPORT.md`

- [ ] **Step 1: Write failing gate tests**

```python
def test_qualification_gate_green_requires_threshold_and_spatial_parity():
    result = qualification.qualification_gate(
        sonata={"t_mAP": 0.297, "overall_mAP": 0.40},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=True,
        completed_epochs=450,
    )
    assert result["label"] == "SQ-GREEN"

def test_qualification_gate_yellow_and_red_boundaries():
    assert qualification.qualification_gate(
        sonata={"t_mAP": 0.296, "overall_mAP": 0.40},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=True,
        completed_epochs=450,
    )["label"] == "SQ-YELLOW"
    assert qualification.qualification_gate(
        sonata={"t_mAP": 0.20, "overall_mAP": 0.30},
        concerto={"t_mAP": 0.25, "overall_mAP": 0.40},
        provenance_complete=True,
        completed_epochs=450,
    )["label"] == "SQ-RED"
```

- [ ] **Step 2: Run gate tests and verify RED**

Run: `pytest -q tests/test_sonata_second_qualification.py -k qualification_gate`

Expected: FAIL because `qualification_gate` is absent.

- [ ] **Step 3: Implement matched-run validation, means and gate generation**

Aggregation must require exactly both identities crossed with seeds 45/46/47; equal evaluation contract hashes, dataset/config runtime controls, precision, batch size, worker count and sequence counts; finite six-metric results; complete SS4 provenance; and 450 completed epochs. `SQ-GREEN` requires Sonata mean t-mAP at least `0.297` and Sonata mean overall mAP at least Concerto mean overall mAP. Invalid/incomplete evidence or collapse below Concerto on both primary metrics is `SQ-RED`; other functional non-green cases are `SQ-YELLOW`.

- [ ] **Step 4: Run all six evaluations sequentially on one selected GPU**

Run for each identity and seed: `/home/ww/miniconda3/envs/persist4d/bin/python scripts/evaluate_sonata_second_checkpoint.py evaluate --model <sonata|concerto> --seed <45|46|47> --device 0`

Expected: six immutable JSON records with status `pass` and identical shared-contract bindings.

- [ ] **Step 5: Finalize and verify the qualification artifacts**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python scripts/evaluate_sonata_second_checkpoint.py finalize`

Expected: two CSVs and one report with one explicit `SQ-GREEN`, `SQ-YELLOW`, or `SQ-RED` decision. SS6/SS7 are executed only for `SQ-GREEN`.

### Task 6: Verify and commit SS5

**Files:**
- Modify: only the SS5 script, test, plan and generated checkpoint artifacts listed above.

- [ ] **Step 1: Run focused and relevant regression tests**

Run: `pytest -q tests/test_sonata_second_qualification.py tests/test_sonata_training_evidence.py tests/test_sonata_second_preflight.py`

Expected: PASS.

- [ ] **Step 2: Run static and repository checks**

Run: `ruff check scripts/evaluate_sonata_second_checkpoint.py tests/test_sonata_second_qualification.py`

Expected: PASS.

Run: `git diff --check`

Expected: PASS.

- [ ] **Step 3: Inspect scope and commit**

Run: `git status --short`

Expected: only declared SS5 paths.

Run: `git add docs/superpowers/plans/2026-08-30-sonata-second-ss5-qualification.md scripts/evaluate_sonata_second_checkpoint.py tests/test_sonata_second_qualification.py artifacts/sonata_second_perception_v1/checkpoint && git commit -m "SS5: freeze and qualify Sonata local checkpoint"`

Expected: one SS5 commit; large checkpoint binaries remain ignored and uncommitted.
