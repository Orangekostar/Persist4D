# Persist4D Final Evidence Design

Date: 2026-08-24
Status: approved by standing user authorization
Source prompt: `docs/Persist4D Final Evidence Completion - Capacity, External Validation and Paper Freeze.md`

## Objective

Complete the final evidence package without changing the frozen Persist4D
architecture. The package must measure bounded-state capacity, attempt an
independent Rescan validation under audited dataset semantics, position related
work conservatively, and freeze only claims supported by reproduced evidence.

## Immutable Boundaries

- `ARCHITECTURE_STATUS=FINAL_LOCK`.
- Do not add eviction, compression, lifecycle, or memory modules.
- Do not change association, update, confidence, or model-forward semantics.
- Do not modify `artifacts/reviewer_closure/`.
- Do not integrate LivingScenes into the model or training pipeline.
- Do not invent chronology, label mappings, ambiguity resolution, baselines,
  samples, metrics, confidence intervals, or published values.
- Ground truth is allowed only after inference for evaluation and coordinate
  alignment explicitly provided by the dataset.
- Large datasets, checkpoints, and prediction caches remain outside Git.

## Selected Approach

Use a frozen-observation replay architecture.

1. Validate the final source tree, checkpoint, config, dataset inventory, and
   prediction-cache provenance.
2. Materialize one content-addressed local-observation sidecar only if the
   existing frozen cache cannot expose the required observation tensors.
3. Replay the exact same observations through the unchanged persistent-memory
   implementation for capacities 64, 100, 128, 160, and 200.
4. Recompute task and identity metrics from sufficient evaluator state, record
   raw per-scene/per-sequence measurements, and aggregate only afterward.
5. Audit Rescan source and data before implementing an adapter. Fail closed to
   identity-only or inconclusive evidence when semantics are not defensible.
6. Generate paper-facing tables and figures only from hashed raw artifacts.

Reusing only existing aggregate CSV files is rejected because no controlled
capacity counterfactual exists. Re-running the complete network once per
capacity is rejected because it changes the controlled observation sample and
wastes compute.

## Components

### Provenance Binding

`artifacts/final_evidence/final_evidence_manifest.json` binds:

- source commit and scientific tree hash;
- frozen reviewer-closure tree and artifact hashes;
- checkpoint, model config, protocol, dataset, evaluator, and environment;
- external source commits and dataset inventory hashes;
- every generated table, figure, report, and raw result.

All artifact writers are atomic and refuse symlinks or differing overwrites.

### Capacity Replay

The capacity code audit documents slot allocation, occupied/active/dormant
semantics, birth acceptance/rejection, monotonic occupancy, state tensor shapes,
state byte accounting, and timing boundaries directly from
`models/persistent_memory.py`.

The replay input contains immutable per-stage observation tensors and evaluator
targets. Each entry is keyed by scene cluster, master sequence, order, horizon,
stage, checkpoint/config/source hashes, and content digest. A run is invalid if
the observation digest differs across capacities.

Raw capacity output includes, for every capacity and T2--T5 prefix:

- peak and mean occupied, active, and dormant slots;
- occupancy ratio, births, accepted/rejected births, and rejection rate;
- identity switches, fragmentation, merge, gap recovery accuracy/recall;
- t-mAP, AP50, AP25, and t-REC;
- persistent-state bytes, memory latency, and total update latency.

Aggregation is scene-cluster aware. Occupancy curves show median, IQR, maximum,
and capacity lines. Performance plots retain all requested panels. State bytes
are described as measured linear scaling only when the values support it.

The capacity report emits exactly one terminal gate:

- `CAPACITY_100_OK`
- `CAPACITY_SENSITIVITY_ONLY`
- `CAPACITY_CONFIG_REOPEN`

`CAPACITY_CONFIG_REOPEN` stops all subsequent experimental work.

### Rescan Audit And Adapter

The official Rescan repository is pinned before use. Dataset acquisition uses
the official release path and stores raw data outside Git, preferably under
`/mnt/shared/ww`. The audit records scenes, captures, ordering evidence, stable
instance identifiers, semantic labels, ambiguity groups/permutations, coordinate
systems, and transforms from both source code and sampled real files.

The parser has explicit contracts for XYZ, normals, RGB, semantic class,
instance identity, scan/capture identity, ordering, and transform metadata. The
label map classifies every target as `exact`, `reasonable`, `ambiguous`, or
`unsupported`, with provenance. Ambiguous and unsupported labels are never
silently guessed.

Level A task metrics are enabled only if semantic and coordinate compatibility
are defensible. Level B identity evaluation is always attempted on the eligible
stable-identity subset. Official transforms may normalize coordinates before
inference. Object-level ground-truth transforms are evaluation-only.

The primary online protocol uses local pairs `[S(t-1), S_t]` with persistent
state. It reports Pairwise Feature, Feature-Class where mapping is valid, EMA,
and Persist4D. Full-History and an official Rescan method are optional only when
their exact protocol is executable without material rewrites. Natural absence
defines gap opportunities; no gaps are manufactured.

Statistics use independent scene clusters and scene bootstrap when the sample
size permits. Every scene is also reported individually. The external report
emits exactly one gate:

- `EXTERNAL_SUPPORT`
- `EXTERNAL_PARTIAL`
- `EXTERNAL_INCONCLUSIVE`
- `EXTERNAL_CONTRADICTS`

`EXTERNAL_CONTRADICTS` stops paper-freeze work.

### Optional External Methods

The official Rescan method receives a bounded smoke audit in a separate
environment. Missing assets, proprietary dependencies, ambiguous protocol, or
major rewrites produce `RESCAN_METHOD_NOT_REPRODUCED` and do not block the main
study.

LivingScenes is related work only. Quantitative execution requires all five
prompt prerequisites; otherwise only `LIVINGSCENES_RELATED_WORK_NOTE.md` is
created.

### Paper Freeze

Published standard-protocol values and controlled common-prefix values are kept
in separate tables. Reported numbers are labeled as reported and sourced from
primary papers or official artifacts. External results appear only when their
gate permits them.

`NOVELTY_BOUNDARY.md` disallows unsupported priority claims and frames the
method as bounded perception context with long-horizon identity state:

`T_perception = bounded`, `T_identity = long-horizon`.

The final report evaluates claim gates A--E and emits exactly one classification:

- `PAPER_READY`
- `PAPER_READY_INTERNAL_ONLY`
- `CAPACITY_CONFIG_DECISION_REQUIRED`
- `EXTERNAL_CONTRADICTION`

No work continues after the final report is published.

## Failure Handling

- Missing or mismatched frozen inputs: stop the affected run and record the
  exact missing binding.
- Capacity gate reopens configuration: stop before external evaluation.
- Unavailable or indefensible Rescan data: produce an audit and
  `EXTERNAL_INCONCLUSIVE`; do not substitute another dataset.
- External contradiction: stop before paper freeze and classify accordingly.
- Optional baseline failure: report its bounded audit status and continue.
- Insufficient bootstrap clusters: publish per-scene raw results and state that
  interval estimation is underpowered.
- Disk pressure: copy inactive checkpoints to `/mnt/shared/ww`, verify SHA-256,
  publish a recovery manifest, then remove only the verified local duplicate.

## Verification Strategy

Unit tests cover:

- free and full capacity, rejected birth accounting, state shape/bytes, and
  identical observation digests across capacity values;
- PLY XYZ/normals/RGB/class/instance parsing;
- deterministic capture order and stable identity across captures;
- accepted official ambiguity alternatives;
- unsupported label exclusion;
- absence of ground-truth fields in inference inputs;
- artifact schema, exact gate vocabulary, provenance binding, and atomic writes.

Integration checks replay a small frozen fixture at all five capacities and run
one real Rescan scene before the full eligible set. Final verification includes
targeted tests, relevant regressions with all required external inputs mounted,
artifact hash validation, source-tree diff review, and a clean Git status.

## Known Baseline Deviations

The isolated worktree baseline produced 1651 passes, 8 skips, and 111 failures.
The failures predate this worktree and depend on ignored data/checkpoint/cache
files, old source-commit bindings, or the locally installed official metric
environment. They are recorded rather than repaired because changing frozen
historical contracts is outside this task. New and directly affected tests must
pass independently of those deviations.
