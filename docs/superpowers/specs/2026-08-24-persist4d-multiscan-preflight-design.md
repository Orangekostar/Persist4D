# Persist4D MultiScan Preflight Design

## Status And Scope

This design implements the approved MultiScan execution prompt from frozen
Persist4D commit `487080cf31266f1572257e2aca36767e074b68b6`. MultiScan is a
zero-shot external validation only. `models/`, ReScene, Persist4D parameters,
`artifacts/final_evidence/`, and `artifacts/reviewer_closure/` are immutable.

The experiment is a fail-closed state machine:

```text
METADATA -> GAP -> ALIGNMENT -> PROTOCOL -> COVERAGE -> FULL_EVAL
              |         |          |           |
              +---------+----------+-----------+-> STOP
```

No later state may create artifacts or consume GPU before every earlier gate is
proven from official release files.

## Source And Storage Boundary

Official code is pinned at commit
`697bc9ec86fb7d34d47cb4cdbddcfc3c7f18c605`, tree
`cbeec22d43dfd9def6055cf13c96ec40091d70be`. The official CSV hashes are
recorded in repository artifacts. Raw archives, extracted meshes, annotations,
checkpoints, and inference caches remain under
`/mnt/shared/ww/persist4d-multiscan/`; Git contains only code, tests, hashes,
manifests, tables, figures, and reports.

The official Hugging Face dataset is license-gated. Source examples and docs
may validate parsers but may never substitute for actual released annotations
when counting identities or gaps. Missing release data yields an explicit
unverified prerequisite, never a passing scientific gate.

## Components

### Metadata And Identity Adapter

`datasets/multiscan_adapter.py` owns strict scan-ID parsing, official split
inventory, annotation schema validation, stable `(scene_id, objectId)` keys,
natural maximal-gap extraction, alignment-matrix parsing, geometry-only input,
evaluator-only targets, and recursive no-GT-leakage checks.

Inference contains only XYZ, normals, RGB, and deterministic geometry-only
segment IDs. `objectId`, instance/semantic/part labels, OBBs, mobility, and
correspondences stay in evaluator targets.

### Dataset Audit

`scripts/audit_multiscan_dataset.py` consumes the pinned official CSVs and real
release annotations. It publishes the 117-scene inventory and the predeclared
all-`T>=3` collection, then derives gaps without scene or event selection. A
gap is one maximal `1 -> 0+ -> 1` interval for an eligible stable object.

The audit fails closed if any selected scan annotation is missing, duplicated,
schema-invalid, or inconsistent with the official scan list. It cannot emit a
passing gap gate from partial data.

### Chronology And Semantics

Dataset order and physical chronology are distinct fields. True chronology is
accepted only when actual release metadata supports acquisition order across
captures. Otherwise the status is `DATASET_ORDER_ONLY` or `UNRESOLVED`.

The semantic map enumerates every official MultiScan semantic ID and classifies
it as `exact`, `defensible`, `ambiguous`, or `unsupported` against the frozen
18-class ReScene output taxonomy. Only exact entries enter Level-A metrics;
structural floor/ceiling/wall entries are excluded from identity gaps.

### Alignment

`scripts/audit_multiscan_coordinates.py` parses each column-major 4x4 official
current-to-reference transform. Synthetic tests establish convention. Real
smoke scenes compare raw versus aligned nearest-neighbor, centroid, bounding
box, and structural-overlap measures and publish inspection PLYs. No GT object
transform is accepted.

### Temporal Protocol And Evaluation

`scripts/multiscan_protocol.py` freezes the all-`T>=3` collection and exact
stage windows `[S1]`, `[S1,S2]`, `[S2,S3]`, ... with unchanged K=100,
association threshold 0.5, class weight 0.25, update rate 0.2, and maximum
update rate 0.2.

`scripts/evaluate_multiscan_persist4d.py` is a thin dataset-specific shell over
the existing ReScan preparation, four trackers, identity metrics, task metrics,
scene bootstrap, cache hashing, and artifact writers. It contains no new
association implementation.

## Gates

1. Gap: at least 10 natural events in at least 3 physical scenes.
2. Alignment: official transforms improve or preserve shared-frame geometry
   under a documented fixed rule and manual inspection.
3. Protocol: stable IDs, defensible order, exact local windows, and no GT
   leakage all verify.
4. Coverage: frozen ReScene entity-stage candidate coverage at IoU 0.25 is at
   least 0.10 on the frozen smoke collection.

Only all four passes authorize full inference. A failure generates the exact
required report classification and stops later stages. Full-History is allowed
only after an interpretable supporting external result.

## Verification

Unit tests cover inventory, identity mapping, maximal gaps, chronology,
alignment convention, GT separation, scene isolation, protocol windows, and
boundary gate cases. Integration checks run the official metadata audit, hash
all generated files, preserve both frozen artifact trees, scan Git for large
files, and compare the local/remote final commit after push.

## Approved Decision

Use one shared adapter plus thin scripts that reuse ReScan evaluation semantics.
Do not fork the association or statistics stack. Follow prompt gate order and
stop at the first failed scientific gate.
