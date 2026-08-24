# MultiScan Preflight Report

## 1. Dataset provenance

Official code commit `697bc9ec86fb7d34d47cb4cdbddcfc3c7f18c605` and data repository revision
`c62c9aad850a8638ac3e42605926a65707600125` were bound. CSV, taxonomy, source, checkpoint, and
frozen configuration hashes are recorded in the reproducibility artifacts.
Release-file resolution returned HTTP 401 `GatedRepo` without an authenticated
license-accepted session.

## 2. Longitudinal inventory

The official release contains 273 scans and
117 physical scenes. The frozen T>=3 collection contains
23 scenes and 101 scans. Counts are
T>=3: 23, T>=4: 14, T>=5: 9.

## 3. Stable identity evidence

Not verified. Source code intends `local instance -> inst2obj_id -> objectId`,
but real released PTH/annotations could not be opened. Documentation alone is
insufficient because it defines `objectId` as a per-scan object-list index.

## 4. Gap opportunities

Not computed. Gap event count and gap-bearing scene count remain `null`; they
are not reported as zero and the >=10 / >=3 gate is not evaluable.

## 5. Chronology

`DATASET_ORDER_ONLY`. Numeric scan suffix order is frozen for a possible ordered
revisit protocol; this is not proven physical chronology.

## 6. Alignment

Not run and not authorized because the preceding identity/gap gate is not
evaluable. This is not classified as an alignment failure.

## 7. Semantic compatibility

Of 20 official MultiScan semantic classes, 11 map exactly
to frozen ReScene18 and 9 are unsupported. Main
class-aware evaluation is frozen to exact mappings only.

## 8. GT leakage audit

The interface separates four geometry-only inference fields from evaluator-only
class, instance, and stable-object IDs. Recursive leakage guards and tests pass.

## 9. Frozen ReScene smoke coverage

Not run and not authorized. Coverage, AP, and recall remain `null`; no GPU
inference or MultiScan tuning was performed.

## 10. Final decision

`MULTISCAN_PROTOCOL_FAIL`
