# ReScene4D Full-History Code Audit

## Frozen Checkpoint

- SHA256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`
- Epoch/global step: `404` / `26730`
- RIO train/validation/test temporal horizon: `T2` / `T2` / `T2`
- Auxiliary ScanNet training horizon: `T1`
- Formal system name: **ReScene4D Full-History (Frozen T2 Checkpoint)**
- T3-T5 status: **zero-shot temporal-horizon extension**

## Code Evidence

| ID | File path | Function/class | Line | Relevant behavior | Scientific implication |
|---|---|---|---:|---|---|
| E1 | `datasets/semseg.py` | `SemanticSegmentationDataset.load_scan_indices` | 412 | Validates an explicit non-empty unique scan-index list and delegates it without requiring a fixed length. | The dataset path structurally accepts T1-T5 exact prefixes and does not itself impose T=2. |
| E2 | `datasets/semseg.py` | `SemanticSegmentationDataset._load_scan_sequence` | 470 | Loads every requested scan and assigns a local temporal coordinate in request order. | A T>2 prefix receives distinct causal stage coordinates for every observed visit. |
| E3 | `datasets/pointcept_utils.py` | `voxelize` | 79 | Voxelizes temporal stages separately, then preserves their stage coordinate in the collated sequence. | Collation has no fixed two-stage tensor contract and retains stage membership for metrics. |
| E4 | `models/pointcept.py` | `PointceptBackbone.forward` | 142 | Runs the configured standard and temporal-overlay serializations on the runtime sparse input. | Backbone feature extraction shares information across every stage present in the supplied prefix. |
| E5 | `models/pointcept.py` | `PointceptBackbone.temporal_overlay` | 519 | Reassigns serialization batch identity to the true sequence batch while retaining temporal coordinates. | Temporal sharing is joint over the observed prefix rather than a persistent state transition. |
| E6 | `models/rescene.py` | `ReScene.initialize_queries` | 217 | Initializes 100 non-parametric queries by farthest-point sampling coordinates from the current complete input. | Changing the prefix can change sampled query anchors; raw query index has no guaranteed cross-prefix semantic namespace. |
| E7 | `models/rescene.py` | `ReScene.forward` | 420 | Uses one joint query set to decode all features supplied by the current forward and emits class/mask predictions without track IDs. | Within-prefix joint reasoning is supported, but deployment identity is not persisted between forwards. |
| E8 | `models/rescene.py` | `ReScene.mask_module` | 594 | Produces one class logit vector and one mask logit column per query. | Raw query index is the only model-native identity candidate exposed to the evaluation adapter. |
| E9 | `trainer/trainer.py` | `InstanceSegmentation._get_mask_and_scores` | 1607 | Selects top query-class pairs, converts them back to query indices for masks, then returns scores/classes/masks without those indices. | Official task-quality postprocessing is valid, but a separate adapter must preserve query indices for deployment identity. |
| E10 | `scripts/evaluate_persist4d_p6a.py` | `RealPredictionCacheProducer.__call__` | 311 | Loads only the request-resolved prefix/window and explicitly passes change_file=None before frozen inference. | The system adapter can enforce no-future access and exclude change-label supervision without changing the model. |

## Q1. Does the ReScene4D code path natively accept T>2?

Yes structurally: explicit variable-length prefixes survive dataset loading, collation, backbone serialization, and joint decoding.

Evidence: `E1, E2, E3, E4, E5, E7`.

## Q2. At what temporal horizon was the checkpoint trained?

The RIO training, validation, and test horizon is T2; the mixed ScanNet auxiliary dataset uses T1.

Evidence: `checkpoint metadata`.

## Q3. What are the semantics of T3/T4/T5 evaluation?

They are zero-shot temporal-horizon extension of a frozen T2 checkpoint, not trained long-horizon ReScene4D.

Evidence: `E1, E4, E5, E7`.

## Q4. How is instance identity represented inside one full-history forward?

Each output mask/class is indexed by one raw query in the joint prefix forward; the model emits no separate track ID.

Evidence: `E6, E7, E8, E9`.

## Q5. Is the query or track namespace stable between S1:S4 and S1:S5?

No stability is guaranteed: non-parametric FPS query anchors are recomputed from the changed full prefix.

Evidence: `E6, E7`.

## Q6. Is inference deterministic?

The checkpoint did not enable deterministic training; evaluation will force deterministic controls and requires a three-repeat empirical fingerprint gate.

Evidence: `E6, E7`.

## Q7. Can the evaluator use future information?

Only if given it. The new adapter must bind each output and target to the exact observed prefix and reject later scan IDs or stage coordinates.

Evidence: `E1, E2, E9, E10`.

## Q8. Does the change-label path affect identity evaluation?

No. Full-history and persistent evaluation load change_file=None, the checkpoint disables change loss, and placeholder changes are not used for identity matching.

Evidence: `E10`.

## Evaluation Consequences

Full-History may reason jointly over the exact observed prefix, but it may not access a later prefix. Official task postprocessing and deployment identity are separate: task metrics keep the registered top-k path, while identity analysis preserves the raw query index without adding persistent memory. Change labels remain disabled. Determinism and T2 parity must pass empirical smoke gates before the full run.
