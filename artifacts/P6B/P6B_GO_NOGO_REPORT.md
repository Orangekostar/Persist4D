# Persist4D P6-B GO / NO-GO Report

## 1. What was changed

Implemented threshold-aware association, dormant reactivation, class compatibility, consolidation, and birth-quality choices without changing frozen local predictions.

## 2. Why it was changed

P6-A isolated association continuity and reactivation as the method-level bottlenecks addressed by P6-B.

## 3. Experimental protocol

Four reference clusters were used for tuning; the selected config was frozen before one exactly-once evaluation of 33 orders from two held-out clusters.

## 4. Reproducibility binding

Selection source `49fa34f04160c02e3bd003581310a5a2e703a794`; evaluation source `2024009f60814d02fbe1fb2bcc48d4408ae70fe2`; package source `5266e236b49fd50d5d83976115819140c6e6d88c`; selection `4c20284649b99d9f4e7eddb69c24d87c308b9eb07496f646def914a7c8137628`; raw attempt `e196f96d06790e769da447a31cff2266e875bb263fbf1a0a67558645d8250b59`. Attempt started 2026-08-21T21:42:41.948315Z, ended 2026-08-21T21:53:32.997695Z, events: attempt_token_published, heldout_raw_published.

## 5. Main results

Held-out P6B T5 t-mAP=0.019324, t-REC=0.055907, ID-switch rate=0.154534. Inactive selected components: threshold_aware_assignment, foreground_normalized_class_compatibility, confidence_gated_consolidation.

## 6. Statistical evidence

Paired reference-cluster bootstrap reports mean, sample SD, and deterministic 95% intervals. Only two held-out clusters are available; no significance claim is made.

## 7. Failure analysis

All 64 method/horizon/failure-category cells are included. B4 T2 total failures=769; B4 T3 total failures=1188; B4 T4 total failures=1643; B4 T5 total failures=2100; P6B T2 total failures=784; P6B T3 total failures=1203; P6B T4 total failures=1680; P6B T5 total failures=2136. Protocol deviations:
- none

## 8. What claims are supported

- No P6-B GO claim is supported because one or more preregistered held-out gates failed.

## 9. What claims are NOT supported

- P6-B does not establish SOTA, retraining gains, P7, or P8 claims.

## 10. GO / NO-GO decision

- G6B-1: PASS; registered threshold-aware and GT-free CPU proofs passed; frozen hashes checked
- G6B-2: FAIL; heldout mean T4/T5 ID-switch relative reduction=-0.040768
- G6B-3: FAIL; P6B accuracy=0.728547, B4 accuracy=0.729822, P6B recall=0.466278, B4 recall=0.483097
- G6B-4: PASS; T2 drop gate=True; long-task P6B=0.061653, B4=0.052574
- G6B-5: PASS; required ablations, paired rows, failures, provenance, and manifest checked

## 11. Exact next action

Stop after P6-B and analyze failed held-out gates; do not start P7/P8.

P6B_STOP
