# Oracle-ID Headroom

## OR0 Status

**PASS.** GT identity is introduced only after official local candidate
masks, predicted classes, and scores are frozen. It is used only as the
trajectory linkage key under mask-IoU Hungarian matching at 0.5.

## All-Order FullHistory t-mAP

| Horizon | FullHistory | B2 official | B4 official | Oracle-ID | B4 - B2 | Oracle - B4 |
|---:|---:|---:|---:|---:|---:|---:|
| T2 | 19.100 | 20.727 | 20.724 | 19.787 | -0.003 | -0.937 |
| T3 | 10.790 | 10.231 | 12.310 | 13.029 | +2.079 | +0.719 |
| T4 | 6.900 | 4.529 | 7.023 | 8.611 | +2.494 | +1.588 |
| T5 | 4.534 | 1.823 | 5.250 | 6.111 | +3.427 | +0.860 |

## Interpretation

`B4 - B2` is recovered identity value under identical official local
candidates. `Oracle - B4` is diagnostic linkage headroom; Oracle-ID is
not a method or baseline and cannot improve missing/wrong local candidates.
Predicted class remains unchanged and is part of every persistent key,
so GT class is never substituted for model semantics.

## Invariants

- Candidate mask/class/score source: frozen V2 official task sidecars.
- Matching: one-to-one Hungarian on candidate/GT mask IoU only, threshold 0.5.
- Persistent key: `(oracle_gt_id, predicted_class_id)`.
- Unmatched candidate key: `(stage_index, candidate_index, predicted_class_id)`.
- Score reducer: mean; candidate masks, classes, and scores are unmodified.
- Fresh B2/B4 regression maximum absolute difference: `0.0`.
- FullHistory values are frozen V2 evidence under the same protocol.
