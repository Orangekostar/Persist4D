# Predeclared diagnostic interpretation

Use TRAIN_PANEL.json (hash-only selection, written before any new head training)
and the complete NOAUG CAL population. Never use SEL ground truth for diagnostics.

Q weighted MSE uses repaired binary quality.valid. The constant baseline is the
weighted target mean fit on all fixed TRAIN records; report its panel/CAL errors.
Report B0 and Q2-F/Q-A0/Q-P at steps 0/500/1000/1500. Class ranking comparisons
use candidate pairs within the same input and class with known quality and
unequal targets; aggregate counts by role/class. Ties are reported separately.
Duplicate means multiple known candidates for the same valid GT entity, without
removing any candidate. Preserve candidate key (input, source query, source class,
retained column). Report partial-ignore separately from all-thresholds-unscored.

GT-assisted score diagnostic: replace scores only for MATCHED/VALID_NEGATIVE
relations by score-independent temporal quality. All other relations retain the
original parent score. This default is fixed before assisted metric evaluation.
Keep every original candidate, including duplicates. Report weighted known-subset
statistics separately from full-prediction official AP; neither is an AP upper bound.

Q memorization trigger: both CAL-selected Q-P and Q-A0 have TRAIN-panel weighted
MSE no smaller than the fixed TRAIN-mean baseline. If triggered, fixed key-first
16 supervisable TRAIN candidates, new temporary head, at most 200 updates; no grid.

M diagnostics use the frozen assignment; unmatched candidates have null assigned
GT IoU rather than invented zero quality. Supervision occurrences count actual
positive sampled positions in the shared seed45 plan. Segment error is disagreement
between sign(z) and the weighted segment target majority (>0.5), on positive-weight
segments. Separately count full known-vertex errors, including mixed-target segments.
Strict |z|>2, exactly |z|=2, and FP32 |delta|=2 saturation are separate fields.

Reachability uses the real full materializer T(z-2), T(z+2), asserts L subset
original subset U, and uses the assigned GT for L union (G intersect (U minus L)).
Unmatched/unknown columns remain unchanged in the separate full-set GT-assisted
official metric diagnostic. Ordinary assigned-GT IoU is explicitly distinct from
this official AP. Both are diagnostic-only, never deployed or sampled for main training.

M fit trigger: selected M-N CAL/T2 <= its B0 within 1e-6 and at least 8 matched,
shape-usable TRAIN-panel candidates with original IoU<.9 and loose gain>=.05.
Take fixed key-first16 (or all8–15), seed145, one shared head, <=500 updates,
positive shape mean plus .01 original regularizer, report 0/250/500.

Three M gradient batches use the first supervised planned batch, first supervised
batch at/after update500 and at/after1500. They are read-only backward probes with
no optimizer updates. Report exact plan indices and checkpoint steps used.
