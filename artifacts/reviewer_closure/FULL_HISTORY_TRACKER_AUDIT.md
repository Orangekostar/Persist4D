# Full-History Tracker Audit

Gate I: `TRACKER_REJECTED`.

## Frozen Scope

- Raw tracker artifact: `d6f2480e2787a943b5fd2a7f9c412a14c32bf0106bc23130f363fc72584efbf9`
- Coverage: 43 masters x 3 orders x T2-T5.
- Tracker observations begin at P2; Persist4D retains its frozen P1 initialization.
- Task metrics are inherited unchanged from the frozen system-comparison caches; only identity assignment changes.
- The separate replay task-drift CSV/JSON quantifies official-metric changes caused by audited cross-process CUDA sparse numerical variation; no post-hoc equivalence threshold is applied.

## Strongest Simple Tracker

Selected `B2` (Pairwise Feature-Class Association). The preregistered ranking minimizes pooled T4/T5 normalized ID-switch rate, then maximizes pooled gap-recovery recall, then uses method ID.
B4 is diagnostic and was excluded from selection.

## Replay Task Drift

All 516 replay prediction content digests differ from the immutable reference predictions. `full_history_replay_task_drift.csv` reports signed, absolute, and relative official-metric drift; `full_history_replay_task_drift.json` binds both manifests. Frozen reference-cache task metrics remain primary.

## Pooled Identity Results

| Method | T | ID-switch rate | Gap recovery recall |
|---|---:|---:|---:|
| ReScene4D Full-History | 4 | 0.978066 | 0.003236 |
| ReScene4D Full-History | 5 | 0.977390 | 0.003058 |
| Pairwise Feature Association | 4 | 0.149551 | 0.058252 |
| Pairwise Feature Association | 5 | 0.149871 | 0.048930 |
| Pairwise Feature-Class Association | 4 | 0.141575 | 0.064725 |
| Pairwise Feature-Class Association | 5 | 0.144057 | 0.058104 |
| EMA Temporal Association | 4 | 0.159521 | 0.100324 |
| EMA Temporal Association | 5 | 0.169897 | 0.087156 |
| Full-History + Persistent-State Diagnostic | 4 | 0.123629 | 0.258900 |
| Full-History + Persistent-State Diagnostic | 5 | 0.129845 | 0.256881 |
| Persist4D Persistent Entity State | 4 | 0.103495 | 0.297414 |
| Persist4D Persistent Entity State | 5 | 0.111385 | 0.311985 |

## Paired Six-Cluster Evidence

Differences are Persist4D minus the selected tracker. Negative is favorable for ID-switch rate; positive is favorable for gap recovery.

| Metric | T | Clusters | Tracker | Persist4D | Difference | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| normalized_id_switch_rate | 4 | 6/6 | 0.116992 | 0.114073 | -0.002918 | [-0.067540, +0.079770] |
| normalized_id_switch_rate | 5 | 6/6 | 0.124414 | 0.126250 | +0.001836 | [-0.062995, +0.082653] |
| gap_recovery_recall | 4 | 5/6 | 0.066015 | 0.306805 | +0.240791 | [+0.145131, +0.355775] |
| gap_recovery_recall | 5 | 6/6 | 0.056346 | 0.433692 | +0.377346 | [+0.178324, +0.648590] |

## Gate Decision

`TRACKER_REJECTED` under the frozen CI + order + LOSO rule.
