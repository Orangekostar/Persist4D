# Final-Evidence Figure Captions

## Main Figure 1: Finite Temporal Context Is Not Persistent Entity State

Full-History perception repeatedly expands the observation set, while Persist4D
uses a bounded local pair and retains dormant entity identity in bounded state.
The inset reports measured median update latency over six profile clusters for
the T2-to-T3 horizon-adapted Full-History model and Persist4D. Full-History
latency excludes lightweight tracker overhead. Source:
`artifacts/reviewer_closure/rescene_horizon_compute.csv`.

## Main Figure 2: Long-Horizon Accuracy-Identity-Compute Scaling

Causal-prefix t-mAP, gap-recovery recall, median update latency, and peak
allocated VRAM across T2-T5. Task and identity metrics use 129 common-prefix
sequence/order scopes grouped into six independent scene clusters. Full-History
and horizon-adapted alternatives use Feature-Class tracking for identity panels;
their compute panels profile local perception and exclude tracker overhead.
Sources: `full_history_tracker_aggregate.csv`,
`rescene_horizon_adaptation_results.csv`, and `rescene_horizon_compute.csv` in
the immutable reviewer-closure package.

## Main Figure 3: Why Long-Horizon t-mAP Remains Similar

IoU-threshold sensitivity, observation coverage, and registered failure
decomposition show that local observation, class, mask, capacity, and unresolved
errors dominate the remaining task ceiling more than registered identity
failures. Source: immutable
`artifacts/reviewer_closure/figures/performance_decomposition.*`.

Main Figure 4 is intentionally omitted because external validation is
`EXTERNAL_INCONCLUSIVE`.

## Figure C1: Occupancy vs Temporal Horizon

Peak occupied persistent slots under the frozen common-prefix replay. The line
and band show the median and interquartile range over 129 sequences; squares
show the maximum. Horizontal rules mark every preregistered capacity. The
largest observed occupancy is 30 at T3--T5, below even K=64, and no true birth
is rejected. Source: `capacity/capacity_aggregate.csv`.

## Figure C2: Performance vs Capacity

Frozen capacity sensitivity for (a) pooled causal-prefix t-mAP, (b) pooled
causal-prefix t-REC, (c) pooled normalized deployment ID-switch rate, and (d)
pooled gap-recovery recall. Colors, markers, and line styles identify T2--T5;
the vertical rule marks the frozen main K=100. T2 gap recall is undefined
because no eligible gap opportunity exists. All defined metrics are exactly
constant over K in `{64,100,128,160,200}`. Source:
`capacity/capacity_aggregate.csv`.

## Figure C3: State Bytes vs Capacity

Exact allocated tensor storage for the eight-field persistent state at B=1,
D=128, and C=19. Storage follows `610*K + 8` bytes and is 59.6 KiB at the
frozen K=100. Values exclude model weights, observation masks, allocator
overhead, evaluator state, and VRAM; they are not directly compared with
Full-History explicit input bytes. Source: `capacity/capacity_aggregate.csv`.
