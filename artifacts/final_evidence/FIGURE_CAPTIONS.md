# Final-Evidence Figure Captions

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
