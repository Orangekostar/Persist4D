# P2R Pilot Decision

- Status: `pass`
- Optimizer steps per path: `32`
- Train sample exposures per path: `64`
- Validation subset: `24` supervised T=2 sequences
- Scope: post-P2 controlled fine-tune pilot; not official-like G2 evidence
- Full candidate authorized: `false`
- Selected variant: `None`
- Selection rule: `strict_pareto_improvement_over_P2R-0_on_t_mAP_stage1_mAP_stage2_mAP_then_max_t_mAP`

| Variant | t-mAP | Overall mAP | Stage1 mAP | Stage2 mAP |
|---|---:|---:|---:|---:|
| P2R-0 | 0.203516 | 0.286250 | 0.386096 | 0.320707 |
| P2R-A | 0.146127 | 0.226644 | 0.330612 | 0.290308 |
| P2R-B | 0.200368 | 0.292707 | 0.374595 | 0.323162 |
| P2R-C | 0.187096 | 0.252142 | 0.363368 | 0.314883 |

A full 450-epoch candidate is authorized only by the preregistered
three-metric strict Pareto rule. Pairwise pilot differences are not
official-like 154-sequence G2 results.
