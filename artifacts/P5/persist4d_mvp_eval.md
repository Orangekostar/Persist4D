# Persist4D MVP P5 Evaluation

Purpose: fixed-capacity streaming association diagnosis; metrics are not an official AP target.

Status: `pass`

Conclusion: `P5_MVP_PASS`

Reason: `bounded_execution_with_t4_t5_identity_improvement`

Identity improvements: `T4:t_REC, T4:identity_switches, T4:reactivation_accuracy, T5:t_REC, T5:identity_switches, T5:reactivation_accuracy`

Source commit: `92bab01e93bacbc939606ec7c7f58d3f9b334fe6`

Source tree contract: `pass`

Allowed untracked outputs: `repo:artifacts/P5/persist4d_mvp_eval.json, repo:artifacts/P5/persist4d_mvp_eval.md`

Checkpoint reference: `repo:checkpoints/rescene4d_concerto_t2_repro.ckpt`

Checkpoint SHA-256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`

Legacy predictions unchanged: `true`

Legacy parity verification: `in_evaluator_fixed_t2_sample_toggle`

Query feature shape: `[1, 100, 128]`

Internal baseline identity: `local_query_index`

The internal no-memory baseline reuses each persistent run's same latest-stage valid ReScene observations, masks, and classifications; only the cross-stage identity is the local query index.

## Persistent And Baseline Metrics

| T | Sequences | Method | t-mAP | t-REC | Per-stage AP | Matched ID obs | ID switches | Reactivation events | Correct reactivations | Reactivation accuracy | Rejected births |
|---:|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|
| 2 | 154 | persistent | 0.239323 | 0.305573 | 1=0.357785; 2=0.349188 | 1890 | 70 | 0 | 0 | n/a | 0 |
| 2 | 154 | internal_baseline | 0.008161 | 0.166236 | 1=0.340680; 2=0.342568 | 1890 | 660 | 0 | 0 | n/a | n/a |
| 3 | 120 | persistent | 0.158717 | 0.185742 | 1=0.341770; 2=0.325207; 3=0.332188 | 2218 | 133 | 114 | 93 | 0.815789 | 0 |
| 3 | 120 | internal_baseline | 0.002777 | 0.187963 | 1=0.302365; 2=0.314751; 3=0.309147 | 2218 | 1088 | 114 | 2 | 0.017544 | n/a |
| 4 | 75 | persistent | 0.126216 | 0.172720 | 1=0.379085; 2=0.388418; 3=0.354637; 4=0.376691 | 1913 | 122 | 152 | 125 | 0.822368 | 0 |
| 4 | 75 | internal_baseline | 0.001604 | 0.110718 | 1=0.333417; 2=0.357783; 3=0.333473; 4=0.355346 | 1913 | 1118 | 152 | 2 | 0.013158 | n/a |
| 5 | 43 | persistent | 0.035618 | 0.115809 | 1=0.314674; 2=0.289066; 3=0.294443; 4=0.303937; 5=0.298411 | 1419 | 121 | 227 | 184 | 0.810573 | 0 |
| 5 | 43 | internal_baseline | 0.001392 | 0.042998 | 1=0.247482; 2=0.232606; 3=0.266082; 4=0.250280; 5=0.253304 | 1419 | 866 | 227 | 1 | 0.004405 | n/a |

## Differences

| T | Comparison | Delta t-mAP | Delta t-REC | Delta per-stage AP | Delta matched obs | Delta switches | Delta reactivation events | Delta correct reactivations | Delta reactivation accuracy |
|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|
| 2 | delta (persistent - baseline) | +0.231162 | +0.139337 | 1=+0.017105; 2=+0.006621 | +0 | -590 | +0 | +0 | n/a |
| 3 | delta (persistent - baseline) | +0.155940 | -0.002221 | 1=+0.039406; 2=+0.010456; 3=+0.023041 | +0 | -955 | +0 | +91 | +0.798246 |
| 4 | delta (persistent - baseline) | +0.124612 | +0.062002 | 1=+0.045668; 2=+0.030635; 3=+0.021164; 4=+0.021345 | +0 | -996 | +0 | +123 | +0.809211 |
| 5 | delta (persistent - baseline) | +0.034226 | +0.072811 | 1=+0.067193; 2=+0.056460; 3=+0.028361; 4=+0.053656; 5=+0.045107 | +0 | -745 | +0 | +183 | +0.806167 |

## Resources And State

| T | Peak CUDA bytes | Mean latency (ms) | Throughput (seq/s) | State bytes |
|---:|---:|---:|---:|---:|
| 2 | 3391539200 | 629.834397 | 1.587719 | 63808 |
| 3 | 2714871296 | 962.304048 | 1.039173 | 63808 |
| 4 | 2714983936 | 1373.667209 | 0.727978 | 63808 |
| 5 | 2714959872 | 1811.279708 | 0.552096 | 63808 |

The fixed-capacity state remained constant in shape across T=2/3/4/5.

Constant state shape: `true`

Maximum serialized state bytes: `63808`
