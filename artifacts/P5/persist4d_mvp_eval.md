# Persist4D P5 MVP Evaluation

Purpose: Persist4D MVP engineering and association diagnosis. This report is not an official AP target.

Status: `pass`

Conclusion: `P5_ASSOCIATION_DIAGNOSIS`

T=2 legacy predictions unchanged: `true`.

The fixed-capacity state was evaluated at T=2/3/4/5. The evaluator did not produce an internal no-memory baseline, so `P5_MVP_PASS` is not claimed.

| T | Sequences | t-mAP | t-REC | Per-stage AP | Matched ID obs | ID switches | Reactivation events | Correct reactivations | Reactivation accuracy | Rejected births | Peak CUDA bytes | Mean latency (ms) | Throughput (seq/s) | State bytes |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 154 | 0.230711 | 0.290190 | 1=0.354744; 2=0.339188 | 1875 | 63 | 0 | 0 | n/a | 0 | 3391337472 | 634.048391 | 1.577167 | 63808 |
| 3 | 120 | 0.157121 | 0.174278 | 1=0.337950; 2=0.332704; 3=0.320509 | 2188 | 111 | 115 | 100 | 0.869565 | 0 | 2714904064 | 992.402344 | 1.007656 | 63808 |
| 4 | 75 | 0.126412 | 0.189551 | 1=0.382518; 2=0.382144; 3=0.367916; 4=0.379353 | 1918 | 131 | 170 | 137 | 0.805882 | 0 | 2714928128 | 1407.839631 | 0.710308 | 63808 |
| 5 | 43 | 0.045564 | 0.173741 | 1=0.301534; 2=0.302613; 3=0.309819; 4=0.299940; 5=0.294356 | 1441 | 110 | 217 | 181 | 0.834101 | 0 | 2714982912 | 1841.442156 | 0.543053 | 63808 |

Constant state shape: `true`

Maximum serialized state bytes: `63808`

Checkpoint SHA-256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`
