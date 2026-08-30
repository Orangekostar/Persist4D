# Sonata Qualification Report

- Gate: `SQ-RED`
- Automatic SS6 authorization: `false`
- Decision reason: Sonata is weaker than Concerto on both temporal and spatial metrics
- Evaluation: same local official-like T2 validation harness, seeds 45/46/47
- Runtime: one NVIDIA A40, batch size 1, 4 workers, precision 32-true
- Validation stochasticity: train-style GridSample retained and reported by seed
- Scope: internal reviewer-closure gate; not a publication standard

| Model | Evidence | t_mAP | t_mAP50 | t_mAP25 | overall_mAP | stage1_mAP | stage2_mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| ReScene4D-S paper reported | external | 33.200 |  |  | 40.900 |  |  |
| Our Sonata reimplementation | measured mean | 24.035 | 41.922 | 56.590 | 31.554 | 37.665 | 37.464 |
| Our Concerto reimplementation | measured mean | 28.290 | 45.536 | 59.078 | 36.979 | 42.032 | 43.061 |
| ReScene4D-C paper reported | external | 34.800 |  |  | 43.300 |  |  |

The paper-reported rows are external references and were not substituted
for local measurements. The Sonata row is this task's reimplementation,
not an official ReScene4D-S checkpoint.

Automatic progression stops at SS5. SS6 and SS7 were not run.
