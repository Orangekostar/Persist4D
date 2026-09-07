# Persist4D R1 Downstream Validation Final Report

- Code commit at generation: `43d72c518ab94b71985a9b7a28f45174054fdca8`
- R1 checkpoint: `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd` (completed epoch 390, step 25740)
- Protocol-B: `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`; 43 masters, 3 orders, 129 sequence-order units, 6 reference-scene clusters
- Execution: `COMPLETE`
- Candidate semantics: `PASS`
- Recovery evidence: `SUPPORTED`
- Historical comparison: `HISTORICAL_REFERENCE_ONLY`
- Publication: `PUSH_VERIFIED`

## Recovery Decision

| Horizon | Method | Opportunities | Attempts | Correct | Attempt coverage | Accuracy | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T4 | B2 | 696 | 238 | 38 | 0.341954 | 0.159664 | 0.054598 |
| T4 | B4 | 696 | 238 | 206 | 0.341954 | 0.865546 | 0.295977 |
| T4 | B4-B2 | - | - | - | - | - | +24.138 pp |
| T5 | B2 | 1093 | 397 | 56 | 0.363220 | 0.141058 | 0.051235 |
| T5 | B4 | 1093 | 397 | 339 | 0.363220 | 0.853904 | 0.310156 |
| T5 | B4-B2 | - | - | - | - | - | +25.892 pp |

| Horizon | Reference cluster | B2 opportunities | B4 opportunities | B2 recall | B4 recall | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T4 | `10b17940-3938-2467-8a7a-958300ba83d3` | 85 | 85 | 0.011765 | 0.341176 | +32.941 pp |
| T4 | `137a8158-1db5-2cc0-8003-31c12610471e` | 353 | 353 | 0.065156 | 0.271955 | +20.680 pp |
| T4 | `280d8ebb-6cc6-2788-9153-98959a2da801` | 65 | 65 | 0.061538 | 0.369231 | +30.769 pp |
| T4 | `5630cfcf-12bf-2860-8784-83d28a611a83` | 2 | 2 | 0.000000 | 0.500000 | +50.000 pp |
| T4 | `8eabc45f-5af7-2f32-8528-640861d2a135` | 102 | 102 | 0.029412 | 0.362745 | +33.333 pp |
| T4 | `ddc73797-765b-241a-9e2c-097c5989baf6` | 89 | 89 | 0.078652 | 0.213483 | +13.483 pp |
| T4 | six-cluster equal weight | - | - | - | - | +30.201 pp (6/6 positive) |
| T5 | `10b17940-3938-2467-8a7a-958300ba83d3` | 137 | 137 | 0.014599 | 0.321168 | +30.657 pp |
| T5 | `137a8158-1db5-2cc0-8003-31c12610471e` | 525 | 525 | 0.059048 | 0.268571 | +20.952 pp |
| T5 | `280d8ebb-6cc6-2788-9153-98959a2da801` | 89 | 89 | 0.067416 | 0.359551 | +29.213 pp |
| T5 | `5630cfcf-12bf-2860-8784-83d28a611a83` | 9 | 9 | 0.000000 | 0.333333 | +33.333 pp |
| T5 | `8eabc45f-5af7-2f32-8528-640861d2a135` | 195 | 195 | 0.025641 | 0.451282 | +42.564 pp |
| T5 | `ddc73797-765b-241a-9e2c-097c5989baf6` | 138 | 138 | 0.086957 | 0.224638 | +13.768 pp |
| T5 | six-cluster equal weight | - | - | - | - | +28.415 pp (6/6 positive) |

Decision: `RECOVERY_SUPPORTED` under the frozen pooled-positive and at-least-four-of-six-positive-clusters rule at both T4 and T5.

## Task Evidence

| Method | Reducer | Horizon | t-mAP | t-mAP50 | t-mAP25 | t-REC |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| FullHistory | official | T2 | 0.228777 | 0.368017 | 0.603987 | 0.372957 |
| FullHistory | official | T4 | 0.102745 | 0.187445 | 0.346681 | 0.224228 |
| FullHistory | official | T5 | 0.081566 | 0.148171 | 0.297173 | 0.174699 |
| B2 | mean | T2 | 0.219017 | 0.381749 | 0.549679 | 0.359598 |
| B2 | mean | T4 | 0.040782 | 0.107363 | 0.203719 | 0.158269 |
| B2 | mean | T5 | 0.018288 | 0.071585 | 0.152374 | 0.110204 |
| B2 | latest | T2 | 0.209891 | 0.369300 | 0.537985 | 0.359598 |
| B2 | latest | T4 | 0.037219 | 0.102075 | 0.196769 | 0.158269 |
| B2 | latest | T5 | 0.018592 | 0.070877 | 0.150245 | 0.110204 |
| B2 | max | T2 | 0.199762 | 0.360794 | 0.534427 | 0.359598 |
| B2 | max | T4 | 0.044225 | 0.111511 | 0.211927 | 0.158269 |
| B2 | max | T5 | 0.022809 | 0.078032 | 0.161413 | 0.110204 |
| B4 | mean | T2 | 0.219637 | 0.381728 | 0.552095 | 0.361011 |
| B4 | mean | T4 | 0.080159 | 0.138862 | 0.273972 | 0.201489 |
| B4 | mean | T5 | 0.058535 | 0.104587 | 0.215714 | 0.158670 |
| B4 | latest | T2 | 0.212175 | 0.371283 | 0.542393 | 0.361011 |
| B4 | latest | T4 | 0.065317 | 0.118389 | 0.245018 | 0.201489 |
| B4 | latest | T5 | 0.044166 | 0.084881 | 0.191337 | 0.158670 |
| B4 | max | T2 | 0.198118 | 0.356932 | 0.532992 | 0.361011 |
| B4 | max | T4 | 0.067435 | 0.121866 | 0.253851 | 0.201489 |
| B4 | max | T5 | 0.044011 | 0.086810 | 0.192870 | 0.158670 |

Reducer sensitivity uses a paired six-cluster bootstrap (seed 45, 10,000 replicates); intervals are descriptive, not population-level significance claims.

| Horizon | Reducer | Metric | Equal-cluster B4-B2 | 95% descriptive CI | Positive clusters |
| --- | --- | --- | ---: | --- | ---: |
| T4 | mean | causal_prefix_t_mAP | +5.910 pp | [+3.357, +8.430] pp | 6/6 |
| T4 | mean | causal_prefix_t_REC | +4.842 pp | [+2.207, +7.822] pp | 6/6 |
| T5 | mean | causal_prefix_t_mAP | +5.033 pp | [+2.641, +7.649] pp | 6/6 |
| T5 | mean | causal_prefix_t_REC | +5.201 pp | [+2.926, +7.476] pp | 6/6 |
| T4 | latest | causal_prefix_t_mAP | +4.644 pp | [+2.593, +6.731] pp | 6/6 |
| T4 | latest | causal_prefix_t_REC | +4.842 pp | [+2.207, +7.822] pp | 6/6 |
| T5 | latest | causal_prefix_t_mAP | +3.919 pp | [+2.083, +5.691] pp | 6/6 |
| T5 | latest | causal_prefix_t_REC | +5.201 pp | [+2.926, +7.476] pp | 6/6 |
| T4 | max | causal_prefix_t_mAP | +4.226 pp | [+2.306, +6.152] pp | 6/6 |
| T4 | max | causal_prefix_t_REC | +4.842 pp | [+2.207, +7.822] pp | 6/6 |
| T5 | max | causal_prefix_t_mAP | +3.701 pp | [+2.223, +4.973] pp | 6/6 |
| T5 | max | causal_prefix_t_REC | +5.201 pp | [+2.926, +7.476] pp | 6/6 |

## Direct Local Current

This is official latest-stage local AP and is separate from trajectory `current_stage_AP`.

| Horizon | Direct local AP | AP50 | AP25 | REC |
| --- | ---: | ---: | ---: | ---: |
| T2 | 0.362199 | 0.578001 | 0.768074 | 0.510894 |
| T4 | 0.334830 | 0.558548 | 0.789076 | 0.509474 |
| T5 | 0.367196 | 0.605087 | 0.787692 | 0.524154 |

## Resource Evidence

Latency is the median of six per-unit medians; memory is the maximum observed unit peak.

| Method | Horizon | Latency (ms) | Peak allocated (MiB) | Peak reserved (MiB) | Max occupied slots | Rejected births |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FullHistory | T2 | 647.388 | 2589.5 | 3888.0 | N/A | N/A |
| FullHistory | T4 | 970.622 | 3847.9 | 6046.0 | N/A | N/A |
| FullHistory | T5 | 1106.892 | 4964.9 | 8030.0 | N/A | N/A |
| B4 | T2 | 670.153 | 2555.6 | 3972.0 | 23 | 0 |
| B4 | T4 | 535.688 | 2100.3 | 3096.0 | 24 | 0 |
| B4 | T5 | 482.910 | 2471.4 | 3748.0 | 24 | 0 |

The A40 profile uses the first canonical master per cluster, 5 warmups and 10 measured repeats. It includes model forward and B4 tracking, while excluding I/O, collation, H2D, metric scoring, and tracker preroll.

## Evidence Boundary

Historical raw replay is `HISTORICAL_REPLAY_UNAVAILABLE`. Frozen C-old summaries remain available for descriptive checkpoint-regime comparison, but they do not support causal attribution of old-versus-R1 changes.
The profile covers only the frozen six units through T5. It does not prove constant whole-system storage or indefinite-horizon behavior.
