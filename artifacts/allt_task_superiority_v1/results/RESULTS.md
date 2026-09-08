# Persist4D All-T Protocol-B Results

## Primary result

The preregistered objective was not met. The frozen C2 checkpoint with the
primary `mean` score reducer exceeded the matched FH-adapt checkpoint only at
T2. All decisions below use unrounded metric values and strict zero tolerance.

| T | C2 mean t-mAP | FH-adapt t-mAP | Delta (pp) | R1-B4 mean t-mAP | Delta vs R1-B4 (pp) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.225025 | 0.223600 | +0.1425 | 0.219637 | +0.5388 |
| 3 | 0.141059 | 0.147723 | -0.6664 | 0.135498 | +0.5560 |
| 4 | 0.076000 | 0.103657 | -2.7657 | 0.080159 | -0.4159 |
| 5 | 0.051323 | 0.080308 | -2.8985 | 0.058535 | -0.7213 |

The mean T2-T5 t-mAP is 0.123352 for C2, 0.138822 for FH-adapt, and
0.123457 for R1-B4 mean. C2 therefore remains nearly tied with R1-B4 on the
four-horizon average while shifting quality toward T2-T3 and away from T4-T5.
Against FH-adapt, 5 of the 20 registered task-metric cells are positive.

## Secondary score sensitivity

The non-primary `max` reducer improves over the corresponding historical
R1-B4 `max` reducer at every horizon:

| T | C2 max t-mAP | R1-B4 max t-mAP | Delta (pp) |
| ---: | ---: | ---: | ---: |
| 2 | 0.212424 | 0.198118 | +1.4306 |
| 3 | 0.132335 | 0.116108 | +1.6228 |
| 4 | 0.076211 | 0.067435 | +0.8776 |
| 5 | 0.053708 | 0.044011 | +0.9697 |

This is score-sensitivity evidence only. It cannot replace the frozen `mean`
reducer or rescue the failed comparison with FH-adapt.

FH-adapt itself is effectively tied with the frozen R1 FullHistory baseline:
its T2-T5 t-mAP deltas are -0.5178, -0.0583, +0.0912, and -0.1259 pp.

## Interpretation and limits

The final population reverses the development-only all-T advantage used to
select C2. The early-horizon gain and later-horizon loss are consistent with a
long-horizon memory aggregation or score-calibration limitation, but the
aggregate metrics alone do not identify a causal mechanism. The processed test
partition remains unavailable, so independent generalization is not
established.

The Protocol-B population was evaluated only after selection was frozen. The
seed-46 confirmation remained gate-skipped because the development comparison
against matched FH-adapt was not all-T positive.

## Evidence integrity

- Evaluation source commit: `d288af93cefc7cf8aaabb9b541b92539782c019c`
- Frozen selection hash: `fb805d98d508f5b7e00ebb3da6df0320b721d32e7beea874311ad664f8e0a244`
- C2 checkpoint: `a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724`
- FH-adapt checkpoint: `ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c`
- Coverage: 43 masters, 3 orders, 6 reference clusters, 129 order-units
- External cache validation: 119,209,970,094 C2 bytes and 14,772,582,818
  FH-adapt bytes; all 258 file hashes passed

Machine-readable values and the final verdict are in `all_t_metrics.csv`,
`deltas_all_metrics.csv`, `score_sensitivity.csv`, `verdict.json`, and
`evidence_manifest.json` in this directory.
