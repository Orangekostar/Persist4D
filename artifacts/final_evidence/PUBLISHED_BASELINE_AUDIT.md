# Published And Controlled Baseline Audit

## Table A: Standard Published Protocol

These are reported percentages from the ReScene4D standard published protocol;
none was rerun under the final common-prefix experiment.

| Method | t-mAP | t-mAP50 | t-mAP25 | mAP | mAP50 | mAP25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mask4D | 1.3 | 2.9 | 8.7 | 2.1 | 5.5 | 21.2 |
| Mask4Former | 17.0 | 38.9 | 59.1 | 21.7 | 45.6 | 66.3 |
| Mask3D + Semantic Matching | 20.1 | 32.9 | 38.6 | 25.9 | 42.3 | 73.9 |
| Mask3D + Geometric Matching | 20.7 | 43.1 | 62.4 | 29.7 | 54.1 | 70.9 |
| ReScene4D (C) | 34.8 | 52.5 | 66.8 | 43.3 | 64.3 | 81.9 |

Source: official ReScene4D paper, Table 1. Machine-readable copy:
`tables/table_1_published_4dsis.csv`. These values must not be compared as if
they used the long-horizon common-prefix protocol.

## Table B: Exact Common-Prefix Protocol

`tables/table_2_common_prefix.csv` contains only methods evaluated on the same
129 frozen sequence/order scopes and six independent scene clusters. It uses
paper-facing names and retains T4/T5 task, identity, gap, latency, VRAM, and
historical-state fields. The key T5 comparison is:

| Method | t-mAP | t-REC | IDSW rate | Gap recall | Latency ms | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full-History + Feature-Class | 0.0453 | 0.1366 | 0.1441 | 0.0581 | 1067.6* | 4964 |
| Adapted Full-History + Feature-Class | 0.0454 | 0.1312 | 0.1304 | 0.0581 | 1068.2* | 4770 |
| Persist4D | 0.0445 | 0.1067 | 0.1114 | 0.3120 | 440.4 | 2473 |

`*` Full-History latency and VRAM profile local perception; lightweight tracker
overhead is excluded. This makes the compute comparison conservative rather
than a complete Full-History-plus-tracker runtime.

## Strong Alternative Table

`tables/table_3_strong_alternatives.csv` isolates Full-History + Feature-Class,
Full-History + EMA, T2-to-T3 Horizon-Adapted Full-History + Feature-Class, and
Persist4D at T4/T5. It preserves the same protocol and runtime-scope qualifier.

## External Table Decision

External Table C/Table 4 is omitted. ReScan is `EXTERNAL_INCONCLUSIVE` because
coverage, natural gaps, and gap-scene count all fail their preregistered minima.
The official ReScan method is `RESCAN_METHOD_NOT_REPRODUCED`; LivingScenes is
`NOT_RUN`. Neither is inserted into the common-prefix table.
