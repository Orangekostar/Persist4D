# Protocol Shift Audit

## Frozen inputs

- `repo:artifacts/P6A/protocol_b_manifest.json`: `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`
- `repo:data/processed/rio/sequence_database_sliding_2.yaml`: `974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416`
- `repo:artifacts/system_comparison/per_order_results.csv`: `6d950fa97da08892f5e86c08a6ff24e7710ff5090e46a664e4cb8394d8a64705`
- `repo:artifacts/system_comparison/aggregate_results.csv`: `c70000d35e9d661c9929d25a465d287f778fc594b49fb596aa36d956e68f73a0`

## R0-R5 observations

| ID | Population | Order scope | N | t-mAP |
| --- | --- | --- | ---: | ---: |
| R0 | official T2 benchmark | paper_reported | not reported | 34.800000% |
| R1 | official-like supervised T2 validation | P2 sliding-T2 | 154 | 27.939000% |
| R2 | Protocol B common-T5 masters | canonical | 43 | 20.722659% |
| R3 | Protocol B common-T5 masters | reverse | 43 | 21.109875% |
| R4 | Protocol B common-T5 masters | sha256_seed45 | 43 | 17.118087% |
| R5 | Protocol B common-T5 masters | three_order_pooled | 129 | 19.099636% |

R0 and R1 target the official/official-like T2 benchmark and are the only intended benchmark-level comparison. R2-R4 are directly comparable order diagnostics inside Protocol B. R5 pools the same three Protocol-B orders; it is not an independent benchmark population.

34.8 and 19.10 are not directly comparable: R0 is paper-reported official T2, while R5 is a pooled causal-prefix result over 43 common-T5 masters and three metadata-derived orders.

## Exact matched-subset audit

- Requested canonical T2 pairs: 43
- Exact ordered pairs present in the P2 sliding-T2 DB: 14
- Missing exact ordered pairs: 29
- Exact full 43-pair control: NOT IDENTIFIABLE FROM CURRENT ARTIFACTS

Only exact ordered sequence IDs count as matches. Reverse pairs and pairs from the same scene are not substituted. Because 29 of 43 canonical pairs are absent, an official-like 43-pair evaluation cannot be constructed. The 14 available pairs are retained as inventory evidence only and are not presented as the requested matched control.

## Order effect

- Canonical R2 t-mAP: 20.722659%
- Pooled R5 t-mAP: 19.099636%
- Pooled minus canonical: -1.623023 percentage points
- Per-order max-minus-min spread: 3.991787 percentage points

The canonical-to-pooled difference measures sensitivity to the registered Protocol-B orders, not a model degradation from the paper score. The common-T5 subset effect cannot be isolated exactly with current artifacts. These deltas must not be added into a causal decomposition of 34.8 to 19.10.

## Gate E1

`E1 = PASS`: population, order, evaluator, and comparability boundaries are explicit. The exact 43-pair subset effect remains non-identifiable and is reported as such rather than approximated.
