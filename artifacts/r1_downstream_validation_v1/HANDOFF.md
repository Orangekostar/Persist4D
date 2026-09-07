# Persist4D R1 Downstream Validation Handoff

## 1. 状态

execution_status: COMPLETE
candidate_semantics: PASS
recovery_evidence: SUPPORTED
temporal_task_result: B2_AND_FULLHISTORY_REPORTED_WITH_REDUCER_SENSITIVITY
resource_evidence: SUPPORTED_IN_PROFILE
old_new_comparability: HISTORICAL_REFERENCE_ONLY
publication_status: PUSH_VERIFIED

## 2. 仓库和来源

repository: `git@github.com:Orangekostar/Persist4D.git`
branch: `research/persist4d-r1-downstream-validation-v1`
parent_commit: `39d81b0299e412f558bb3982dd06caab584518f7`
code_commit_at_generation: `43d72c518ab94b71985a9b7a28f45174054fdca8`
publication_commit: resolve with `git rev-parse HEAD` after checking out the remote branch; it is intentionally not self-recorded.
checkpoint_sha256: `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`
protocol_sha256: `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`
algorithm_semantic_hash: `3950865866acbfef88e28ed1a1448b8edd625ea6ded1cd49c23d55408551691e`
evaluator_source_hashes:
- `conf/p6a/default.yaml`: `dd7a9fccc098fc7e5faecc059d6202ca7f8a762aaee3e7182738a2d7f800aac2`
- `models/persistent_memory.py`: `111da9366ad873741cff5c9481c96f39a119b84b02e13c8fbedf7a2e32c8cbf8`
- `scripts/evaluate_persist4d_p6a.py`: `844cb976529bd541d3d3ac02bcd2e118cb527e05b02ae590544b078186de8614`
- `scripts/system_comparison_v2_inference.py`: `49787b1babff24b4f2892538e3bcf6694f3b9a1c003da36aadec67c273b2381b`
- `scripts/system_comparison_v3_identity.py`: `6ad16bb5ad0abf1370b383f56934e402340b64a8d224b2edde1cff2fb84108c6`

## 3. 科学问题与固定条件

在不训练、不修改 B2/B4、不改变 Protocol-B/V3 评价语义的条件下，检验 R1 下 B4 相对 B2 的 T4/T5 gap-recovery 优势。固定 seed 45、FP32、batch size 1、43 masters、3 orders、129 units、6 reference clusters、T1-T5 连续更新，报告 T2/T4/T5。
限制：不是跨 backbone、不是外部泛化、没有三个训练 seed，不证明 checkpoint 间 feature 表示相同，也不允许结果驱动调阈值。

## 4. 实际修改与命令

本分支相对 parent 的文件：
- `artifacts/r1_downstream_validation_v1/CODE_MAP.md`
- `artifacts/r1_downstream_validation_v1/EXPERIMENT_CONTRACT.md`
- `artifacts/r1_downstream_validation_v1/cache_manifest.json`
- `artifacts/r1_downstream_validation_v1/historical_replay_status.json`
- `artifacts/r1_downstream_validation_v1/input_manifest.json`
- `artifacts/r1_downstream_validation_v1/metrics/checkpoint_regime_comparison.csv`
- `artifacts/r1_downstream_validation_v1/metrics/gap_event_ledger.csv`
- `artifacts/r1_downstream_validation_v1/metrics/identity_aggregate.csv`
- `artifacts/r1_downstream_validation_v1/metrics/identity_per_cluster.csv`
- `artifacts/r1_downstream_validation_v1/metrics/identity_per_sequence.csv`
- `artifacts/r1_downstream_validation_v1/metrics/local_current.csv`
- `artifacts/r1_downstream_validation_v1/metrics/score_sensitivity.csv`
- `artifacts/r1_downstream_validation_v1/metrics/task_aggregate.csv`
- `artifacts/r1_downstream_validation_v1/metrics/task_per_cluster.csv`
- `artifacts/r1_downstream_validation_v1/metrics/task_per_sequence.csv`
- `artifacts/r1_downstream_validation_v1/profile/samples.csv`
- `artifacts/r1_downstream_validation_v1/profile/summary.csv`
- `artifacts/r1_downstream_validation_v1/runtime_config.yaml`
- `artifacts/r1_downstream_validation_v1/smoke_and_parity.json`
- `configs/r1_downstream_validation/default.yaml`
- `docs/superpowers/plans/2026-09-07-persist4d-r1-downstream-validation-v1.md`
- `scripts/analyze_r1_downstream_validation.py`
- `scripts/finalize_r1_downstream_validation.py`
- `scripts/profile_r1_downstream_validation.py`
- `scripts/r1_downstream_context.py`
- `scripts/run_r1_downstream_validation.py`
- `tests/test_r1_downstream_analysis.py`
- `tests/test_r1_downstream_cache.py`
- `tests/test_r1_downstream_context.py`
- `tests/test_r1_downstream_finalizer.py`
- `tests/test_r1_downstream_profile.py`

完整执行入口（外部路径由环境变量提供）：
```bash
PERSIST4D_PYTHON=${PERSIST4D_PYTHON:-python}
$PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py audit --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" --cache-root "$R1_CACHE_ROOT"
CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py smoke --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" --cache-root "$R1_CACHE_ROOT"
CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py cache-local --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" --cache-root "$R1_CACHE_ROOT"
CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py cache-full --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" --cache-root "$R1_CACHE_ROOT"
$PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py finalize-cache --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" --cache-root "$R1_CACHE_ROOT"
$PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py cache-parity --cache-root "$R1_CACHE_ROOT"
$PERSIST4D_PYTHON scripts/analyze_r1_downstream_validation.py --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" --cache-root "$R1_CACHE_ROOT"
CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/profile_r1_downstream_validation.py --device cuda:0 --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" --cache-root "$R1_CACHE_ROOT"
$PERSIST4D_PYTHON scripts/finalize_r1_downstream_validation.py --publication-status PUSH_VERIFIED
```

## 5. 覆盖和 parity

- local raw / sidecar / FullHistory: 645 / 645 / 645
- masters / orders / reference clusters: 43 / 3 / 6
- report horizons: T2/T4/T5; actual state updates: T1-T5
- six-cluster live T2 smoke: 6/6 pass
- cached T2 official candidate parity: 129/129 pass
- local-current invariance: PASS; B2/B4 consume the same 645 raw+sidecar candidates
- old replay: `HISTORICAL_REPLAY_UNAVAILABLE`; old raw/sidecar 0/645
- missing/failed: historical C-old raw replay only; it does not block R1-internal conclusions

## 6. R1 主结果

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

## 7. 任务结果与 reducer

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

Reducer sensitivity:

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

Direct local current（不等同于 trajectory current slice）：

| Horizon | Direct local AP | AP50 | AP25 | REC |
| --- | ---: | ---: | ---: | ---: |
| T2 | 0.362199 | 0.578001 | 0.768074 | 0.510894 |
| T4 | 0.334830 | 0.558548 | 0.789076 | 0.509474 |
| T5 | 0.367196 | 0.605087 | 0.787692 | 0.524154 |

## 8. 旧新对照

| Channel | Metric | Method | Reducer | T | C-old | R1 | R1-C-old |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| task | causal_prefix_t_mAP | B4 | mean | T2 | 0.2072410136461258 | 0.2196367383003235 | 0.012395724654197693 |
| task | causal_prefix_t_REC | B4 | mean | T2 | 0.34868067502975464 | 0.3610108196735382 | 0.01233014464378357 |
| task | causal_prefix_t_mAP | B4 | mean | T4 | 0.07023176550865173 | 0.08015920221805573 | 0.009927436709403992 |
| task | causal_prefix_t_REC | B4 | mean | T4 | 0.17581450939178467 | 0.20148883759975433 | 0.025674328207969666 |
| task | causal_prefix_t_mAP | B4 | mean | T5 | 0.05250302702188492 | 0.05853525176644325 | 0.006032224744558334 |
| task | causal_prefix_t_REC | B4 | mean | T5 | 0.13842646777629852 | 0.15866969525814056 | 0.02024322748184204 |
| identity | gap_recovery_recall | B2 | not_applicable | T2 | N/A | N/A | N/A |
| identity | gap_recovery_recall | B4 | not_applicable | T2 | N/A | N/A | N/A |
| identity | gap_recovery_recall | B2 | not_applicable | T4 | 0.09770114942528736 | 0.05459770114942529 | -0.04310344827586207 |
| identity | gap_recovery_recall | B4 | not_applicable | T4 | 0.2974137931034483 | 0.2959770114942529 | -0.0014367816091954144 |
| identity | gap_recovery_recall | B2 | not_applicable | T5 | 0.08508691674290943 | 0.05123513266239707 | -0.03385178408051236 |
| identity | gap_recovery_recall | B4 | not_applicable | T5 | 0.3119853613906679 | 0.3101555352241537 | -0.0018298261665142257 |
| local_current | local_current_AP | DirectLocalCurrent | not_applicable | T2 | 0.371692419052124 | 0.36219850182533264 | -0.009493917226791382 |
| local_current | local_current_AP | DirectLocalCurrent | not_applicable | T4 | 0.36625537276268005 | 0.3348296284675598 | -0.03142574429512024 |
| local_current | local_current_AP | DirectLocalCurrent | not_applicable | T5 | 0.3829203248023987 | 0.3671956956386566 | -0.015724629163742065 |

可比性：`HISTORICAL_REFERENCE_ONLY`。C-old 仅有冻结摘要，缺少 raw/sidecar；表中差值是描述性 checkpoint-regime 参照，不是 AP 或 feature 变化的因果效应。

## 9. 资源

同一 A40、FP32、六簇各第一 canonical master、T2/T4/T5、5 warmups + 10 measured repeats。计时包含 CUDA 同步的模型 forward；B4 另含 CPU tracker，排除 I/O、collate、H2D、metric scoring 和 tracker preroll。360 个逐次 latency 在 `profile/samples.csv`。

| Reference | Master | T | Method | Median ms | Peak alloc bytes | Peak reserved bytes | Input scans | Input points | State bytes | Occupied slots | Rejected births |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10b17940-3938-2467-8a7a-958300ba83d3` | `scene0069_00-scene0069_02-scene0069_04-scene0069_03-scene0069_01` | T2 | B4 | 490.776 | 1600377856 | 2336227328 | 2 | 87611 | 61008 | 11 | 0 |
| `10b17940-3938-2467-8a7a-958300ba83d3` | `scene0069_00-scene0069_02-scene0069_04-scene0069_03-scene0069_01` | T4 | B4 | 470.981 | 1332816896 | 1868562432 | 2 | 66735 | 61008 | 12 | 0 |
| `10b17940-3938-2467-8a7a-958300ba83d3` | `scene0069_00-scene0069_02-scene0069_04-scene0069_03-scene0069_01` | T5 | B4 | 442.938 | 1482817024 | 2076180480 | 2 | 79393 | 61008 | 14 | 0 |
| `10b17940-3938-2467-8a7a-958300ba83d3` | `scene0069_00-scene0069_02-scene0069_04-scene0069_03-scene0069_01` | T2 | FullHistory | 505.818 | 1600566272 | 2298478592 | 2 | 87611 | N/A | N/A | N/A |
| `10b17940-3938-2467-8a7a-958300ba83d3` | `scene0069_00-scene0069_02-scene0069_04-scene0069_03-scene0069_01` | T4 | FullHistory | 647.546 | 2270589952 | 3433037824 | 4 | 154346 | N/A | N/A | N/A |
| `10b17940-3938-2467-8a7a-958300ba83d3` | `scene0069_00-scene0069_02-scene0069_04-scene0069_03-scene0069_01` | T5 | FullHistory | 813.378 | 2818883584 | 4383047680 | 5 | 205984 | N/A | N/A | N/A |
| `137a8158-1db5-2cc0-8003-31c12610471e` | `scene0079_00-scene0079_09-scene0079_05-scene0079_04-scene0079_03` | T2 | B4 | 561.154 | 1912354816 | 2801795072 | 2 | 113770 | 61008 | 17 | 0 |
| `137a8158-1db5-2cc0-8003-31c12610471e` | `scene0079_00-scene0079_09-scene0079_05-scene0079_04-scene0079_03` | T4 | B4 | 472.459 | 1500756480 | 2151677952 | 2 | 76310 | 61008 | 18 | 0 |
| `137a8158-1db5-2cc0-8003-31c12610471e` | `scene0079_00-scene0079_09-scene0079_05-scene0079_04-scene0079_03` | T5 | B4 | 387.639 | 1208503808 | 1671430144 | 2 | 48972 | 61008 | 18 | 0 |
| `137a8158-1db5-2cc0-8003-31c12610471e` | `scene0079_00-scene0079_09-scene0079_05-scene0079_04-scene0079_03` | T2 | FullHistory | 583.133 | 1945516032 | 2847932416 | 2 | 113770 | N/A | N/A | N/A |
| `137a8158-1db5-2cc0-8003-31c12610471e` | `scene0079_00-scene0079_09-scene0079_05-scene0079_04-scene0079_03` | T4 | FullHistory | 813.861 | 2747249664 | 4294967296 | 4 | 190080 | N/A | N/A | N/A |
| `137a8158-1db5-2cc0-8003-31c12610471e` | `scene0079_00-scene0079_09-scene0079_05-scene0079_04-scene0079_03` | T5 | FullHistory | 898.628 | 3096528384 | 4961861632 | 5 | 224746 | N/A | N/A | N/A |
| `280d8ebb-6cc6-2788-9153-98959a2da801` | `scene0119_00-scene0119_02-scene0119_03-scene0119_01-scene0119_04` | T2 | B4 | 779.151 | 2584049152 | 3869245440 | 2 | 184174 | 61008 | 22 | 0 |
| `280d8ebb-6cc6-2788-9153-98959a2da801` | `scene0119_00-scene0119_02-scene0119_03-scene0119_01-scene0119_04` | T4 | B4 | 671.754 | 2202375168 | 3246391296 | 2 | 142278 | 61008 | 22 | 0 |
| `280d8ebb-6cc6-2788-9153-98959a2da801` | `scene0119_00-scene0119_02-scene0119_03-scene0119_01-scene0119_04` | T5 | B4 | 749.677 | 2591440896 | 3930062848 | 2 | 189099 | 61008 | 23 | 0 |
| `280d8ebb-6cc6-2788-9153-98959a2da801` | `scene0119_00-scene0119_02-scene0119_03-scene0119_01-scene0119_04` | T2 | FullHistory | 711.643 | 2607096320 | 3898605568 | 2 | 184174 | N/A | N/A | N/A |
| `280d8ebb-6cc6-2788-9153-98959a2da801` | `scene0119_00-scene0119_02-scene0119_03-scene0119_01-scene0119_04` | T4 | FullHistory | 1127.383 | 4034769408 | 6339690496 | 4 | 326452 | N/A | N/A | N/A |
| `280d8ebb-6cc6-2788-9153-98959a2da801` | `scene0119_00-scene0119_02-scene0119_03-scene0119_01-scene0119_04` | T5 | FullHistory | 1494.682 | 5206044672 | 8420065280 | 5 | 454986 | N/A | N/A | N/A |
| `5630cfcf-12bf-2860-8784-83d28a611a83` | `scene0219_00-scene0219_03-scene0219_02-scene0219_01-scene0219_04` | T2 | B4 | 555.824 | 1841427968 | 2692743168 | 2 | 108716 | 61008 | 13 | 0 |
| `5630cfcf-12bf-2860-8784-83d28a611a83` | `scene0219_00-scene0219_03-scene0219_02-scene0219_01-scene0219_04` | T4 | B4 | 497.381 | 1676399616 | 2451570688 | 2 | 93812 | 61008 | 17 | 0 |
| `5630cfcf-12bf-2860-8784-83d28a611a83` | `scene0219_00-scene0219_03-scene0219_02-scene0219_01-scene0219_04` | T5 | B4 | 522.882 | 1719294976 | 2499805184 | 2 | 100521 | 61008 | 17 | 0 |
| `5630cfcf-12bf-2860-8784-83d28a611a83` | `scene0219_00-scene0219_03-scene0219_02-scene0219_01-scene0219_04` | T2 | FullHistory | 506.498 | 1924826624 | 2774532096 | 2 | 108716 | N/A | N/A | N/A |
| `5630cfcf-12bf-2860-8784-83d28a611a83` | `scene0219_00-scene0219_03-scene0219_02-scene0219_01-scene0219_04` | T4 | FullHistory | 796.297 | 2843488768 | 4359979008 | 4 | 202528 | N/A | N/A | N/A |
| `5630cfcf-12bf-2860-8784-83d28a611a83` | `scene0219_00-scene0219_03-scene0219_02-scene0219_01-scene0219_04` | T5 | FullHistory | 979.692 | 3442065408 | 5433720832 | 5 | 263273 | N/A | N/A | N/A |
| `8eabc45f-5af7-2f32-8528-640861d2a135` | `scene0309_00-scene0309_02-scene0309_01-scene0309_04-scene0309_03` | T2 | B4 | 785.689 | 2679774720 | 4164943872 | 2 | 180028 | 61008 | 19 | 0 |
| `8eabc45f-5af7-2f32-8528-640861d2a135` | `scene0309_00-scene0309_02-scene0309_01-scene0309_04-scene0309_03` | T4 | B4 | 573.995 | 1951689216 | 2805989376 | 2 | 109486 | 61008 | 20 | 0 |
| `8eabc45f-5af7-2f32-8528-640861d2a135` | `scene0309_00-scene0309_02-scene0309_01-scene0309_04-scene0309_03` | T5 | B4 | 571.183 | 1951066112 | 2782920704 | 2 | 107348 | 61008 | 20 | 0 |
| `8eabc45f-5af7-2f32-8528-640861d2a135` | `scene0309_00-scene0309_02-scene0309_01-scene0309_04-scene0309_03` | T2 | FullHistory | 757.322 | 2715262976 | 4020240384 | 2 | 180028 | N/A | N/A | N/A |
| `8eabc45f-5af7-2f32-8528-640861d2a135` | `scene0309_00-scene0309_02-scene0309_01-scene0309_04-scene0309_03` | T4 | FullHistory | 1137.600 | 3930477568 | 6230638592 | 4 | 289514 | N/A | N/A | N/A |
| `8eabc45f-5af7-2f32-8528-640861d2a135` | `scene0309_00-scene0309_02-scene0309_01-scene0309_04-scene0309_03` | T5 | FullHistory | 1271.097 | 4392021504 | 7186939904 | 5 | 331656 | N/A | N/A | N/A |
| `ddc73797-765b-241a-9e2c-097c5989baf6` | `scene0449_00-scene0449_04-scene0449_05-scene0449_03-scene0449_01` | T2 | B4 | 800.029 | 2612613632 | 4009754624 | 2 | 184916 | 61008 | 23 | 0 |
| `ddc73797-765b-241a-9e2c-097c5989baf6` | `scene0449_00-scene0449_04-scene0449_05-scene0449_03-scene0449_01` | T4 | B4 | 648.151 | 2079701504 | 3185573888 | 2 | 139246 | 61008 | 24 | 0 |
| `ddc73797-765b-241a-9e2c-097c5989baf6` | `scene0449_00-scene0449_04-scene0449_05-scene0449_03-scene0449_01` | T5 | B4 | 423.920 | 1301706752 | 1799356416 | 2 | 70052 | 61008 | 24 | 0 |
| `ddc73797-765b-241a-9e2c-097c5989baf6` | `scene0449_00-scene0449_04-scene0449_05-scene0449_03-scene0449_01` | T2 | FullHistory | 756.885 | 2652210176 | 4076863488 | 2 | 184916 | N/A | N/A | N/A |
| `ddc73797-765b-241a-9e2c-097c5989baf6` | `scene0449_00-scene0449_04-scene0449_05-scene0449_03-scene0449_01` | T4 | FullHistory | 1187.871 | 3980517888 | 6306136064 | 4 | 324162 | N/A | N/A | N/A |
| `ddc73797-765b-241a-9e2c-097c5989baf6` | `scene0449_00-scene0449_04-scene0449_05-scene0449_03-scene0449_01` | T5 | FullHistory | 1234.093 | 4187227136 | 6822035456 | 5 | 359322 | N/A | N/A | N/A |

## 10. 解释

R1 下 T4/T5 的 pooled gap-recovery 与六簇方向均支持预注册恢复结论。task 与 identity 是不同通道；B4 相对 B2 的 identity 增益不能改写为对 FullHistory 的质量胜出。资源结果只描述固定 K、局部窗口、相近单 scan 规模和 T<=5；不覆盖归档、host RAM、metric accumulator 或无限时间行为。

## 11. 文件和复现条件

Git 产物位于 `artifacts/r1_downstream_validation_v1/`：合同、代码图、输入/缓存/smoke manifest、10 个 metrics CSV、profile 两表、FINAL_REPORT、HANDOFF、FINAL_MANIFEST。
外部未入 Git：`R1_CHECKPOINT`、`CONCERTO_PRETRAINED`、`RIO_METADATA`、`PERSIST4D_DATA_ROOT`、`R1_CACHE_ROOT`（逻辑别名 `external:r1_downstream_validation_v1/cache`）。不提交凭据或私人绝对路径。
重跑条件：相同 checkpoint/protocol/config SHA、同一代码提交、seed 45、FP32、batch size 1、单 A40；任何绑定漂移必须重新 audit，不能复用结果。

## 12. 验证与资源释放

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_r1_downstream_*.py
$PERSIST4D_PYTHON -m ruff check scripts/r1_downstream_context.py scripts/run_r1_downstream_validation.py scripts/analyze_r1_downstream_validation.py scripts/profile_r1_downstream_validation.py scripts/finalize_r1_downstream_validation.py tests/test_r1_downstream_*.py
$PERSIST4D_PYTHON scripts/finalize_r1_downstream_validation.py --publication-status PUSH_VERIFIED
git diff --check
nvidia-smi
```
终验仅使用提示词规定的相关测试组，不以全仓测试作为本实验门禁。实验进程退出后释放其 CUDA 上下文；不终止其他用户进程。

## 13. GitHub

branch_url: `https://github.com/Orangekostar/Persist4D/tree/research/persist4d-r1-downstream-validation-v1`
```bash
git rev-parse HEAD
git ls-remote --heads origin refs/heads/research/persist4d-r1-downstream-validation-v1
git show origin/research/persist4d-r1-downstream-validation-v1:artifacts/r1_downstream_validation_v1/HANDOFF.md
git show origin/research/persist4d-r1-downstream-validation-v1:artifacts/r1_downstream_validation_v1/FINAL_MANIFEST.json
```
PR: 未创建；本轮要求是分支推送与远端回读，不虚构 PR。

## 14. 下一步

唯一最高优先级任务：若需要严格 old/new 配对结论，先恢复 C-old 的 645 raw + 645 sidecar，并在同一代码/runtime 下只做历史回放；不要继续训练、换 checkpoint 或调 B4 阈值。
