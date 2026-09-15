# TaskMemory Retention V2 Handoff

## 1. Goal and verdict

- EXECUTION: `COMPLETE`
- TMAP_ALL_T_VS_R1: `FAIL`
- TMAP_ALL_T_VS_MATCHED_FH: `PASS`
- TASK_METRICS_ALL_T: `PASS`
- RETENTION: `IMPROVED`
- RESOURCE: `TRADEOFF`
- MECHANISM: `PARTIAL`
- GENERALIZATION: `BASE_EXPOSED_ONLY`
- PUBLICATION: `PUSH_VERIFIED`

## 2. Repository

Parent: `32a51e11b51043ec5a5825215669ef7b0ea03bc2`. Branch: `research/persist4d-task-memory-retention-v2`. Results commit E: `3027b942a5d76dea1dad3f42b3c67c0a9c9383ee`. Publication tip is resolved from the remote branch containing this file.

## 3. What was actually implemented

Implemented causal TaskMemory route/commit state, TALA sequence supervision, bounded visual memory, compact evaluation caches, frozen training/evaluation entry points, final analysis, and one-A40 resource profiling. Numeric artifacts in FINAL_MANIFEST.json were actually run; items named under limitations were not.

## 4. Knowledge and upstream evidence

Upstream code and literature identities, immutable revisions/blobs, and license-use boundaries are recorded in EVIDENCE_MAP.md and references_inventory.csv. Reviewed parent is fixed above.

## 5. Data and output policy

Primary confirmation population is frozen Protocol-B: 6 physical references, 43 masters, and 129 deterministic orders. Primary output uses lag1/mean; commit0 is a separate policy control. Development, Protocol-B, and independent-native evidence are not pooled.

## 6. Model state and training

Selected M3-V-CORE uses K=100, r=8 and measured permanent state 489816 bytes. It was selected at update 1500 by the frozen development rule. FH-CONT is the matched full-history continuation control; training seed and evaluation seed are both 45 and remain distinct fields.

## 7. Experiments completed and not run

Completed M2 W-BASE/Q-INDEP/Q-TALA/FH-MATCH, M3 BASE-CONT/V-LAST/V-CORE, FH-CONT, the unified fourteen-method Protocol-B table, base-exposed native evaluation, long-memory controls, per-reference analysis, and resource profiling. M4 was conditionally skipped as `SKIPPED_BUDGET` because `paired_CONT_and_KD_lower_bound_exceeds_remaining_budget_before_teacher_forward`. Overall execution is `COMPLETE`; scientific limitations remain in section 13.

## 8. Primary table

| Variant | T2 | T3 | T4 | T5 | Mean |
|---|---:|---:|---:|---:|---:|
| FH-R1-native | 0.228777 | 0.148305 | 0.102745 | 0.081566 | 0.140349 |
| B4-commit0 | 0.219637 | 0.135498 | 0.080159 | 0.058535 | 0.123457 |
| R1+B4-lag1 | 0.244751 | 0.123902 | 0.061478 | 0.034896 | 0.116257 |
| FH-R1-lag1 | 0.243306 | 0.105958 | 0.047023 | 0.016563 | 0.103213 |
| W-BASE | 0.219235 | 0.091059 | 0.036850 | 0.011986 | 0.089783 |
| Q-TALA | 0.226566 | 0.134196 | 0.077771 | 0.046475 | 0.121252 |
| M3-BASE-CONT | 0.223988 | 0.133415 | 0.078771 | 0.048050 | 0.121056 |
| M3-V-CORE | 0.223172 | 0.135569 | 0.067608 | 0.038498 | 0.116212 |
| FH-MATCH | 0.224733 | 0.097008 | 0.037274 | 0.013837 | 0.093213 |
| FH-CONT | 0.208039 | 0.085259 | 0.032787 | 0.012343 | 0.084607 |
| D-LAST-commit0 | 0.220178 | 0.138127 | 0.080269 | 0.057070 | 0.123911 |
| D-LAST-lag1 | 0.249229 | 0.155346 | 0.090114 | 0.064331 | 0.139755 |
| D-EMA-commit0 | 0.220178 | 0.138226 | 0.078855 | 0.059468 | 0.124182 |
| D-EMA-lag1 | 0.249229 | 0.154903 | 0.088704 | 0.063843 | 0.139170 |

- M3-V-CORE_vs_B4-commit0: tMAP `FAIL`, 20-cell `12/20`, minimum tMAP delta `-0.020037`, failed tMAP horizons `T4, T5`.
- M3-V-CORE_vs_R1+B4-lag1: tMAP `FAIL`, 20-cell `14/20`, minimum tMAP delta `-0.021579`, failed tMAP horizons `T2`.
- M3-V-CORE_vs_FH-R1-native: tMAP `FAIL`, 20-cell `4/20`, minimum tMAP delta `-0.043068`, failed tMAP horizons `T2, T3, T4, T5`.
- M3-V-CORE_vs_FH-R1-lag1: tMAP `FAIL`, 20-cell `16/20`, minimum tMAP delta `-0.020134`, failed tMAP horizons `T2`.
- M3-V-CORE_vs_W-BASE: tMAP `PASS`, 20-cell `18/20`, minimum tMAP delta `0.003936`, failed tMAP horizons `none`.
- M3-V-CORE_vs_Q-TALA: tMAP `FAIL`, 20-cell `9/20`, minimum tMAP delta `-0.010163`, failed tMAP horizons `T2, T4, T5`.
- M3-V-CORE_vs_M3-BASE-CONT: tMAP `FAIL`, 20-cell `5/20`, minimum tMAP delta `-0.011163`, failed tMAP horizons `T2, T4, T5`.
- M3-V-CORE_vs_FH-MATCH: tMAP `FAIL`, 20-cell `17/20`, minimum tMAP delta `-0.001561`, failed tMAP horizons `T2`.
- M3-V-CORE_vs_FH-CONT: tMAP `PASS`, 20-cell `20/20`, minimum tMAP delta `0.015133`, failed tMAP horizons `none`.
- M3-V-CORE_vs_D-LAST-lag1: tMAP `FAIL`, 20-cell `1/20`, minimum tMAP delta `-0.026057`, failed tMAP horizons `T2, T3, T4, T5`.
- M3-V-CORE_vs_D-EMA-lag1: tMAP `FAIL`, 20-cell `1/20`, minimum tMAP delta `-0.026057`, failed tMAP horizons `T2, T3, T4, T5`.

## 9. Mechanism and failure evidence

Mechanism verdict is `PARTIAL`. The final development real/read-off/previous/unrelated visual-content comparison is in visual/paired_content_ablation.csv; route, gap, birth, reactivation, fragmentation, merge, and ID-switch evidence remains separate from task AP in final/identity_counts.csv.

## 10. Cost and storage

Recorded campaign training GPU-hours after FH-CONT: 112.574297 / 120. Permanent state budget: 489816 / 2097152 bytes.

Resource verdict: `TRADEOFF`.
- M3-V-CORE T4: panel median model_update 1414.789 ms; absolute peak 1694087936 bytes.
- M3-V-CORE T5: panel median model_update 1341.676 ms; absolute peak 1509274880 bytes.
- FH-CONT T4: panel median model_update 993.039 ms; absolute peak 3245945344 bytes.
- FH-CONT T5: panel median model_update 1219.371 ms; absolute peak 3671237632 bytes.

## 11. Selected and resume checkpoints

M3-V-CORE selected checkpoint: `1b883db3faf26f168274853857719786125d640e24377f99d427d2b8aec7793f`, logical location `external:run_root/training/formal/M3-V-CORE/update=1500.ckpt`. FH-CONT selected checkpoint: `6672569c9524d38a0f932c9d8351d6606b9b26a574236a59d4799f225178d4f5`, logical location `external:run_root/training/formal/FH-CONT/update=1500.ckpt`. Selected checkpoints are not presented as optimizer-complete resume checkpoints; their run directories retain separate `last.ckpt` files.

## 12. Reproduction commands

Public environment-variable commands are in COMMANDS.md. Final direct verification used:
- `correctness_suite`: `conda run -n persist4d python -m pytest -q tests/test_persist4d_task_memory.py tests/test_object_visual_memory.py tests/test_train_task_memory.py tests/test_task_memory_*.py tests/test_analyze_task_memory_final.py tests/test_publish_task_memory_v2.py tests/test_profile_task_memory.py`
- `real_gpu_gate`: `jq -e '.status == "PASS" and .device == "NVIDIA A40" and .warmups == 5 and .repeats == 10 and .measurement_rows == 1920 and .summary_rows == 192 and .permanent_state_bytes == 489816' artifacts/task_memory_retention_v2/resources/run_summary.json`
- `ruff_changed_python`: `ruff check scripts/run_task_memory_controls.py scripts/analyze_task_memory_final.py scripts/publish_task_memory_v2.py tests/test_task_memory_controls.py tests/test_analyze_task_memory_final.py tests/test_publish_task_memory_v2.py`
- `git_diff_check`: `git diff --check`
- `manifest_and_hash`: `conda run -n persist4d python -c 'import csv,json; from pathlib import Path; from scripts.task_memory_contracts import canonical_json_sha256; r=Path("artifacts/task_memory_retention_v2"); files=("final/status.json","resources/run_summary.json","evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json","evaluation/M5/protocol_b/baseline/control_observation_manifest.json"); assert all((lambda v:v["content_sha256"]==canonical_json_sha256({k:x for k,x in v.items() if k!="content_sha256"}))(json.loads((r/p).read_text())) for p in files); paths=("final/all_t_metrics.csv","evaluation/M5/protocol_b/baseline/long_memory_controls.csv","evaluation/M5/protocol_b/R1-B4-policy/policy_comparison.csv","final/independent_reference_results.csv"); counts=tuple(sum(1 for _ in csv.DictReader((r/p).open())) for p in paths); assert counts==(56,16,8,9); print("four canonical manifests and formal row coverage PASS", counts)'`
- `secret_scan`: `! git diff --name-only -z 6933f635f1834849de3e00aef60cd737f0aa1cba 3027b942a5d76dea1dad3f42b3c67c0a9c9383ee | xargs -0 rg -q --no-messages -e 'ghp_[A-Za-z0-9]{30,}' -e 'github_pat_[A-Za-z0-9_]{30,}' -e '-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----' -e 'Bearer [A-Za-z0-9_.-]{30,}'`

## 13. Tests and limitations

All eight correctness classes and the real A40 gate are detailed in TEST_REPORT.md. Remaining limitations:
- single training seed
- native adaptation-holdout references were exposed to the original R1 base
- native adaptation-holdout evidence covers T2-T4 only
- FH_ENCODER_CACHE_NOT_ESTABLISHED
- new-model capacity saturation was not entered; capacity effect is not established

## 14. Claims supported / not supported

Supported claims are limited to the exact all-T, retention, mechanism, and resource statuses in section 1. Not supported: indefinite-horizon retention, replicated training stability, truly unseen-base generalization, or attribution of policy effects to memory content alone.

## 15. GitHub and next exact action

Results commit E publication status: `PUSH_VERIFIED`. FINAL_MANIFEST.json SHA256: `7d94f79cff190d7c4e28f54ac6b3b518da284b73f038a56daa73d0a85b13bdb9`. Next exact action: replicate M3-V-CORE and its strongest matched baseline with a second training seed before making a stability claim.
