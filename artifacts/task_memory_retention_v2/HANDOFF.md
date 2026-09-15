# TaskMemory Retention V2 Handoff

## 1. Goal and verdict

- EXECUTION: `PARTIAL`
- TMAP_ALL_T_VS_R1: `FAIL`
- TMAP_ALL_T_VS_MATCHED_FH: `PASS`
- TASK_METRICS_ALL_T: `PASS`
- RETENTION: `IMPROVED`
- RESOURCE: `TRADEOFF`
- MECHANISM: `PARTIAL`
- GENERALIZATION: `NOT_ESTABLISHED`
- PUBLICATION: `PUSH_VERIFIED`

## 2. Repository

Parent: `32a51e11b51043ec5a5825215669ef7b0ea03bc2`. Branch: `research/persist4d-task-memory-retention-v2`. Results commit E: `fb4b6df8f46449981bcb344913de04208e312ba5`. Publication tip is resolved from the remote branch containing this file.

## 3. What was actually implemented

Implemented causal TaskMemory route/commit state, TALA sequence supervision, bounded visual memory, compact evaluation caches, frozen training/evaluation entry points, final analysis, and one-A40 resource profiling. Numeric artifacts in FINAL_MANIFEST.json were actually run; items named under limitations were not.

## 4. Knowledge and upstream evidence

Upstream code and literature identities, immutable revisions/blobs, and license-use boundaries are recorded in EVIDENCE_MAP.md and references_inventory.csv. Reviewed parent is fixed above.

## 5. Data and output policy

Primary confirmation population is frozen Protocol-B: 6 physical references, 43 masters, and 129 deterministic orders. Primary output uses lag1/mean; commit0 is a separate policy control. Development, Protocol-B, and independent-native evidence are not pooled.

## 6. Model state and training

Selected M3-V-CORE uses K=100, r=8 and measured permanent state 489816 bytes. It was selected at update 1500 by the frozen development rule. FH-CONT is the matched full-history continuation control; training seed and evaluation seed are both 45 and remain distinct fields.

## 7. Experiments completed and not run

Completed M2 W-BASE/Q-INDEP/Q-TALA/FH-MATCH, M3 BASE-CONT/V-LAST/V-CORE, FH-CONT, final three-model Protocol-B inference, per-reference analysis, and resource profiling. M4 was conditionally skipped as `SKIPPED_BUDGET` because `paired_CONT_and_KD_lower_bound_exceeds_remaining_budget_before_teacher_forward`. Overall execution remains PARTIAL for the explicit limitations in section 13.

## 8. Primary table

| Variant | T2 | T3 | T4 | T5 | Mean |
|---|---:|---:|---:|---:|---:|
| R1+B4 | 0.219637 | 0.135498 | 0.080159 | 0.058535 | 0.123457 |
| FH-R1 | 0.228777 | 0.148305 | 0.102745 | 0.081566 | 0.140349 |
| M3-BASE-CONT | 0.223988 | 0.133415 | 0.078771 | 0.048050 | 0.121056 |
| FH-CONT | 0.208039 | 0.085259 | 0.032787 | 0.012343 | 0.084607 |
| M3-V-CORE | 0.223172 | 0.135569 | 0.067608 | 0.038498 | 0.116212 |

- M3-V-CORE_vs_R1+B4: tMAP `FAIL`, 20-cell `12/20`, minimum tMAP delta `-0.020037`, failed tMAP horizons `T4, T5`.
- M3-V-CORE_vs_FH-R1: tMAP `FAIL`, 20-cell `4/20`, minimum tMAP delta `-0.043068`, failed tMAP horizons `T2, T3, T4, T5`.
- M3-V-CORE_vs_M3-BASE-CONT: tMAP `FAIL`, 20-cell `5/20`, minimum tMAP delta `-0.011163`, failed tMAP horizons `T2, T4, T5`.
- M3-V-CORE_vs_FH-CONT: tMAP `PASS`, 20-cell `20/20`, minimum tMAP delta `0.015133`, failed tMAP horizons `none`.

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
- `correctness_suite`: `python -m pytest -q tests/test_persist4d_task_memory.py tests/test_task_memory_episode.py tests/test_task_memory_routing.py tests/test_task_memory_output.py tests/test_task_memory_criterion.py tests/test_object_visual_memory.py tests/test_task_memory_trainer.py tests/test_train_task_memory.py tests/test_task_memory_evaluation.py tests/test_analyze_task_memory_final.py tests/test_profile_task_memory.py tests/test_publish_task_memory_v2.py`
- `real_gpu_gate`: `jq -e '.status == "PASS" and .device == "NVIDIA A40" and .warmups == 5 and .repeats == 10 and .measurement_rows == 1920 and .summary_rows == 192 and .permanent_state_bytes == 489816' artifacts/task_memory_retention_v2/resources/run_summary.json`
- `ruff_changed_python`: `git diff --name-only 32a51e -- '*.py' | rg -v '^(datasets/pointcept_utils.py|datasets/semseg.py)$' | xargs -r ruff check && for file in datasets/pointcept_utils.py datasets/semseg.py; do diff -u <(git show "32a51e:$file" | ruff check --stdin-filename "$file" --output-format=json - | jq -S 'group_by(.code) | map({code: .[0].code, count: length})') <(ruff check "$file" --output-format=json | jq -S 'group_by(.code) | map({code: .[0].code, count: length})'); done`
- `git_diff_check`: `git diff --cached --check`
- `manifest_and_hash`: `python -c 'import csv,hashlib,json; from pathlib import Path; from scripts.task_memory_contracts import canonical_json_sha256; root=Path("artifacts/task_memory_retention_v2"); a=json.loads((root/"final/status.json").read_text()); p=json.loads((root/"resources/run_summary.json").read_text()); assert a["content_sha256"]==canonical_json_sha256({k:v for k,v in a.items() if k!="content_sha256"}); assert p["content_sha256"]==canonical_json_sha256({k:v for k,v in p.items() if k!="content_sha256"}); assert all(hashlib.sha256((base/item["path"]).read_bytes()).hexdigest()==item["sha256"] for owner,base in ((a,root/"final"),(p,root/"resources")) for item in owner["outputs"]); counts=tuple(sum(1 for _ in csv.DictReader((root/path).open())) for path in ("final/all_t_metrics.csv","final/paired_deltas.csv","final/per_reference_metrics.csv","resources/per_update_measurements.csv","resources/profile_summary.csv","resources/profile_units.csv")); assert counts==(20,80,48,1920,192,6); print("content hashes and row counts passed", counts)'`
- `secret_scan`: `set -o pipefail; ! git diff --text 32a51e -- | rg -n -i -- '-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}'; ! git diff --cached --text -- artifacts/task_memory_retention_v2 | rg -n -- "$PERSIST4D_PRIVATE_PATHS_PATTERN"`

## 13. Tests and limitations

All eight correctness classes and the real A40 gate are detailed in TEST_REPORT.md. Remaining limitations:
- final candidate independent-native evaluation not run
- not all preregistered long-memory controls were rerun on Protocol-B
- single training seed

## 14. Claims supported / not supported

Supported claims are limited to the exact all-T, retention, mechanism, and resource statuses in section 1. Not supported: indefinite-horizon retention, replicated training stability, independent generalization, or attribution of policy effects to memory content alone.

## 15. GitHub and next exact action

Results commit E publication status: `PUSH_VERIFIED`. FINAL_MANIFEST.json SHA256: `1627e864f00b2be97a12d938034ca72540a7d1e682b1566f1e04179002f63460`. Next exact action: evaluate this frozen M3-V-CORE checkpoint on the registered independent-native population before making a generalization claim.
