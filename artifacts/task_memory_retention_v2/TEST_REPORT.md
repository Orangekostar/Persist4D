# TaskMemory Retention V2 Test Report

Results commit: `3027b942a5d76dea1dad3f42b3c67c0a9c9383ee`.

## Correctness classes

- `r1_load_and_shutdown_parity`: PASS; tests/test_persist4d_task_memory.py plus PASS A40 two-update smoke and real-model profile load audit
- `data_and_supervision_isolation`: PASS; tests/test_task_memory_episode.py in the 212-test direct suite
- `route_and_commit`: PASS; routing and lag-one output tests in the direct suite
- `tala_supervision`: PASS; tests/test_task_memory_criterion.py in the direct suite
- `visual_memory`: PASS; tests/test_object_visual_memory.py plus the formal VCORE profile path
- `training_path`: PASS; trainer and training-entrypoint tests plus recorded real two-update gradient smoke
- `output_and_metric`: PASS; task-memory output, evaluation, and final-analysis tests in the direct suite
- `publication`: PASS; profile and publication contract tests in the direct suite

## Direct verification

### `correctness_suite`

```bash
conda run -n persist4d python -m pytest -q tests/test_persist4d_task_memory.py tests/test_object_visual_memory.py tests/test_train_task_memory.py tests/test_task_memory_*.py tests/test_analyze_task_memory_final.py tests/test_publish_task_memory_v2.py tests/test_profile_task_memory.py
```

Observed: 212 passed in 19.87s; one third-party SciPy deprecation warning (exit code 0).

### `real_gpu_gate`

```bash
jq -e '.status == "PASS" and .device == "NVIDIA A40" and .warmups == 5 and .repeats == 10 and .measurement_rows == 1920 and .summary_rows == 192 and .permanent_state_bytes == 489816' artifacts/task_memory_retention_v2/resources/run_summary.json
```

Observed: formal A40 profile passed with 1920 measurements, 192 summaries, and 489816 permanent-state bytes (exit code 0).

### `ruff_changed_python`

```bash
ruff check scripts/run_task_memory_controls.py scripts/analyze_task_memory_final.py scripts/publish_task_memory_v2.py tests/test_task_memory_controls.py tests/test_analyze_task_memory_final.py tests/test_publish_task_memory_v2.py
```

Observed: six changed Python code and test files passed with no lint errors (exit code 0).

### `git_diff_check`

```bash
git diff --check
```

Observed: no tracked whitespace errors after results commit E (exit code 0).

### `manifest_and_hash`

```bash
conda run -n persist4d python -c 'import csv,json; from pathlib import Path; from scripts.task_memory_contracts import canonical_json_sha256; r=Path("artifacts/task_memory_retention_v2"); files=("final/status.json","resources/run_summary.json","evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json","evaluation/M5/protocol_b/baseline/control_observation_manifest.json"); assert all((lambda v:v["content_sha256"]==canonical_json_sha256({k:x for k,x in v.items() if k!="content_sha256"}))(json.loads((r/p).read_text())) for p in files); paths=("final/all_t_metrics.csv","evaluation/M5/protocol_b/baseline/long_memory_controls.csv","evaluation/M5/protocol_b/R1-B4-policy/policy_comparison.csv","final/independent_reference_results.csv"); counts=tuple(sum(1 for _ in csv.DictReader((r/p).open())) for p in paths); assert counts==(56,16,8,9); print("four canonical manifests and formal row coverage PASS", counts)'
```

Observed: four canonical manifests verified; formal main, D-control, policy, and native tables contain 56, 16, 8, and 9 rows (exit code 0).

### `secret_scan`

```bash
! git diff --name-only -z 6933f635f1834849de3e00aef60cd737f0aa1cba 3027b942a5d76dea1dad3f42b3c67c0a9c9383ee | xargs -0 rg -q --no-messages -e 'ghp_[A-Za-z0-9]{30,}' -e 'github_pat_[A-Za-z0-9_]{30,}' -e '-----BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY-----' -e 'Bearer [A-Za-z0-9_.-]{30,}'
```

Observed: no credential or private-key patterns in newly committed files (exit code 0).

Passing engineering checks do not change failed or inconclusive scientific
statuses.
