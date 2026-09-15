# TaskMemory Retention V2 Test Report

Results commit: `fb4b6df8f46449981bcb344913de04208e312ba5`.

## Correctness classes

- `r1_load_and_shutdown_parity`: PASS; tests/test_persist4d_task_memory.py plus the formal real-model profile load audit
- `data_and_supervision_isolation`: PASS; tests/test_task_memory_episode.py in the 175-test direct suite
- `route_and_commit`: PASS; routing and lag-one output tests in the direct suite
- `tala_supervision`: PASS; tests/test_task_memory_criterion.py in the direct suite
- `visual_memory`: PASS; tests/test_object_visual_memory.py plus the formal VCORE profile path
- `training_path`: PASS; trainer and training-entrypoint tests plus recorded real two-update gradient smoke
- `output_and_metric`: PASS; task-memory output, evaluation, and final-analysis tests in the direct suite
- `publication`: PASS; profile and publication contract tests in the direct suite

## Direct verification

### `correctness_suite`

```bash
python -m pytest -q tests/test_persist4d_task_memory.py tests/test_task_memory_episode.py tests/test_task_memory_routing.py tests/test_task_memory_output.py tests/test_task_memory_criterion.py tests/test_object_visual_memory.py tests/test_task_memory_trainer.py tests/test_train_task_memory.py tests/test_task_memory_evaluation.py tests/test_analyze_task_memory_final.py tests/test_profile_task_memory.py tests/test_publish_task_memory_v2.py
```

Observed: 175 passed in 18.38s; one third-party SciPy deprecation warning (exit code 0).

### `real_gpu_gate`

```bash
jq -e '.status == "PASS" and .device == "NVIDIA A40" and .warmups == 5 and .repeats == 10 and .measurement_rows == 1920 and .summary_rows == 192 and .permanent_state_bytes == 489816' artifacts/task_memory_retention_v2/resources/run_summary.json
```

Observed: formal A40 profile passed with 1920 measurements, 192 summaries, and 489816 permanent-state bytes (exit code 0).

### `ruff_changed_python`

```bash
git diff --name-only 32a51e -- '*.py' | rg -v '^(datasets/pointcept_utils.py|datasets/semseg.py)$' | xargs -r ruff check && for file in datasets/pointcept_utils.py datasets/semseg.py; do diff -u <(git show "32a51e:$file" | ruff check --stdin-filename "$file" --output-format=json - | jq -S 'group_by(.code) | map({code: .[0].code, count: length})') <(ruff check "$file" --output-format=json | jq -S 'group_by(.code) | map({code: .[0].code, count: length})'); done
```

Observed: all branch Python outside two inherited-debt files passed; the two excluded files retained exactly the parent's 10 and 36 diagnostics with no added rule counts (exit code 0).

### `git_diff_check`

```bash
git diff --cached --check
```

Observed: staged results diff had no whitespace errors before results commit E (exit code 0).

### `manifest_and_hash`

```bash
python -c 'import csv,hashlib,json; from pathlib import Path; from scripts.task_memory_contracts import canonical_json_sha256; root=Path("artifacts/task_memory_retention_v2"); a=json.loads((root/"final/status.json").read_text()); p=json.loads((root/"resources/run_summary.json").read_text()); assert a["content_sha256"]==canonical_json_sha256({k:v for k,v in a.items() if k!="content_sha256"}); assert p["content_sha256"]==canonical_json_sha256({k:v for k,v in p.items() if k!="content_sha256"}); assert all(hashlib.sha256((base/item["path"]).read_bytes()).hexdigest()==item["sha256"] for owner,base in ((a,root/"final"),(p,root/"resources")) for item in owner["outputs"]); counts=tuple(sum(1 for _ in csv.DictReader((root/path).open())) for path in ("final/all_t_metrics.csv","final/paired_deltas.csv","final/per_reference_metrics.csv","resources/per_update_measurements.csv","resources/profile_summary.csv","resources/profile_units.csv")); assert counts==(20,80,48,1920,192,6); print("content hashes and row counts passed", counts)'
```

Observed: analysis/profile canonical hashes and all declared output hashes passed; row counts were 20, 80, 48, 1920, 192, and 6 (exit code 0).

### `secret_scan`

```bash
set -o pipefail; ! git diff --text 32a51e -- | rg -n -i -- '-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}'; ! git diff --cached --text -- artifacts/task_memory_retention_v2 | rg -n -- "$PERSIST4D_PRIVATE_PATHS_PATTERN"
```

Observed: credential and public-local-path scan passed (exit code 0).

Passing engineering checks do not change failed or inconclusive scientific
statuses.
