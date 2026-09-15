# Commands

Run from the project root in the `persist4d` Conda environment. External paths
are resolved locally by the untracked
`artifacts/task_memory_retention_v2/external_assets.local.json`; replace the
example environment variables below with the same assets for reproduction.

```bash
export PERSIST4D_DATA_ROOT=/actual/data/root
export PERSIST4D_RIO_METADATA=/actual/3RScan.json
export PERSIST4D_R1_CHECKPOINT=/actual/R1.ckpt
export PERSIST4D_CONCERTO_PRETRAINED=/actual/concerto_base.pth
export PERSIST4D_RUN_ROOT=/actual/new-run-root

python -m scripts.prepare_task_memory_v2 \
  --config conf/task_memory_v2/experiment.yaml \
  --data-root "$PERSIST4D_DATA_ROOT" \
  --rio-metadata "$PERSIST4D_RIO_METADATA" \
  --r1-checkpoint "$PERSIST4D_R1_CHECKPOINT" \
  --concerto-pretrained "$PERSIST4D_CONCERTO_PRETRAINED" \
  --run-root "$PERSIST4D_RUN_ROOT" \
  --output artifacts/task_memory_retention_v2

python -m scripts.preflight_task_memory_episode \
  --data-root "$PERSIST4D_DATA_ROOT" \
  --rio-metadata "$PERSIST4D_RIO_METADATA" \
  --data-contract artifacts/task_memory_retention_v2/DATA_CONTRACT.json \
  --output artifacts/task_memory_retention_v2/implementation/preflight_real_sequence.json
```

## Verified Task 7 commands

The real two-update gradient smoke was executed on one A40:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_task_memory \
  --variant Q-TALA --smoke --devices 1 \
  --gradient-accumulation 1 --stop-after-updates 2

python -m scripts.train_task_memory --materialize-contracts
```

## Verified Task 8 commands

The compact evaluator was tested on two development H5 masters for Q-TALA and
FH-MATCH. Dense cache payloads stay under the external run root.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_task_memory \
  --variant Q-TALA --checkpoint "$PERSIST4D_Q_TALA_CHECKPOINT" \
  --device cuda:0 --smoke-masters 2 \
  --cache-root "$PERSIST4D_RUN_ROOT/evaluation_cache/smoke/Q-TALA" \
  --output artifacts/task_memory_retention_v2/implementation/evaluation_smoke/Q-TALA

CUDA_VISIBLE_DEVICES=1 python -m scripts.evaluate_task_memory \
  --variant FH-MATCH --checkpoint "$PERSIST4D_R1_CHECKPOINT" \
  --device cuda:0 --smoke-masters 2 \
  --cache-root "$PERSIST4D_RUN_ROOT/evaluation_cache/smoke/FH-MATCH" \
  --output artifacts/task_memory_retention_v2/implementation/evaluation_smoke/FH-MATCH

python -m pytest -q \
  tests/test_task_memory_evaluation.py tests/test_task_memory_selection.py \
  tests/test_rescene_task_postprocess.py tests/test_system_comparison_metrics.py \
  tests/test_task_memory_output.py
```

## Completed M2 commands

Run each 300-update exact-schedule prefix in an isolated directory:

```bash
for variant in W-BASE Q-INDEP Q-TALA FH-MATCH; do
  python -m scripts.train_task_memory \
    --variant "$variant" --external-root "$PERSIST4D_RUN_ROOT" \
    --devices 2 --gradient-accumulation 4 --stop-after-updates 300
done
```

Resume each technically valid prefix to the frozen 3,000-update endpoint:

```bash
for variant in W-BASE Q-INDEP Q-TALA FH-MATCH; do
  python -m scripts.train_task_memory \
    --variant "$variant" --external-root "$PERSIST4D_RUN_ROOT" \
    --devices 2 --gradient-accumulation 4 --stop-after-updates 3000 \
    --resume "$PERSIST4D_RUN_ROOT/training/formal/$variant/last.ckpt"
done
```

Exact per-variant commands and resolved-config hashes are frozen in
`training/variants.json`. All four arms reached update 3000; the development
selection fixed W-BASE at 3000, Q-INDEP and Q-TALA at 1500, and FH-MATCH at
0750. M3 continues from Q-TALA/1500; FH-CONT continues from FH-MATCH/0750.
The exact checkpoint identities and retained optimizer state are recorded in
`training/selected_checkpoints.json`.

## Verified Task 12 commands

Run the three full Protocol-B evaluations. They may run concurrently on three
A40 devices, or sequentially with each command mapped to `cuda:0`:

```bash
for variant in M3-V-CORE M3-BASE-CONT FH-CONT; do
  CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_task_memory \
    --variant "$variant" \
    --checkpoint "$PERSIST4D_RUN_ROOT/training/formal/$variant/update=1500.ckpt" \
    --population protocol_b_43_masters_3_orders --reducers mean \
    --device cuda:0 \
    --cache-root "$PERSIST4D_RUN_ROOT/evaluation_cache/M5/protocol_b/$variant" \
    --output "artifacts/task_memory_retention_v2/evaluation/M5/protocol_b/$variant"
done
```

The same Protocol-B population was also evaluated for W-BASE, Q-TALA,
FH-MATCH, and the frozen R1 weights through the FH-MATCH model path with
`lag1/mean`. The original FH-R1-native reference retains its separate official
output policy; do not relabel it as the same-policy FH-R1-lag1 measurement.

## Formal prediction-only controls

The formal R1+B4 cache manifest is in
`evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json` and binds all 129
orders to 115 unique sequence cache files. Shard production by unique sequence
so that separate GPUs cannot write the same observation file:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.run_task_memory_controls \
  --mode produce --population protocol_b_43_masters_3_orders \
  --base-manifest artifacts/task_memory_retention_v2/evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json \
  --output-root artifacts/task_memory_retention_v2/evaluation/M5/protocol_b \
  --shard-index 0 --shard-count 2 --device cuda:0

CUDA_VISIBLE_DEVICES=2 python -m scripts.run_task_memory_controls \
  --mode produce --population protocol_b_43_masters_3_orders \
  --base-manifest artifacts/task_memory_retention_v2/evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json \
  --output-root artifacts/task_memory_retention_v2/evaluation/M5/protocol_b \
  --shard-index 1 --shard-count 2 --device cuda:0

python -m scripts.run_task_memory_controls \
  --mode analyze --population protocol_b_43_masters_3_orders \
  --base-manifest artifacts/task_memory_retention_v2/evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json \
  --output-root artifacts/task_memory_retention_v2/evaluation/M5/protocol_b
```

The development CSV files under `baseline/` are not the formal Protocol-B
policy or control inputs. The final analyzer defaults to the files under
`evaluation/M5/protocol_b/`.

Compute the two six-reference tables independently, then merge them with the
aggregate results:

```bash
mkdir -p "$PERSIST4D_RUN_ROOT/analysis_intermediate"

python -m scripts.analyze_task_memory_final \
  --data-root "$PERSIST4D_DATA_ROOT" --external-root "$PERSIST4D_RUN_ROOT" \
  --candidate-cache "$PERSIST4D_RUN_ROOT/evaluation_cache/M5/protocol_b/M3-V-CORE" \
  --reference-only M3-V-CORE \
  --reference-output "$PERSIST4D_RUN_ROOT/analysis_intermediate/M3-V-CORE.csv"

python -m scripts.analyze_task_memory_final \
  --data-root "$PERSIST4D_DATA_ROOT" --external-root "$PERSIST4D_RUN_ROOT" \
  --fh-cache "$PERSIST4D_RUN_ROOT/evaluation_cache/M5/protocol_b/FH-CONT" \
  --reference-only FH-CONT \
  --reference-output "$PERSIST4D_RUN_ROOT/analysis_intermediate/FH-CONT.csv"
```

Profile the frozen candidate and matched full-history checkpoint sequentially
on one A40:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.profile_task_memory \
  --comparison-contract artifacts/task_memory_retention_v2/PROFILE_CONTRACT.json \
  --protocol-manifest artifacts/P6A/protocol_b_manifest.json \
  --data-contract artifacts/task_memory_retention_v2/DATA_CONTRACT.json \
  --variant-manifest artifacts/task_memory_retention_v2/training/variants.json \
  --data-root "$PERSIST4D_DATA_ROOT" \
  --rio-metadata "$PERSIST4D_RIO_METADATA" \
  --pretrained "$PERSIST4D_CONCERTO_PRETRAINED" \
  --external-root "$PERSIST4D_RUN_ROOT" \
  --candidate-checkpoint "$PERSIST4D_RUN_ROOT/training/formal/M3-V-CORE/update=1500.ckpt" \
  --fh-checkpoint "$PERSIST4D_RUN_ROOT/training/formal/FH-CONT/update=1500.ckpt" \
  --device cuda:0 --warmup 5 --repeats 10 \
  --output artifacts/task_memory_retention_v2/resources
```

Generate the final tables after the resource profile exists:

```bash
python -m scripts.analyze_task_memory_final \
  --candidate-cache "$PERSIST4D_RUN_ROOT/evaluation_cache/M5/protocol_b/M3-V-CORE" \
  --base-cache "$PERSIST4D_RUN_ROOT/evaluation_cache/M5/protocol_b/M3-BASE-CONT" \
  --fh-cache "$PERSIST4D_RUN_ROOT/evaluation_cache/M5/protocol_b/FH-CONT" \
  --candidate-reference "$PERSIST4D_RUN_ROOT/analysis_intermediate/M3-V-CORE.csv" \
  --fh-reference "$PERSIST4D_RUN_ROOT/analysis_intermediate/FH-CONT.csv"
```
