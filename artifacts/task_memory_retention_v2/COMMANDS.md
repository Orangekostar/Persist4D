# Commands

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

## Frozen M2 commands (not yet run)

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
`training/variants.json`.
