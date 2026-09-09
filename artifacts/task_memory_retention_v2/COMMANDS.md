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
