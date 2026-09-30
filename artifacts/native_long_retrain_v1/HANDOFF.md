# Native Long Retrain Handoff

Experiment A: `9ea09b3484f6c030b103c70b524ff90f0765bbd6`. Branch: `research/rescene-native-long-retrain-v1`. Delivery tag: `rescene-native-long-retrain-v1`. Base: `c6e9d01edfe2b3c832374a424fc45e44bdadfb68`.

Scientific status is PARTIAL_FULL_RETRAIN_BUDGET; publication is CODE_ONLY. All four arms have zero actual training updates. No main AP, CAL/SEL lock, seed46, formal confirmation, trained profile or gain exists. The measured full E0 forecast already exceeds the remaining lifetime budget. Do not shrink U or convert temporary preflight optimizer updates into a training result.

Runtime: `/home/ww/persist4d_runs/native_long_retrain_v1`. Read FINAL_REPORT.md, REQUIREMENT_REVIEW.md, CODE_BINDINGS.md and RUNTIME_NOTE.md. The encoder is the separately obtained fixed Concerto SHA in SOURCE_AND_INITIALIZATION.json. Initialization contains no old task warm start. Current encoder has zero registered buffers; the loader explicitly preserves every nonencoder tensor and any registered encoder buffers.

Git includes untrained F initialization parameters and official initialized TRAIN-A sufficient statistics. Nonencoder task initialization bundle remains local: `/home/ww/persist4d_runs/native_long_retrain_v1/initialization/task_reload_initialization.pt`, 106912359 bytes, SHA `3790a88856ccef2b08a3619911ae56870043d8fd65cf958fd58314864943edcb`. It is not a trained deployment model. No Release authorization was available; no trained weights were generated. Raw data, raw GT and original pretrained files are not redistributed.

Commands with the existing persist4d environment, from this worktree:

```bash
export RESCENE_NATIVE_LONG_ROOT='/home/ww/persist4d_runs/native_long_retrain_v1'
python -m scripts.native_long_campaign status --root "$RESCENE_NATIVE_LONG_ROOT"
python -m scripts.native_long_campaign run --root "$RESCENE_NATIVE_LONG_ROOT" --resume
python -m scripts.native_long_campaign report --root "$RESCENE_NATIVE_LONG_ROOT"
python -m scripts.native_long_campaign publish --root "$RESCENE_NATIVE_LONG_ROOT" --resume
```

The run command revalidates bindings and retains the empty budget-authorized trajectory set. A new training budget needs an explicit new plan; current authorization cannot sustain full retraining. Seed45 training/CAL/SEL code has no production-run validation; positive-plan replication/formal/profile and trained-result reporting remain outstanding. These are not certified as complete.

Once Release authorization is explicitly available, the existing GitHubReleaseClient can upload the single task initialization package with its manifest bytes/SHA; an empty Release or an LFS pointer is not delivery. Current missing-asset details are in the external receipt.

```bash
python -m scripts.native_long_campaign publish --root "$RESCENE_NATIVE_LONG_ROOT" --resume --upload-assets
```

B is recorded only in the external publication/PUBLICATION_RECEIPT.json after actual push. ARTIFACT_MANIFEST.json excludes itself; this handoff does not contain its own commit SHA.
