# Commands

```bash
export PERSIST4D_RUN_ROOT=/mnt/shared/ww/persist4d-crosswindow-evidence-v1
conda run -n persist4d python -m scripts.crosswindow_campaign preflight --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume --read-only
conda run -n persist4d python -m scripts.crosswindow_campaign run --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --through E4 --resume
conda run -n persist4d python -m scripts.crosswindow_campaign conditional-train --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume
conda run -n persist4d python -m scripts.crosswindow_campaign lock --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume
conda run -n persist4d python -m scripts.crosswindow_campaign confirm --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT" --resume
conda run -n persist4d python -m scripts.crosswindow_campaign report --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT"
conda run -n persist4d python -m scripts.crosswindow_campaign publish --config configs/crosswindow_evidence_v1.yaml --external-root "$PERSIST4D_RUN_ROOT"
```
