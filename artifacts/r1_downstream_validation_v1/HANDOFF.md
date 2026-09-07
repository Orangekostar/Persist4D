# R1 Downstream Validation Handoff

## Result

The experiment is `EXECUTION_COMPLETE` and the registered recovery decision is `RECOVERY_SUPPORTED`.
T4 B4-B2 gap-recovery recall: `+0.241379` with `6/6` positive clusters.
T5 B4-B2 gap-recovery recall: `+0.258920` with `6/6` positive clusters.

## Frozen Identities

- Source commit used for finalization: `b7eb9d0afcc64024e617845f5a1dc6ff37f293eb`
- R1 checkpoint SHA256: `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`
- Protocol-B SHA256: `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`
- Runtime: one A40, float32, batch size 1, seed 45

## Evidence Layout

- `metrics/`: task, direct-local, identity, event, sensitivity, and old-new tables
- `profile/`: all 360 measured samples and 36 six-unit summary cells
- `FINAL_REPORT.md`: reviewer-facing compact interpretation
- `FINAL_MANIFEST.json`: authoritative SHA256 inventory and claim boundary

## Verification

```bash
PERSIST4D_PYTHON=/home/ww/miniconda3/envs/persist4d/bin/python
$PERSIST4D_PYTHON -m pytest -q tests/test_r1_downstream_*.py
$PERSIST4D_PYTHON scripts/finalize_r1_downstream_validation.py
git diff --check
git status --short
```

Large tensor caches are not in Git. Their compact identity is `cache_manifest.json`; the local external reference is `/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache`.

## Claim Boundary

Publication status is `RECOVERY_CLAIM_AUTHORIZED`. Historical comparison is `FROZEN_SUMMARY_COMPARISON_ONLY`; do not make causal old-versus-R1 claims without reconstructing the missing C-old raw cache.
