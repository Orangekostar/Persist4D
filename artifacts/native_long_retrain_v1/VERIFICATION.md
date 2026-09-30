# Development Verification

Primary-agent verification on the final numerical source, 2026-09-30. These checks validate the budget-limited development path; they do not certify trained accuracy or completion of the four-arm study.

## CPU Regression

Executed with CUDA_VISIBLE_DEVICES='' and OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=2:

```bash
/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_native_long_budget.py tests/test_native_long_collation.py \
  tests/test_native_long_data.py tests/test_native_long_initialization.py \
  tests/test_native_long_modules.py tests/test_native_long_resume.py \
  tests/test_native_long_sampling.py tests/test_native_long_selection.py \
  tests/test_objective_semantics.py tests/test_perception_gain_model.py
```

Result: 27 passed, 5 warnings, 13.56s after the budget-resume repair. The actual Lightning CPU split-resume fixture checks optimizer-boundary draw position, all parameters, learning rate and scheduler against the continuous trajectory. This fixture does not exercise full native two-GPU DDP. Existing dependency deprecation and Lightning dataloader/resume warnings remain; the explicit equality assertions passed.

## Source Checks

Ruff passed for all six scripts/native_long_*.py files, datasets/native_long_dataset.py, models/native_long_modules.py, trainer/native_long_trainer.py and all eight tests/test_native_long_*.py files. Compileall passed for those implementation files plus models/pointcept.py, models/rescene.py and datasets/pointcept_utils.py. git diff --check exited zero.

## Actual CLI And Native Evidence

The run command with --resume --through report exited zero after binding the exact base, encoder bytes, installed library sources, raw data content, fresh seed45 initialization, measured preflight and immutable empty budget plan. A subsequent run reused the verified numerical preflight identity and refreshed the CPU task-package code binding. Separate status and report commands were also exercised.

The first delivered B was 6e393786b974f79ca5416f7a7712698e6d9efc99 under rescene-native-long-retrain-v1. A post-publication run --resume exposed integer horizon keys versus persisted JSON string keys in the budget comparison, which incorrectly revised the correction lineage and was rejected by the evidence seal. A byte-preservation regression failed before the fix and passed after canonicalizing the forecast representation. The original correction record was preserved from the committed evidence. Two subsequent real CLI resumes left the lock SHA unchanged; see resources/BUDGET_RESUME_AUDIT.json. The numerical preflight was reused, only the CPU package binding was refreshed, and GPUh did not increase. The first tag remains immutable; the repaired delivery uses a new tag and external receipt.

All recorded input, mapping, embedding, encoder, decoder, maskF, query and final tensors are byte-identical between the two fresh-process repeats of each A, BA and ABA protocol; see runtime/REPEATABILITY_ALL_STAGES.json. Cross-protocol history sensitivity remains RUNTIME_CONDITIONAL. Real F identity/late-update/two-step gradient, S cap quotas, raw_sum gradient equivalence, mixed FULL/LOW collation, largest T5 backward and task reconstruction evidence are stored separately.

The budget ledger includes failed probes and the conservatively charged early CPU-fixture three-device CUDA capture. The final CPU regression hides CUDA and the native resume hook now captures only the rank-local GPU.

## Completion Boundary

All four official training trajectories remain at zero updates. Main CAL/SEL expected denominators are frozen and completed denominators remain zero; scores are null. Full native training/DDP, seed46 execution, formal confirmation, trained diagnostics/profile and positive-plan reporting/integration remain outstanding. Initialization state and temporary preflight updates are not trained deployment models or accuracy results.

PRIMARY_REVIEW.json seals the current code and reviewed files. The actual A/B commits, remote branch/tag and Git asset bytes are certified only by publication/PUBLICATION_RECEIPT.json outside Git after push. The task initialization package is local-only because the single startup Release authorization check was unavailable.
