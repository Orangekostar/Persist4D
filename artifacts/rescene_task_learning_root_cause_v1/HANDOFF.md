# ReScene Task-Learning Root-Cause Handoff

## 1. Repository / branch / start SHA / end SHA / remote SHA

Repository: `Orangekostar/Persist4D`. Branch:
`research/persist4d-rescene-task-learning-root-cause-v1`. Start SHA:
`29bb228ad5b090797045fa3b3fc55cb973f001be`. Evidence commit before final
package publication: `ac20730e285970859b241e431203285e54fdacaf`. End/local and remote are the
same verified branch tip containing this file; resolve them with the commands
in section 30 because a Git commit cannot embed its own SHA without changing
that SHA.

## 2. Scientific question

Determine whether an evidenced ReScene training-semantic difference explains
part of the local Concerto reproduction gap, then test the lowest-risk native
local-perception strengthening switch without using Persist4D outcomes.

## 3. External evidence frozen

ReScene4D paper revision: arXiv `2601.11508v2`. Official ReScene source commit:
`fb2fe42eb8f1e926567c48eea9acb874e608ee10`. Concerto source revision:
`c31f993a56129f2ba9c5d06a35957e3f05bff710`. The official repository stated
that task checkpoints were not released when START_STATE was frozen.

## 4. Current Concerto/Sonata baseline metrics

Frozen Concerto three-seed means: t-mAP `0.2829008499781291`, overall mAP
`0.3697936435540517`, stage1 mAP `0.4203239579995473`, stage2 mAP
`0.43060773611068726`, SpatialStageMean `0.4254658470551173`. Frozen Sonata
means: t-mAP approximately `0.2404`, overall mAP approximately `0.3155`.

## 5. Exact files changed

See `FINAL_MANIFEST.json` for the complete artifact inventory and hashes. Code
changes cover loss-objective selection, root-cause preflight/launch/evaluation,
diagnostics, strong-local A1 support, recovery/finalization contracts, and their
tests. Frozen V1/V2/V3/Sonata evidence was not modified.

## 6. Root-cause hypotheses audited

The released-code raw-sum objective differs materially from the local weighted
objective. EOS 0.1 was not authorized. Label 255 was material but its R4 curve
failed with a deterministic nonfinite objective. Physical-global batch 8 was
diagnostically authorized but its R2 full-data launch OOMed. DDP sampler
corruption was rejected. Frozen-encoder stochasticity was confirmed but R3 was
not selected under the registered two-conditional-slot rule.

## 7. DDP sampler verdict

PASS. Lightning resolves `DistributedSamplerWrapper -> WeightedRandomSampler`.
Both ranks receive 1056 draws; reconstructed global draw positions have zero
mismatches. Cross-rank repeated values arise from the registered replacement
stream, not accidental rank duplication.

## 8. Label-255 verdict

Material. Label 255 accounts for 34,731 target instances and 62,688,083 points,
or 55.49% of target instances and 39.48% of supervised target points. R4 was
not a usable candidate because its controlled training objective became
deterministically nonfinite.

## 9. Encoder stochasticity verdict

Confirmed. Eight repeated passes triggered all six decoder-relevant levels;
mean cosine ranged from 0.869569 to 0.964102 and relative RMS deviation from
0.250786 to 0.478557. Disabling DropPath reduced but did not remove variation.

## 10. Physical-batch gradient verdict

Physical-global batch 8 was feasible diagnostically and triggered five
registered gradient groups relative to batch 4. Batch 16 OOMed; batch 32 was
not attempted. The later full-data R2 launch at batch 8 also OOMed, so no R2
learning curve was used.

## 11. Short-curve variants actually run

R0 was the current local weighted-objective control. R1 changed only the
optimized objective to the public released-code raw sum. Both ran from the same
initial state and frozen 29,700-step schedule to completed epoch 90. R2 and R4
were recorded as runtime-infeasible and did not contribute fabricated metrics.

## 12. Common initialization SHA

`d941b59ce95a8bb27bf5627f621cafe7c399a7a66b71ce05460782560fe98d4f`,
540,886,910 bytes, created at
`68373a3cc6ba2415359d69dc0f9158a2e287e2d2` from Concerto pretrained SHA256
`845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.

## 13. Scheduler/full-trajectory verification

All curves use the frozen 450-epoch, 29,700-optimizer-step OneCycle trajectory.
R1 full finalization contains exactly 30 completed-epoch validation boundaries.
Its allocator replay resumes completed epoch 390 and selects logger version 3
after that cutover; the superseded version-2 epoch-405 row remains preserved in
the signed lineage rather than silently merged.

## 14. Epoch-60/90 results

At epoch 60, R1 minus R0 paired SpatialStageMean was `0.05072592695554098`;
R1 means were SpatialStageMean `0.314528947075208`, overall mAP
`0.28228097160657245`, and t-mAP `0.20248334109783173`. At epoch 90, the paired
spatial gain was `0.018439258138338726`; R1 means were SpatialStageMean
`0.35234537720680237`, overall mAP `0.3126864234606425`, and t-mAP
`0.22579213480154672`. All paired seed deltas were positive.

## 15. Full-run authorization decision

R1 passed all five registered gates and was the only root-cause candidate
authorized to 450 epochs. A1 independently passed its epoch-90 strong-local
gate. Neither decision used Persist4D evidence.

## 16. Full candidate result if run

R1 completed epoch 450. Verdict: `ROOTCAUSE-CONFIRMED`. Selected epoch/step:
`390` / `25740`. Three-seed means: t-mAP `0.3106326957543691`, overall mAP
`0.3963398337364197`, SpatialStageMean `0.4399757186571757`; paired spatial
delta `0.01450987160205841`.

## 17. Query initialization diagnostics

Across 154 scenes, FPS query foreground fraction is `0.3492207777045377` and
background fraction `0.6507792205779583`. GT coverage is `0.7890519025650892`
overall but only `0.08108108108108109` for instances below 100 points. Query
content is initially zero for all sampled queries. This authorized A1.

## 18. Query conflict diagnostics

The mean competed-active-query fraction is `0.8235045625740549`; competing
query pairwise IoU is `0.6620520193117638`; mean queries per GT at IoU 0.25 is
`8.14773367783982`. A query-competition design is supported diagnostically but
was not authorized for implementation.

## 19. Mask-attention recall diagnostics

Earliest-layer allowed-GT fraction is `0.055661157151657564`; severe earliest
recall occurs for `0.9219185722253207` of measured cases, and post-sample reset
fraction is `0.11151515151515151`. Attention relaxation is supported only as a
future diagnostic-gated design and was not implemented.

## 20. Strong-local variants actually run

A1 (`use_np_features=true`) ran to epoch 450; its selected epoch-435 three-seed
verdict is `STRONG-LOCAL-PARTIAL`. A2 was gate-skipped after A1 passed, per the
registered STOP rule. A1+A2 and high-risk decoder modules were not run.

## 21. Best local checkpoint

Root-cause R1: completed epoch `390`, step `25740`, SHA256
`629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`, bytes
`754813672`, config SHA256
`3846415b6de3ea12b7b08d39ea20398916dd685c99ce7ad33e42df17d9ac6536`.
ReScene-Strong A1: completed epoch 435, step 28710, SHA256
`0bf2476a6fc53768f97aad7e14a4ac8779661f302725325993d5b37442a7864f`,
755,165,656 bytes, config SHA256
`4b193162acc5c3a54ea57c5b14b6c9909745e68d16fe7f92687462fadb283d9c`.

## 22. Exact claims supported

R1 improves the controlled short-curve local spatial metric and is the sole
authorized reproduction-compatible full candidate. At full budget, switching
to the released-code raw-sum objective improves every registered three-seed
mean over the frozen Concerto reimplementation, including t-mAP by
`0.02773184577624001` and SpatialStageMean by `0.01450987160205841`, with a
positive paired spatial delta for every seed. It explains part of the local
reproduction gap but does not establish an official reproduction. Feature-seeded
FPS queries improve mean local metrics at full budget but are seed-sensitive.

## 23. Exact claims NOT supported

The study does not prove an official ReScene4D reproduction, does not establish
the paper's physical batch implementation, does not show that EOS or label 255
causes the reported gap, and does not support choosing local models with
Persist4D metrics. A1 is ReScene-Strong, not an official reproduction.

## 24. Remaining risks

The official task checkpoint and complete training recipe remain unavailable.
R1 and A1 each experienced one deterministic large-sample allocator-fragmentation
OOM before exact-state recovery. A1 has one negative paired seed at full budget.
R1's final residual gap and seed variance are recorded in the full verdict.

## 25. All artifact hashes

`FINAL_MANIFEST.json` is the authoritative complete path-to-SHA256 inventory.
Key signed content hashes: short decision
`c71d7f770fe72dbeb3816fc1bb2e910dd73f46f5b6fbb18b522f12603f4b3def`,
decoder diagnostics
`b84c7cd95522b878cf966b07693bf08ddd2c0596d40d9fa9fa919824cef341de`,
A1 full verdict
`97d1afe8f664f9c312c6a61a77c7fe4c9eb63d85ea9576ccf26724769395c629`,
and R1 full verdict
`27cebebd7dfdd481a0c596aa717733a50e2258e3a8eb83e1ea84f8f8a989aba9`.

## 26. Test/lint results

Main suite: `195 passed`. Adjacent regression suite: `7 passed, 1 skipped`; the
skip is the canonical GPU parity gate requiring `P5_VERIFY_GPU_ARTIFACTS=1`.
Targeted Ruff: pass. `git diff --check`: pass. Final artifact verifier: pass in
the atomic finalizer and the independent post-publication rerun.

## 27. External files not in Git and how to locate them by hash

No `.ckpt`, `.pth`, raw dataset, token, or private path is committed. Exact
portable references, SHA256, bytes, creating commit, config SHA, and selected
epoch/step are in `FINAL_MANIFEST.json`. Locate each external file by its SHA256
under the experiment checkpoint store or shared archive.

## 28. Exact reproduction commands

```bash
python -m pytest -q \
  tests/test_rootcause*.py tests/test_rescene_rootcause*.py \
  tests/test_rescene_strong*.py tests/test_objective_semantics.py \
  tests/test_ddp_sampler_runtime_contract.py tests/test_filter255_inventory.py \
  tests/test_encoder_stochasticity_audit.py \
  tests/test_physical_batch_gradient_audit.py \
  tests/test_common_initialization_contract.py \
  tests/test_query_initialization_diagnostics.py \
  tests/test_query_conflict_diagnostics.py \
  tests/test_attention_mask_recall_diagnostics.py \
  tests/test_np_feature_query_variant.py tests/test_adaptive_scatter_variant.py
python -m pytest -q \
  tests/test_rescene_query_features.py tests/test_rescene_task_postprocess.py
python -m ruff check \
  scripts/finalize_rescene_rootcause_full_training.py \
  scripts/finalize_rescene_task_learning_root_cause.py \
  tests/test_rootcause_full_training_finalization.py \
  tests/test_rescene_rootcause_handoff.py
python scripts/verify_rescene_rootcause_handoff.py artifacts/rescene_task_learning_root_cause_v1
```

Evaluation commands and exact inputs are encoded in the selected-checkpoint,
training, evaluation-provenance, and authorization artifacts.

## 29. Recommended next stage

Use the stronger local checkpoints only as prospectively frozen downstream
validation assets. Do not retroactively cherry-pick existing Persist4D results.
Prioritize a preregistered local-checkpoint substitution evaluation before
considering query competition or attention-relaxation modules.

## 30. GitHub branch URL / PR URL

Branch: `https://github.com/Orangekostar/Persist4D/tree/research/persist4d-rescene-task-learning-root-cause-v1`.
PR: `https://github.com/Orangekostar/Persist4D/pull/new/research/persist4d-rescene-task-learning-root-cause-v1`.
Verify identical end/local and remote SHA values with:

```bash
git rev-parse refs/heads/research/persist4d-rescene-task-learning-root-cause-v1
git ls-remote origin refs/heads/research/persist4d-rescene-task-learning-root-cause-v1
```
