# Persist4D All-T Handoff

## 1. Scientific Goal

The goal was one frozen Persist4D checkpoint that strictly exceeds matched ReScene at t-mAP T2-T5. Execution is complete; the scientific goal is **not achieved**.

| Status | Value |
| --- | --- |
| execution_status | COMPLETE |
| baseline_comparability | MATCHED |
| tmap_all_t_vs_R1 | FAIL |
| tmap_all_t_vs_matched_rescene | FAIL |
| task_metrics_all_t | FAIL |
| seed_confirmation | NOT_RUN |
| independent_generalization | NOT_ESTABLISHED |
| publication_status | NOT_ATTEMPTED |

## 2. Repository And Commits

- Repository: `Orangekostar/Persist4D`
- Branch: `research/persist4d-allt-task-superiority-v1`
- Start SHA: `2c7494b982eff84886aef3a7274bae43752a1485`
- Publication-code SHA: `b35701fb276ef47e2d0111e49385a584d657ce4d`
- Formal evaluation SHA: `d288af93cefc7cf8aaabb9b541b92539782c019c`
- Final cache-analysis SHA: `a4c8744323f2c039c3a89dd0dcc7040b6e7a7215`
- Resource-profile SHA: `4931840a874c895f5798582b8fb4e8c7f3b861ac`
- Result/publication commit: resolve from remote branch HEAD after push.

## 3. Frozen Inputs And Checkpoints

| Object | Logical reference | SHA256/content SHA256 | Bytes |
| --- | --- | --- | --- |
| R1 checkpoint | external:r1_checkpoint | 629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd | 754813672 |
| Concerto pretrained | external:concerto_pretrained | 845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07 | 433987358 |
| C0 selected checkpoint | external:allt_task_superiority_v1/training/formal/C0/update=0400.ckpt | 574ea496db467f1208eea934d2aa29595b36057c7bad27dcd5cadc88793cdb25 | 754819304 |
| C1 selected checkpoint | external:allt_task_superiority_v1/training/formal/C1/update=0100.ckpt | 44afc0faa72d9e4215b8c27b1d6f0b15a94dd5e9d63da34add9a4868e7f6da0d | 755623752 |
| C2 selected checkpoint | external:allt_task_superiority_v1/training/formal/C2/update=0200.ckpt | a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724 | 756029616 |
| FH-adapt selected checkpoint | external:allt_task_superiority_v1/training/formal/FH-adapt/update=0100.ckpt | ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c | 754819240 |
| C2 Protocol-B cache | external:allt_task_superiority_v1/evaluation_cache/protocol_b_43_masters_3_orders/C2/a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724 | a620e2e17d3f9b3f874254f032f6919c88daabc4fc792259f64b0c2f10e5be8c | 119209970094 |
| FH-adapt Protocol-B cache | external:allt_task_superiority_v1/evaluation_cache/protocol_b_43_masters_3_orders/FH-adapt/ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c | bafb8a16d6b53d3dce8ce5afe73fe1f0ea83eb57476c2e681513d5d9a213c90f | 14772582818 |

The sole primary candidate is C2 update 200, SHA256 `a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724`. The matched comparator is FH-adapt update 100, SHA256 `ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c`.

## 4. Data Roles

RIO adaptation uses 36 references/215 masters; development uses 8 references/47 masters and is R1-base exposed. Protocol-B uses 6 references/43 masters x 3 orders = 129 correlated order units and is historical-benchmark exposed, final-only. No processed independent test split was available, so independent generalization is `NOT_ESTABLISHED`.

## 5. Implementation Surface

- `models/persist4d_allt.py`: `Persist4DAllT.after_decoder_stage/forward` implement optional L/M and prediction-only persistent reads.
- `datasets/persist4d_sequence_dataset.py`: `build_episode_draw_plan/build_episode_masters` provide deterministic continuous episodes.
- `trainer/persist4d_allt_trainer.py`: `Persist4DAllTTrainer.training_step/on_before_optimizer_step` enforce all-T training, immediate commits, and gradient audits.
- `scripts/train_persist4d_allt.py`: `main` binds frozen two-GPU budgets and external checkpoint manifests.
- `scripts/evaluate_persist4d_allt.py`: `apply_memory_read_policy/run_evaluation` implement continuous T1-T5 metrics, cache provenance, and read diagnostics.
- `scripts/analyze_persist4d_allt_final.py`: `analyze_final_caches` streams per-reference, identity, and equal-cluster analysis.
- `scripts/profile_persist4d_allt.py`: `run_profile` executes the bounded real-A40 profile.
- `scripts/publish_persist4d_allt.py`: `validate_completion_inputs/publish_final_package` fail closed and build reports/manifests.

Launch commands are in section 16.

## 6. Variant States

| Variant | State | Selected update | Trainable params | GPU-h | Exposure | Checkpoint/reason |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | COMPLETE | 400 | 26688019 | 2.128 | 3200 episodes / 7642 stages | 574ea496db467f1208eea934d2aa29595b36057c7bad27dcd5cadc88793cdb25 |
| C1 | COMPLETE | 100 | 26754195 | 2.255 | 3200 episodes / 7642 stages | 44afc0faa72d9e4215b8c27b1d6f0b15a94dd5e9d63da34add9a4868e7f6da0d |
| C2 | COMPLETE | 200 | 26787475 | 2.232 | 3200 episodes / 7642 stages | a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724 |
| C3 | GATE_SKIPPED | N/A | N/A | N/A | N/A | C1 did not strictly improve both T2 and T3 versus selected C0, so the preregistered complementarity gate failed closed. |
| FH-adapt | COMPLETE | 100 | 26688019 | 2.551 | 3200 episodes / 7642 stages | ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c |
| FH-L | GATE_SKIPPED | N/A | N/A | N/A | N/A | The frozen Persist4D candidate does not use L, so matched FH-L training is not authorized. |

C0/C1/C2/FH-adapt completed. C3 was gate-skipped because C1 did not improve both T2/T3 over C0. FH-L was gate-skipped because the frozen candidate did not use L.

## 7. Budget And Exposure

All executed variants used two A40s, 400 optimizer updates, physical batch 1/GPU, accumulation 4, effective batch 8, and `32-true` precision. Every plan contains 3,200 global episodes, 1,600 groups, 1,778 RIO episodes, 1,422 ScanNet episodes, and 7,642 supervised stages. C0/C1/C2 each scanned 12,084 encoder inputs; FH-adapt scanned 16,524. Exact trainable parameters and GPU-hours are in section 6.

## 8. Single-Checkpoint Main Results

| Horizon | C2 mean | R1 B4 mean | Delta vs R1 B4 | FH-adapt | Delta vs FH-adapt | FH-R1 |
| --- | --- | --- | --- | --- | --- | --- |
| T2 | 0.225025 | 0.219637 | 0.005388 | 0.223600 | 0.001425 | 0.228777 |
| T3 | 0.141059 | 0.135498 | 0.005560 | 0.147723 | -0.006664 | 0.148305 |
| T4 | 0.076000 | 0.080159 | -0.004159 | 0.103657 | -0.027657 | 0.102745 |
| T5 | 0.051323 | 0.058535 | -0.007213 | 0.080308 | -0.028985 | 0.081566 |

All 20 C2-vs-FH-adapt task cells:

| Metric | Horizon | Raw delta | Strict > 0 |
| --- | --- | --- | --- |
| t_mAP | T2 | 0.001425 | PASS |
| t_mAP | T3 | -0.006664 | FAIL |
| t_mAP | T4 | -0.027657 | FAIL |
| t_mAP | T5 | -0.028985 | FAIL |
| t_mAP50 | T2 | -0.004089 | FAIL |
| t_mAP50 | T3 | 0.000601 | PASS |
| t_mAP50 | T4 | -0.052444 | FAIL |
| t_mAP50 | T5 | -0.052216 | FAIL |
| t_mAP25 | T2 | -0.026373 | FAIL |
| t_mAP25 | T3 | -0.051469 | FAIL |
| t_mAP25 | T4 | -0.079664 | FAIL |
| t_mAP25 | T5 | -0.071030 | FAIL |
| t_REC | T2 | -0.003363 | FAIL |
| t_REC | T3 | 0.011039 | PASS |
| t_REC | T4 | -0.007575 | FAIL |
| t_REC | T5 | 0.000584 | PASS |
| prefix_overall_mAP | T2 | 0.005919 | PASS |
| prefix_overall_mAP | T3 | -0.013475 | FAIL |
| prefix_overall_mAP | T4 | -0.037826 | FAIL |
| prefix_overall_mAP | T5 | -0.049288 | FAIL |

## 9. Baseline Verdicts

C2 vs R1 B4 t-mAP all-T: `FAIL` (T4/T5). C2 vs matched FH-adapt t-mAP all-T: `FAIL` (T3/T4/T5). C2 vs matched FH-adapt all 20 task cells: `FAIL` (only 5/20 positive).

## 10. L/M Evidence

| Variant | Update | T2 | T3 | T4 | T5 | Minimum |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 400 | 0.507136 | 0.389127 | 0.365171 | 0.334946 | 0.334946 |
| C1 | 100 | 0.506292 | 0.396667 | 0.356264 | 0.330555 | 0.330555 |
| C2 | 200 | 0.512769 | 0.395971 | 0.376152 | 0.348153 | 0.348153 |
| FH-adapt | 100 | 0.508799 | 0.452106 | 0.400897 | 0.384912 | 0.384912 |

| Effect | T2 | T3 | T4 | T5 |
| --- | --- | --- | --- | --- |
| L: C1-C0 | -0.000844 | 0.007541 | -0.008906 | -0.004391 |
| M: C2-C0 | 0.005633 | 0.006844 | 0.010981 | 0.013207 |

L did not pass its short-horizon gate. M improved selected C0 across development T2-T5 but exceeded matched FH-adapt only at T2. C3 complementarity therefore was not tested. C2's W=2 current input reads prediction-only persistent K=100 slots that can originate before the current window; the deployment state does not grow with T.

D0 contains 827 official-valid trajectory rows (220 unique GT trajectories). Its non-additive labels include mask-insufficient=319, full-history-success/B4-failure=112, class-change=25, fragmentation=21, merge=28, and no-candidate=10. T1 cold-start contributed to 133/350 failed rows (0.380).

## 11. Evaluation Semantics

Primary C2 score reducer: `mean`; `latest/max` are sensitivity only. `CandidateTrajectoryKey=(track_id,class_id)` is unchanged. Each episode executes T1 then T2/T3/T4/T5 continuously, with prediction-driven B4 state committed immediately after each stage; no offline smoothing or GT enters state/prediction.

## 12. Seeds And Statistical Units

Training/adaptation seed 45 is primary. Seed 46 was not run because the first-seed development candidate failed the matched all-T gate. Evaluation seed 45 is used for the frozen final run; planned 46/47 repeats are not independent training seeds. The primary population is pooled official evaluation over 129 order units clustered within six references; the 1,000-resample equal-cluster intervals are descriptive only.

## 13. Resources And Limits

| Model | Horizon | Median of cell medians (ms) | Cell range (ms) | Max incremental MiB | Median voxel points |
| --- | --- | --- | --- | --- | --- |
| C2 | T2 | 598.834 | 448.086-731.518 | 1831.2 | 107015 |
| C2 | T3 | 533.940 | 398.578-736.524 | 1815.9 | 88838 |
| C2 | T4 | 490.112 | 370.437-576.188 | 1336.2 | 77082 |
| C2 | T5 | 435.099 | 342.438-688.280 | 1703.4 | 65276 |
| FH-adapt | T2 | 598.495 | 444.197-723.567 | 1829.7 | 107015 |
| FH-adapt | T3 | 781.470 | 546.748-1007.175 | 2693.8 | 150270 |
| FH-adapt | T4 | 923.107 | 624.384-1112.516 | 3028.7 | 185549 |
| FH-adapt | T5 | 1055.358 | 775.224-1453.270 | 4136.0 | 220421 |

Scope: 5 warmups + 10 repeats for C2 and FH-adapt on one canonical sequence from each of six references at T2-T5, all sequentially on one NVIDIA A40. Includes forward, C2 memory read, observation extraction, and B4 update; excludes I/O, collation, H2D, metrics, and C2 preroll. It proves only this bounded deployment operation, not end-to-end throughput or multi-GPU scaling, and cannot offset the failed accuracy goal.

C2 also has fewer ID switches/fragments and materially higher gap-recovery rates than FH-adapt in `results/identity_counts.csv`; this identity advantage did not translate into all-T task accuracy.

## 14. Tests And Real Failures

See `TEST_REPORT.md`. Direct model/evaluator/analyzer/profiler/publisher/selection/finalizer tests and Ruff are required. Retained failures include the scientific gates, the corrected NFS diagnostic-write permission failure, and the profile-smoke T2 input mismatch that was fixed before the formal profile.

## 15. External Storage Reconstruction

Logical `external:allt_task_superiority_v1/...` resolves under `$ALLT_EXTERNAL_ROOT`. Current binding is `/mnt/shared/ww/persist4d-allt-task-superiority-v1`; `/mnt/shared` is the node-mounted NFS export `192.168.100.102:/mnt/data/shared`. Checkpoint and cache hashes/bytes are in section 3 and their tracked manifests. No credentials or large binaries are stored in Git.

## 16. Minimal Commands

```bash
export ALLT_EXTERNAL_ROOT=/mnt/shared/ww/persist4d-allt-task-superiority-v1
export R1_CHECKPOINT=/mnt/shared/ww/persist4d-rescene-finalization/checkpoints/629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd.ckpt
export CONCERTO_PRETRAINED=/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth
export RIO_METADATA=/home/ww/3RScan.json
export PERSIST4D_DATA_ROOT="$PWD/data"

python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_persist4d_allt.py --variant C2 \
  --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" \
  --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" \
  --external-root "$ALLT_EXTERNAL_ROOT/training" \
  --artifact-root artifacts/allt_task_superiority_v1/training

python scripts/evaluate_persist4d_allt.py --variant C2 \
  --checkpoint "$ALLT_EXTERNAL_ROOT/training/formal/C2/update=0200.ckpt" \
  --population protocol_b --device cuda:0 --reducers mean latest max \
  --cache-root "$ALLT_EXTERNAL_ROOT/evaluation_cache" \
  --output-root artifacts/allt_task_superiority_v1/evaluation/protocol_b/C2/update=0200

python scripts/analyze_persist4d_allt_final.py \
  --c2-cache-directory "$ALLT_EXTERNAL_ROOT/evaluation_cache/protocol_b_43_masters_3_orders/C2/a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724" \
  --fh-cache-directory "$ALLT_EXTERNAL_ROOT/evaluation_cache/protocol_b_43_masters_3_orders/FH-adapt/ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c"
python scripts/profile_persist4d_allt.py --device cuda:0
python scripts/publish_persist4d_allt.py
```

## 17. One Next-Round Bottleneck

Change only memory-read quality scoring/gating. Keep reducer, losses, K=100, W=2, selection, and evaluation fixed. The diagnostic read restrictions improved T3-T5 slightly, so low-quality occupied-slot reads are the best-supported single hypothesis; the diagnostic is not causal proof.

## 18. GitHub Publication

- Branch: <https://github.com/Orangekostar/Persist4D/tree/research/persist4d-allt-task-superiority-v1>
- Commit: resolve from remote branch HEAD.
- Tracked artifact publication status in this immutable package: `NOT_ATTEMPTED`.
- `PUSH_VERIFIED` may be reported only in the external publication receipt/final response after local HEAD equals remote HEAD and Git readback of this file plus `FINAL_MANIFEST.json` matches local bytes.
