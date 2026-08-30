# ReScene Task-Learning Root-Cause V1 Start State

Status: `PASS`

## Source

- Task parent: `29bb228ad5b090797045fa3b3fc55cb973f001be` on `research/persist4d-sonata-second-perception-v1`.
- Task branch: `research/persist4d-rescene-task-learning-root-cause-v1`.
- Official ReScene source: `fb2fe42eb8f1e926567c48eea9acb874e608ee10`.
- Official task checkpoints remain unavailable in the pinned repository README.

## Runtime

The registered environment is Python 3.10.20, PyTorch 2.6.0+cu126, CUDA 12.6, PyTorch Lightning 2.6.5, and Hydra 1.3.4. Three NVIDIA A40 devices are visible. Training topology remains two devices unless a preregistered physical-batch diagnostic authorizes a dedicated variant.

## Baseline Validation

The first complete suite exposed one missing repository-ignored reviewer-closure cache binding: 2,008 tests passed and one cache-backed test failed. The frozen replay and sidecar manifests were then bound to 516 entries each, all entry hashes were verified, and the failed test passed in isolation. The complete suite was rerun without source changes and passed with 2,009 tests passed, 11 skipped, and 94 warnings in 1,402.91 seconds. The external payload remains outside Git; only its committed manifest hashes, counts, and byte sizes are recorded in `START_STATE.json`.

## Inputs

The Concerto pretrained encoder is bound to SHA-256 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`. The existing Concerto and Sonata task reimplementations are bound to `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e` and `3d6432711dd9639d9e9203134d846d9a1a29f09b7fb3fbb85375e2127945a199` respectively. These task checkpoints are evidence inputs only; root-cause curves must start from one new common pretrained-encoder initialization.

The active filtered mix contains 1,174 RIO T2 sequences and 1,199 ScanNet T1 scans. Its replacement sampler draws 2,112 examples per epoch. The RIO T2 sequence database SHA-256 is `974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416`.

## Baselines

| Model | Evidence | t-mAP | overall mAP | stage1 mAP | stage2 mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| ReScene4D-C | paper-reported | 0.348 | 0.433 | not used | not used |
| Concerto reimplementation | measured local three-seed mean | 0.2829008499781291 | 0.3697936435540517 | 0.4203239579995473 | 0.43060773611068726 |
| ReScene4D-S | paper-reported | 0.332 | 0.409 | not used | not used |
| Sonata reimplementation | measured local three-seed mean | 0.24035188059012094 | 0.31553595264752704 | 0.37665464480717975 | 0.37463711698849994 |

Paper rows are external references. Local rows come from `repo:artifacts/sonata_second_perception_v1/checkpoint/official_like_summary.csv` and are not official checkpoints.

## Frozen Evidence

The task must not modify `artifacts/P2`, `artifacts/P6A`, `artifacts/system_comparison`, `artifacts/system_comparison_v2`, `artifacts/reviewer_closure_v3`, or `artifacts/sonata_second_perception_v1`. Full file and input hashes are in `START_STATE.json`.

Model and checkpoint selection is local-perception only. Protocol-B, B2/B3/B4, gap, identity, persistent-memory, latency, and VRAM outcomes are prohibited selection signals.
