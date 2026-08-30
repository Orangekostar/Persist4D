# Final Sonata Second-Perception Report

1. Branch / start commit / generation commit
   - `research/persist4d-sonata-second-perception-v1` / `e5d7f4e96fedc76c0c6d414ab293f54909c61df3` / `86eaa74369008c0d1be837f5b4aebeab90637a38`.
2. Changed files
   - `53` tracked paths changed since the captured start commit; core additions are Sonata provenance, preflight, smoke/training evidence, SS5 evaluator/tests, and final synthesis artifacts.
3. External evidence re-verified
   - ReScene4D `fb2fe42eb8f1e926567c48eea9acb874e608ee10` still declared checkpoints `Coming soon`; no official task checkpoint was substituted.
4. Sonata source revision
   - Code `18c09ff8d713494f78a8213792262b910977a65d`; weight repository `df99897472c09f91ba9288da0a034aacffc0b010`.
5. Sonata pretrained weight
   - Immutable revision `df99897472c09f91ba9288da0a034aacffc0b010`, SHA256 `c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50`, `434008287` bytes, license `CC-BY-NC-4.0`.
6. Load-key audit
   - `453` encoder keys loaded; `248` allowlisted decoder keys missing; `0` unexpected keys.
7. Resolved Sonata ReScene config
   - Sonata PTv3, T2, 100 non-parametric queries, mixed ST serialization, temporal masking ON, contrastive OFF, EOS 0.2, frozen encoder, AdamW/OneCycle, max LR 5e-4.
8. Data / split / mix provenance
   - 3RScan T2 + ScanNet T1 mixed training at 1.0:0.8; official-like qualification uses all 154 filtered 3RScan T2 validation sequences.
9. Hardware / batch
   - Two NVIDIA A40 GPUs; physical batch `2` per GPU, accumulation `8`, effective batch `32`.
10. Smoke + gradient contract
   - SSMOKE-PASS; query interface compatible; frozen encoder gradients absent; all trainable decoder/head gradients finite and nonzero.
11. Formal training
   - Seed `45`, `450` epochs, `29700` optimizer steps, `3` interruptions / `3` resumes.
12. Selected checkpoint
   - Epoch `449`, SHA256 `3d6432711dd9639d9e9203134d846d9a1a29f09b7fb3fbb85375e2127945a199`, highest local val_mean_t-AP `0.2430925965309143`; no Protocol-B selection leakage.
13. Official-like Sonata local results
   - 45: t-mAP=0.241607, overall=0.315465; 46: t-mAP=0.242057, overall=0.323937; 47: t-mAP=0.237392, overall=0.307206. Mean t-mAP `0.240352`, overall `0.315536`.
14. Matched current-Concerto results
   - Mean t-mAP `0.282901`, overall `0.369794` under the same seeds/runtime.
15. SQ gate
   - `SQ-RED`: Sonata is weaker than Concerto on both temporal and spatial qualification metrics and is below the 0.297 t-mAP threshold.
16. Conditional SQ-GREEN evidence
   - Not applicable. SS6/SS7 were not authorized; no Sonata Protocol-B, invariance, robustness, reducer, or compute values were generated.
17. Concerto-vs-Sonata synthesis
   - Frozen Concerto V3 remains positive, but this Sonata checkpoint did not qualify. The persistent-state benefit is not cross-backbone validated by this experiment.
18. Tests / lint / diff-check
   - `95` Sonata tests and `57` frozen V3 regressions passed; task-owned Python files pass Ruff and `git diff --check` passes. The same `36` Ruff findings remain in `trainer/trainer.py` at both the captured start commit and this revision.
19. Artifact and checkpoint hashes
   - Upstream/output hashes are enumerated in `FINAL_MANIFEST.json`; checkpoint hashes are in `checkpoint/CHECKPOINT_MANIFEST.json`.
20. Remaining external-asset failures
   - Official ReScene4D-S/C task checkpoints remain unavailable; no missing local asset blocked SS0-SS5.
21. SR gate
   - `SR-RED`, derived from the failed SQ gate; robustness was gate-skipped rather than measured as zero.
22. Claims now authorized
   - A provenance-locked 450-epoch Sonata local task reimplementation was trained and negatively qualified against matched Concerto evidence.
23. Claims still forbidden
   - Official 33.2 reproduction, Sonata Protocol-B performance, cross-backbone persistence benefit, backbone universality, and SOTA.
24. Recommended next research stage
   - Diagnose the local Sonata task-learning gap under a separately preregistered training study; do not tune Persist4D/B4 on these final qualification results.
