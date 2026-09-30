# Native Long Retrain Result

Scientific status: **PARTIAL_FULL_RETRAIN_BUDGET**. Publication: **CODE_ONLY**. No trained accuracy result or module gain is available.

| Arm | Actual optimizer updates | Status | T1-T5 official AP | Net gain |
|---|---:|---|---|---|
| E0 | 0 / 29700 | NOT_RUN_BUDGET | Not evaluated | Not measured |
| E1 | 0 / 29700 | NOT_RUN_BUDGET | Not evaluated | Not measured |
| E2 | 0 / 29700 | NOT_RUN_BUDGET | Not evaluated | Not measured |
| E3 | 0 / 29700 | NOT_RUN_BUDGET | Not evaluated | Not measured |

All four share a new seed45 task initialization with the fixed pretrained embedding/encoder. E0 uses short T2; E1 long curriculum; E2 long+S; E3 long+F. No combination or old task warm start. Temporary optimizer updates only measure feasibility/cost and do not count toward these trajectories.

Budget mode: **BASELINE_RECOVERY_ONLY**, authorized full arms `[]`. Prior 80.735038750 GPUh; this campaign 0.218209667 GPUh; lifetime 80.953248417/192 GPUh; current remainder 111.046751583 GPUh. At least 8 GPUh is reserved.

Full forecasts include the single 1.25 training margin and separate CAL proxy:

- E0: 379.97 GPUh.
- E1: 400.31 GPUh.
- E2: 410.49 GPUh.
- E3: 416.37 GPUh.

Even E0 plus the reserve exceeds the available lifetime budget, so section10 forbids starting it. Forecasts are finite measurements with approximate point-count scaling, not promised runtimes.

Recommended configuration from single-A40 preflight: two A40, batch1/rank, accumulation16, effective batch32, 0 workers, FP32/TF32 off, voxel0.02, Q100, U29700, AdamW5e-4 and fixed full OneCycle. Batch2 remained OOM after exact F checkpointing; largest T5 batch1 used 38.58 GiB allocated at peak. Full native DDP has not been exercised.

Population is TRAIN361/CAL8/SEL8. Complete expected input counts T1..T5 are CAL `[31, 19, 14, 14, 6]`, SEL `[39, 21, 21, 15, 15]`. All completed counts are zero for main evaluation; see evaluation/COVERAGE.json. No seed46 pair, formal lock, PB/LOCAL/ADDITIONAL accuracy, trained nested-prefix diagnosis or deployment profile exists.

Real development evidence: corrected mixed-H FULL inverse-map slicing; exact common tensors; F initial identity and late output changes; zero first qkv gradient then nonzero second gradient; S fixed-cap stage quotas; raw_sum scalar/gradient equality; actual largest backward and IO timing. Native status remains RUNTIME_CONDITIONAL. The initialized TRAIN A diagnostic has zero AP in all orders while BA changes candidate lineage; this is not evidence of a trained-model gain.

New common state SHA: `8d3dfbc4e5c78d707ae5d6cb913123727456698c06d7c1e5077d944772c423e1`. R1 is historical-only and was not reevaluated as a common-runtime bridge, so no historical AP comparison is claimed.

Git includes the F initialization state and compressed official initialized TRAIN-A sufficient statistics. No trained model was generated. The genuine nonencoder initialization task package is local-only; Release authorization was unavailable at the single startup check. Full common state and raw traces remain local. No raw data/GT or original pretrained checkpoint is redistributed. See HANDOFF.md, ARTIFACT_MANIFEST.json and the external publication receipt for exact delivery.
