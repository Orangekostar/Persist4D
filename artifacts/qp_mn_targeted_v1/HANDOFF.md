# Q-P / M-N handoff

Experiment commit A: `d9644d01d7e88ff9a74dff975f4f158d5deddd79`. Branch: `research/rescene-qp-mn-targeted-v1`. New delivery tag: `rescene-qp-mn-targeted-v1`.

Repaired base: `465f37a46972e81584c1563bba57e1c05c911c1a`. Runtime: `/home/ww/persist4d_runs/qp_mn_targeted_v1`.

Read `FINAL_REPORT.md`, `REQUIREMENT_REVIEW.md`, and `resources/NATIVE_RUNTIME_NOTE.md`. All five selected single-module bundles are under `training/<arm>/seed45/deployment.pt`; all 22 inference checkpoints are under `training/<arm>/seed<seed>/checkpoints/`. They require the separately held original R1, Concerto and legal dataset assets. Neither raw parent weights, raw data nor raw GT are redistributed.

Confirmed development modules: []. Provisional engineering candidates: ['M_C']. Q-P and M-N do not beat B0. M-C is positive in both tested seeds but fails the seed45 reference-consistency gate. Keep B0 as the default; no combinations or formal test claims.

Commands (existing persist4d environment):

```bash
python -m scripts.qp_mn_campaign status
python -m scripts.qp_mn_campaign run --base-commit 465f37a --resume
python -m scripts.qp_mn_campaign report
python -m scripts.qp_mn_campaign publish --resume
```

The artifact manifest excludes itself and contains no self-referential B hash. The external `publication/PUBLICATION_RECEIPT.json` under the runtime root records actual A/B/tag and byte-verified remote delivery after the push. Git carries all required small assets; no Release is required.
