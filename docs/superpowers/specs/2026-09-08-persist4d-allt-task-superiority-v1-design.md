# Persist4D All-T Task Superiority V1 Design

## Authority

The authoritative experiment specification is
`/home/ww/paper5/docs/Persist4D_Codex_AllT_Task_Superiority_V1.md`. This
document resolves implementation choices without weakening its gates.

## Evidence boundary

- Initialize every trainable system from the frozen R1 epoch-390 checkpoint
  with SHA256 `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`.
- Preserve all R1 code, caches, metrics, and checkpoint bytes as read-only
  evidence.
- The only complete H=5 validation population is the six-reference
  Protocol-B population. It remains final evaluation data and is never used
  for model selection, stopping, or configuration decisions.
- The 44 complete H=5 RIO training references are deterministically ordered by
  `sha256("allt-v1:45:" + reference_id)`. The first 8 form the development
  reference holdout and the remaining 36 form adaptation training. All
  sequence orders and shorter prefixes inherit their reference's role.
- The development holdout is exposed to R1 base training. It is useful for
  controlled adaptation selection but is not independent evidence.
- Processed RIO test scans are absent. Therefore
  `independent_generalization=NOT_ESTABLISHED` regardless of Protocol-B gains.

## Model boundary

`Persist4DAllT` extends ReScene while preserving its parameter names and exact
off-path behavior. ReScene receives one protected, parameter-free decoder-stage
hook. The base implementation is identity. The new model overrides it and
applies adapters in the fixed order decoder, optional competition adapter,
optional memory read.

The memory read occurs once per forward pass, after the first complete pass
through `hlevels`, after its FFN, and before the next mask prediction. The hook
receives both the monotonically increasing execution stage and the shared
decoder parameter index so shared decoder weights cannot collapse stage
identity.

The differentiable state is a detached tensor snapshot with memory embeddings
`[B, 100, 128]` and occupancy `[B, 100]`. It is read before the current
prediction and replaced only after the current loss has been formed. The
official state remains `PersistentMemoryState`, is built only from detached
predictions and metadata, and never enters autograd.

`PersistentMemoryRead` uses normalized projected query-memory attention, an
explicit learned abstain token, an all-masked-safe softmax, a sigmoid gate, and
a zero-initialized output projection. Empty memory is an exact identity. Q/K/V
and gate parameters receive finite nonzero initialization; the output
projection is the only zero-initialized transform, allowing gradients to reach
upstream read parameters after two real optimizer updates.

The competition adapter L is gated by the epoch-390 diagnostic. If substantial
current-frame query competition is reproduced, L is the bounded QCL-inspired
current-prediction adapter. Otherwise, if the earliest mask gate excludes
matched ground truth widely, L is first-cross-attention mask relaxation. Only
one L design may be implemented. If neither condition is met, L is
`NOT_JUSTIFIED` and its variants are gate-skipped.

## Episode and loss semantics

An episode has a real H in `{2, 3, 4, 5}` and never crosses a reference or data
role. T1 consumes X1 alone. Each later step consumes only `[X_(t-1), X_t]`.
All scans in an episode share one augmentation draw whose parameters and
normalization statistics are independent of future scans. Temporal coordinates
are regenerated locally for each model input.

No optimizer step occurs inside an episode. Per-stage losses retain the R1
`raw_sum` semantics plus explicitly named new losses. For H >= 2 the episode
objective is the exact prefix-balanced loss

`mean_{T=2..H} (1/T) * sum_{s=1..T} loss_s`.

Equivalently, stage `s` has coefficient
`mean_{T=max(2,s)..H}(1/T)` and the coefficients sum to one. H=1 uses the
unmodified stage loss. Encoder parameters remain frozen and in evaluation
mode. The same augmentation and draw plans are reused across variants.

## Variant matrix

- `FH-R1`: frozen FullHistory baseline.
- `P0`: frozen local-window R1/B4 baseline.
- `C0`: prefix-balanced episode training, L off, M off.
- `C1`: C0 plus L.
- `C2`: C0 plus differentiable persistent memory read M.
- `FH-adapt`: FullHistory episode adaptation under the same loss and draws.
- `C3`: L plus M, only when C1 and C2 show complementary development gains.
- `FH-L`: FullHistory plus L, only when the selected Persist4D candidate uses L.

All formal variants use training seed 45 first. The preflight budget is 4000
optimizer updates, evaluations at 0/1000/2000/3000/4000, 5% warmup, cosine
decay to 1% of the initial rate, inherited non-encoder LR `5e-5`, adapter LR
`1e-4`, and R1 AdamW fields. Effective episode batch is 8. At most two A40s may
be used. One amendment to physical batch, accumulation, or total updates is
allowed after a real unlabeled throughput trial and before the first formal
candidate update; the amended contract is then immutable.

## Selection and evaluation

Development selection uses one checkpoint per variant. The primary key is
`min_T(candidate_t_mAP - baseline_t_mAP)` over T=2,3,4,5, followed by mean
delta and then the preregistered task metrics. Protocol-B never participates.
The selected Persist4D checkpoint and strongest matched ReScene checkpoint are
then evaluated sequentially on all 129 Protocol-B sequence-order units with
evaluation seeds 45, 46, and 47 where required. Seed-46 training confirmation
is run only if seed 45 is promising, and uses identical selection rules.

Mean score reduction is primary; latest and max are sensitivity analyses. The
same generated predictions are reused across reducers and B2/B4 state methods.
Every cache key binds checkpoint, module configuration, data/order/prefix,
evaluation seed, and postprocessing. The final all-T claim passes only if the
single selected checkpoint has strictly positive T=2,3,4,5 t-mAP deltas against
the stated comparator. No average, recovery, or speed result compensates for a
failed horizon.

## Resource and publication boundary

Checkpoints, caches, and temporary predictions live under
`/mnt/shared/ww/persist4d-allt-task-superiority-v1/`; Git contains only source,
small contracts, tables, manifests, and reports. Training jobs use only free
GPU devices and release their CUDA contexts when complete. Final publication
requires a clean scoped diff, relevant tests, manifest hash verification,
remote branch push, remote SHA equality, and remote readback of `HANDOFF.md`
and `FINAL_MANIFEST.json`.
