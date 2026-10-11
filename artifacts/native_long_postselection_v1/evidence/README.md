# Evidence and measurement units

The selection tables preserve every CAL endpoint and the six locked SEL rows.
`FORMAL_AUDIT.json` verifies four fixed checkpoint rows across the three formal
populations: 3,560 inputs, 596 official states, and 32 independent pooled metric
recomputations. AP/REC and deltas in machine-readable JSON/CSV are fractions in
[0, 1]. A displayed percentage is multiplied by 100; a percentage-point delta
is a fractional difference multiplied by 100.
`FORMAL_METRICS.csv` separates `role_total_inputs` from the actual per-horizon
`metric_inputs`. The latter is empty for S_long, which averages three separately
pooled horizon AP values and has no single pooled input denominator.
`../FORMAL_BY_REFERENCE.csv` preserves all 564 physical-reference/horizon rows
with their actual completed/expected denominators. Missing AP remains missing.

`RUNTIME_AUDIT.json` records the complete 12-job deployment queue, including
fixed-CAL profiles and repeated fresh-process A versus B-then-A probes. Its
profile network/IO/transfer/postprocess components and runtime-order differences
must retain their respective scopes. The raw prediction tensors remain in the
runtime storage; they are not included in this public statistics package.
`../PROFILE.csv` contains 20 horizon/checkpoint summaries; `../PROFILE_RAW.csv`
retains all 80 observations including the 20 explicitly labelled warmups.
`../PROFILE_SETUP.csv` records each of the four process setups once. Timing
columns use seconds and memory columns use bytes.

`training_logs.zip` preserves every collected logger version for all four
training arms, together with resolved configurations and sampled diagnostics.
`TRAINING_LOG_MANIFEST.json` binds each uncompressed member and the complete
archive. `TRAINING_LOG_COLLECTION_AUDIT.json` verifies all 109 files against
the remote source bytes and confirms the trained numerical sources remain
unchanged. All four trajectories completed 29,700 optimizer updates.

`../TRAINING_LOSS.csv` extracts logged total step and epoch loss without
deduplicating or smoothing attempts. It preserves the logger version and row
index. The recorded Lightning `step` is not an independently inferred update
count; a final logged step of 29,699 is compatible with the verified checkpoint
at optimizer update 29,700. Empty CSV fields represent unreported observations.
An epoch loss and a step loss are different summaries and should be plotted
separately. Repeated-step trace differences remain visible in
`TRAINING_MECHANISM_TRACE_AUDIT.json`.

The public package contains configurations, scalar diagnostics and official
metric sufficient states. It excludes raw scans, per-point ground-truth tensors,
optimizer checkpoints and the separately licensed pretrained encoder.

`MECHANISM_AUDIT.json` independently verifies all 152 fixed CAL jobs, 75 pooled
and by-reference official states, all 30 feedback pairs, six sampling cases,
30 nested trajectories and two trained gradient probes. The seven mechanism
CSV files preserve all measurements. Candidate IoU events and raw query-slot
changes are distinct from official AP and persistent tracking ID errors.

Two original feedback pairs have upstream numerical differences. All 30 remain
in the metrics; only 28 satisfy strict upstream alignment. `MECHANISM_REPLAY_AUDIT.json`
verifies the original four runs plus eight bounded repeats on those two inputs,
retaining all 30 pairwise comparisons. Same-mode comparisons show tensor and
official-state differences despite equal scalar metrics on these two inputs.
`FRESH_PROCESS_RUNTIME_LIMITATION.json` supersedes any generalization of the
earlier single-input repeatability check. The exact upstream operator remains
unlocalized; no byte tolerance or replacement of original outcomes is used.
In the pair audit, `computed_feedback_exact` compares the recorded capture fields
(input hashes, norms and counts), not a saved complete feedback-update tensor.
`attribution_eligible` means these measured upstream conditions passed; it does
not certify every later operation's numerical repeatability.

Both trained-memory checkpoints completed 129 units and 1,032 forwards each.
`BRIDGE_AUDIT.json` verifies 184 official states and independently recomputes 40
pooled/current-frame metrics. Six MEMORY CSV files preserve task/identity rates,
actual denominators, paired reference intervals, resources and capacity.
The official-state archive contains 1,093 states and uses Git LFS; run `git lfs pull`
before opening it. Measurement receipts (444 JSON files) and training logs (109
files) are separate archives. All archives have member and complete-byte SHA-256
checks. `COST_AUDIT.json` records final accounting with zero active reservations.

A pooled formal AP can be recomputed without point
clouds or model weights. Use the bound stmetrics version and the CPU environment:

```python
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.native_long_confirmation_audit import recompute
from scripts.native_long_evidence_package import validate_state

assets = Path("artifacts/native_long_postselection_v1")
with ZipFile(assets / "evidence/official_metric_states.zip") as archive:
    state = json.loads(archive.read(
        "formal/E1/evaluation/E1/seed45/PB-23760-T5-official-state.json"
    ))
validate_state(state)
print(recompute(state, assets / "deployment/rio.yaml"))
```

The statistic archive excludes the prediction/target tensors used in the private
mechanism audits. Raw-data inference reproduction requires the original licensed
datasets; metric reconstruction uses only the published sufficient statistics.
