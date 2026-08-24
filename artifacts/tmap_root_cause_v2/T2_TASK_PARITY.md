# T2 Official-Task Current-Stage Parity

- Status: `pass`
- Source commit: `eeda7717904a272244c9f3b3629428c0d05e3fa5`
- Scope: `43 masters x 3 preregistered orders x T2 = 129 units`
- Units: `129`
- Passed: `129`
- Failed: `0`
- Maximum score absolute difference: `0.0`
- Maximum per-unit AP absolute difference: `0.0`
- Aggregate FullHistory current-stage AP: `0.37503349781036377`
- Aggregate local-window current-stage AP: `0.37503349781036377`
- Aggregate AP absolute difference: `0.0`
- Per-unit rows SHA256: `aa28f573fb63ebec2422b4ae4335014997830e8f353466950c7f6cfbb08eaaac`
- Sidecar manifest SHA256: `08b99ecf5d5e0083015f220ee2ac72032efe0a23f8539e641dce8c82401d02d7`
- Sidecar storage: `external:system_comparison_v2/task_sidecars/entries`
- Matched raw storage: `external:system_comparison_v2/raw_predictions/entries`
- Frozen V1 local raw bytewise replays: `0/129`
- Frozen V1 FullHistory bytewise replays: `0/129`

## Gate Semantics

Each local task prediction is produced from the same forward pass as its
content-addressed V2 raw observation. FullHistory and Local are replayed in
the same deterministic process, and the existing T2 observation fingerprint
regression must pass before task comparison. The gate then requires exact
candidate count, masks, and classes; scores use `rtol=0, atol=1e-7`; official
raw-local AP uses the same current-stage target on both sides.

Frozen V1 content hashes are retained as diagnostics rather than replay gates:
the original audit established same-process repeatability, not cross-process
bitwise replay. This gate tests evaluator/input parity only. It does not
establish causal t-mAP parity or tune task post-processing/memory behavior.
