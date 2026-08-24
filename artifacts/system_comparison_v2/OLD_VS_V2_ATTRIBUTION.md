# Old vs System Comparison V2 Attribution

- Status: `pass`
- V2 task candidates: official local ReScene candidates
- Identity linkage: unchanged B4 raw-query tracker
- Persistent trajectory key: `(track_id, official_class_id)`
- Unmatched candidates: retained with stage-local ephemeral keys
- Primary score reducer: mean official per-stage candidate score

| Horizon | FullHistory t-mAP | Legacy Persist4D t-mAP | V2 Persist4D t-mAP | V2 - legacy | Legacy current AP | V2 current AP |
|---:|---:|---:|---:|---:|---:|---:|
| T2 | 0.190996 | 0.158636 | 0.207241 | +0.048605 | 0.290030 | 0.361744 |
| T3 | 0.107897 | 0.098894 | 0.123102 | +0.024208 | 0.300169 | 0.369052 |
| T4 | 0.068999 | 0.059570 | 0.070232 | +0.010662 | 0.299144 | 0.358928 |
| T5 | 0.045340 | 0.044497 | 0.052503 | +0.008006 | 0.309149 | 0.371405 |

The V2-minus-legacy differences isolate a task-prediction semantics change
while keeping the registered B4 tracker algorithm and identity reporting
path unchanged. They are not an additive causal decomposition of the total
FullHistory gap. Frozen V1 identity fields are copied only after exact
keyed regression and therefore remain byte-for-byte numerically unchanged.
