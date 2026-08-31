# ReScene Root-Cause Short-Curve Decision

Status: `authorized`
Selected full candidate: `R1`

Primary metric: epoch-90 mean `SpatialStageMean`.
Persist4D metrics were not used.

## Gates

### R1

- `positive_for_all_paired_seeds`: `PASS`
- `mean_spatial_gain_at_least_one_point`: `PASS`
- `overall_map_not_lower_than_r0`: `PASS`
- `leads_validation_at_75_and_90`: `PASS`
- `contract_integrity`: `PASS`

All gates: `PASS`

## Runtime-Infeasible Variants

- `R2`: `full_dataset_cuda_oom`
- `R4`: `deterministic_nonfinite_objective`
